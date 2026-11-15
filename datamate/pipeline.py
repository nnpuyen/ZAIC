from __future__ import annotations

import json
import math
import re
import tempfile
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import duckdb

from .execution import execute_plan_with_trace
from .ingestion.loader import load_business_data, repair_mojibake
from .context import build_context
from .planning.schemas import ParseResult, QueryPlan
from .planning.plan_extraction import extract_query_plan
from .numeric import number_sql_expression


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")


def _iter_questions(path: str | Path):
    """Read JSONL incrementally and preserve malformed records as failures."""
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError("record must be a JSON object")
                if not all(key in item for key in ("id", "business_id", "question", "as_of_date")):
                    raise ValueError("record must contain id, business_id, question, and as_of_date")
                item["question"] = repair_mojibake(str(item["question"]))
                yield item
            except (json.JSONDecodeError, ValueError, TypeError) as error:
                yield {"id": None, "_input_error": f"line {line_number}: {error}"}


def _write_answer_records(path: Path, records: list[dict[str, Any]]) -> None:
    if path.suffix.lower() == ".json":
        content = json.dumps(records, ensure_ascii=False, indent=2, default=str) + "\n"
    else:
        content = "\n".join(
            json.dumps(item, ensure_ascii=False, default=str) for item in records
        ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f"{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(content)
        temporary_path = Path(handle.name)

    try:
        for attempt in range(5):
            try:
                temporary_path.replace(path)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * (2**attempt))
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass


class _QuestionStatusError(ValueError):
    """Carry an explicit parser status through the per-question error path."""

    def __init__(self, answer_type: str, reason: str) -> None:
        super().__init__(reason)
        self.answer_type = answer_type


def _safe_directory_name(value: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", value).strip(" .")
    return name or "question"


def _is_integer_type(column_type: str) -> bool:
    """Return whether a DuckDB type represents an integer-valued column."""
    normalized = re.sub(r"\s+", " ", str(column_type).upper()).strip()
    return normalized.startswith((
        "TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT",
        "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT",
    ))


def _answer_decimal_places(plan: QueryPlan, database: Any | None) -> int:
    """Choose answer precision from the source column's actual values."""
    metric = plan.metric

    # The final value of a percentage/ratio is a computed DOUBLE, even when
    # its internal numerator and denominator happen to be counts.  Do this
    # before the count check: percentage plans commonly use count_distinct as
    # their underlying aggregation.
    if (
        plan.answer_type == "percent"
        or metric is not None
        and (
            metric.aggregation in {"percentage", "ratio"}
            or metric.is_percent_of_total
        )
    ):
        return 2

    # Counts are inherently integer-valued, including count(*) and distinct
    # counts, when they are the final answer rather than percentage inputs.
    if metric is not None and metric.aggregation in {"count", "count_distinct"}:
        return 0
    if database is None or metric is None or metric.column == "*":
        return 2
    metadata = getattr(database, "connection", None)
    if metadata is None:
        return 2
    try:
        table = str(plan.table).replace('"', '""')
        column = str(metric.column).rsplit(".", 1)[-1].replace('"', '""')
        column_type = metadata.execute(
            f'DESCRIBE SELECT "{column}" FROM "{table}"'
        ).fetchone()[1]
        # Pandas/DuckDB may promote integer-looking source data to DOUBLE.
        # Inspect the stored values as well, so formatting follows the data's
        # effective precision rather than only the physical column type.
        parsed_number = number_sql_expression(f'"{column}"')
        fractional_count = metadata.execute(
            f'''SELECT COUNT(*)
                FROM "{table}"
                WHERE {parsed_number} IS NOT NULL
                  AND {parsed_number} <> FLOOR({parsed_number})'''
        ).fetchone()[0]
        row_count = metadata.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
    except Exception:
        # Formatting must never make an otherwise valid answer fail.
        return 2
    # An empty DOUBLE column gives us no value evidence; in that case retain
    # the physical type instead of treating the absence of fractions as proof
    # that the metric is integer-valued.
    return 0 if _is_integer_type(column_type) or (row_count > 0 and fractional_count == 0) else 2


def _format_answer(
    answer_type: str,
    rows: list[dict[str, Any]],
    plan: QueryPlan,
    database: Any | None = None,
) -> Any:
    """Convert execution rows to the canonical shape for the declared type."""
    if answer_type == "list":
        if not rows:
            raise ValueError("Query returned no rows for a list answer; review the source and filters")

        columns = list(rows[0])
        if plan.group_by_period:
            value_column = "group_period"
        elif plan.group_by:
            value_column = plan.group_by[0]
        elif plan.select_columns:
            value_column = plan.select_columns[0]
        elif plan.metric is not None:
            value_column = plan.metric.alias or "metric"
        elif plan.metrics:
            value_column = plan.metrics[0].alias or "metric"
        elif len(columns) == 1:
            value_column = columns[0]
        else:
            raise ValueError("A list answer must resolve to one output column")

        if value_column not in columns and len(columns) == 1:
            value_column = columns[0]
        if value_column not in columns:
            raise ValueError(
                f"List answer column {value_column!r} is missing from the query result"
            )
        values = [row[value_column] for row in rows]
        return [
            value.isoformat() if isinstance(value, (date, datetime)) else _finite_or_none(value)
            for value in values
        ]

    if answer_type == "table":
        if not rows:
            raise ValueError("Query returned no rows for a table answer; review the source and filters")
        columns = list(rows[0]) if rows else _planned_result_columns(plan)
        return {
            "columns": columns,
            "rows": [
                [_finite_or_none(row.get(column)) for column in columns]
                for row in rows
            ],
        }

    if answer_type not in {"number", "percent", "string", "date", "boolean"}:
        raise ValueError(f"Unsupported answer type: {answer_type}")
    if not rows:
        raise ValueError(
            f"Query returned no rows for a {answer_type} answer; review the source and date filters"
        )
    if len(rows) != 1:
        raise ValueError(
            f"Answer type {answer_type!r} requires one value, but the query returned {len(rows)} rows"
        )

    row = rows[0]
    if answer_type == "string":
        # Ranked name queries often return both the requested name and the
        # metric used to rank it. Project the declared target column only.
        target_columns = [
            *(["group_period"] if plan.group_by_period else []),
            *plan.group_by,
            *plan.select_columns,
        ]
        target_column = next((column for column in target_columns if column in row), None)
        if target_column is None:
            value_columns = [name for name in row if name != "metric"]
            if len(value_columns) == 1:
                target_column = value_columns[0]
            elif set(row) == {"metric"}:
                target_column = "metric"
        if target_column is None:
            raise ValueError(
                "A string answer must resolve to one requested output column"
            )
        value = row[target_column]
        if isinstance(value, (dict, list, tuple, set)):
            raise ValueError("A string answer must contain one scalar value")
        if value is None:
            raise ValueError(
                "Query returned NULL for a string answer; review the source and filters"
            )
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        return str(value)

    value_column = None
    if answer_type in {"number", "percent", "boolean"}:
        metric = plan.metric or (plan.metrics[0] if len(plan.metrics) == 1 else None)
        if metric is not None:
            expected_alias = metric.alias or "metric"
            if expected_alias in row:
                value_column = expected_alias
    elif answer_type == "date":
        target_columns = [
            *(["group_period"] if plan.group_by_period else []),
            *plan.group_by,
            *plan.select_columns,
        ]
        value_column = next((column for column in target_columns if column in row), None)
        if value_column is None and plan.metric is not None:
            expected_alias = plan.metric.alias or "metric"
            if expected_alias in row:
                value_column = expected_alias

    if value_column is None:
        if len(row) != 1:
            raise ValueError(
                f"Answer type {answer_type!r} requires one output value, but the query returned columns {list(row)}"
            )
        value_column = next(iter(row))
    value = row[value_column]
    if isinstance(value, (dict, list, tuple, set)):
        raise ValueError(f"Answer type {answer_type!r} must contain one scalar value")
    if value is None:
        raise ValueError(
            f"Query returned NULL for a {answer_type} answer; review the source and filters"
        )
    if answer_type in {"number", "percent"} and _is_non_finite_number(value):
        raise ValueError(
            f"Query returned non-finite numeric value {value!r}; review the source and filters"
        )
    if answer_type in {"number", "percent"} and value is not None and (
        not isinstance(value, (int, float)) or isinstance(value, bool)
    ):
        raise ValueError(f"Answer type {answer_type!r} requires a numeric query result")
    if answer_type == "boolean" and value is not None and not isinstance(value, bool):
        raise ValueError("Answer type 'boolean' requires a boolean query result")
    if answer_type in {"number", "percent"} and isinstance(value, (int, float)) and not isinstance(value, bool):
        places = _answer_decimal_places(plan, database)
        return int(round(value)) if places == 0 else round(value, 2)
    if answer_type == "date":
        if isinstance(value, datetime):
            return value.date().isoformat()
        if isinstance(value, date):
            return value.isoformat()
        if hasattr(value, "isoformat"):
            return value.isoformat()
    return value


def _finite_or_none(value: Any) -> Any:
    """Reject non-finite values so they cannot silently become JSON null."""
    if _is_non_finite_number(value):
        raise ValueError(
            f"Query returned non-finite numeric value {value!r}; review the source and filters"
        )
    return value


def _is_non_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return not math.isfinite(value)
    except (TypeError, ValueError, OverflowError):
        return False


def _planned_result_columns(plan: QueryPlan) -> list[str]:
    """Infer result column labels for an empty grouped or selected result."""
    columns = (
        ["group_period"]
        if plan.group_by_period
        else list(plan.group_by)
    )
    metrics = ([plan.metric] if plan.metric is not None else []) + list(plan.metrics)
    if metrics:
        columns.extend(
            metric.alias or ("metric" if index == 0 else f"metric_{index}")
            for index, metric in enumerate(metrics)
        )
    elif plan.select_columns:
        columns.extend(plan.select_columns)
    return columns


def _write_question_evidence(
    evidence_root: Path,
    question: dict[str, Any],
    database: Any,
    context: dict[str, Any],
    plan: QueryPlan | None = None,
    query_trace: list[dict[str, Any]] | None = None,
    rows: list[dict[str, Any]] | None = None,
    answer: Any = None,
    answer_type: str | None = None,
    error: str | None = None,
    parser: Any | None = None,
    token_usage: dict[str, int] | None = None,
) -> None:
    question_dir = evidence_root / _safe_directory_name(str(question["id"]))
    evidence_dir = question_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    _write_json(evidence_dir / "01_question.json", question)
    _write_json(
        evidence_dir / "02_loaded_data.json",
        {
            "tables": {
                name: {
                    "source": metadata.source,
                    "sheet": metadata.sheet,
                    "rows": metadata.rows,
                    "columns": metadata.columns,
                    "raw_columns": metadata.raw_columns,
                    "column_mappings": metadata.column_mappings,
                }
                for name, metadata in database.tables.items()
            },
            "load_errors": database.errors,
        },
    )
    evidence_context = (
        getattr(parser, "last_retrieved_context", None)
        if parser is not None
        else None
    )
    _write_json(
        evidence_dir / "03_context.json",
        evidence_context if evidence_context is not None else context,
    )
    if parser is not None and getattr(parser, "last_response_content", None) is not None:
        (evidence_dir / "04_planner_response.txt").write_text(
            str(parser.last_response_content), encoding="utf-8"
        )
    if parser is not None and getattr(parser, "last_raw_plan", None) is not None:
        _write_json(evidence_dir / "05_planner_plan.json", parser.last_raw_plan)
    if plan is not None:
        _write_json(evidence_dir / "06_validated_plan.json", plan.model_dump(mode="json"))
    if query_trace is not None:
        _write_json(evidence_dir / "07_query_trace.json", query_trace)
        sql_index = 0
        for query in query_trace:
            if "sql" not in query:
                continue
            sql_index += 1
            query_path = evidence_dir / f"07_query_{sql_index}_{query['kind']}.sql"
            query_path.write_text(
                f"-- params: {json.dumps(query['params'], ensure_ascii=False, default=str)}\n"
                f"{query['sql']}\n",
                encoding="utf-8",
            )
    if rows is not None:
        _write_json(evidence_dir / "08_query_result.json", rows)
    if error is not None:
        _write_json(evidence_dir / "09_error.json", {"error": error})
        _write_json(
            question_dir / "answer.json",
            {
                "id": question["id"],
                "type": answer_type or "abstain",
                "reason": error,
            },
        )
    else:
        (evidence_dir / "09_error.json").unlink(missing_ok=True)
        answer_record = {
            "id": question["id"],
            "type": answer_type,
            "answer": answer,
        }
        _write_json(question_dir / "answer.json", answer_record)
    _write_json(
        question_dir / "token_usage.json",
        {
            "question_id": question["id"],
            **(
                token_usage
                or {
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                }
            ),
        },
    )


def answer_questions(
    questions_path: str | Path,
    data_root: str | Path,
    output_path: str | Path,
    parser: Any | None = None,
    schema_output_path: str | Path | None = None,
    trace_output_path: str | Path | None = None,
    rulebook_path: str | Path | None = None,
    calendar_path: str | Path | None = None,
    evidence_output_dir: str | Path | None = None,
    token_usage_output_path: str | Path | None = None,
) -> None:
    """Run a parser/executor pair and write competition-compatible JSONL answers.

    ``parser`` must expose ``parse(question, database, as_of_date, context=...)``
    and return a ``QueryPlan`` or a dictionary accepted by ``QueryPlan``.
    Model inference remains replaceable while execution stays deterministic.
    """
    output_path = Path(output_path)
    output: list[dict[str, Any]] = []

    def checkpoint_answers() -> None:
        _write_answer_records(output_path, output)

    schemas: dict[str, Any] = {}
    traces: list[dict[str, Any]] = []
    databases: dict[str, Any] = {}
    contexts: dict[str, dict[str, Any]] = {}
    evidence_root = (
        Path(evidence_output_dir)
        if evidence_output_dir is not None
        else Path(output_path).with_name(Path(output_path).stem + "_questions")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    token_usage_path = Path(token_usage_output_path or Path(output_path).parent / "token_usage.jsonl")
    token_usage_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        for question in _iter_questions(questions_path):
            stale_completed_plan = (
                evidence_root
                / _safe_directory_name(str(question.get("id")))
                / "evidence"
                / "06_completed_plan.json"
            )
            stale_completed_plan.unlink(missing_ok=True)
            business_id = question.get("business_id")
            database = None
            plan = None
            rows = None
            query_trace = None
            try:
                if question.get("_input_error"):
                    raise ValueError(question["_input_error"])
                if business_id not in databases:
                    database = load_business_data(Path(data_root) / str(business_id))
                    if database.errors:
                        errors = "; ".join(
                            f"{item['source']}: {item['error']}" for item in database.errors
                        )
                        database.close()
                        database = None
                        raise ValueError(f"Business data could not be fully loaded: {errors}")
                    databases[business_id] = database
                    contexts[business_id] = build_context(
                        database,
                        Path(data_root) / str(business_id),
                        rulebook_path=rulebook_path,
                        calendar_path=calendar_path,
                    )
                    schemas[business_id] = contexts[business_id]
                database = databases[business_id]
                if not database.tables:
                    raise ValueError(f"No supported business data found for {business_id!r}")
                if parser is None:
                    raise ValueError("No question parser configured; supply an offline parser")
                if hasattr(parser, "parse_result"):
                    parse_result = parser.parse_result(
                        question["question"], database, question["as_of_date"]
                    )
                    if isinstance(parse_result, ParseResult):
                        if parse_result.status != "answerable":
                            raise _QuestionStatusError(
                                parse_result.status,
                                parse_result.reason or parse_result.status,
                            )
                        if parse_result.plan is None:
                            raise _QuestionStatusError(
                                "abstain",
                                parse_result.reason or "Parser returned no plan",
                            )
                        raw_plan = parse_result.plan
                    else:
                        raw_plan = parse_result
                else:
                    raw_plan = parser.parse(
                        question["question"],
                        database,
                        question["as_of_date"],
                        context=contexts[business_id],
                    )
                plan = extract_query_plan(raw_plan)
                rows, query_trace = execute_plan_with_trace(plan, database)
                answer_type = plan.answer_type
                answer = _format_answer(answer_type, rows, plan, database)
                output.append({"id": question["id"], "type": answer_type, "answer": answer})
                _write_question_evidence(
                    evidence_root, question, database, contexts[business_id],
                    plan=plan,
                    query_trace=query_trace, rows=rows, answer=answer,
                    answer_type=answer_type,
                    parser=parser,
                    token_usage=getattr(parser, "last_token_usage", None),
                )
                _append_jsonl(
                    token_usage_path,
                    {
                        "id": question.get("id"),
                        "business_id": business_id,
                        "usage": getattr(parser, "last_token_usage", None),
                    },
                )
                trace = {
                    "id": question.get("id"),
                    "business_id": business_id,
                    "question": question.get("question"),
                    "plan": plan.model_dump(mode="json"),
                    "queries": query_trace,
                }
                llm_plan = getattr(parser, "last_raw_plan", None)
                if llm_plan is not None:
                    trace["llm_plan"] = llm_plan
                traces.append(trace)
            except Exception as error:
                answer_type = (
                    error.answer_type
                    if isinstance(error, _QuestionStatusError)
                    else "abstain"
                )
                output.append({
                    "id": question.get("id"),
                    "type": answer_type,
                    "reason": str(error),
                })
                if database is not None:
                    _write_question_evidence(
                        evidence_root, question, database, contexts.get(business_id, {}),
                        plan=plan,
                        query_trace=query_trace,
                        rows=rows,
                        answer_type=answer_type,
                        error=str(error), parser=parser,
                        token_usage=getattr(parser, "last_token_usage", None),
                    )
                _append_jsonl(
                    token_usage_path,
                    {
                        "id": question.get("id"),
                        "business_id": business_id,
                        "usage": getattr(parser, "last_token_usage", None),
                    },
                )
                trace = {
                    "id": question.get("id"),
                    "business_id": business_id,
                    "question": question.get("question"),
                    "error": str(error),
                }
                if plan is not None:
                    trace["plan"] = plan.model_dump(mode="json")
                if query_trace is not None:
                    trace["queries"] = query_trace
                llm_plan = getattr(parser, "last_raw_plan", None)
                llm_response = getattr(parser, "last_response_content", None)
                if llm_plan is not None:
                    trace["llm_plan"] = llm_plan
                if llm_response is not None:
                    trace["llm_response"] = llm_response
                traces.append(trace)
            checkpoint_answers()
    finally:
        for database in databases.values():
            database.close()
    _write_answer_records(output_path, output)
    if schema_output_path is not None:
        Path(schema_output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(schema_output_path).write_text(
            json.dumps(schemas, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
    if trace_output_path is not None:
        Path(trace_output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(trace_output_path).write_text(
            "\n".join(json.dumps(item, ensure_ascii=False, default=str) for item in traces) + "\n",
            encoding="utf-8",
        )
