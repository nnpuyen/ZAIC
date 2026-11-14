from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import duckdb

from .execution import execute_plan_with_trace
from .ingestion.loader import load_business_data, repair_mojibake
from .context import build_context
from .planning.schemas import QueryPlan
from .planning.completion import complete_plan


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
        fractional_count = metadata.execute(
            f'''SELECT COUNT(*)
                FROM "{table}"
                WHERE TRY_CAST(REPLACE(CAST("{column}" AS VARCHAR), ',', '') AS DOUBLE) IS NOT NULL
                  AND TRY_CAST(REPLACE(CAST("{column}" AS VARCHAR), ',', '') AS DOUBLE)
                      <> FLOOR(TRY_CAST(REPLACE(CAST("{column}" AS VARCHAR), ',', '') AS DOUBLE))'''
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
    if answer_type in {"number", "percent", "boolean"} and len(rows) == 1:
        # Aggregate aliases are arbitrary (for example, ``tong_doanh_thu``),
        # so scalar answers must be detected by result shape rather than by
        # expecting the executor's default ``metric`` name.
        if len(rows[0]) == 1:
            value = next(iter(rows[0].values()))
            if answer_type in {"number", "percent"} and isinstance(value, (int, float)) and not isinstance(value, bool):
                places = _answer_decimal_places(plan, database)
                return int(round(value)) if places == 0 else round(value, 2)
            return value

    if answer_type == "string" and len(rows) == 1:
        row = rows[0]
        value_columns = [name for name in row if name != "metric"]
        if len(value_columns) == 1:
            return row[value_columns[0]]
        if set(row) == {"metric"}:
            return row["metric"]

    return rows


def _write_question_evidence(
    evidence_root: Path,
    question: dict[str, Any],
    database: Any,
    context: dict[str, Any],
    plan: QueryPlan | None = None,
    original_plan: QueryPlan | None = None,
    query_trace: list[dict[str, Any]] | None = None,
    rows: list[dict[str, Any]] | None = None,
    answer: Any = None,
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
    if original_plan is not None:
        _write_json(
            evidence_dir / "06_validated_plan.json",
            original_plan.model_dump(mode="json"),
        )
    if plan is not None:
        if original_plan is None:
            _write_json(evidence_dir / "06_validated_plan.json", plan.model_dump(mode="json"))
        if original_plan is None or plan.model_dump(mode="json") != original_plan.model_dump(mode="json"):
            _write_json(
                evidence_dir / "06_completed_plan.json",
                plan.model_dump(mode="json"),
            )
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
                "answer": None,
                "reason": error,
            },
        )
    else:
        answer_record = {
            "id": question["id"],
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
    output: list[dict[str, Any]] = []
    schemas: dict[str, Any] = {}
    traces: list[dict[str, Any]] = []
    databases: dict[str, Any] = {}
    contexts: dict[str, dict[str, Any]] = {}
    evidence_root = (
        Path(evidence_output_dir)
        if evidence_output_dir is not None
        else Path(output_path).with_name(Path(output_path).stem + "_questions")
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    token_usage_path = Path(token_usage_output_path or Path(output_path).parent / "token_usage.jsonl")
    token_usage_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        for question in _iter_questions(questions_path):
            business_id = question.get("business_id")
            database = None
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
                raw_plan = parser.parse(
                    question["question"],
                    database,
                    question["as_of_date"],
                    context=contexts[business_id],
                )
                plan = raw_plan if isinstance(raw_plan, QueryPlan) else QueryPlan.model_validate(raw_plan)
                completion = complete_plan(
                    plan,
                    question["question"],
                    question["as_of_date"],
                    contexts[business_id],
                    database,
                )
                # Keep any repairs made during completion even if its
                # advisory name checks report unresolved columns. The
                # execution layer validates the resulting query against the
                # actual database schema.
                plan = completion.plan
                rows, query_trace = execute_plan_with_trace(plan, database)
                if completion.changes:
                    query_trace.insert(0, {
                        "kind": "plan_completion",
                        "original_plan": completion.original_plan.model_dump(mode="json"),
                        "changes": completion.changes,
                    })
                answer_type = plan.answer_type
                answer = _format_answer(answer_type, rows, plan, database)
                output.append({"id": question["id"], "answer": answer})
                _write_question_evidence(
                    evidence_root, question, database, contexts[business_id],
                    plan=plan, original_plan=completion.original_plan,
                    query_trace=query_trace, rows=rows, answer=answer,
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
                output.append({"id": question.get("id"), "answer": None, "reason": str(error)})
                if database is not None:
                    _write_question_evidence(
                        evidence_root, question, database, contexts.get(business_id, {}),
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
                llm_plan = getattr(parser, "last_raw_plan", None)
                llm_response = getattr(parser, "last_response_content", None)
                if llm_plan is not None:
                    trace["llm_plan"] = llm_plan
                if llm_response is not None:
                    trace["llm_response"] = llm_response
                traces.append(trace)
    finally:
        for database in databases.values():
            database.close()
    output_path = Path(output_path)
    output_path = Path(output_path)
    if output_path.suffix.lower() == ".json":
        output_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
    else:
        output_path.write_text(
            "\n".join(json.dumps(item, ensure_ascii=False, default=str) for item in output) + "\n",
            encoding="utf-8",
        )
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
