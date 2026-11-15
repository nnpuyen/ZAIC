from __future__ import annotations

import re
import unicodedata
import calendar
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from ..ingestion.loader import BusinessDatabase
from .dates import resolve_date_range
from .schemas import Filter, FilterGroup, Join, Metric, QueryPlan


_SEMANTIC_KEYS = (
    "name", "type", "canonical_name", "raw_name", "aliases", "role", "definition",
)
_NUMERIC_AGGREGATIONS = {"sum", "avg", "mean", "min", "max", "median"}
_ENTITY_TOKENS = {
    "don", "don_hang", "hoa_don", "bill", "invoice", "order",
    "khach", "khach_hang", "customer", "san_pham", "product",
}
_REVENUE_HINTS = (
    "doanh thu", "ban duoc", "tien hang", "tien ban", "so don",
    "ban chay", "ban nhieu", "khach da mua", "thanh tien",
)
_CANCEL_QUESTION_HINTS = (
    "ti le huy", "ti le don huy", "don huy", "bi huy", "huy don",
    "tra hang", "don tra",
)
_STATUS_CLASSES = {
    "cancelled": ("huy", "cancel"),
    "returned": ("tra hang", "returned", "doi tra", "hoan tra"),
    "completed": ("hoan thanh", "completed", "done", "da giao"),
    "shipping": ("dang giao", "shipping", "delivering"),
}
_DATE_NAME_HINTS = (
    "date", "datetime", "time", "created", "updated", "ngay", "thoi gian", "gio",
)
_COMPARISON_HINTS = (
    "cung ky", "thang truoc", "cao hay thap hon", "hon hay kem",
    "so sanh", "tang hay giam", "chenh lech", "so voi",
)


def _fold(value: str) -> str:
    for _ in range(2):
        try:
            repaired = value.encode("latin1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            break
        if repaired == value:
            break
        value = repaired
    normalized = unicodedata.normalize("NFD", value.lower().replace("đ", "d"))
    return " ".join(
        "".join(char for char in normalized if unicodedata.category(char) != "Mn").split()
    )


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", _fold(value)))


def _contains_status_term(value: str, term: str) -> bool:
    """Return whether *term* occurs as a complete word/phrase in *value*."""
    value_tokens = re.findall(r"[a-z0-9]+", _fold(value))
    term_tokens = re.findall(r"[a-z0-9]+", _fold(term))
    if not term_tokens or len(term_tokens) > len(value_tokens):
        return False
    width = len(term_tokens)
    return any(
        value_tokens[index:index + width] == term_tokens
        for index in range(len(value_tokens) - width + 1)
    )


@dataclass
class CompletionResult:
    plan: QueryPlan
    original_plan: QueryPlan
    changes: list[dict[str, Any]] = field(default_factory=list)
    status: str = "unchanged"
    reason: str | None = None
    fatal: bool = False


def _schema_tables(context: dict[str, Any]) -> list[dict[str, Any]]:
    schema = context.get("schema", {})
    tables = schema.get("tables", []) if isinstance(schema, dict) else []
    if isinstance(schema, dict) and not tables:
        tables = [
            {
                "table": table_name,
                "columns": [{"name": column} for column in details.get("columns", [])],
            }
            for table_name, details in schema.items()
            if isinstance(details, dict)
        ]
    return [table for table in tables if isinstance(table, dict)]


def _schema_table(context: dict[str, Any], table: str) -> dict[str, Any] | None:
    folded = _fold(table)
    matches: list[dict[str, Any]] = []
    for candidate in _schema_tables(context):
        names = [str(candidate.get("table", "")), str(candidate.get("canonical_name") or "")]
        aliases = candidate.get("aliases") or []
        if isinstance(aliases, list):
            names.extend(str(alias) for alias in aliases)
        if any(_fold(name) == folded for name in names if name):
            matches.append(candidate)
    if len(matches) == 1:
        return matches[0]
    for candidate in matches:
        if str(candidate.get("table", "")) == table:
            return candidate
    return None


def _columns_for(context: dict[str, Any], database: BusinessDatabase, table: str) -> list[dict[str, Any]]:
    schema_table = _schema_table(context, table)
    if schema_table is not None:
        columns = [
            column for column in schema_table.get("columns", [])
            if isinstance(column, dict) and column.get("name")
        ]
        if columns:
            return columns
    if table in database.tables:
        metadata = database.tables[table]
        mapping = {
            str(item.get("canonical_name")): item
            for item in metadata.column_mappings
        }
        return [
            {
                "name": name,
                **{
                    key: mapping.get(name, {})[key]
                    for key in _SEMANTIC_KEYS
                    if key in mapping.get(name, {})
                },
            }
            for name in metadata.columns
        ]
    return []


def _plan_cte_columns(plan: QueryPlan, name: str) -> list[dict[str, Any]]:
    for cte in plan.ctes:
        if cte.name != name:
            continue
        query = cte.query
        columns = (
            [{"name": "group_period"}]
            if query.group_by_period
            else [{"name": column} for column in query.group_by]
        )
        metrics = ([query.metric] if query.metric is not None else []) + list(query.metrics)
        if metrics:
            columns.extend(
                {"name": metric.alias or ("metric" if index == 0 else f"metric_{index}")}
                for index, metric in enumerate(metrics)
            )
        else:
            columns.extend({"name": column} for column in query.select_columns)
        return columns
    return []


def _unique_column_match(name: str, columns: list[dict[str, Any]]) -> str | None:
    # Plans may contain SQL-style qualified references such as
    # ``Table.Column`` while metadata stores only the physical column name.
    folded = _fold(name.rsplit(".", 1)[-1])
    matches: list[str] = []
    for column in columns:
        candidates = [
            column.get("name"),
            column.get("canonical_name"),
            column.get("raw_name"),
            *(column.get("aliases") or [] if isinstance(column.get("aliases"), list) else []),
        ]
        if any(_fold(str(candidate)) == folded for candidate in candidates if candidate):
            matches.append(str(column["name"]))
    unique = list(dict.fromkeys(matches))
    return unique[0] if len(unique) == 1 else None


def _role_columns(columns: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
    return [
        column for column in columns
        if column.get("role") == role and column.get("name")
    ]


def _best_role_column(
    columns: list[dict[str, Any]], role: str, question: str
) -> str | None:
    matches = _role_columns(columns, role)
    if len(matches) == 1:
        return str(matches[0]["name"])
    if not matches:
        return None
    question_tokens = _tokens(question)
    scored: list[tuple[int, str]] = []
    for column in matches:
        blob = " ".join(
            [
                str(column.get("name", "")),
                str(column.get("canonical_name", "")),
                str(column.get("definition", "")),
                *(column.get("aliases") or [] if isinstance(column.get("aliases"), list) else []),
            ]
        )
        scored.append((len(question_tokens & _tokens(blob)), str(column["name"])))
    scored.sort(key=lambda item: item[0], reverse=True)
    if scored[0][0] > 0 and (len(scored) == 1 or scored[0][0] > scored[1][0]):
        return scored[0][1]
    return None


def _entity_requested(question: str) -> bool:
    return bool(_tokens(question) & _ENTITY_TOKENS)


def _needs_revenue_status_exclusion(question: str) -> bool:
    folded = _fold(question)
    if any(term in folded for term in _CANCEL_QUESTION_HINTS):
        return "doanh thu" in folded or "ban duoc" in folded
    return any(term in folded for term in _REVENUE_HINTS)


def _metric_filter_present(metric: Metric) -> bool:
    return metric.filter is not None


def _percentage_count_pair(plan: QueryPlan) -> tuple[Metric, Metric] | None:
    """Identify a numerator and denominator in a two-count percentage plan."""
    if plan.answer_type != "percent" or plan.group_by or plan.group_by_period:
        return None
    metrics = ([plan.metric] if plan.metric is not None else []) + list(plan.metrics)
    if len(metrics) != 2:
        return None

    numerator, denominator = metrics
    count_types = {"count", "count_distinct"}
    if (
        numerator.aggregation not in count_types
        or denominator.aggregation not in count_types
        or numerator.column != denominator.column
        or (numerator.aggregation == "count_distinct" or numerator.distinct)
        != (denominator.aggregation == "count_distinct" or denominator.distinct)
    ):
        return None

    # Prefer explicit conditional-count structure. Aliases are a fallback for
    # planners that label the pair but omit the metric-level filters.
    first_has_filter = _metric_filter_present(metrics[0])
    second_has_filter = _metric_filter_present(metrics[1])
    if first_has_filter != second_has_filter:
        numerator, denominator = (metrics[0], metrics[1]) if first_has_filter else (metrics[1], metrics[0])
        if not _metric_filter_present(numerator):
            return None
        return numerator, denominator

    numerator_hints = ("cancel", "huy", "return", "tra", "numerator")
    denominator_hints = ("total", "tong", "denominator", "all", "tat ca")
    first_name = _fold(metrics[0].alias or "")
    second_name = _fold(metrics[1].alias or "")
    first_is_num = any(hint in first_name for hint in numerator_hints)
    second_is_num = any(hint in second_name for hint in numerator_hints)
    first_is_den = any(hint in first_name for hint in denominator_hints)
    second_is_den = any(hint in second_name for hint in denominator_hints)
    if first_is_num and second_is_den:
        return metrics[0], metrics[1]
    if second_is_num and first_is_den:
        return metrics[1], metrics[0]
    return None


def _align_answer_shape(plan: QueryPlan, changes: list[dict[str, Any]]) -> str | None:
    """Make scalar plans produce one target value, or explain the mismatch."""
    metrics = ([plan.metric] if plan.metric is not None else []) + list(plan.metrics)

    pair = _percentage_count_pair(plan)
    if pair is not None:
        numerator, denominator = pair
        plan.metric = Metric(
            column=numerator.column,
            aggregation="percentage",
            alias="metric",
            distinct=(
                numerator.aggregation == "count_distinct" or numerator.distinct
            ),
            filter=numerator.filter,
            denominator_filter=denominator.filter,
        )
        plan.metrics = []
        metrics = [plan.metric]
        changes.append({
            "kind": "answer_shape",
            "reason": "percentage_counts_collapsed_to_scalar_metric",
            "numerator_filter": numerator.filter is not None,
            "denominator_filter": denominator.filter is not None,
        })

    if plan.answer_type in {"list", "table"}:
        return None

    if len(metrics) > 1:
        return (
            f"Scalar answer type {plan.answer_type!r} requires one metric; "
            f"the plan contains {len(metrics)}"
        )

    is_group_count = (
        plan.answer_type == "number"
        and len(metrics) == 1
        and metrics[0].aggregation in {"count", "count_distinct"}
    )
    if plan.group_by or plan.group_by_period:
        # A single ranked result is scalar. Number plans that count distinct
        # groups are already reduced to one count by the executor.
        if plan.limit != 1 and not is_group_count:
            return (
                f"Scalar answer type {plan.answer_type!r} cannot use grouping "
                "unless the plan selects one result"
            )
    if not metrics:
        if plan.answer_type in {"number", "percent"} and plan.group_by:
            return f"Answer type {plan.answer_type!r} requires a numeric metric, not a grouped label"
        if not plan.group_by and len(plan.select_columns) != 1:
            return "A scalar selection plan must return exactly one selected column"
        if plan.limit != 1 and not is_group_count:
            return "A scalar selection plan must limit its result to one value"

    if plan.answer_type == "string":
        target_columns = list(dict.fromkeys([*plan.group_by, *plan.select_columns]))
        if plan.group_by_period:
            target_columns.insert(0, "group_period")
        if len(target_columns) > 1:
            return "A string answer must select exactly one target column"
        if (plan.group_by or not metrics) and plan.limit != 1:
            return "A string answer must limit its plan to one selected value"

    return None


def _status_column(columns: list[dict[str, Any]]) -> dict[str, Any] | None:
    role_matches = _role_columns(columns, "status")
    if len(role_matches) == 1:
        return role_matches[0]
    named = [
        column for column in columns
        if any(term in _fold(str(column.get("name", ""))) for term in ("status", "trang thai"))
    ]
    return named[0] if len(named) == 1 else None


def _excluded_status_values(column: dict[str, Any]) -> list[Any]:
    samples = column.get("sample_values") or []
    excluded: list[Any] = []
    for value in samples:
        folded = _fold(str(value))
        is_cancelled = any(
            _contains_status_term(folded, term) for term in ("huy", "cancel")
        ) and not _contains_status_term(folded, "tra")
        is_returned = any(
            _contains_status_term(folded, term)
            for term in ("tra hang", "returned", "doi tra", "hoan tra")
        )
        if is_cancelled or is_returned:
            excluded.append(value)
    return excluded


def _expand_status_value(value: Any, samples: list[Any]) -> Any:
    if not isinstance(value, str) or not samples:
        return value
    folded = _fold(value)
    exact = [sample for sample in samples if _fold(str(sample)) == folded]
    if len(exact) == 1:
        return exact[0]
    class_terms: tuple[str, ...] | None = None
    for label, terms in _STATUS_CLASSES.items():
        if folded == _fold(label) or folded in terms:
            class_terms = terms
            break
    if class_terms is None:
        return value
    matches = [
        sample for sample in samples
        if any(_contains_status_term(str(sample), term) for term in class_terms)
    ]
    return matches[0] if len(matches) == 1 else value


def _iter_filters(items: list[Filter | FilterGroup]) -> list[Filter]:
    filters: list[Filter] = []
    for item in items:
        if isinstance(item, FilterGroup):
            filters.extend(_iter_filters(item.conditions))
        else:
            filters.append(item)
    return filters


def _has_column_filter(items: list[Filter | FilterGroup], column: str) -> bool:
    return any(item.column == column for item in _iter_filters(items))


def _is_date_column(column: dict[str, Any]) -> bool:
    return column.get("role") == "transaction_date" or any(
        hint in _fold(str(column.get(key, "")))
        for key in ("name", "canonical_name", "definition")
        for hint in _DATE_NAME_HINTS
    )


def _date_column_for_plan(
    plan: QueryPlan, columns: list[dict[str, Any]], question: str
) -> str | None:
    date_columns = [column for column in columns if _is_date_column(column)]
    names = {str(column.get("name")) for column in date_columns if column.get("name")}
    # Prefer a date field the planner already filtered. This usually captures
    # the transaction date while allowing a separate time-of-day predicate.
    existing = [
        item.column
        for item in [*_iter_filters(plan.filters), *_iter_filters(plan.comparison_filters)]
        if item.column in names
    ]
    if existing:
        return existing[0]

    role_match = _best_role_column(columns, "transaction_date", question)
    if role_match is not None:
        return role_match
    if len(date_columns) == 1:
        return str(date_columns[0]["name"])

    # Prefer invoice/order/sale dates over creation or update timestamps when
    # the question only supplies a relative period such as "last week".
    scored: list[tuple[int, str]] = []
    for column in date_columns:
        name = _fold(" ".join(str(column.get(key, "")) for key in ("name", "definition")))
        score = 0
        if any(term in name for term in ("ngay dat hang", "ngay hoa don", "ngay giao dich", "transaction date", "order date")):
            score += 3
        if any(term in name for term in ("gio tao", "created", "updated")):
            score -= 2
        scored.append((score, str(column["name"])))
    scored.sort(reverse=True)
    if scored and scored[0][0] > 0 and (len(scored) == 1 or scored[0][0] > scored[1][0]):
        return scored[0][1]
    return None


def _is_comparison_question(question: str) -> bool:
    folded = _fold(question)
    return any(hint in folded for hint in _COMPARISON_HINTS)


def _parsed_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def _is_full_day_window(item: Filter) -> bool:
    """Recognize planner-added calendar windows, not hour-of-day filters."""
    values = item.value if isinstance(item.value, list) else [item.value]
    if item.operator == "between" and len(values) == 2:
        parsed = [_parsed_date(value) for value in values]
        if None in parsed:
            return False
        raw_values = [str(value) for value in values]
        return all(
            len(raw) <= 10 or raw[10:].strip(" T0:") == ""
            for raw in raw_values
        )
    if item.operator in {"eq", "gte", "gt", "lte", "lt"} and len(values) == 1:
        raw = str(values[0])
        return _parsed_date(values[0]) is not None and (
            len(raw) <= 10 or raw[10:].strip(" T0:") == ""
        )
    return False


def _reanchor_time_window(
    item: Filter, date_range: tuple[str, str]
) -> list[Any] | None:
    if item.operator != "between" or not isinstance(item.value, list) or len(item.value) != 2:
        return None
    parsed: list[datetime] = []
    for value in item.value:
        if isinstance(value, datetime):
            parsed.append(value)
            continue
        if not isinstance(value, str):
            return None
        try:
            parsed.append(datetime.fromisoformat(value.replace("Z", "+00:00")))
        except ValueError:
            return None
    if all(value.time().replace(tzinfo=None) == datetime.min.time() for value in parsed):
        return None
    anchor = date.fromisoformat(date_range[0])
    original_anchor = parsed[0].date()
    result: list[Any] = []
    for value, raw in zip(parsed, item.value):
        day_offset = (value.date() - original_anchor).days
        shifted = datetime.combine(anchor.fromordinal(anchor.toordinal() + day_offset), value.time())
        if value.tzinfo is not None:
            shifted = shifted.replace(tzinfo=value.tzinfo)
        result.append(shifted if isinstance(raw, datetime) else shifted.isoformat())
    return result


def _column_mentioned(question: str, column: dict[str, Any]) -> bool:
    folded_question = _fold(question)
    candidates = [
        column.get("name"), column.get("canonical_name"), column.get("definition"),
        *(column.get("aliases") or [] if isinstance(column.get("aliases"), list) else []),
    ]
    return any(
        (folded := _fold(str(candidate))).strip()
        and len(folded) >= 5
        and folded in folded_question
        for candidate in candidates
        if candidate
    )


def _canonicalize_question_date(
    plan: QueryPlan,
    question: str,
    date_range: tuple[str, str] | None,
    columns: list[dict[str, Any]],
    changes: list[dict[str, Any]],
) -> str | None:
    """Replace conflicting planner calendar windows with the resolved window."""
    if date_range is None or _is_comparison_question(question):
        return None
    target = _date_column_for_plan(plan, columns, question)
    if target is None:
        return None

    date_columns = {
        str(column["name"]): column
        for column in columns
        if column.get("name") and _is_date_column(column)
    }
    replaced = 0
    removed: list[str] = []
    reanchored: list[str] = []

    def rewrite(items: list[Filter | FilterGroup], inside_or: bool = False) -> list[Filter | FilterGroup]:
        nonlocal replaced
        result: list[Filter | FilterGroup] = []
        for item in items:
            if isinstance(item, FilterGroup):
                nested = rewrite(item.conditions, inside_or or item.logic == "or")
                if nested:
                    item.conditions = nested
                    result.append(item)
                continue
            if item.column == target:
                if inside_or:
                    result.append(item)
                    continue
                item.operator = "between"
                item.value = list(date_range)
                replaced += 1
                continue
            column = date_columns.get(item.column)
            shifted_values = (
                _reanchor_time_window(item, date_range)
                if column is not None and item.column != target
                else None
            )
            if shifted_values is not None and not _column_mentioned(question, column):
                item.value = shifted_values
                reanchored.append(item.column)
            if (
                not inside_or
                and column is not None
                and item.column != target
                and _is_full_day_window(item)
                and not _column_mentioned(question, column)
            ):
                removed.append(item.column)
                continue
            result.append(item)
        return result

    plan.filters = rewrite(plan.filters)  # type: ignore[assignment]
    plan.comparison_filters = [
        item for item in rewrite(plan.comparison_filters) if isinstance(item, Filter)
    ]
    if replaced == 0:
        plan.filters.append(
            Filter(column=target, operator="between", value=list(date_range))
        )
    changes.append({
        "kind": "filter",
        "reason": "canonical_question_date_range",
        "column": target,
        "operator": "between",
        "value": list(date_range),
        "replaced_filters": replaced,
        "removed_conflicting_date_columns": sorted(set(removed)),
        "reanchored_time_columns": sorted(set(reanchored)),
    })
    return target


def _table_period(metadata: Any) -> tuple[date, date] | None:
    labels = " ".join(
        str(value)
        for value in (
            getattr(metadata, "table", ""),
            Path(str(getattr(metadata, "source", ""))).stem,
            getattr(metadata, "sheet", "") or "",
        )
    )
    month_match = re.search(
        r"(?:thang|month|t)[ _./-]*0?([1-9]|1[0-2])[ _./-]*(20\d{2})",
        labels,
        flags=re.IGNORECASE,
    )
    if month_match is None:
        month_match = re.search(
            r"(20\d{2})[ _./-]*(?:thang|month|t)[ _./-]*0?([1-9]|1[0-2])",
            labels,
            flags=re.IGNORECASE,
        )
        if month_match:
            year, month = int(month_match.group(1)), int(month_match.group(2))
        else:
            year_match = re.search(r"(?<!\d)(20\d{2})(?!\d)", labels)
            if year_match is None:
                return None
            year = int(year_match.group(1))
            return date(year, 1, 1), date(year, 12, 31)
    else:
        month, year = int(month_match.group(1)), int(month_match.group(2))
    return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])


def _table_family(metadata: Any) -> str:
    label = str(getattr(metadata, "table", "")).casefold()
    label = re.sub(
        r"(?:thang|month|t)[ _./-]*0?(?:[1-9]|1[0-2])[ _./-]*20\d{2}",
        " ", label,
    )
    label = re.sub(r"20\d{2}[ _./-]*(?:thang|month|t)[ _./-]*0?(?:[1-9]|1[0-2])", " ", label)
    label = re.sub(r"20\d{2}", " ", label)
    return re.sub(r"[^a-z0-9]+", "", label)


def _observed_period(
    database: BusinessDatabase, table: str, column: str
) -> tuple[date, date] | None:
    quote = lambda value: '"' + value.replace('"', '""') + '"'
    try:
        start, end = database.connection.execute(
            f"SELECT MIN(TRY_CAST({quote(column)} AS DATE)), "
            f"MAX(TRY_CAST({quote(column)} AS DATE)) FROM {quote(table)}"
        ).fetchone()
    except Exception:
        return None
    parsed_start, parsed_end = _parsed_date(start), _parsed_date(end)
    if parsed_start is None or parsed_end is None:
        return None
    return parsed_start, parsed_end


def _ranges_overlap(
    period: tuple[date, date], bounds: tuple[date | None, date | None]
) -> bool:
    start, end = bounds
    period_start, period_end = period
    lower = start or end or date.min
    upper = end or start or date.max
    return period_start <= upper and period_end >= lower


def _filter_date_bounds(
    filters: list[Filter | FilterGroup], column: str
) -> tuple[date | None, date | None] | None:
    bounds: list[tuple[date | None, date | None]] = []
    for item in _iter_filters(filters):
        if item.column != column:
            continue
        values = item.value if isinstance(item.value, list) else [item.value]
        if item.operator == "between" and len(values) == 2:
            bounds.append((_parsed_date(values[0]), _parsed_date(values[1])))
        elif item.operator == "eq" and values:
            value = _parsed_date(values[0])
            bounds.append((value, value))
        elif item.operator in {"gt", "gte"} and values:
            bounds.append((_parsed_date(values[0]), None))
        elif item.operator in {"lt", "lte"} and values:
            bounds.append((None, _parsed_date(values[0])))
    if not bounds:
        return None
    starts = [start for start, _ in bounds if start is not None]
    ends = [end for _, end in bounds if end is not None]
    return (max(starts) if starts else None, min(ends) if ends else None)


def _period_matches(
    period: tuple[date, date],
    bounds: tuple[date | None, date | None],
    *,
    allow_partial: bool,
) -> bool:
    start, end = bounds
    period_start, period_end = period
    if allow_partial:
        lower = start or end or date.min
        upper = end or start or date.max
        return period_start <= upper and period_end >= lower
    return (start is None or period_start <= start) and (end is None or period_end >= end)


def _repair_plan_source_coverage(
    plan: QueryPlan,
    database: BusinessDatabase,
    changes: list[dict[str, Any]],
    *,
    allow_partial: bool = False,
) -> str | None:
    """Repoint a period table only when one compatible source covers its dates."""
    if plan.table in database.tables:
        metadata = database.tables[plan.table]
        columns = _columns_for({}, database, plan.table)
        date_names = {
            str(column["name"])
            for column in columns
            if column.get("name") and _is_date_column(column)
        }
        filter_names = [
            item.column
            for item in _iter_filters(plan.filters)
            if item.column in date_names
        ]
        if not filter_names:
            filter_names = [
                item.column
                for item in _iter_filters(plan.comparison_filters)
                if item.column in date_names
            ]
        if filter_names:
            date_column = filter_names[0]
            bounds = _filter_date_bounds(plan.filters, date_column)
            if bounds is None:
                bounds = _filter_date_bounds(plan.comparison_filters, date_column)
            period = _table_period(metadata)
            closed_range = (
                bounds is not None
                and bounds[0] is not None
                and bounds[1] is not None
            )
            if closed_range:
                partial = allow_partial or bool(plan.set_operations)
                observed = period or _observed_period(
                    database, metadata.table, date_column
                )
                current_matches = (
                    _period_matches(period, bounds, allow_partial=partial)
                    if period is not None
                    else observed is None or _ranges_overlap(observed, bounds)
                )
            else:
                partial = allow_partial or bool(plan.set_operations)
                observed = period
                current_matches = True

            if closed_range and not current_matches:
                candidates = []
                for candidate in database.tables.values():
                    if (
                        candidate.table == metadata.table
                        or _table_family(candidate) != _table_family(metadata)
                        or set(candidate.columns) != set(metadata.columns)
                    ):
                        continue
                    candidate_period = _table_period(candidate)
                    is_partition_label = candidate_period is not None
                    if candidate_period is None:
                        candidate_period = _observed_period(
                            database, candidate.table, date_column
                        )
                    if candidate_period is None:
                        continue
                    candidate_matches = (
                        _period_matches(
                            candidate_period, bounds, allow_partial=partial
                        )
                        if is_partition_label
                        else _ranges_overlap(candidate_period, bounds)
                    )
                    if candidate_matches:
                        candidates.append(candidate)
                if len(candidates) != 1:
                    start, end = bounds
                    requested = f"{start or 'open'}..{end or 'open'}"
                    if len(candidates) > 1:
                        return (
                            f"Date range {requested} is not uniquely covered by a source table "
                            f"matching {metadata.table!r}"
                        )
                    current_range = (
                        f"{observed[0]}..{observed[1]}" if observed is not None else "an unknown period"
                    )
                    return (
                        f"Source table {metadata.table!r} covers {current_range}, outside "
                        f"requested date range {requested}; no compatible source was found"
                    )
                replacement = candidates[0]
                old_table = plan.table
                plan.table = replacement.table
                for item in [*_iter_filters(plan.filters), *_iter_filters(plan.comparison_filters)]:
                    if item.table == old_table:
                        item.table = replacement.table
                changes.append({
                    "kind": "table",
                    "reason": "source_date_coverage",
                    "from": old_table,
                    "to": replacement.table,
                    "date_column": date_column,
                    "requested_range": [
                        bounds[0].isoformat() if bounds[0] else None,
                        bounds[1].isoformat() if bounds[1] else None,
                    ],
                })

    for cte in plan.ctes:
        reason = _repair_plan_source_coverage(cte.query, database, changes)
        if reason:
            return reason
    for branch in plan.set_operations:
        reason = _repair_plan_source_coverage(
            branch.query, database, changes, allow_partial=True
        )
        if reason:
            return reason
    return None


def _rewrite_filters(
    items: list[Filter | FilterGroup],
    columns: list[dict[str, Any]],
    table_columns: set[str],
    changes: list[dict[str, Any]],
    unresolved: list[str],
    column_tables: dict[str, str] | None = None,
) -> None:
    for item in items:
        if isinstance(item, FilterGroup):
            _rewrite_filters(item.conditions, columns, table_columns, changes, unresolved, column_tables)
            continue
        resolved = _resolve_column(item.column, columns, table_columns, changes, unresolved)
        item.column = resolved
        if item.table is None and column_tables is not None:
            owner = column_tables.get(resolved)
            if owner is not None:
                item.table = owner
        status = next((column for column in columns if column.get("name") == resolved), None)
        samples = list(status.get("sample_values") or []) if status else []
        if item.operator in {"eq", "neq", "in", "not_in"} and samples:
            original = item.value
            if isinstance(item.value, list):
                item.value = [_expand_status_value(value, samples) for value in item.value]
            else:
                item.value = _expand_status_value(item.value, samples)
            if item.value != original:
                changes.append({
                    "kind": "filter_value",
                    "reason": "canonical_or_stored_value",
                    "column": item.column,
                    "original_value": original,
                    "value": item.value,
                })


def _resolve_column(
    name: str,
    columns: list[dict[str, Any]],
    table_columns: set[str],
    changes: list[dict[str, Any]],
    unresolved: list[str],
    *,
    allow_star: bool = False,
) -> str:
    if name in {"*", ""}:
        if allow_star:
            return name
        unresolved.append(name)
        return name
    if name in table_columns:
        return name
    matched = _unique_column_match(name, columns)
    if matched is not None:
        if matched != name:
            changes.append({
                "kind": "alias",
                "reason": "canonical_column_alias",
                "from": name,
                "to": matched,
            })
        return matched
    unresolved.append(name)
    return name


def _rewrite_metric(
    metric: Metric | None,
    columns: list[dict[str, Any]],
    table_columns: set[str],
    changes: list[dict[str, Any]],
    unresolved: list[str],
    *,
    include_conditions: bool = True,
) -> None:
    if metric is None:
        return
    metric.column = _resolve_column(
        metric.column, columns, table_columns, changes, unresolved, allow_star=True,
    )
    if metric.denominator_column:
        metric.denominator_column = _resolve_column(
            metric.denominator_column, columns, table_columns, changes, unresolved,
        )
    if include_conditions:
        for condition in (metric.filter, metric.denominator_filter):
            if isinstance(condition, FilterGroup):
                _rewrite_filters(condition.conditions, columns, table_columns, changes, unresolved)
            elif isinstance(condition, Filter):
                _rewrite_filters([condition], columns, table_columns, changes, unresolved)


def _plan_referenced_columns(plan: QueryPlan) -> list[str]:
    names: list[str] = []
    names.extend(plan.select_columns)
    names.extend(plan.group_by)
    for item in plan.filters + plan.having_filters + plan.qualify_filters:
        names.extend(filter_item.column for filter_item in _iter_filters([item]))
    for item in plan.comparison_filters:
        names.append(item.column)
    for metric in ([plan.metric] if plan.metric is not None else []) + list(plan.metrics):
        names.append(metric.column)
        if metric.denominator_column:
            names.append(metric.denominator_column)
        for condition in (metric.filter, metric.denominator_filter):
            if condition is not None:
                names.extend(filter_item.column for filter_item in _iter_filters([condition]))
    return [name for name in names if name not in {"", "*"}]


def _implicit_joins_for_missing_columns(
    plan: QueryPlan,
    question: str,
    context: dict[str, Any],
    database: BusinessDatabase,
    columns: list[dict[str, Any]],
    table_columns: set[str],
    changes: list[dict[str, Any]],
) -> None:
    """Add only high-confidence joins needed by an otherwise valid column.

    This handles the common normalized-data shape where a transaction/detail
    table stores an item ID while the requested item name lives in a product
    master table.  A candidate must expose the missing column and share a
    key with the current relation; ties are left unresolved for safety.
    """
    used_tables = {plan.table} | {
        join.table for join in plan.joins if join.table in database.tables
    }
    available_columns = set(table_columns)
    question_folded = _fold(question)

    for requested in _plan_referenced_columns(plan):
        if requested in available_columns or _unique_column_match(requested, columns):
            continue
        candidates: list[tuple[int, str, str, str]] = []
        for table_name, metadata in database.tables.items():
            if table_name in used_tables:
                continue
            candidate_columns = list(metadata.columns)
            resolved = _unique_column_match(requested, [{"name": c} for c in candidate_columns])
            if resolved is None:
                continue
            shared = set(candidate_columns) & available_columns
            if not shared:
                continue
            key_candidates = sorted(
                shared,
                key=lambda key: (
                    2 if any(token in _fold(key) for token in ("ma ", " id", "code", "sku")) else 0,
                    -len(key),
                ),
                reverse=True,
            )
            join_key = key_candidates[0]
            score = 0
            table_folded = _fold(table_name)
            if any(token in table_folded for token in ("san pham", "sanpham", "hang hoa", "hanghoa", "product")):
                score += 20
            if any(token in _fold(requested) for token in ("ten hang", "ten san pham", "product name")):
                score += 10
            if any(token in question_folded for token in ("mat hang", "san pham", "ten hang")):
                score += 1
            if any(token in _fold(join_key) for token in ("ma ", " id", "code", "sku")):
                score += 5
            candidates.append((score, table_name, resolved, join_key))

        if not candidates:
            continue
        candidates.sort(reverse=True)
        if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
            continue
        score, table_name, resolved, join_key = candidates[0]
        del score
        plan.joins.append(Join(
            table=table_name,
            on_conditions=[{"left_column": join_key, "right_column": join_key}],
        ))
        used_tables.add(table_name)
        joined_columns = _columns_for(context, database, table_name)
        columns.extend(joined_columns)
        table_columns.update(database.tables[table_name].columns)
        available_columns.update(database.tables[table_name].columns)
        changes.append({
            "kind": "join",
            "reason": "missing_column_shared_key",
            "table": table_name,
            "column": resolved,
            "left_column": join_key,
            "right_column": join_key,
        })


def complete_plan(
    plan: QueryPlan,
    question: str,
    as_of_date: str,
    context: dict[str, Any],
    database: BusinessDatabase,
) -> CompletionResult:
    """Complete only high-confidence omissions grounded in canonical metadata."""
    original = plan.model_copy(deep=True)
    completed = plan.model_copy(deep=True)
    changes: list[dict[str, Any]] = []
    unresolved: list[str] = []

    cte_columns = _plan_cte_columns(completed, completed.table)
    schema_table = _schema_table(context, completed.table)
    if completed.table not in database.tables and not cte_columns:
        if schema_table is not None and str(schema_table.get("table", "")) in database.tables:
            resolved_table = str(schema_table["table"])
            changes.append({
                "kind": "table",
                "reason": "canonical_table_alias",
                "from": completed.table,
                "to": resolved_table,
            })
            completed.table = resolved_table
        else:
            return CompletionResult(
                plan=completed,
                original_plan=original,
                status="rejected",
                reason=f"Unknown table: {completed.table}",
            )

    columns = (
        cte_columns
        if cte_columns
        else _columns_for(context, database, completed.table)
    )
    table_columns = (
        set(database.tables[completed.table].columns)
        if completed.table in database.tables
        else {str(column["name"]) for column in columns}
    )

    completed.select_columns = [
        _resolve_column(name, columns, table_columns, changes, unresolved)
        for name in completed.select_columns
    ]
    completed.group_by = [
        _resolve_column(name, columns, table_columns, changes, unresolved)
        for name in completed.group_by
    ]
    if completed.group_by_period_column:
        completed.group_by_period_column = _resolve_column(
            completed.group_by_period_column, columns, table_columns, changes, unresolved,
        )
    for join in completed.joins:
        if join.table not in database.tables:
            cte_columns = _plan_cte_columns(completed, join.table)
            if cte_columns:
                join_columns = cte_columns
                join_table_columns = {str(column["name"]) for column in join_columns}
                for condition in join.on_conditions:
                    condition.left_column = _resolve_column(
                        condition.left_column, columns, table_columns, changes, unresolved,
                    )
                    condition.right_column = _resolve_column(
                        condition.right_column, join_columns, join_table_columns, changes, unresolved,
                    )
                continue
            join_schema = _schema_table(context, join.table)
            resolved_join = (
                str(join_schema.get("table", "")) if join_schema is not None else ""
            )
            if resolved_join in database.tables:
                changes.append({
                    "kind": "table",
                    "reason": "canonical_table_alias",
                    "from": join.table,
                    "to": resolved_join,
                })
                join.table = resolved_join
            else:
                return CompletionResult(
                    plan=completed,
                    original_plan=original,
                    status="rejected",
                    reason=f"Unknown join table: {join.table}",
                )
        join_columns = _columns_for(context, database, join.table)
        join_table_columns = (
            set(database.tables[join.table].columns)
            if join.table in database.tables
            else {str(column["name"]) for column in join_columns}
        )
        for condition in join.on_conditions:
            condition.left_column = _resolve_column(
                condition.left_column, columns, table_columns, changes, unresolved,
            )
            condition.right_column = _resolve_column(
                condition.right_column, join_columns, join_table_columns, changes, unresolved,
            )

    _implicit_joins_for_missing_columns(
        completed,
        question,
        context,
        database,
        columns,
        table_columns,
        changes,
    )

    # Filters and conditional metrics may refer to columns on joined tables.
    # Resolve them against the same union of tables that execution validates.
    all_columns = list(columns)
    all_table_columns = set(table_columns)
    for join in completed.joins:
        join_columns = (
            _columns_for(context, database, join.table)
            if join.table in database.tables
            else _plan_cte_columns(completed, join.table)
        )
        all_columns.extend(join_columns)
        all_table_columns.update(
            database.tables[join.table].columns
            if join.table in database.tables
            else (column["name"] for column in join_columns)
        )
    column_tables: dict[str, str] = {}
    for name in table_columns:
        column_tables[name] = completed.table
    for join in completed.joins:
        join_columns = database.tables[join.table].columns if join.table in database.tables else []
        for name in join_columns:
            column_tables.setdefault(name, join.table)
    _rewrite_filters(completed.filters, all_columns, all_table_columns, changes, unresolved, column_tables)
    _rewrite_filters(completed.comparison_filters, all_columns, all_table_columns, changes, unresolved, column_tables)
    _rewrite_filters(completed.having_filters, all_columns, all_table_columns, changes, unresolved, column_tables)
    _rewrite_filters(completed.qualify_filters, all_columns, all_table_columns, changes, unresolved, column_tables)
    for metric in ([completed.metric] if completed.metric is not None else []) + list(completed.metrics):
        _rewrite_metric(metric, all_columns, all_table_columns, changes, unresolved)

    if unresolved:
        unknown = sorted({name for name in unresolved if name not in {"*", ""}})
        if unknown:
            return CompletionResult(
                plan=completed,
                original_plan=original,
                status="rejected",
                reason=f"Unknown columns in {completed.table}: {unknown}",
            )

    # Apply the same conservative recovery to both the legacy single-metric
    # field and the newer multi-metric field.  A planner can legally emit both,
    # so completing only ``metric`` would leave an otherwise valid plan
    # semantically incomplete.
    metrics = ([completed.metric] if completed.metric is not None else []) + list(completed.metrics)
    for metric in metrics:
        if metric.aggregation == "count" and metric.column in {"*", ""} and _entity_requested(question):
            identifier = _best_role_column(columns, "identifier", question)
            if identifier is not None:
                metric.column = identifier
                metric.aggregation = "count_distinct"
                changes.append({
                    "kind": "metric",
                    "reason": "entity_count_requires_identifier",
                    "column": identifier,
                    "aggregation": "count_distinct",
                })

        if metric.column in {"*", ""} and metric.aggregation in _NUMERIC_AGGREGATIONS:
            measure = _best_role_column(columns, "measure", question)
            if measure is not None:
                metric.column = measure
                changes.append({
                    "kind": "metric",
                    "reason": "single_canonical_measure",
                    "column": metric.column,
                })
            else:
                return CompletionResult(
                    plan=completed,
                    original_plan=original,
                    status="rejected",
                    reason="Numeric aggregation is missing a unique canonical measure",
                )

    shape_error = _align_answer_shape(completed, changes)
    if shape_error is not None:
        return CompletionResult(
            plan=completed,
            original_plan=original,
            changes=changes,
            status="rejected",
            reason=shape_error,
            fatal=True,
        )

    date_range = resolve_date_range(question, as_of_date)
    transaction_date = _canonicalize_question_date(
        completed, question, date_range, columns, changes
    )
    # Canonicalization installs a date filter (rewriting an existing predicate
    # or adding one when absent). Preserve that fact explicitly: the as-of
    # horizon is only a fallback for plans without a resolved date range.
    date_range_applied = transaction_date is not None
    if transaction_date is None:
        transaction_date = _best_role_column(columns, "transaction_date", question)
    has_date_filter = bool(
        date_range_applied
        or (
            transaction_date
            and (
                _has_column_filter(completed.filters, transaction_date)
                or _has_column_filter(completed.comparison_filters, transaction_date)
            )
        )
    )
    if transaction_date and not has_date_filter:
        completed.filters.append(
            Filter(column=transaction_date, operator="lte", value=as_of_date)
        )
        changes.append({
            "kind": "filter",
            "reason": "as_of_date_horizon",
            "column": transaction_date,
            "operator": "lte",
            "value": as_of_date,
        })
    if completed.group_by_period and not completed.group_by and transaction_date:
        completed.group_by = [transaction_date]
        completed.group_by_period_column = transaction_date
        changes.append({
            "kind": "group_by",
            "reason": "period_requires_transaction_date",
            "column": transaction_date,
        })

    source_error = _repair_plan_source_coverage(completed, database, changes)
    if source_error:
        return CompletionResult(
            plan=completed,
            original_plan=original,
            changes=changes,
            status="rejected",
            reason=source_error,
            fatal=True,
        )

    status_column = _status_column(columns)
    if (
        status_column is not None
        and _needs_revenue_status_exclusion(question)
        and not _has_column_filter(completed.filters, str(status_column["name"]))
    ):
        excluded = _excluded_status_values(status_column)
        if excluded:
            completed.filters.append(
                Filter(
                    column=str(status_column["name"]),
                    operator="not_in",
                    value=excluded,
                )
            )
            changes.append({
                "kind": "filter",
                "reason": "rulebook_exclude_cancelled_returned",
                "column": str(status_column["name"]),
                "operator": "not_in",
                "value": excluded,
            })

    # Keep the resolved calendar window in the final executable plan. Earlier
    # repairs can rebuild filter collections, so enforce the canonical range
    # after those repairs have finished as well.
    if (
        date_range_applied
        and transaction_date is not None
        and date_range is not None
        and not _has_column_filter(completed.filters, transaction_date)
    ):
        completed.filters.append(
            Filter(
                column=transaction_date,
                operator="between",
                value=list(date_range),
                table=column_tables.get(transaction_date),
            )
        )
        changes.append({
            "kind": "filter",
            "reason": "restore_canonical_question_date_range",
            "column": transaction_date,
            "operator": "between",
            "value": list(date_range),
        })

    status = "completed" if changes else "unchanged"
    return CompletionResult(
        plan=completed,
        original_plan=original,
        changes=changes,
        status=status,
    )
