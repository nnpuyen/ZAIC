from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any
import re
import unicodedata

from .ingestion.loader import BusinessDatabase
from .numeric import number_sql_expression, parse_measure_number
from .planning.schemas import Filter, FilterGroup, Metric, QueryPlan


# Keep planner-generated SQL flowing to DuckDB without the local schema
# reference validator. Set this to True to restore fail-fast plan-reference checks.
VALIDATE_PLAN_REFERENCES = False
# Controls the join-key checks within _validate when reference validation is on.
VALIDATE_JOIN_COLUMNS = False


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _column_key(value: str) -> str:
    """Return the stable key used when resolving a column reference.

    Physical column names are preserved, but lookup is tolerant of casing,
    Unicode composition, and repeated/leading/trailing whitespace.
    """
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"\s+", " ", value).strip()


def _match_column(name: str, columns: set[str]) -> str | None:
    if name in columns:
        return name
    name = name.rsplit(".", 1)[-1]
    key = _column_key(name)
    matches = [column for column in columns if _column_key(column) == key]
    return matches[0] if len(matches) == 1 else None


def _normalize_plan_columns(plan: QueryPlan, database: BusinessDatabase) -> QueryPlan:
    """Rewrite plan column references to the exact physical column names."""
    normalized = plan.model_copy(deep=True)
    for cte in normalized.ctes:
        cte.query = _normalize_plan_columns(cte.query, database)
    for operation in normalized.set_operations:
        operation.query = _normalize_plan_columns(operation.query, database)

    base_columns = (
        set(database.tables[normalized.table].columns)
        if normalized.table in database.tables
        else _cte_columns(normalized, normalized.table)
    )
    joined_columns = [
        (
            join.table,
            set(database.tables[join.table].columns)
            if join.table in database.tables
            else _cte_columns(normalized, join.table),
        )
        for join in normalized.joins
    ]

    def resolve(name: str, owner: str | None = None) -> str:
        if name in {"", "*"}:
            return name
        if owner in {normalized.table, normalized.alias} or owner is None:
            match = _match_column(name, base_columns)
            if match is not None:
                return match
        if owner is not None:
            for join, (table, columns) in zip(normalized.joins, joined_columns):
                if owner == table or owner == join.alias:
                    match = _match_column(name, columns)
                    return match or name
            return name
        for _, columns in joined_columns:
            match = _match_column(name, columns)
            if match is not None:
                return match
        return name

    normalized.select_columns = [resolve(name) for name in normalized.select_columns]
    normalized.group_by = [resolve(name) for name in normalized.group_by]
    if normalized.group_by_period_column:
        normalized.group_by_period_column = resolve(normalized.group_by_period_column)

    for join, (_, join_columns) in zip(normalized.joins, joined_columns):
        for condition in join.on_conditions:
            condition.left_column = resolve(condition.left_column)
            condition.right_column = _match_column(condition.right_column, join_columns) or condition.right_column

    def normalize_filter(item: Filter | FilterGroup) -> None:
        if isinstance(item, FilterGroup):
            for condition in item.conditions:
                normalize_filter(condition)
        else:
            item.column = resolve(item.column, item.table)

    for item in normalized.filters + normalized.having_filters + normalized.qualify_filters:
        normalize_filter(item)
    for item in normalized.comparison_filters:
        item.column = resolve(item.column, item.table)

    metrics = ([normalized.metric] if normalized.metric is not None else []) + list(normalized.metrics)
    if normalized.pivot is not None:
        metrics.extend(normalized.pivot.using)
    for metric in metrics:
        metric.column = resolve(metric.column)
        if metric.denominator_column:
            metric.denominator_column = resolve(metric.denominator_column)
        for condition in (metric.filter, metric.denominator_filter):
            if condition is not None:
                normalize_filter(condition)
    return normalized


def _iter_filters(items: list[Filter | FilterGroup]) -> list[Filter]:
    filters: list[Filter] = []
    for item in items:
        if isinstance(item, FilterGroup):
            filters.extend(_iter_filters(item.conditions))
        else:
            filters.append(item)
    return filters


def _fold_value(value: str) -> str:
    import unicodedata

    normalized = unicodedata.normalize("NFD", value.lower().replace("đ", "d"))
    return "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Mn"
    )


def _repair_mojibake(value: str) -> str:
    """Repair the common UTF-8-as-Latin-1 corruption used by old exports."""
    for _ in range(2):
        try:
            repaired = value.encode("latin1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            break
        if repaired == value:
            break
        value = repaired
    return value


def _column_type(database: BusinessDatabase, table: str, column: str) -> str:
    quoted_table = _quote(table)
    quoted_column = _quote(column)
    result = database.connection.execute(
        f"DESCRIBE SELECT {quoted_column} FROM {quoted_table}"
    ).fetchone()
    return str(result[1]).upper() if result else ""


def _stored_value(
    database: BusinessDatabase, table: str, column: str, value: Any
) -> Any:
    if not isinstance(value, str):
        return value
    quoted_table = _quote(table)
    quoted_column = _quote(column)
    matches = database.connection.execute(
        f"SELECT DISTINCT {quoted_column} FROM {quoted_table} "
        f"WHERE {quoted_column} IS NOT NULL"
    ).fetchall()
    folded = _fold_value(value.strip())
    for (candidate,) in matches:
        if isinstance(candidate, str) and _fold_value(candidate.strip()) == folded:
            return candidate
    return value


def _date_only(value: Any) -> date | None:
    """Return a calendar date only when the filter value has no time part."""
    if isinstance(value, datetime):
        return None
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    try:
        return date.fromisoformat(normalized)
    except ValueError:
        pass
    for format_string in ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d thg %m %Y"):
        try:
            return datetime.strptime(normalized, format_string).date()
        except ValueError:
            continue
    return None


def _parse_filter_datetime(value: Any) -> datetime | None:
    """Parse planner date/time literals without discarding a supplied time."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    date_value = _date_only(normalized)
    if date_value is not None:
        return datetime.combine(date_value, datetime.min.time())
    try:
        return datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError:
        pass
    if normalized.replace(".", "", 1).isdigit():
        try:
            serial = float(normalized)
            if 20_000 <= serial <= 60_000:
                return datetime(1899, 12, 30) + timedelta(days=serial)
        except ValueError:
            pass
    formats = (
        "%d/%m/%Y %H:%M:%S.%f", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M",
        "%d/%m/%Y %I:%M:%S %p", "%d/%m/%Y %I:%M %p",
        "%d-%m-%Y %H:%M:%S.%f", "%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M",
        "%d-%m-%Y %I:%M:%S %p", "%d-%m-%Y %I:%M %p",
        "%d.%m.%Y %H:%M:%S.%f", "%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M",
        "%d.%m.%Y %I:%M:%S %p", "%d.%m.%Y %I:%M %p",
        "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
        "%Y-%m-%d %I:%M:%S %p", "%Y-%m-%d %I:%M %p",
        "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M",
    )
    for format_string in formats:
        try:
            return datetime.strptime(normalized, format_string)
        except ValueError:
            continue
    return None


def _parse_text_timestamp_sql(column: str) -> str:
    """Build a tolerant SQL timestamp expression for legacy text date columns."""
    value = f"TRIM(CAST({column} AS VARCHAR))"
    timezone_formats = [
        "%Y-%m-%d %H:%M:%S.%f%z",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%d/%m/%Y %H:%M:%S.%f%z",
        "%d/%m/%Y %H:%M:%S%z",
        "%d/%m/%Y %H:%M%z",
        "%d-%m-%Y %H:%M:%S.%f%z",
        "%d-%m-%Y %H:%M:%S%z",
        "%d-%m-%Y %H:%M%z",
    ]
    formats = [
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %I:%M:%S %p",
        "%Y-%m-%d %I:%M %p",
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%d/%m/%Y %H:%M:%S.%f",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y %I:%M:%S %p",
        "%d/%m/%Y %I:%M %p",
        "%d/%m/%Y",
        "%d-%m-%Y %H:%M:%S.%f",
        "%d-%m-%Y %H:%M:%S",
        "%d-%m-%Y %H:%M",
        "%d-%m-%Y %I:%M:%S %p",
        "%d-%m-%Y %I:%M %p",
        "%d-%m-%Y",
        "%d.%m.%Y %H:%M:%S.%f",
        "%d.%m.%Y %H:%M:%S",
        "%d.%m.%Y %H:%M",
        "%d.%m.%Y %I:%M:%S %p",
        "%d.%m.%Y %I:%M %p",
        "%d.%m.%Y",
        "%d thg %m %Y %H:%M:%S",
        "%d thg %m %Y",
    ]
    format_list = "[" + ", ".join(
        "'" + format_string.replace("'", "''") + "'"
        for format_string in formats
    ) + "]"
    timezone_format_list = "[" + ", ".join(
        "'" + format_string.replace("'", "''") + "'"
        for format_string in timezone_formats
    ) + "]"
    excel_serial = f"TRY_CAST({value} AS DOUBLE)"
    return (
        "COALESCE("
        f"TRY_STRPTIME({value}, {timezone_format_list}), "
        f"TRY_STRPTIME({value}, {format_list}), "
        f"TRY_CAST({value} AS TIMESTAMPTZ), "
        f"TRY_CAST({value} AS TIMESTAMP), "
        "CASE WHEN "
        f"{excel_serial} BETWEEN 20000 AND 60000 "
        "THEN TIMESTAMP '1899-12-30' + "
        f"({excel_serial} * INTERVAL '1 day') END"
        ")"
    )


def _normalize_scalar(
    value: Any, database: BusinessDatabase, table: str, column: str, column_type: str
) -> Any:
    if value is None:
        return value
    if "DATE" in column_type or "TIMESTAMP" in column_type:
        if isinstance(value, (date, datetime)):
            return value
        if isinstance(value, str):
            parsed_day = _date_only(value)
            if parsed_day is not None:
                return parsed_day
            parsed_datetime = _parse_filter_datetime(value)
            if parsed_datetime is not None:
                return (
                    parsed_datetime.date()
                    if "DATE" in column_type and "TIME" not in column_type
                    else parsed_datetime
                )
            try:
                return (
                    datetime.fromisoformat(value).date()
                    if "DATE" in column_type and "TIME" not in column_type
                    else datetime.fromisoformat(value)
                )
            except ValueError:
                return value
    if any(token in column_type for token in ("INT", "DECIMAL", "NUMERIC", "DOUBLE", "FLOAT", "REAL")):
        if isinstance(value, (int, float, Decimal)):
            return value
        if isinstance(value, str):
            numeric = parse_measure_number(value)
            return numeric if numeric is not None else value
    return _stored_value(database, table, column, value)


def _normalize_filter(
    item: Filter | FilterGroup,
    database: BusinessDatabase,
    table: str,
    changes: list[dict[str, Any]],
    column_tables: dict[str, str] | None = None,
    table_aliases: dict[str, str] | None = None,
) -> None:
    if isinstance(item, FilterGroup):
        for condition in item.conditions:
            _normalize_filter(
                condition, database, table, changes, column_tables, table_aliases
            )
        return
    if item.operator in {"is_null", "is_not_null"}:
        return
    # A filter may refer to a joined relation.  The old implementation always
    # described it on the base table, which turned valid joined filters into a
    # DuckDB Binder Error before SQL compilation.
    owner = item.table or (column_tables or {}).get(item.column, table)
    source_table = (table_aliases or {}).get(owner, owner)
    if (
        source_table not in database.tables
        or item.column not in database.tables[source_table].columns
    ):
        return
    column_type = _column_type(database, source_table, item.column)
    values = item.value if isinstance(item.value, list) else [item.value]
    normalized_values = [
        _normalize_scalar(value, database, source_table, item.column, column_type)
        for value in values
    ]
    normalized_value: Any = (
        normalized_values if isinstance(item.value, list) else normalized_values[0]
    )
    if normalized_value != item.value:
        changes.append(
            {
                "table": source_table,
                "column": item.column,
                "operator": item.operator,
                "original_value": item.value,
                "normalized_value": normalized_value,
                "column_type": column_type,
            }
        )
        item.value = normalized_value


def _normalize_metric(
    metric: Any,
    database: BusinessDatabase,
    table: str,
    column_tables: dict[str, str],
    changes: list[dict[str, Any]],
    table_aliases: dict[str, str],
) -> None:
    for condition in (metric.filter, metric.denominator_filter):
        if condition is not None:
            _normalize_filter(
                condition, database, table, changes, column_tables, table_aliases
            )


def normalize_plan(
    plan: QueryPlan, database: BusinessDatabase
) -> tuple[QueryPlan, list[dict[str, Any]]]:
    """Normalize condition values against the loaded schema and stored values."""
    normalized = plan.model_copy(deep=True)
    changes: list[dict[str, Any]] = []
    for cte in normalized.ctes:
        cte.query, cte_changes = normalize_plan(cte.query, database)
        changes.extend(cte_changes)
    for operation in normalized.set_operations:
        operation.query, operation_changes = normalize_plan(operation.query, database)
        changes.extend(operation_changes)
    column_tables = _column_tables(normalized, database)
    table_aliases = {
        alias: source
        for source, alias in [
            (normalized.table, normalized.alias),
            *((join.table, join.alias) for join in normalized.joins),
        ]
        if alias is not None
    }
    for item in normalized.filters:
        _normalize_filter(
            item, database, normalized.table, changes, column_tables, table_aliases
        )
    for item in normalized.comparison_filters:
        _normalize_filter(
            item, database, normalized.table, changes, column_tables, table_aliases
        )
    for item in normalized.having_filters:
        _normalize_filter(
            item, database, normalized.table, changes, column_tables, table_aliases
        )
    for item in normalized.qualify_filters:
        _normalize_filter(
            item, database, normalized.table, changes, column_tables, table_aliases
        )
    metrics = list(normalized.metrics)
    if normalized.metric is not None:
        metrics.append(normalized.metric)
    for metric in metrics:
        _normalize_metric(
            metric, database, normalized.table, column_tables, changes, table_aliases
        )
    if normalized.pivot is not None:
        for metric in normalized.pivot.using:
            _normalize_metric(
                metric, database, normalized.table, column_tables, changes, table_aliases
            )
    return normalized, changes


def _plan_output_columns(plan: QueryPlan) -> list[str]:
    """Return the column names emitted by a query plan in projection order."""
    columns = ["group_period"] if plan.group_by_period else list(plan.group_by)
    metrics = ([plan.metric] if plan.metric is not None else []) + list(plan.metrics)
    if metrics:
        columns.extend(
            metric.alias or ("metric" if index == 0 else f"metric_{index}")
            for index, metric in enumerate(metrics)
        )
    else:
        columns.extend(plan.select_columns)
    return columns


def _cte_columns(plan: QueryPlan, name: str) -> set[str]:
    for cte in plan.ctes:
        if cte.name == name:
            return set(_plan_output_columns(cte.query))
    return set()


def _column_tables(plan: QueryPlan, database: BusinessDatabase) -> dict[str, str]:
    tables: dict[str, str] = {}
    if plan.table in database.tables:
        for column in database.tables[plan.table].columns:
            tables.setdefault(column, plan.table)
    for join in plan.joins:
        join_columns = (
            database.tables[join.table].columns
            if join.table in database.tables
            else _cte_columns(plan, join.table)
        )
        owner = join.table
        for column in join_columns:
            tables.setdefault(column, owner)
    return tables


def _validate(plan: QueryPlan, database: BusinessDatabase) -> None:
    physical_or_cte = set(database.tables) | {cte.name for cte in plan.ctes}
    table_aliases = {
        alias: source
        for source, alias in [
            (plan.table, plan.alias),
            *((join.table, join.alias) for join in plan.joins),
        ]
        if alias is not None
    }
    if plan.table not in physical_or_cte:
        raise ValueError(f"Unknown table: {plan.table}")
    base_columns = (
        set(database.tables[plan.table].columns)
        if plan.table in database.tables
        else _cte_columns(plan, plan.table)
    )
    columns = set(base_columns)
    for join in plan.joins:
        if join.table not in physical_or_cte:
            raise ValueError(f"Unknown join table: {join.table}")
        join_columns = (
            set(database.tables[join.table].columns)
            if join.table in database.tables
            else _cte_columns(plan, join.table)
        )
        if VALIDATE_JOIN_COLUMNS:
            for condition in join.on_conditions:
                if condition.left_column not in columns:
                    raise ValueError(
                        f"Unknown left join column in {plan.table}: {condition.left_column}"
                    )
                if condition.right_column not in join_columns:
                    raise ValueError(
                        f"Unknown right join column in {join.table}: {condition.right_column}"
                    )
        columns.update(join_columns)
    metrics = ([plan.metric] if plan.metric is not None else []) + list(plan.metrics)
    referenced = set(plan.group_by)
    referenced.update(plan.select_columns)
    if plan.group_by_period_column:
        referenced.add(plan.group_by_period_column)
    referenced.update(item.column for item in _iter_filters(plan.filters))
    referenced.update(item.column for item in plan.comparison_filters)
    referenced.update(item.column for item in _iter_filters(plan.having_filters))
    referenced.update(item.column for item in _iter_filters(plan.qualify_filters))
    for metric in metrics:
        referenced.add(metric.column)
        if metric.weight_column:
            referenced.add(metric.weight_column)
        if metric.denominator_column:
            referenced.add(metric.denominator_column)
        for condition in (metric.filter, metric.denominator_filter):
            if condition is not None:
                referenced.update(item.column for item in _iter_filters([condition]))
    referenced.discard("*")
    unknown = referenced - columns
    if unknown:
        raise ValueError(f"Unknown columns in {plan.table}: {sorted(unknown)}")
    filters = (
        plan.filters
        + plan.comparison_filters
        + plan.having_filters
        + plan.qualify_filters
    )
    for metric in metrics:
        filters.extend(
            condition
            for condition in (metric.filter, metric.denominator_filter)
            if condition is not None
        )
    for item in _iter_filters(filters):
        if item.table is not None:
            owner = table_aliases.get(item.table, item.table)
            if owner not in physical_or_cte:
                raise ValueError(f"Unknown filter table: {item.table}")
            owner_columns = (
                set(database.tables[owner].columns)
                if owner in database.tables
                else _cte_columns(plan, owner)
            )
            if item.column not in owner_columns:
                raise ValueError(f"Unknown filter column in {item.table}: {item.column}")

    _validate_set_operations_in(plan, database)


def _validate_set_operations_in(plan: QueryPlan, database: BusinessDatabase) -> None:
    """Validate set-operation branches, including branches nested in CTEs."""
    if plan.set_operations:
        expected_columns = _plan_output_columns(plan)
        if not expected_columns:
            raise ValueError("Set operations require a query that projects columns")
        for operation in plan.set_operations:
            _validate(operation.query, database)
            branch_columns = _plan_output_columns(operation.query)
            if branch_columns != expected_columns:
                raise ValueError(
                    "Set-operation branches must project the same columns in the same order; "
                    f"expected {expected_columns}, got {branch_columns}"
                )
    for cte in plan.ctes:
        _validate_set_operations_in(cte.query, database)

def _compile_filter(
    item: Filter | FilterGroup,
    params: list[Any],
    column_reference: Any,
    column_type_reference: Any | None = None,
) -> str:
    if isinstance(item, FilterGroup):
        if not item.conditions:
            return "TRUE" if item.logic == "and" else "FALSE"
        operator = " AND " if item.logic == "and" else " OR "
        return "(" + operator.join(
            _compile_filter(
                condition, params, column_reference, column_type_reference
            )
            for condition in item.conditions
        ) + ")"

    column = column_reference(item.column, item.table)
    column_type = (
        str(column_type_reference(item.column, item.table)).upper()
        if column_type_reference is not None
        else ""
    )
    is_text_column = any(
        marker in column_type for marker in ("VARCHAR", "CHAR", "STRING", "TEXT")
    )
    date_column = _parse_text_timestamp_sql(column) if is_text_column else column

    def day_bounds(value: Any) -> tuple[datetime, datetime] | None:
        value_date = _date_only(value)
        if value_date is None:
            return None
        start = datetime.combine(value_date, datetime.min.time())
        return start, start + timedelta(days=1)

    # A date-only equality means the whole calendar day for timestamp columns.
    # The same parsed expression also lets legacy VARCHAR date columns match
    # dd/mm/yyyy, ISO date/time, and Excel serial representations.
    if item.operator in {"eq", "neq", "gt", "gte", "lt", "lte"}:
        bounds = day_bounds(item.value)
        if bounds is not None and ("TIMESTAMP" in column_type or is_text_column):
            start, end = bounds
            if item.operator == "eq":
                params.extend((start, end))
                return f"({date_column} >= ? AND {date_column} < ?)"
            if item.operator == "neq":
                params.extend((start, end))
                return f"NOT ({date_column} >= ? AND {date_column} < ?)"
            if item.operator == "gt":
                params.append(end)
                return f"{date_column} >= ?"
            if item.operator == "gte":
                params.append(start)
                return f"{date_column} >= ?"
            if item.operator == "lt":
                params.append(start)
                return f"{date_column} < ?"
            params.append(end)
            return f"{date_column} < ?"

    if item.operator in {"is_null", "is_not_null"}:
        return f"{column} IS {'NOT ' if item.operator == 'is_not_null' else ''}NULL"
    if item.operator in {"in", "not_in"}:
        values = item.value if isinstance(item.value, list) else [item.value]
        if (
            values
            and ("TIMESTAMP" in column_type or is_text_column)
            and all(day_bounds(value) is not None for value in values)
        ):
            day_terms = []
            for value in values:
                start, end = day_bounds(value)  # type: ignore[misc]
                params.extend((start, end))
                day_terms.append(f"({date_column} >= ? AND {date_column} < ?)")
            joined_days = "(" + " OR ".join(day_terms) + ")"
            return f"NOT {joined_days}" if item.operator == "not_in" else joined_days
        if values and is_text_column:
            parsed_values = [_parse_filter_datetime(value) for value in values]
            if all(value is not None for value in parsed_values):
                params.extend(parsed_values)
                operator = "NOT IN" if item.operator == "not_in" else "IN"
                return (
                    f"{date_column} {operator} "
                    f"({', '.join('?' for _ in parsed_values)})"
                )
    if item.operator == "between":
        if not isinstance(item.value, list) or len(item.value) != 2:
            raise ValueError("between filter requires two values")
        start_bounds, end_bounds = day_bounds(item.value[0]), day_bounds(item.value[1])
        if "TIMESTAMP" in column_type or is_text_column:
            start_value = (
                start_bounds[0]
                if start_bounds is not None
                else _parse_filter_datetime(item.value[0])
            )
            end_value = (
                end_bounds[1]
                if end_bounds is not None
                else _parse_filter_datetime(item.value[1])
            )
            if start_value is not None and end_value is not None:
                params.extend((start_value, end_value))
                upper_operator = "<" if end_bounds is not None else "<="
                return (
                    f"({date_column} >= ? AND {date_column} "
                    f"{upper_operator} ?)"
                )
        params.extend(item.value)
        return f"{column} BETWEEN ? AND ?"
    if item.operator in {"in", "not_in"}:
        values = item.value if isinstance(item.value, list) else [item.value]
        if not values:
            return "TRUE" if item.operator == "not_in" else "FALSE"
        params.extend(values)
        operator = "NOT IN" if item.operator == "not_in" else "IN"
        return f"{column} {operator} ({', '.join('?' for _ in values)})"
    if item.operator == "contains":
        params.append(f"%{item.value}%")
        return f"LOWER(CAST({column} AS VARCHAR)) LIKE LOWER(?)"
    if is_text_column and item.operator in {"eq", "neq", "gt", "gte", "lt", "lte"}:
        parsed_value = _parse_filter_datetime(item.value)
        if parsed_value is not None:
            operators = {
                "eq": "=",
                "neq": "<>",
                "gt": ">",
                "gte": ">=",
                "lt": "<",
                "lte": "<=",
            }
            params.append(parsed_value)
            return f"{date_column} {operators[item.operator]} ?"
    operators = {
        "eq": "=",
        "neq": "<>",
        "gt": ">",
        "gte": ">=",
        "lt": "<",
        "lte": "<=",
    }
    if item.operator not in operators:
        raise ValueError(f"Unsupported filter operator: {item.operator}")
    params.append(item.value)
    return f"{column} {operators[item.operator]} ?"


def compile_query(plan: QueryPlan, database: BusinessDatabase) -> tuple[str, list[Any]]:
    plan = _normalize_plan_columns(plan, database)
    if VALIDATE_PLAN_REFERENCES:
        _validate(plan, database)
    return _compile_query(plan, database, include_ctes=True)


def _compile_query(
    plan: QueryPlan, database: BusinessDatabase, *, include_ctes: bool
) -> tuple[str, list[Any]]:
    """Compile a plan, including its local CTE namespace."""
    table = _quote(plan.table)

    # A plan stores column names without an owning table.  Once joins are
    # present, an unqualified name can bind to more than one relation.  Keep
    # the base table as the conservative default (this preserves the legacy
    # meaning of plan columns), and use the first joined relation only when a
    # column does not exist in the base table.
    base_columns = (
        set(database.tables[plan.table].columns)
        if plan.table in database.tables
        else _cte_columns(plan, plan.table)
    )
    joined_columns = [
        (
            join.alias or f"_join_{index}",
            (
                set(database.tables[join.table].columns)
                if join.table in database.tables
                else _cte_columns(plan, join.table)
            ),
        )
        for index, join in enumerate(plan.joins)
    ]

    def column_reference(column: str, owner: str | None = None) -> str:
        if column in {"", "*"}:
            return _quote(column)
        if owner is not None:
            if owner in {plan.table, plan.alias}:
                return f"{table}.{_quote(column)}"
            for index, join in enumerate(plan.joins):
                alias = join.alias or f"_join_{index}"
                if owner == join.table or owner == join.alias:
                    return f"{_quote(alias)}.{_quote(column)}"
        if column in base_columns:
            return f"{table}.{_quote(column)}"
        for alias, columns in joined_columns:
            if column in columns:
                return f"{_quote(alias)}.{_quote(column)}"
        # _validate() should have rejected this already. Keep a quoted
        # fallback so this helper remains safe for unusual table metadata.
        return _quote(column)

    column_type_cache: dict[tuple[str, str], str] = {}

    def column_type_reference(column: str, owner: str | None = None) -> str:
        source_table: str | None = None
        if owner in {plan.table, plan.alias}:
            source_table = plan.table
        elif owner is not None:
            for index, join in enumerate(plan.joins):
                alias = join.alias or f"_join_{index}"
                if owner in {join.table, join.alias, alias}:
                    source_table = join.table
                    break
        elif column in base_columns:
            source_table = plan.table
        else:
            for join, (_, columns) in zip(plan.joins, joined_columns):
                if column in columns:
                    source_table = join.table
                    break

        if (
            source_table is None
            or source_table not in database.tables
            or column not in database.tables[source_table].columns
        ):
            return ""
        key = (source_table, column)
        if key not in column_type_cache:
            column_type_cache[key] = _column_type(
                database, source_table, column
            )
        return column_type_cache[key]

    def having_column_reference(column: str, owner: str | None = None) -> str:
        """Reference a HAVING column at the grouped grain.

        A planner can express an order-level threshold against a line-item
        amount column.  Such a column is not part of GROUP BY, so it must be
        aggregated before the HAVING predicate is applied.  SUM is the
        intended order-level meaning for numeric threshold filters.
        """
        reference = column_reference(column, owner)
        if column in plan.group_by:
            return reference
        return f"SUM({number_sql_expression(reference)})"

    params: list[Any] = []

    def compile_metric(metric: Metric) -> str:
        metric_column = column_reference(metric.column)
        numeric = number_sql_expression(metric_column)

        def aggregate_with_filter(expression: str, condition: Any | None) -> str:
            aggregate = f"SUM({expression})"
            if condition is not None:
                predicate = _compile_filter(
                    condition, params, column_reference, column_type_reference
                )
                aggregate += f" FILTER (WHERE {predicate})"
            return aggregate

        def count_expression(condition: Any | None) -> str:
            if metric.column in {"", "*"}:
                expression = "COUNT(*)"
            elif metric.distinct or metric.aggregation == "count_distinct":
                expression = f"COUNT(DISTINCT {metric_column})"
            else:
                expression = "COUNT(*)"
            if condition is not None:
                predicate = _compile_filter(
                    condition, params, column_reference, column_type_reference
                )
                expression += f" FILTER (WHERE {predicate})"
            return expression

        if metric.aggregation == "count":
            return count_expression(metric.filter)
        if metric.aggregation == "count_distinct":
            return count_expression(metric.filter)
        if metric.aggregation == "percentage":
            numerator = count_expression(metric.filter)
            denominator = count_expression(metric.denominator_filter)
            return f"ROUND(100.0 * ({numerator}) / NULLIF(({denominator}), 0), 2)"
        if metric.aggregation == "ratio":
            if not metric.denominator_column:
                raise ValueError("ratio aggregation requires denominator_column")
            numerator = aggregate_with_filter(numeric, metric.filter)
            denominator_column = column_reference(metric.denominator_column)
            denominator_numeric = number_sql_expression(denominator_column)
            denominator = aggregate_with_filter(
                denominator_numeric, metric.denominator_filter
            )
            return f"ROUND(100.0 * ({numerator}) / NULLIF(({denominator}), 0), 2)"
        if metric.aggregation in {"weighted_avg", "weighted_average"}:
            if not metric.weight_column:
                raise ValueError(
                    "weighted_avg aggregation requires weight_column"
                )
            weight_reference = column_reference(metric.weight_column)
            weight = number_sql_expression(weight_reference)
            numerator = aggregate_with_filter(f"({numeric}) * ({weight})", metric.filter)
            denominator = aggregate_with_filter(weight, metric.filter)
            return f"{numerator} / NULLIF({denominator}, 0)"
        function = "AVG" if metric.aggregation == "mean" else metric.aggregation.upper()
        aggregate = f"{function}({numeric})"
        if metric.filter is not None:
            predicate = _compile_filter(
                metric.filter, params, column_reference, column_type_reference
            )
            aggregate += f" FILTER (WHERE {predicate})"
        return aggregate

    metrics = ([plan.metric] if plan.metric is not None else []) + list(plan.metrics)
    metric_sqls = [compile_metric(metric) for metric in metrics]
    if plan.group_by_period:
        if not plan.group_by:
            raise ValueError("group_by_period requires a date column in group_by")
        period = "month" if plan.group_by_period == "month" else "day"
        group_expression = f"DATE_TRUNC('{period}', CAST({column_reference(plan.group_by[0])} AS DATE))"
        select_parts = [f"{group_expression} AS group_period"]
    else:
        select_parts = [column_reference(column) for column in plan.group_by]
    if metrics:
        for index, (metric, metric_sql) in enumerate(zip(metrics, metric_sqls)):
            alias = metric.alias or ("metric" if index == 0 else f"metric_{index}")
            select_parts.append(f"{metric_sql} AS {_quote(alias)}")
    elif plan.select_columns:
        select_parts.extend(column_reference(column) for column in plan.select_columns)
    else:
        raise ValueError("Query plan requires metric, metrics, or select_columns")
    distinct = " DISTINCT" if plan.distinct else ""
    sql = f"SELECT{distinct} {', '.join(select_parts)} FROM {table}"
    joined_before: list[tuple[str, set[str]]] = []
    for index, join in enumerate(plan.joins):
        join_table = _quote(join.table)
        join_alias = _quote(join.alias or f"_join_{index}")
        sql += f" {join.join_type.upper()} JOIN {join_table} AS {join_alias}"
        if join.join_type != "cross":
            if join.on_conditions:
                operators = {
                    "eq": "=",
                    "neq": "<>",
                    "gt": ">",
                    "gte": ">=",
                    "lt": "<",
                    "lte": "<=",
                }
                clauses = []
                for condition in join.on_conditions:
                    if condition.operator not in operators:
                        raise ValueError(
                            f"Unsupported join operator: {condition.operator}"
                        )
                    if condition.left_column in base_columns:
                        left_reference = f"{table}.{_quote(condition.left_column)}"
                    else:
                        left_reference = next(
                            (
                                f"{_quote(alias)}.{_quote(condition.left_column)}"
                                for alias, columns in joined_before
                                if condition.left_column in columns
                            ),
                            None,
                        )
                    if left_reference is None:
                        if VALIDATE_JOIN_COLUMNS:
                            raise ValueError(
                                "Unknown left join column before joining "
                                f"{join.table!r}: {condition.left_column}"
                            )
                        # Keep compiling when schema-level validation is off;
                        # DuckDB will decide whether this base-table reference exists.
                        left_reference = f"{table}.{_quote(condition.left_column)}"
                    clauses.append(
                        f"{left_reference} "
                        f"{operators[condition.operator]} "
                        f"{join_alias}.{_quote(condition.right_column)}"
                    )
                sql += " ON " + " AND ".join(clauses)
            else:
                raise ValueError(f"Join {join.table!r} requires structured on_conditions")
        joined_table_columns = (
            set(database.tables[join.table].columns)
            if join.table in database.tables
            else _cte_columns(plan, join.table)
        )
        joined_before.append((join.alias or f"_join_{index}", joined_table_columns))
    predicates = [
        _compile_filter(item, params, column_reference, column_type_reference)
        for item in plan.filters
    ]
    if predicates:
        sql += " WHERE " + " AND ".join(predicates)
    if plan.group_by:
        group_expression = (
            f"DATE_TRUNC('{plan.group_by_period}', CAST({column_reference(plan.group_by[0])} AS DATE))"
            if plan.group_by_period
            else ", ".join(column_reference(column) for column in plan.group_by)
        )
        sql += " GROUP BY " + group_expression
    having_predicates = [
        _compile_filter(
            item, params, having_column_reference, column_type_reference
        )
        for item in plan.having_filters
    ]
    if having_predicates:
        sql += " HAVING " + " AND ".join(having_predicates)
    if plan.set_operations:
        operators = {
            "union": "UNION",
            "union_all": "UNION ALL",
            "intersect": "INTERSECT",
            "except": "EXCEPT",
        }
        combined_sql = sql
        combined_params = params
        for index, operation in enumerate(plan.set_operations):
            branch_sql, branch_params = _compile_query(
                operation.query, database, include_ctes=False
            )
            operator = operators[operation.op_type]
            combined_sql = (
                f"SELECT * FROM ({combined_sql}) AS {_quote(f'_set_left_{index}')} "
                f"{operator} "
                f"SELECT * FROM ({branch_sql}) AS {_quote(f'_set_right_{index}')}"
            )
            combined_params = combined_params + branch_params
        sql = combined_sql
        params = combined_params

    if plan.order_by:
        order = plan.order_by[0]
        if plan.set_operations:
            if order.expression == "metric":
                first_metric = plan.metric or (plan.metrics[0] if plan.metrics else None)
                expression = (first_metric.alias or "metric") if first_metric else order.expression
            elif order.expression == "group" and plan.group_by:
                expression = plan.group_by[0]
            elif plan.group_by_period and order.expression in {"group_period", "group"}:
                expression = "group_period"
            elif order.expression.isdigit():
                expression = order.expression
            else:
                expression = _quote(order.expression)
        else:
            expression = "metric" if order.expression == "metric" else (
                "group_period" if plan.group_by_period else column_reference(plan.group_by[0])
            )
        sql += f" ORDER BY {expression} {order.direction.upper()}"
    if plan.limit is not None:
        sql += " LIMIT ?"
        params.append(plan.limit)
    if include_ctes and plan.ctes:
        cte_sql: list[str] = []
        cte_params: list[Any] = []
        for cte in plan.ctes:
            query_sql, query_params = _compile_query(
                cte.query, database, include_ctes=False
            )
            cte_sql.append(f"{_quote(cte.name)} AS ({query_sql})")
            cte_params.extend(query_params)
        sql = "WITH " + ", ".join(cte_sql) + " " + sql
        params = cte_params + params
    return sql, params


def execute_plan(plan: QueryPlan, database: BusinessDatabase) -> list[dict[str, Any]]:
    rows, _ = execute_plan_with_trace(plan, database)
    return rows


def execute_plan_with_trace(
    plan: QueryPlan, database: BusinessDatabase
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Execute a plan and return the SQL statements used for its calculation."""
    normalized_plan, changes = normalize_plan(plan, database)
    sql, params = compile_query(normalized_plan, database)
    result = database.connection.execute(sql, params).fetchdf()
    trace: list[dict[str, Any]] = [
        {"kind": "normalization", "changes": changes}
    ]
    trace.append({"kind": "current", "sql": sql, "params": params})
    records = result.to_dict(orient="records")
    metrics = (
        [normalized_plan.metric]
        if normalized_plan.metric is not None
        else list(normalized_plan.metrics)
    )
    if (
        normalized_plan.answer_type == "number"
        and normalized_plan.group_by
        and len(metrics) == 1
        and metrics[0].aggregation in {"count", "count_distinct"}
    ):
        records = [{"metric": len(records)}]
    if not normalized_plan.comparison_filters:
        return records, trace
    comparison_plan = normalized_plan.model_copy(
        update={
            "filters": normalized_plan.comparison_filters,
            "comparison_filters": [],
        }
    )
    comparison_sql, comparison_params = compile_query(comparison_plan, database)
    comparison_result = database.connection.execute(
        comparison_sql, comparison_params
    ).fetchdf()
    trace.append(
        {
            "kind": "comparison",
            "sql": comparison_sql,
            "params": comparison_params,
        }
    )
    comparison_rows = comparison_result.to_dict(orient="records")
    if normalized_plan.group_by or normalized_plan.group_by_period:
        raise ValueError("Comparison with grouping is not supported")
    current = result.iloc[0]["metric"] if not result.empty else 0
    previous = comparison_rows[0]["metric"] if comparison_rows else 0
    if normalized_plan.answer_type == "percent":
        value = None if not previous else round((current - previous) * 100 / previous, 2)
    else:
        value = current - previous
    return [{"metric": value}], trace
