from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
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
        if query.select_columns:
            return [{"name": column} for column in query.select_columns]
        metrics = ([query.metric] if query.metric is not None else []) + list(query.metrics)
        return [{"name": metric.alias or ("metric" if index == 0 else f"metric_{index}")}
                for index, metric in enumerate(metrics)]
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

    schema_table = _schema_table(context, completed.table)
    if completed.table not in database.tables:
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

    columns = _columns_for(context, database, completed.table)
    table_columns = set(database.tables[completed.table].columns) if completed.table in database.tables else {
        str(column["name"]) for column in columns
    }

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

    transaction_date = _best_role_column(columns, "transaction_date", question)
    has_date_filter = bool(
        transaction_date and _has_column_filter(completed.filters, transaction_date)
    )
    date_range = resolve_date_range(question, as_of_date)
    if transaction_date and date_range and not has_date_filter:
        completed.filters.append(
            Filter(column=transaction_date, operator="between", value=list(date_range))
        )
        changes.append({
            "kind": "filter",
            "reason": "question_time_range",
            "column": transaction_date,
            "operator": "between",
            "value": list(date_range),
        })
        has_date_filter = True
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

    status = "completed" if changes else "unchanged"
    return CompletionResult(
        plan=completed,
        original_plan=original,
        changes=changes,
        status=status,
    )
