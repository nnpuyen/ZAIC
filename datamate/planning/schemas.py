from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ------------------------------------------------------------------------------
# Filters & Logical Expressions
# ------------------------------------------------------------------------------

FilterOperator = Literal[
    "eq",
    "neq",
    "gt",
    "gte",
    "lt",
    "lte",
    "in",
    "not_in",
    "contains",
    "not_contains",
    "starts_with",
    "ends_with",
    "like",
    "ilike",
    "regex",
    "glob",
    "between",
    "is_null",
    "is_not_null",
]


class Filter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str
    operator: FilterOperator
    value: Any = None
    # Set during plan completion when a column belongs to a joined table.
    table: str | None = None


class FilterGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    logic: Literal["and", "or"] = "and"
    conditions: list[Union[Filter, FilterGroup]] = Field(default_factory=list)


# ------------------------------------------------------------------------------
# Aggregations, Metrics & Window Functions
# ------------------------------------------------------------------------------

AggregationType = Literal[
    "sum",
    "avg",
    "mean",
    "weighted_avg",
    "weighted_average",
    "count",
    "count_distinct",
    "min",
    "max",
    "median",
    "mode",
    "stddev",
    "variance",
    "quantile",
    "string_agg",
    "list",
    "first",
    "last",
    "arg_min",
    "arg_max",
    "corr",
    "covar",
    "percentage",  # Ratio of matching condition vs total (or column A vs B)
    "ratio",
]


class Metric(BaseModel):
    column: str = "*"
    aggregation: AggregationType
    alias: str | None = None
    distinct: bool = False

    # Required for weighted_avg/weighted_average.  The weighted mean is
    # SUM(column * weight_column) / SUM(weight_column), ignoring rows whose
    # numeric value cannot be parsed (as the other numeric aggregations do).
    weight_column: str | None = None
    
    # Conditional Numerator Filter (e.g., COUNT(*) FILTER (WHERE status = 'shipped'))
    filter: FilterGroup | Filter | None = None  
    
    # Explicit Percentage Enhancements
    is_percent_of_total: bool = False  # Triggers SUM(col) * 100.0 / SUM(SUM(col)) OVER ()
    denominator_column: str | None = None  # For ratio between two columns: (col_a / col_b) * 100
    denominator_filter: FilterGroup | Filter | None = None  # For ratio between two filtered counts

    @model_validator(mode="after")
    def validate_metric_dependencies(self) -> "Metric":
        if self.aggregation in {"weighted_avg", "weighted_average"} and not self.weight_column:
            raise ValueError(
                "weighted_avg aggregation requires weight_column"
            )
        if self.aggregation not in {"weighted_avg", "weighted_average"} and self.weight_column:
            raise ValueError(
                "weight_column is only valid with weighted_avg aggregation"
            )
        if self.aggregation == "ratio" and not self.denominator_column:
            raise ValueError("ratio aggregation requires denominator_column")
        return self


class WindowFunction(BaseModel):
    function: Literal[
        "row_number",
        "rank",
        "dense_rank",
        "percent_rank",
        "cume_dist",
        "ntile",
        "lag",
        "lead",
        "first_value",
        "last_value",
        "nth_value",
        "sum",
        "avg",
        "count",
        "min",
        "max",
    ]
    column: str | None = None
    partition_by: list[str] = Field(default_factory=list)
    order_by: list[OrderBy] = Field(default_factory=list)
    frame: str | None = None  # e.g., "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
    alias: str | None = None


# ------------------------------------------------------------------------------
# Joins
# ------------------------------------------------------------------------------

JoinType = Literal[
    "inner",
    "left",
    "right",
    "full",
    "cross",
    "semi",
    "anti",
    "asof",
    "positional",
]


class JoinCondition(BaseModel):
    left_column: str
    right_column: str
    operator: FilterOperator = "eq"


class Join(BaseModel):
    table: str  # Table name, file function read_parquet(...), or CTE alias
    alias: str | None = None
    join_type: JoinType = "inner"
    on_conditions: list[JoinCondition] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_columns(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        if "on_conditions" in value or not {
            "left_column",
            "right_column",
        }.issubset(value):
            return value
        migrated = dict(value)
        migrated["on_conditions"] = [{
            "left_column": migrated.pop("left_column"),
            "right_column": migrated.pop("right_column"),
        }]
        return migrated


# ------------------------------------------------------------------------------
# Grouping & Ordering
# ------------------------------------------------------------------------------

class OrderBy(BaseModel):
    expression: str  # Metric alias, column name, window function, or ordinal index
    direction: Literal["asc", "desc"] = "desc"
    nulls: Literal["first", "last"] | None = None


DatePeriod = Literal[
    "year",
    "quarter",
    "month",
    "week",
    "day",
    "hour",
    "minute",
    "second",
]


# ------------------------------------------------------------------------------
# CTEs, Pivot & Set Operations
# ------------------------------------------------------------------------------

class PivotSpec(BaseModel):
    on: list[str]
    using: list[Metric]
    group_by: list[str] = Field(default_factory=list)


class CTE(BaseModel):
    name: str
    query: QueryPlan


# ------------------------------------------------------------------------------
# Query Plan & Parse Result
# ------------------------------------------------------------------------------

OperationType = Literal[
    "select",
    "aggregation",
    "ranking",
    "grouping",
    "filtering",
    "join",
    "pivot",
    "unpivot",
    "set_operation",
]


class QueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: OperationType = "select"
    table: str  # Table name, CTE name, or file path reader (e.g., read_parquet('...'))
    alias: str | None = None

    # CTEs (WITH clauses)
    ctes: list[CTE] = Field(default_factory=list)
    # Set operations are applied left-to-right to this query and each branch.
    set_operations: list[SetOperationBranch] = Field(default_factory=list)

    # Selections & Aggregations
    select_columns: list[str] = Field(default_factory=list)
    metric: Metric | None = None  # Single metric backward compatibility
    metrics: list[Metric] = Field(default_factory=list)  # Multiple metrics support
    window_functions: list[WindowFunction] = Field(default_factory=list)

    # Joins
    joins: list[Join] = Field(default_factory=list)

    # Filter Clauses
    filters: list[Union[Filter, FilterGroup]] = Field(default_factory=list)
    comparison_filters: list[Filter] = Field(default_factory=list)
    having_filters: list[Union[Filter, FilterGroup]] = Field(default_factory=list)
    qualify_filters: list[Union[Filter, FilterGroup]] = Field(default_factory=list)

    # Grouping
    group_by: list[str] = Field(default_factory=list)
    group_by_period: DatePeriod | None = None
    group_by_period_column: str | None = None
    grouping_sets: Literal["standard", "rollup", "cube"] = "standard"

    # Ordering & Pagination
    order_by: list[OrderBy] = Field(default_factory=list)
    limit: int | None = Field(default=None, ge=1, le=1000000)
    offset: int | None = Field(default=None, ge=0)

    # DuckDB Native Extensions
    pivot: PivotSpec | None = None
    sample: str | None = None  # e.g., "10%", "1000 rows (bernoulli)"
    distinct: bool = False

    # Metadata
    answer_type: Literal["number", "percent", "string", "date", "list", "table", "boolean"] = "number"

    @model_validator(mode="before")
    @classmethod
    def normalize_filter_shapes(cls, value: Any) -> Any:
        return normalize_plan_filter_shapes(value)

    @model_validator(mode="after")
    def require_set_operation_branches(self) -> "QueryPlan":
        if self.operation == "set_operation" and not self.set_operations:
            raise ValueError("operation='set_operation' requires at least one set_operations branch")
        return self


class SetOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op_type: Literal["union", "union_all", "intersect", "except"]
    left_query: QueryPlan
    right_query: QueryPlan


class SetOperationBranch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op_type: Literal["union", "union_all", "intersect", "except"] = "union"
    query: QueryPlan


QueryPlan.model_rebuild()


class ParseResult(BaseModel):
    status: Literal["answerable", "unanswerable", "abstain"]
    plan: QueryPlan | SetOperation | None = None
    reason: str | None = None


def normalize_plan_filter_shapes(raw_plan: Any) -> Any:
    """Normalize planner filter and set-operation shapes to the native schema."""
    if not isinstance(raw_plan, dict):
        return raw_plan

    normalized = dict(raw_plan)

    # Older planner responses encode extra UNION branches as ``unions``. Map
    # those to the typed set-operation representation before strict validation.
    legacy_unions = normalized.pop("unions", None)
    set_operations = normalized.get("set_operations")
    if legacy_unions is not None:
        if set_operations:
            raise ValueError("Planner response cannot combine 'unions' and 'set_operations'")
        if not isinstance(legacy_unions, list):
            raise ValueError("Planner field 'unions' must be a list of query plans")
        set_operations = [
            {"op_type": "union", "query": branch}
            for branch in legacy_unions
        ]
        normalized["set_operations"] = set_operations

    if isinstance(set_operations, list):
        normalized["set_operations"] = [
            {
                **step,
                "query": normalize_plan_filter_shapes(step["query"]),
            }
            if isinstance(step, dict) and isinstance(step.get("query"), dict)
            else step
            for step in set_operations
        ]
    for key in ("filters", "comparison_filters", "having_filters", "qualify_filters"):
        filters = normalized.get(key)
        if isinstance(filters, dict):
            filters = [filters]
        if isinstance(filters, list):
            normalized[key] = [_normalize_filter_shape(item) for item in filters]

    if "metric" in normalized:
        normalized["metric"] = _normalize_metric_filter_shapes(normalized["metric"])
    metrics = normalized.get("metrics")
    if isinstance(metrics, list):
        normalized["metrics"] = [
            _normalize_metric_filter_shapes(metric) for metric in metrics
        ]

    pivot = normalized.get("pivot")
    if isinstance(pivot, dict) and isinstance(pivot.get("using"), list):
        normalized_pivot = dict(pivot)
        normalized_pivot["using"] = [
            _normalize_metric_filter_shapes(metric) for metric in pivot["using"]
        ]
        normalized["pivot"] = normalized_pivot

    ctes = normalized.get("ctes")
    if isinstance(ctes, list):
        normalized_ctes = []
        for cte in ctes:
            if isinstance(cte, dict) and isinstance(cte.get("query"), dict):
                normalized_cte = dict(cte)
                normalized_cte["query"] = normalize_plan_filter_shapes(cte["query"])
                normalized_ctes.append(normalized_cte)
            else:
                normalized_ctes.append(cte)
        normalized["ctes"] = normalized_ctes

    for key in ("left_query", "right_query", "query_plan"):
        if isinstance(normalized.get(key), dict):
            normalized[key] = normalize_plan_filter_shapes(normalized[key])

    return normalized


def _normalize_metric_filter_shapes(raw_metric: Any) -> Any:
    if not isinstance(raw_metric, dict):
        return raw_metric
    metric = dict(raw_metric)
    for key in ("filter", "denominator_filter"):
        if key in metric:
            metric[key] = _normalize_filter_shape(metric[key])
    return metric


def _normalize_filter_shape(item: Any) -> Any:
    if not isinstance(item, dict):
        return item

    # Some planners serialize union variants with explicit tags. Unwrap only
    # single-key tags; extra keys remain for strict validation to reject.
    if len(item) == 1 and "filter_group" in item:
        return _normalize_filter_shape(item["filter_group"])
    if len(item) == 1 and "filter" in item:
        return _normalize_filter_shape(item["filter"])

    normalized = dict(item)
    if normalized.get("operator") == "not in":
        normalized["operator"] = "not_in"
    conditions = normalized.get("conditions")
    if isinstance(conditions, dict):
        conditions = [conditions]
    if isinstance(conditions, list):
        normalized["conditions"] = [
            _normalize_filter_shape(condition) for condition in conditions
        ]
    return normalized
# from __future__ import annotations

# from typing import Any, Literal

# from pydantic import BaseModel, Field


# FilterOperator = Literal[
#     "eq", "neq", "gt", "gte", "lt", "lte", "in", "not_in", "contains",
#     "between", "is_null", "is_not_null",
# ]


# class Filter(BaseModel):
#     column: str
#     operator: FilterOperator
#     value: Any = None


# class Metric(BaseModel):
#     column: str
#     aggregation: Literal[
#         "sum", "avg", "count", "count_distinct", "min", "max",
#     ]


# class OrderBy(BaseModel):
#     expression: Literal["metric", "group"] = "metric"
#     direction: Literal["asc", "desc"] = "desc"


# class Join(BaseModel):
#     table: str
#     left_column: str
#     right_column: str
#     join_type: Literal["inner", "left", "right", "full"] = "inner"


# class QueryPlan(BaseModel):
#     operation: Literal["aggregation", "ranking", "grouping", "filtering", "join"] = "aggregation"
#     table: str
#     metric: Metric
#     joins: list[Join] = Field(default_factory=list)
#     filters: list[Filter] = Field(default_factory=list)
#     comparison_filters: list[Filter] = Field(default_factory=list)
#     group_by: list[str] = Field(default_factory=list)
#     group_by_period: Literal["month", "day"] | None = None
#     order_by: list[OrderBy] = Field(default_factory=list)
#     limit: int | None = Field(default=None, ge=1, le=1000)
#     answer_type: Literal["number", "percent", "string", "list", "table"] = "number"


# class ParseResult(BaseModel):
#     status: Literal["answerable", "unanswerable", "abstain"]
#     plan: QueryPlan | None = None
#     reason: str | None = None
