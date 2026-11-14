from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, Field, model_validator


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
    column: str
    operator: FilterOperator
    value: Any = None
    # Set during plan completion when a column belongs to a joined table.
    table: str | None = None


class FilterGroup(BaseModel):
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
    operation: OperationType = "select"
    table: str  # Table name, CTE name, or file path reader (e.g., read_parquet('...'))
    alias: str | None = None

    # CTEs (WITH clauses)
    ctes: list[CTE] = Field(default_factory=list)

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
    answer_type: Literal["number", "percent", "string", "list", "table", "boolean"] = "number"


class SetOperation(BaseModel):
    op_type: Literal["union", "union_all", "intersect", "except"]
    left_query: QueryPlan
    right_query: QueryPlan


class ParseResult(BaseModel):
    status: Literal["answerable", "unanswerable", "abstain"]
    plan: QueryPlan | SetOperation | None = None
    reason: str | None = None
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
