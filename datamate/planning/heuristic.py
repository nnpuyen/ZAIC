from __future__ import annotations

import calendar
import re
import unicodedata
from datetime import date, timedelta
from typing import Any

from ..ingestion.loader import BusinessDatabase
from .dates import resolve_date_range
from .schemas import Filter, Metric, OrderBy, ParseResult, QueryPlan


def _fold(value: str) -> str:
    value = unicodedata.normalize("NFD", value.lower())
    return "".join(char for char in value if unicodedata.category(char) != "Mn")


def _column_score(column: str, aliases: tuple[str, ...]) -> int:
    folded = _fold(column).replace("_", " ")
    return max((len(alias) for alias in aliases if _fold(alias) in folded), default=0)


def _best_column(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    ranked = sorted(
        (( _column_score(column, aliases), -index, column) for index, column in enumerate(columns)),
        reverse=True,
    )
    return ranked[0][2] if ranked and ranked[0][0] else None


DATE_ALIASES = ("ngay", "date", "datetime", "thoi gian", "created", "time")
REVENUE_ALIASES = ("thanh tien", "doanh thu", "revenue", "total", "amount", "gia tri", "cong")
QUANTITY_ALIASES = ("quy ra", "so luong", "soluong", "quantity", "qty", "sl")
PRODUCT_ALIASES = ("noi dung", "ten hang", "san pham", "mat hang", "product", "item", "hang hoa")
GROUP_ALIASES = ("nhom hang", "category", "group", "loai hang")


class HeuristicParser:
    """Conservative Vietnamese parser for the six v0.1 query capabilities."""

    def parse(
        self, question: str, database: BusinessDatabase, as_of_date: str,
        context: dict[str, Any] | None = None,
    ) -> QueryPlan:
        result = self.parse_result(question, database, as_of_date)
        if result.status != "answerable" or result.plan is None:
            raise ValueError(result.reason or result.status)
        return result.plan

    def parse_result(self, question: str, database: BusinessDatabase, as_of_date: str) -> ParseResult:
        text = _fold(question)
        table, columns = self._choose_table(text, database)
        if table is None:
            return ParseResult(status="abstain", reason="No unambiguous table or metric matched")
        date_column = _best_column(columns, DATE_ALIASES)
        metric_column, aggregation = self._metric(text, columns)
        if metric_column is None or aggregation is None:
            return ParseResult(status="unanswerable", reason="Required metric is not available")

        filters: list[Filter] = []
        current_range = self._date_filter(text, as_of_date)
        if current_range and date_column:
            filters.append(Filter(column=date_column, operator="between", value=list(current_range)))
        elif self._mentions_time(text) and not date_column:
            return ParseResult(status="unanswerable", reason="Question has a time filter but no date column")

        product_column = _best_column(columns, PRODUCT_ALIASES)
        if product_column:
            product_value = self._known_value(question, text, database, table, product_column)
            if product_value is not None:
                filters.append(Filter(column=product_column, operator="eq", value=product_value))

        group_column = self._group_column(text, columns, product_column)
        is_ranking = any(term in text for term in ("ban chay", "ban nhieu", "hot nhat", "cao nhat", "thap nhat"))
        is_grouping = any(term in text for term in ("theo tung", "theo moi", "phan theo", "moi thang", "tung thang"))
        limit = self._limit(text) if is_ranking else None
        if is_ranking and group_column is None:
            group_column = product_column
        comparison_filters: list[Filter] = []
        if self._is_comparison(text):
            if not current_range or not date_column:
                return ParseResult(status="unanswerable", reason="Comparison requires a resolvable date range")
            comparison_range = self._previous_range(current_range)
            comparison_filters = [Filter(column=date_column, operator="between", value=list(comparison_range))]

        group_by_period = "month" if is_grouping and "thang" in text and date_column else None
        group_by = [group_column] if group_column and not group_by_period else ([date_column] if group_by_period else [])
        answer_type = "string" if is_ranking and aggregation == "count" else (
            "percent" if "%" in text or "phan tram" in text or "ti le" in text else "number"
        )
        if is_grouping:
            # Grouped answers include both the grouping label and its measure.
            answer_type = "table"
        if is_ranking and group_column:
            answer_type = "string" if limit == 1 else "table"
        operation = "ranking" if is_ranking else ("grouping" if is_grouping else "aggregation")
        return ParseResult(
            status="answerable",
            plan=QueryPlan(
                operation=operation,
                table=table,
                metric=Metric(column=metric_column, aggregation=aggregation),
                filters=filters,
                comparison_filters=comparison_filters,
                group_by=group_by,
                group_by_period=group_by_period,
                order_by=[OrderBy(expression="metric", direction="desc")] if is_ranking else [],
                limit=limit,
                answer_type=answer_type,
            ),
        )

    def _choose_table(self, text: str, database: BusinessDatabase) -> tuple[str | None, list[str]]:
        candidates: list[tuple[int, str]] = []
        for name, metadata in database.tables.items():
            columns = metadata.columns
            score = max((_column_score(column, REVENUE_ALIASES + QUANTITY_ALIASES) for column in columns), default=0)
            if any(term in text for term in ("doanh thu", "ban duoc", "ban chay", "so luong", "mat hang")):
                score += 2 if _best_column(columns, REVENUE_ALIASES if "doanh thu" in text else QUANTITY_ALIASES) else 0
            candidates.append((score, name))
        candidates.sort(reverse=True)
        if not candidates or candidates[0][0] == 0:
            return None, []
        if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
            return None, []
        return candidates[0][1], database.tables[candidates[0][1]].columns

    def _metric(self, text: str, columns: list[str]) -> tuple[str | None, str | None]:
        if any(term in text for term in ("bao nhieu don", "so don", "bao nhieu hoa don", "ti le don")):
            return _best_column(columns, ("ma hoa don", "ma don", "so ct", "invoice", "bill", "id")), "count_distinct"
        if any(term in text for term in ("so luong", "bao nhieu san pham", "ban chay", "ban nhieu", "don vi")):
            column = _best_column(columns, QUANTITY_ALIASES)
            return (column, "sum") if column else (None, None)
        column = _best_column(columns, REVENUE_ALIASES)
        return (column, "sum") if column else (None, None)

    def _group_column(self, text: str, columns: list[str], product_column: str | None) -> str | None:
        if any(term in text for term in ("san pham", "mat hang", "hang nao", "hang hoa")):
            return product_column
        if "nhom hang" in text or "nhom" in text:
            return _best_column(columns, GROUP_ALIASES)
        return None

    def _date_filter(self, text: str, as_of_date: str) -> tuple[str, str] | None:
        return resolve_date_range(text, as_of_date)

    def _mentions_time(self, text: str) -> bool:
        return any(term in text for term in ("hom nay", "hom qua", "tuan", "thang", "nam nay", "tu ngay"))

    def _is_comparison(self, text: str) -> bool:
        return any(term in text for term in ("tang bao nhieu", "giam bao nhieu", "so voi", "tang truong", "thay doi"))

    def _previous_range(self, current: tuple[str, str]) -> tuple[str, str]:
        start, end = date.fromisoformat(current[0]), date.fromisoformat(current[1])
        days = (end - start).days + 1
        previous_end = start - timedelta(days=1)
        return (previous_end - timedelta(days=days - 1)).isoformat(), previous_end.isoformat()

    def _limit(self, text: str) -> int:
        match = re.search(r"(?:top|top\s*)?(\d+)", text)
        return int(match.group(1)) if match else 1

    def _known_value(
        self, original: str, folded_question: str, database: BusinessDatabase,
        table: str, column: str,
    ) -> str | None:
        cache_key = (table, column)
        if cache_key not in database.distinct_values:
            escaped_column = column.replace('"', '""')
            escaped_table = table.replace('"', '""')
            database.distinct_values[cache_key] = [
                value
                for (value,) in database.connection.execute(
                    f'SELECT DISTINCT "{escaped_column}" FROM "{escaped_table}" '
                    f'WHERE "{escaped_column}" IS NOT NULL LIMIT 2000'
                ).fetchall()
            ]
        for value in database.distinct_values[cache_key]:
            if isinstance(value, str) and _fold(value) in folded_question:
                return value
        return None
