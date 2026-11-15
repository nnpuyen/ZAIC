from __future__ import annotations

import csv
import io
import json
import re
import unicodedata
from datetime import date, timedelta
from typing import Any

from .dates import resolve_date_range


_SECTION_NAMES = {
    1: "revenue",
    2: "time",
    3: "ranking",
    4: "debt",
    5: "inventory",
}
_SECTION_TRIGGERS = {
    "revenue": ("doanh thu", "ban duoc", "tien", "don", "thanh toan", "huy", "tra hang"),
    "time": (
        "hom nay", "hom qua", "tuan", "thang", "quy", "nam", "ngay", "tet",
        "am lich", "trang", "ca sang", "ca chieu",
    ),
    "ranking": (
        "top", "xep hang", "ban chay", "ban nhieu", "nhat", "san pham",
        "mat hang", "khach", "nhan vien",
    ),
    "debt": ("no", "cong no", "tra no", "thu no", "tuoi no", "con no"),
    "inventory": ("ton kho", "nhap hang", "gia von", "loi nhuan", "bao hanh", "hang ton"),
}
_LUNAR_HINTS = ("tet", "am lich", "lunar", "thang chap", "thang gieng", "mung")
_ORDER_STATUS_HINTS = ("don", "ban duoc", "doanh thu", "huy", "tra hang")


def _fold(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value.lower().replace("đ", "d"))
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def _tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9_]+", _fold(value)) if len(token) > 1}


def split_rulebook_sections(rulebook: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current: list[str] = []
    current_name = "general"
    for line in rulebook.splitlines():
        match = re.match(r"\s*##\s*(\d+)\.", line)
        if match:
            if current:
                sections[current_name] = "\n".join(current).strip()
            current_name = _SECTION_NAMES.get(int(match.group(1)), f"section_{match.group(1)}")
            current = [line]
        else:
            current.append(line)
    if current:
        sections[current_name] = "\n".join(current).strip()
    return sections


def _column_search_text(column: dict[str, Any]) -> str:
    parts = [
        str(column.get("name", "")),
        str(column.get("canonical_name", "")),
        str(column.get("raw_name", "")),
        str(column.get("role", "")),
        str(column.get("definition", "")),
    ]
    aliases = column.get("aliases") or []
    if isinstance(aliases, list):
        parts.extend(str(alias) for alias in aliases)
    return " ".join(part for part in parts if part)


_GENERIC_QUERY_TOKENS = {
    "bao", "duoc", "hom", "nay", "ngay", "nhieu", "so", "thang", "nam",
    "tuan", "quy", "tong", "trong", "value", "total", "today", "what", "how",
}
_SEMANTIC_TABLE_GROUPS = (
    (("doanh thu", "revenue", "sales", "ban hang", "ban duoc"),
     ("doanh thu", "revenue", "sales", "ban hang", "hoa don", "don hang", "tong tien", "thanh tien")),
    (("nhap hang", "phieu nhap", "mua hang", "nha cung cap", "ncc"),
     ("nhap hang", "phieu nhap", "mua hang", "nha cung cap", "ncc")),
    (("ton kho", "hang ton", "ton cuoi ky", "inventory"),
     ("ton kho", "hang ton", "ton cuoi ky", "baocao ton kho", "inventory")),
    (("khach hang", "khach", "customer"),
     ("khach hang", "customer", "khach")),
    (("cong no", "con no", "tuoi no", "tra no", "debt"),
     ("con no", "cong no", "phieu thu", "payment", "debt")),
    (("loi nhuan", "gia von", "profit", "cost"),
     ("loi nhuan", "gia von", "profit", "cost", "thanh tien")),
    (("san pham", "mat hang", "hang hoa", "product"),
     ("san pham", "mat hang", "hang hoa", "product", "chi tiet hoa don")),
)


def _table_score(question: str, table: dict[str, Any]) -> int:
    question_tokens = _tokens(question) - _GENERIC_QUERY_TOKENS
    table_parts = [
        str(table.get("table", "")),
        str(table.get("canonical_name", "")),
        str(table.get("grain", "")),
        str(table.get("sheet", "")),
    ]
    aliases = table.get("aliases") or []
    if isinstance(aliases, list):
        table_parts.extend(str(alias) for alias in aliases)
    column_text = " ".join(
        _column_search_text(column)
        for column in table.get("columns", [])
        if isinstance(column, dict)
    )
    table_text = " ".join(table_parts)
    table_tokens = _tokens(table_text)
    column_tokens = _tokens(column_text)
    score = 5 * len(question_tokens & table_tokens) + len(question_tokens & column_tokens)
    folded_question = _fold(question)
    folded_table = _fold(table_text + " " + column_text)
    semantic_match = any(
        any(term in folded_question for term in question_terms)
        and any(term in folded_table for term in table_terms)
        for question_terms, table_terms in _SEMANTIC_TABLE_GROUPS
    )
    if semantic_match:
        score += 8

    # A field role can strengthen an existing semantic match, but does not by
    # itself make every date or amount table look relevant to every question.
    if not score:
        return 0
    question_wants_time = any(term in folded_question for term in _SECTION_TRIGGERS["time"])
    question_wants_metric = any(
        term in folded_question
        for term in ("doanh thu", "ban duoc", "so luong", "bao nhieu", "gia tri", "tong")
    )
    question_wants_status = any(term in folded_question for term in _ORDER_STATUS_HINTS)
    for column in table.get("columns", []):
        if not isinstance(column, dict):
            continue
        role = column.get("role")
        if role == "transaction_date" and question_wants_time:
            score += 2
        elif role == "measure" and question_wants_metric:
            score += 2
        elif role == "status" and question_wants_status:
            score += 2
    return score


def _column_schema_key(table: dict[str, Any]) -> str:
    """Return a stable key for a table's column schema, excluding profile data."""
    columns = table.get("columns", [])
    schema_columns: list[dict[str, Any]] = []
    for column in columns:
        if isinstance(column, dict):
            schema_columns.append({
                key: value
                for key, value in column.items()
                if key not in {"sample_values", "distinct_count", "row_count"}
            })
        else:
            schema_columns.append({"name": column})

    # Column order and table metadata do not change the underlying schema.
    schema_columns.sort(
        key=lambda column: json.dumps(
            column, ensure_ascii=False, sort_keys=True, default=str
        )
    )
    return json.dumps(schema_columns, ensure_ascii=False, sort_keys=True, default=str)


def _sheet_period(table: dict[str, Any]) -> tuple[date, date] | None:
    label = f"{table.get('sheet') or ''} {table.get('table') or ''}"
    year_month = re.search(r"(?<!\d)(20\d{2})[-_/ .](0?[1-9]|1[0-2])(?!\d)", label)
    month_year = re.search(r"(?<!\d)(0?[1-9]|1[0-2])[-_/ .](20\d{2})(?!\d)", label)
    if year_month:
        year, month = int(year_month.group(1)), int(year_month.group(2))
    elif month_year:
        month, year = int(month_year.group(1)), int(month_year.group(2))
    else:
        year_match = re.search(r"(?<!\d)(20\d{2})(?!\d)", label)
        if not year_match:
            return None
        year = int(year_match.group(1))
        return date(year, 1, 1), date(year, 12, 31)
    start = date(year, month, 1)
    next_month = date(year + (month == 12), month % 12 + 1, 1)
    return start, next_month - timedelta(days=1)


def _date_value(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _sheet_relevance(
    table: dict[str, Any],
    time_range: tuple[str, str] | None,
    as_of_date: str,
) -> tuple[int, int]:
    period = _sheet_period(table)
    if period is None:
        return 0, 0
    period_start, period_end = period
    if time_range:
        start, end = _date_value(time_range[0]), _date_value(time_range[1])
        if start and end:
            if period_end < start or period_start > end:
                return -100, 0
            overlap = (min(period_end, end) - max(period_start, start)).days + 1
            return 100, overlap
    as_of = _date_value(as_of_date)
    if as_of and period_start <= as_of:
        # Prefer the latest sheet when the question does not identify a period.
        return 1, period_start.toordinal()
    return -1, -period_start.toordinal()


def _calendar_evidence(calendar_text: str, question: str) -> list[dict[str, Any]]:
    folded = _fold(question)
    lunar_requested = any(term in folded for term in _LUNAR_HINTS)
    if not lunar_requested:
        return []
    rows = list(csv.DictReader(io.StringIO(calendar_text)))
    year_match = re.search(r"\b(20\d{2})\b", question)
    lunar_year = int(year_match.group(1)) if year_match else None
    lunar_date_match = re.search(
        r"(?:mùng\s*)?(\d{1,2})\s+tháng\s+(\d{1,2})(?:\s+âm)?", folded
    )
    requested_day = int(lunar_date_match.group(1)) if lunar_date_match else None
    requested_month = int(lunar_date_match.group(2)) if lunar_date_match else None
    selected: list[dict[str, Any]] = []
    for row in rows:
        try:
            row_year = int(row["lunar_year"])
            month = int(row["lunar_month"])
            day = int(row["lunar_day"])
        except (KeyError, TypeError, ValueError):
            continue
        wanted = False
        if "tet" in folded:
            wanted = (month == 12 and day == 23) or (month == 1 and day == 10)
            if lunar_year is not None:
                wanted = wanted and row_year in {lunar_year - 1, lunar_year}
        elif requested_day is not None and requested_month is not None:
            wanted = day == requested_day and month == requested_month
            if lunar_year is not None:
                wanted = wanted and row_year == lunar_year
        else:
            wanted = False
        if wanted:
            selected.append({
                "date": row.get("date"),
                "lunar_day": day,
                "lunar_month": month,
                "lunar_year": row_year,
                "holiday": row.get("holiday", ""),
            })
    return selected


def retrieve_context(
    question: str,
    as_of_date: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Build a compact, question-specific schema and business context."""
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
    folded_question = _fold(question)
    sections = context.get("rulebook_sections") or split_rulebook_sections(
        str(context.get("rulebook", ""))
    )
    selected_sections = {
        name: text for name, text in sections.items()
        if name == "general" or any(trigger in folded_question for trigger in _SECTION_TRIGGERS.get(name, ()))
    }
    if not selected_sections and sections:
        selected_sections = {"general": sections.get("general", "")}

    time_range = resolve_date_range(question, as_of_date)
    calendar_rows = _calendar_evidence(str(context.get("calendar", "")), question)
    calendar_requested = any(term in folded_question for term in _LUNAR_HINTS)
    scored_tables = [
        (table, _table_score(question, table))
        for table in tables
        if isinstance(table, dict)
    ]
    groups: dict[str, list[tuple[dict[str, Any], int]]] = {}
    for table, score in scored_tables:
        groups.setdefault(_column_schema_key(table), []).append((table, score))

    selected_tables: list[tuple[dict[str, Any], int]] = []
    for candidates in groups.values():
        relevant = [(table, score) for table, score in candidates if score > 0]
        choices = relevant or candidates

        # A date range can span multiple sheets with the same schema. Keep all
        # matching sheets so aggregation can cover the full requested period.
        period_matches = [
            item for item in choices
            if time_range and _sheet_relevance(item[0], time_range, as_of_date)[0] == 100
        ]
        if period_matches:
            selected_tables.extend(period_matches)
            continue

        # Otherwise keep the best matching sheet, or one representative for a
        # schema with no lexical match. This guarantees each schema is retained.
        selected_tables.append(max(
            choices,
            key=lambda item: (
                item[1] + _sheet_relevance(item[0], time_range, as_of_date)[0],
                _sheet_relevance(item[0], time_range, as_of_date)[1],
                str(item[0].get("table", "")),
            ),
        ))

    compact_tables: list[dict[str, Any]] = []
    canonical_tables: list[dict[str, Any]] = []
    for table, _ in sorted(selected_tables, key=lambda item: str(item[0].get("table", ""))):
        source_columns = [
            column if isinstance(column, dict) else {"name": str(column)}
            for column in table.get("columns", [])
        ]
        # Filter tables by question, but preserve the complete column catalog
        # for every selected table so identifiers and join keys remain visible.
        compact_columns: list[dict[str, Any]] = []
        for column in source_columns:
            compact = {
                key: column[key]
                for key in (
                    "name", "type", "role", "aliases", "definition", "distinct_count",
                )
                if key in column
            }
            is_unnamed = (
                str(column.get("name", "")).strip().casefold().startswith("unnamed:")
            )
            if "sample_values" in column:
                sample_values = column["sample_values"]
                if is_unnamed:
                    sample_limit = 5
                else:
                    try:
                        distinct_count = int(column.get("distinct_count"))
                    except (TypeError, ValueError):
                        distinct_count = None
                    sample_limit = (
                        None if distinct_count is not None and distinct_count <= 10 else 3
                    )
                compact["sample_values"] = (
                    sample_values[:sample_limit]
                    if sample_limit is not None and isinstance(sample_values, list)
                    else sample_values
                )
            compact_columns.append(compact)
        compact_table: dict[str, Any] = {
            "table": table.get("table"),
            "columns": compact_columns,
        }
        row_count = table.get("row_count", table.get("rows"))
        if row_count is not None:
            compact_table["row_count"] = row_count
        compact_tables.append(compact_table)

        table_canonical: dict[str, Any] = {"table": table.get("table")}
        if table.get("canonical_name") != table.get("table"):
            table_canonical["canonical_name"] = table.get("canonical_name")
        for key in ("aliases", "grain"):
            value = table.get(key)
            if value:
                table_canonical[key] = value
        if len(table_canonical) > 1:
            canonical_tables.append(table_canonical)

    return {
        "business_description": str(context.get("business_description", "")),
        "schema": {"tables": compact_tables},
        "canonical": {"tables": canonical_tables},
        "time": {
            "start_date": time_range[0] if time_range else None,
            "end_date": time_range[1] if time_range else None,
        },
        "rulebook_sections": selected_sections,
        "calendar_evidence": calendar_rows,
    }
