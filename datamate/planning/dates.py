from __future__ import annotations

import calendar
import re
import unicodedata
from datetime import date, timedelta


def _fold(value: str) -> str:
    """Normalize Vietnamese text, including common UTF-8-as-Latin-1 damage."""
    for _ in range(2):
        try:
            repaired = value.encode("latin1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            break
        if repaired == value:
            break
        value = repaired
    value = value.lower().replace("đ", "d")
    decomposed = unicodedata.normalize("NFD", value)
    return " ".join(
        "".join(char for char in decomposed if unicodedata.category(char) != "Mn").split()
    )


def _month_bounds(year: int, month: int) -> tuple[date, date]:
    return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])


def _year_for_month_day(month: int, day: int, anchor: date) -> int:
    try:
        candidate = date(anchor.year, month, day)
    except ValueError:
        return anchor.year
    return anchor.year if candidate <= anchor else anchor.year - 1


def resolve_date_range(expression: str, as_of_date: str) -> tuple[str, str] | None:
    """Resolve common explicit and relative Vietnamese date expressions."""
    anchor = date.fromisoformat(as_of_date)
    text = _fold(expression)

    if any(term in text for term in ("hom nay", "hom ni", "bua nay")):
        return anchor.isoformat(), anchor.isoformat()
    if any(term in text for term in ("hom qua", "bua qua")):
        day = anchor - timedelta(days=1)
        return day.isoformat(), day.isoformat()

    if any(term in text for term in ("tuan truoc", "tuan roi", "tuan vua roi")):
        start = anchor - timedelta(days=anchor.weekday() + 7)
        return start.isoformat(), (start + timedelta(days=6)).isoformat()
    if "tuan nay" in text:
        start = anchor - timedelta(days=anchor.weekday())
        return start.isoformat(), anchor.isoformat()

    if "thang nay" in text:
        return date(anchor.year, anchor.month, 1).isoformat(), anchor.isoformat()
    if any(term in text for term in ("dau nam", "tu dau nam")):
        return date(anchor.year, 1, 1).isoformat(), anchor.isoformat()

    quarter_match = re.search(r"(?:quy)\s*([1-4])(?:\s*/\s*(\d{4})|\s+(\d{4}))?", text)
    if quarter_match:
        quarter = int(quarter_match.group(1))
        year = int(quarter_match.group(2) or quarter_match.group(3) or anchor.year)
        quarter_start_month = (quarter - 1) * 3 + 1
        if not quarter_match.group(2) and not quarter_match.group(3):
            if date(year, quarter_start_month, 1) > anchor:
                year -= 1
        start, _ = _month_bounds(year, quarter_start_month)
        _, end = _month_bounds(year, quarter_start_month + 2)
        return start.isoformat(), end.isoformat()

    month_match = re.search(
        r"(?:thang|th)\s*(\d{1,2})(?:\s*/\s*|\s+nam\s+)(20\d{2}|\d{2})", text
    )
    if month_match:
        month, year = int(month_match.group(1)), int(month_match.group(2))
        if year < 100:
            year += 2000
        if 1 <= month <= 12:
            start, end = _month_bounds(year, month)
            return start.isoformat(), end.isoformat()

    month_match = re.search(r"(?:thang|th)\s*(\d{1,2})(?!\s*/\s*20\d{2})", text)
    if month_match:
        month = int(month_match.group(1))
        if 1 <= month <= 12:
            year = anchor.year if month <= anchor.month else anchor.year - 1
            start, end = _month_bounds(year, month)
            return start.isoformat(), end.isoformat()

    # A bare month/day such as 8/26 has no year. If the first value cannot be
    # a day, interpret it as month/day; otherwise follow Vietnamese day/month.
    date_match = re.search(
        r"(?<!\d)(\d{1,2})\s*/\s*(\d{1,2})(?:\s*/\s*(20\d{2}))?(?!\d)",
        text,
    )
    if date_match:
        first, second = int(date_match.group(1)), int(date_match.group(2))
        year = int(date_match.group(3)) if date_match.group(3) else None
        if first > 12 and second <= 12:
            day, month = first, second
        else:
            month, day = first, second
        if 1 <= month <= 12 and 1 <= day <= 31:
            if year is None:
                year = _year_for_month_day(month, day, anchor)
            try:
                resolved = date(year, month, day)
            except ValueError:
                return None
            return resolved.isoformat(), resolved.isoformat()

    return None
