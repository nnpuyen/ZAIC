from __future__ import annotations

import calendar
import re
from datetime import date, timedelta


def _month_bounds(year: int, month: int) -> tuple[date, date]:
    return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])


def resolve_date_range(expression: str, as_of_date: str) -> tuple[str, str] | None:
    """Resolve the Gregorian date expressions used by the baseline rulebook."""
    anchor = date.fromisoformat(as_of_date)
    text = expression.lower().strip()
    if "hôm nay" in text or "hom nay" in text or text == "hôm nay":
        return anchor.isoformat(), anchor.isoformat()
    if "hôm qua" in text or "hom qua" in text:
        day = anchor - timedelta(days=1)
        return day.isoformat(), day.isoformat()
    if "tuần trước" in text or "tuan truoc" in text or "tuần rồi" in text or "tuan roi" in text:
        start = anchor - timedelta(days=anchor.weekday() + 7)
        return start.isoformat(), (start + timedelta(days=6)).isoformat()
    if "tuần này" in text or "tuan nay" in text:
        start = anchor - timedelta(days=anchor.weekday())
        return start.isoformat(), anchor.isoformat()
    if "tháng này" in text or "thang nay" in text:
        return date(anchor.year, anchor.month, 1).isoformat(), anchor.isoformat()
    quarter_match = re.search(r"(?:quý|quy)\s*([1-4])(?:\s*/\s*(\d{4})|\s+(\d{4}))?", text)
    if quarter_match:
        quarter = int(quarter_match.group(1))
        year = int(quarter_match.group(2) or quarter_match.group(3) or anchor.year)
        if not quarter_match.group(2) and not quarter_match.group(3):
            quarter_start_month = (quarter - 1) * 3 + 1
            if date(year, quarter_start_month, 1) > anchor:
                year -= 1
        start_month = (quarter - 1) * 3 + 1
        start, _ = _month_bounds(year, start_month)
        _, end = _month_bounds(year, start_month + 2)
        return start.isoformat(), end.isoformat()
    match = re.search(r"(?:tháng|thang)\s*(\d{1,2})(?:\s*/\s*|\s+)(\d{4})", text)
    if match:
        start, end = _month_bounds(int(match.group(2)), int(match.group(1)))
        return start.isoformat(), end.isoformat()
    match = re.search(r"(?:tháng|thang)\s*(\d{1,2})(?!\s*/)", text)
    if match:
        month = int(match.group(1))
        year = anchor.year if month <= anchor.month else anchor.year - 1
        start, end = _month_bounds(year, month)
        return start.isoformat(), end.isoformat()
    if "đầu năm" in text or "dau nam" in text:
        return date(anchor.year, 1, 1).isoformat(), anchor.isoformat()
    return None
