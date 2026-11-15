from __future__ import annotations

import math
import re
import unicodedata
from decimal import Decimal, DecimalException
from numbers import Real
from typing import Any


NUMBER_SQL_FUNCTION = "datamate_parse_number"

_MAGNITUDE_SUFFIXES = {
    "k": Decimal("1000"),
    "nghin": Decimal("1000"),
    "ngan": Decimal("1000"),
    "thousand": Decimal("1000"),
    "tr": Decimal("1000000"),
    "trieu": Decimal("1000000"),
    "million": Decimal("1000000"),
}
_CURRENCY_PATTERN = re.compile(
    r"vnd|vnđ|đồng|đ|₫|usd|eur|gbp|jpy|cny|cad|aud|\$|€|£|¥|%",
    re.IGNORECASE,
)
_NUMERIC_LITERAL_PATTERN = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$"
)
_DECORATION_PATTERN = re.compile(
    r"[,\.']|vnd|vnđ|đồng|đ|₫|usd|eur|gbp|jpy|cny|cad|aud|\$|€|£|¥|%|"
    r"(?:triệu|trieu|tr|nghìn|nghin|ngàn|ngan|thousand|million|k)\s*$",
    re.IGNORECASE,
)


def _fold_for_suffix(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value.casefold())
    return "".join(
        character
        for character in decomposed
        if unicodedata.category(character) != "Mn"
    )


def has_number_format_marker(value: Any) -> bool:
    """Return whether text carries a decimal, currency, or scale marker."""
    if not isinstance(value, str):
        return False
    return _DECORATION_PATTERN.search(unicodedata.normalize("NFKC", value)) is not None


def _normalize_separators(value: str, vietnamese_locale: bool) -> str | None:
    """Normalize a localized mantissa to a Decimal-compatible spelling."""
    value = value.replace("'", "").replace("’", "")
    value = "".join(character for character in value if not character.isspace())
    if not value:
        return None

    sign = ""
    if value[0] in "+-":
        sign, value = value[0], value[1:]
    if not value or any(character not in "0123456789.," for character in value):
        return None

    separators = {character for character in value if character in ".,"}
    if not separators:
        normalized = value
    elif len(separators) == 2:
        decimal_separator = "." if value.rfind(".") > value.rfind(",") else ","
        decimal_index = value.rfind(decimal_separator)
        whole = value[:decimal_index].replace(".", "").replace(",", "")
        fraction = value[decimal_index + 1:]
        if not whole or not whole.isdigit() or not fraction.isdigit():
            return None
        normalized = f"{whole}.{fraction}"
    else:
        separator = next(iter(separators))
        parts = value.split(separator)
        if len(parts) > 2:
            if not parts[0].isdigit() or not 1 <= len(parts[0]) <= 3:
                return None
            if all(len(part) == 3 and part.isdigit() for part in parts[1:]):
                normalized = "".join(parts)
            else:
                if not all(
                    len(part) == 3 and part.isdigit()
                    for part in parts[1:-1]
                ):
                    return None
                whole = "".join(parts[:-1])
                fraction = parts[-1]
                if not whole.isdigit() or not fraction.isdigit():
                    return None
                normalized = f"{whole}.{fraction}"
        else:
            whole, fraction = parts
            if not whole and not fraction:
                return None
            if not whole:
                whole = "0"
            if not whole.isdigit():
                return None
            if not fraction:
                normalized = whole
            elif len(fraction) == 3 and not (
                vietnamese_locale and separator == ","
            ):
                normalized = whole + fraction
            else:
                if not fraction.isdigit():
                    return None
                normalized = f"{whole}.{fraction}"

    normalized = sign + normalized
    return normalized if _NUMERIC_LITERAL_PATTERN.fullmatch(normalized) else None


def _parse_decimal_token(value: str, vietnamese_locale: bool) -> Decimal | None:
    exponent_parts = re.split(r"([eE][+-]?\d+)$", value, maxsplit=1)
    mantissa = exponent_parts[0]
    exponent = exponent_parts[1] if len(exponent_parts) == 3 else ""
    normalized = _normalize_separators(mantissa, vietnamese_locale)
    if normalized is None:
        return None
    try:
        return Decimal(normalized + exponent)
    except DecimalException:
        return None


def parse_measure_number(value: Any) -> float | None:
    """Parse numeric measures, including common Vietnamese money formats.

    Examples: ``1.048.000đ`` -> ``1048000``, ``484k`` -> ``484000``,
    and ``1,5tr`` -> ``1500000``. Unknown text and non-finite values return
    ``None`` rather than being partially stripped into a different number.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        try:
            number = float(value)
        except (OverflowError, ValueError):
            return None
        return number if math.isfinite(number) else None
    if isinstance(value, Real):
        try:
            number = float(value)
        except (OverflowError, ValueError):
            return None
        return number if math.isfinite(number) else None
    if not isinstance(value, str):
        return None

    text = unicodedata.normalize("NFKC", value).strip().casefold()
    if not text:
        return None

    negative_parentheses = text.startswith("(") and text.endswith(")")
    if negative_parentheses:
        text = text[1:-1].strip()
    elif "(" in text or ")" in text:
        return None

    text = text.replace("−", "-").replace("–", "-")
    text = _CURRENCY_PATTERN.sub("", text)
    text = _fold_for_suffix(text)
    text = "".join(character for character in text if not character.isspace())

    multiplier = Decimal("1")
    locale_hint = False

    # Vietnamese shorthand such as 1tr5 means 1.5 triệu and 1tr500 means
    # 1.500 triệu. Grouped bases (1.000tr5) are supported too.
    compact_tr = re.fullmatch(
        r"([+-]?\d+(?:[.,]\d{3})*)tr(\d{1,3})", text
    )
    if compact_tr:
        base_text, fractional_tail = compact_tr.groups()
        base = _parse_decimal_token(base_text, vietnamese_locale=True)
        if base is None:
            return None
        try:
            sign = Decimal("-1") if base < 0 else Decimal("1")
            number = (abs(base) + Decimal(f"0.{fractional_tail}")) * Decimal("1000000")
            result = float(number * sign)
        except (DecimalException, OverflowError, ValueError):
            return None
        if negative_parentheses:
            result = -abs(result)
        return result if math.isfinite(result) else None

    folded_suffix = _fold_for_suffix(text)
    suffix_match = re.search(r"(triệu|tr|nghìn|ngàn|thousand|million|k)$", text)
    if suffix_match is None:
        suffix_match = re.search(r"(trieu|nghin|ngan)$", folded_suffix)
    if suffix_match is not None:
        suffix = _fold_for_suffix(suffix_match.group(1))
        multiplier = _MAGNITUDE_SUFFIXES[suffix]
        text = text[:suffix_match.start()]
        locale_hint = suffix in {"tr", "trieu", "nghin", "ngan", "k"}

    if re.search(r"(?:đ|₫|vnd|vnđ|đồng)", value, flags=re.IGNORECASE):
        locale_hint = True

    number = _parse_decimal_token(text, vietnamese_locale=locale_hint)
    if number is None:
        return None
    if negative_parentheses:
        number = -abs(number)
    try:
        result = float(number * multiplier)
    except (DecimalException, OverflowError, ValueError):
        return None
    return result if math.isfinite(result) else None


def number_sql_expression(expression: str) -> str:
    """Return SQL that applies the shared parser to a database expression."""
    return f"{NUMBER_SQL_FUNCTION}(CAST(({expression}) AS VARCHAR))"
