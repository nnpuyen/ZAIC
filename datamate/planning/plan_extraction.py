"""Extract and validate a QueryPlan from common planner response shapes.

This module repairs response framing only. It does not invent missing query
semantics or rewrite metrics, filters, or column names.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from .schemas import ParseResult, QueryPlan


_PLAN_FIELDS = {
    "operation",
    "table",
    "alias",
    "ctes",
    "set_operations",
    "unions",
    "select_columns",
    "metric",
    "metrics",
    "window_functions",
    "joins",
    "filters",
    "comparison_filters",
    "having_filters",
    "qualify_filters",
    "group_by",
    "group_by_period",
    "group_by_period_column",
    "grouping_sets",
    "order_by",
    "limit",
    "offset",
    "pivot",
    "sample",
    "distinct",
    "answer_type",
}
_WRAPPER_KEYS = (
    "query_plan",
    "plan",
    "query",
    "result",
    "data",
    "output",
    "response",
    "answer",
    "parsed",
    "payload",
    "content",
)


def extract_query_plan(value: Any) -> QueryPlan:
    """Return a validated plan from a model, mapping, or planner response text.

    Supported framing includes JSON code fences, prose around a JSON object,
    common ``plan``/``query_plan`` envelopes, answer envelopes with a plan,
    and the malformed ``{"table": {<QueryPlan>}}`` shape. Conflicting plan
    candidates are rejected instead of silently choosing one.
    """
    if isinstance(value, QueryPlan):
        return value
    if isinstance(value, ParseResult):
        if value.status != "answerable" or value.plan is None:
            raise ValueError(
                f"Cannot extract a QueryPlan from ParseResult status {value.status!r}"
            )
        if not isinstance(value.plan, QueryPlan):
            raise ValueError("ParseResult contains a non-QueryPlan plan")
        return value.plan

    if isinstance(value, str):
        candidates = _decode_json_candidates(value)
        if not candidates:
            raise ValueError("Planner response contains no JSON object for a QueryPlan")
        extracted: list[QueryPlan] = []
        errors: list[str] = []
        for candidate in candidates:
            try:
                plan = _extract_mapping(candidate)
                if not any(plan.model_dump(mode="json") == prior.model_dump(mode="json") for prior in extracted):
                    extracted.append(plan)
            except (TypeError, ValueError, ValidationError) as error:
                errors.append(str(error))
        if not extracted:
            detail = errors[0] if errors else "no valid plan candidate"
            raise ValueError(f"Planner response does not contain a valid QueryPlan: {detail}")
        if len(extracted) > 1:
            raise ValueError("Planner response contains multiple conflicting QueryPlan objects")
        return extracted[0]

    return _extract_mapping(value)


def _decode_json_candidates(text: str) -> list[Any]:
    stripped = text.strip()
    if not stripped:
        return []

    # A response may put the JSON in a fenced block with a language label.
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", stripped, flags=re.IGNORECASE | re.DOTALL)
    search_texts = [*fenced, stripped]
    decoder = json.JSONDecoder()
    candidates: list[Any] = []
    seen: set[str] = set()

    for search_text in search_texts:
        candidate_text = search_text.strip()
        try:
            parsed = json.loads(candidate_text)
            possible = parsed if isinstance(parsed, list) else [parsed]
            for item in possible:
                _append_unique(candidates, seen, item)
            continue
        except json.JSONDecodeError:
            pass

        # Locate complete JSON values embedded in surrounding commentary.
        # Advancing to the end of a decoded value skips its nested CTE/query
        # objects, which must not be mistaken for separate candidate plans.
        index = 0
        while index < len(candidate_text):
            if candidate_text[index] not in "{[":
                index += 1
                continue
            try:
                parsed, end = decoder.raw_decode(candidate_text, index)
            except json.JSONDecodeError:
                index += 1
                continue
            possible = parsed if isinstance(parsed, list) else [parsed]
            for item in possible:
                _append_unique(candidates, seen, item)
            index = end

    return candidates


def _append_unique(candidates: list[Any], seen: set[str], value: Any) -> None:
    if not isinstance(value, (dict, list)):
        return
    try:
        marker = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        marker = repr(value)
    if marker not in seen:
        seen.add(marker)
        candidates.append(value)


def _extract_mapping(value: Any) -> QueryPlan:
    if isinstance(value, QueryPlan):
        return value
    if isinstance(value, str):
        return extract_query_plan(value)
    if isinstance(value, ParseResult):
        if value.status != "answerable" or not isinstance(value.plan, QueryPlan):
            raise ValueError(f"ParseResult status {value.status!r} has no QueryPlan")
        return value.plan
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("Planner response list must contain exactly one plan")
        return _extract_mapping(value[0])
    if not isinstance(value, dict):
        raise TypeError(f"Expected a QueryPlan object, got {type(value).__name__}")

    # The direct schema shape wins whenever table is already a table name.
    if isinstance(value.get("table"), str):
        return QueryPlan.model_validate(value)

    candidates: list[Any] = []

    # Common malformed form: the complete plan is stored as the value of the
    # QueryPlan.table field. Move the inner plan object back to the root.
    nested_table = value.get("table")
    if isinstance(nested_table, dict) and _looks_like_plan(nested_table):
        candidates.append(_merge_envelope_fields(value, nested_table, "table"))

    for key in _WRAPPER_KEYS:
        nested = value.get(key)
        if isinstance(nested, dict):
            if _looks_like_plan(nested):
                candidates.append(_merge_envelope_fields(value, nested, key))
            elif any(wrapper in nested for wrapper in _WRAPPER_KEYS):
                candidates.append(nested)
        elif isinstance(nested, list) and len(nested) == 1 and isinstance(nested[0], dict):
            if _looks_like_plan(nested[0]):
                candidates.append(_merge_envelope_fields(value, nested[0], key))
            elif any(wrapper in nested[0] for wrapper in _WRAPPER_KEYS):
                candidates.append(nested[0])
        elif isinstance(nested, str):
            candidates.append(nested)

    if not candidates:
        # A partial plan still gets schema validation so callers receive the
        # field-level Pydantic error (for example, a missing table).
        return QueryPlan.model_validate(value)

    plans: list[QueryPlan] = []
    errors: list[str] = []
    for candidate in candidates:
        try:
            plan = _extract_mapping(candidate)
        except (TypeError, ValueError, ValidationError) as error:
            errors.append(str(error))
            continue
        if not any(plan.model_dump(mode="json") == prior.model_dump(mode="json") for prior in plans):
            plans.append(plan)
    if not plans:
        detail = errors[0] if errors else "no valid QueryPlan candidate"
        raise ValueError(f"Planner wrapper does not contain a valid QueryPlan: {detail}")
    if len(plans) != 1:
        raise ValueError("Planner response contains multiple conflicting QueryPlan objects")
    return plans[0]


def _merge_envelope_fields(
    envelope: dict[str, Any], nested: dict[str, Any], wrapper_key: str
) -> dict[str, Any]:
    """Keep schema fields placed beside a wrapped plan when unambiguous."""
    merged = dict(nested)
    for key, outer_value in envelope.items():
        if (
            key == wrapper_key
            or key not in _PLAN_FIELDS
            or (key == "table" and not isinstance(outer_value, str))
        ):
            continue
        if key in merged and merged[key] != outer_value:
            raise ValueError(f"Conflicting {key!r} values in wrapped QueryPlan response")
        merged.setdefault(key, outer_value)
    return merged


def _looks_like_plan(value: dict[str, Any]) -> bool:
    if isinstance(value.get("table"), str):
        return True
    return bool((_PLAN_FIELDS - {"table"}).intersection(value))
