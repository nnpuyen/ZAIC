from __future__ import annotations

import json
import os
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..ingestion.loader import BusinessDatabase
from .plan_extraction import extract_query_plan
from .retrieval import retrieve_context
from .schemas import QueryPlan


PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "local": {
        "base_url": "http://localhost:8000/v1",
        "model": "gemma-local",
        "api_key_env": "LLM_API_KEY",
    },
    "gemma4-local": {
        "base_url": "http://localhost:8001/v1",
        "model": "gemma4-12b-planner",
        "api_key_env": "LLM_API_KEY",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "qwen/qwen3.5-9b",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "huggingface": {
        "base_url": "https://router.huggingface.co/v1",
        # Gemma 4 12B is not currently deployed by HF Inference Providers;
        # keep the hosted preset on a chat-capable model.
        "model": "google/gemma-3-12b-it",
        "api_key_env": "HF_TOKEN",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "qwen/qwen3-32b",
        "api_key_env": "GROQ_API_KEY",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-3.5-flash-lite",
        "api_key_env": "GEMINI_API_KEY",
    },
}

MAX_PLANNER_ATTEMPTS = 3


class LLMParser:
    """Plan questions with an OpenAI-compatible language-model endpoint."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        provider: str | None = None,
        request_delay: float | None = None,
    ) -> None:
        self.provider = (provider or os.getenv("LLM_PROVIDER", "local")).lower()
        if self.provider not in PROVIDER_DEFAULTS:
            supported = ", ".join(sorted(PROVIDER_DEFAULTS))
            raise ValueError(f"Unknown LLM provider {self.provider!r}; choose from {supported}")
        defaults = PROVIDER_DEFAULTS[self.provider]
        configured_base_url = (
            base_url
            or os.getenv("LLM_BASE_URL")
            or defaults["base_url"]
        )
        if not configured_base_url:
            raise ValueError(
                "An OpenAI-compatible endpoint URL is required; set "
                "LLM_BASE_URL or pass --llm-base-url"
            )
        self.base_url = configured_base_url.rstrip("/")
        self.model = model or os.getenv("LLM_MODEL", defaults["model"])
        self.api_key = (
            api_key
            if api_key is not None
            else os.getenv("LLM_API_KEY", os.getenv(defaults["api_key_env"], ""))
        )
        configured_timeout = os.getenv("LLM_TIMEOUT", "600")
        self.timeout = timeout if timeout is not None else float(configured_timeout)
        default_max_tokens = "1024"
        configured_max_tokens = os.getenv("LLM_MAX_TOKENS", default_max_tokens)
        self.max_tokens = int(configured_max_tokens)
        configured_request_delay = os.getenv("LLM_REQUEST_DELAY", "1")
        self.request_delay = (
            request_delay
            if request_delay is not None
            else float(configured_request_delay)
        )
        if self.request_delay < 0:
            raise ValueError("LLM request delay must be non-negative")
        self._last_request_at: float | None = None
        self.last_response_content: str | None = None
        self.last_raw_plan: Any | None = None
        self.last_retrieved_context: dict[str, Any] | None = None
        self.last_token_usage: dict[str, int] | None = None

    def parse(
        self,
        question: str,
        database: BusinessDatabase,
        as_of_date: str,
        context: dict[str, Any] | None = None,
    ) -> QueryPlan:
        if context is None:
            raise ValueError("LLM planning requires pipeline context")
        self.last_response_content = None
        self.last_raw_plan = None
        self.last_token_usage = None
        relevant_context = retrieve_context(question, as_of_date, context)
        self.last_retrieved_context = relevant_context
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": self._prompt(as_of_date, relevant_context)},
                {"role": "user", "content": question},
            ],
        }
        if self.provider in {"local", "gemma4-local"}:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        errors: list[str] = []

        for attempt in range(1, MAX_PLANNER_ATTEMPTS + 1):
            request = Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            self._wait_between_requests()
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))

                usage = body.get("usage")
                if isinstance(usage, dict):
                    token_usage = {
                        key: usage[key]
                        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                        if isinstance(usage.get(key), int)
                    }
                    self.last_token_usage = token_usage or None

                try:
                    content = body["choices"][0]["message"]["content"]
                    if isinstance(content, list):
                        content = "".join(part.get("text", "") for part in content)
                    self.last_response_content = (
                        content if isinstance(content, str) else str(content)
                    )
                    try:
                        raw_plan = json.loads(content)
                    except json.JSONDecodeError:
                        # Some endpoints add a short preamble or JSON fence
                        # despite requesting JSON mode. The extractor handles
                        # that framing while still validating the plan itself.
                        raw_plan = content
                    self.last_raw_plan = raw_plan
                except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
                    raise ValueError("LLM response did not contain a JSON QueryPlan") from error
                return extract_query_plan(raw_plan)
            except HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")
                failure = f"LLM endpoint returned HTTP {error.code}: {detail}"
            except URLError as error:
                failure = f"Could not reach LLM endpoint: {error.reason}"
            except TimeoutError as error:
                failure = (
                    f"LLM endpoint timed out after {self.timeout:g} seconds; "
                    "increase LLM_TIMEOUT or reduce LLM_MAX_TOKENS"
                )
            except json.JSONDecodeError:
                failure = "LLM endpoint returned invalid JSON"
            except ValueError as error:
                failure = str(error)

            errors.append(f"attempt {attempt}: {failure}")
            if attempt < MAX_PLANNER_ATTEMPTS:
                # Back off after failures so retries do not amplify a provider outage.
                delay = min(8.0, 2 ** (attempt - 1))
                time.sleep(delay)

        raise ValueError(
            f"LLM planner failed after {MAX_PLANNER_ATTEMPTS} attempts: "
            + "; ".join(errors)
        )

    def _wait_between_requests(self) -> None:
        """Space consecutive questions to avoid bursting the provider endpoint."""
        now = time.monotonic()
        if self._last_request_at is not None:
            remaining = round(self.request_delay - (now - self._last_request_at), 10)
            if remaining > 0:
                time.sleep(remaining)
        self._last_request_at = time.monotonic()

    def _prompt(self, as_of_date: str, context: dict[str, Any]) -> str:
        """Build a concise prompt around the retrieved context."""
        instructions = f"""You plan safe queries for Vietnamese business data.

Return exactly one complete JSON object strictly matching the QueryPlan schema, do not add any wrapper for the returend JSON object. Do not return SQL, explanations, or markdown.

Schema shape and executable-plan rules:
- A CTE is `{{"name": "cte_name", "query": {{...complete QueryPlan object...}}}}`; its `query` is itself a QueryPlan with a string `table` at its root.
- To combine queries, keep the first query as the base QueryPlan and add ordered `set_operations` entries shaped as `{{"op_type": "union|union_all|intersect|except", "query": {{...complete QueryPlan object...}}}}`. Each branch must project the same columns in the same order. Use `union_all` for disjoint partitions such as monthly files; use `union` when duplicate rows should be removed. Use `intersect` or `except` only when the question asks for a common set or a set difference.
- In a grouped query, every selected non-aggregated column must also be in `group_by`. For a scalar aggregate CTE, omit raw `select_columns` unless they are grouped.
- Plan-level `filters` apply to all metrics in that query. Metric-level `filter` and `denominator_filter` are supported for `count`, `count_distinct`, and count-based `percentage`; do not attach them to `sum`, `avg`, `min`, or `max` expecting a conditional aggregate.
- `percentage` computes a filtered count numerator divided by a filtered count denominator. Do not use it to compare monetary sums.
- To compare two conditionally filtered sums, create one scalar CTE per sum, put each condition set in that CTE's plan-level `filters`, and give each sum metric an alias. Cross join the scalar CTEs, then use one outer `ratio` metric with `column` set to the numerator alias and `denominator_column` set to the denominator alias. Set `answer_type` to `percent` for a requested percentage. Do not group these scalar CTEs.
- Use `answer_type: "percent"` for a requested percentage, even when the underlying calculation is a ratio of amounts.

Interpretation priority:
1. Follow any explicit rule in the user's question. For aspects not explicitly specified, apply the relevant business rules from the supplied rulebook, such as the definition of total order count.
2. Apply relevant business rules from the supplied rulebook, using contextual evidence and common-sense reasoning where necessary, as long as they do not contradict the rulebook. For revenue and order counts, exclude orders marked as cancelled or returned when the rulebook says to exclude them. Also exclude a failed-delivery status such as "Không giao được" when it appears in the data, since it does not represent a completed sale. Ground excluded labels in the rulebook and actual data; do not infer that other statuses should be excluded. A NULL or blank status is not evidence that an order was cancelled or invalid, so include it unless an explicit rule says otherwise. Preserve blank statuses in the filter: use an OR group that keeps NULL (and the empty string if blank cells are stored that way) or statuses NOT IN the excluded labels. Never combine multiple `status <> value` checks with OR; use `not_in` or combine the `neq` checks with AND. When a status applies to an entire order, ensure excluding it removes all lines of that order from the calculation.

Question understanding: 
- Normalize the user's question first Interpret natural Vietnamese, abbreviations, shorthand, typos, and informal business terminology using context. Preserve the original intent.
- Should understanding the question using the business_description in database context, follow the rules in business_description if any.

Planning constraints:
- Use exact table and column names from the supplied schema. Choose the simplest valid plan and only join tables when the question requires it and the supplied keys support it.
- The schema in the context contains only tables that may be relevant to the question; use business_description for the broader overview of the business.
- Use the date field that represents the requested event. Apply the supplied date range and exclude data after {as_of_date}.
- Count requested entities with count_distinct on their identifier, not line-item rows.
- Ground status and category filters in the rulebook and actual sample values. Do not invent stored labels or relationships.
- Use only supported filter operators: eq, neq, gt, gte, lt, lte, in, not_in, contains, between, is_null, is_not_null.
- Match the QueryPlan filter schema exactly: use `logic` and `conditions` for groups, and put filter fields directly in each condition. Do not wrap them in `filter_group` or `filter` keys.
- Use weighted averages when the underlying observations have different quantities. Do not add limits to aggregations unless requested.
- Correct obvious language errors only when unambiguous; preserve codes, names, numbers, dates, and quoted values exactly.
- When the same data come from different sources, should priority the most clear and clean source, can use join operator if needed to take the data source.

Counting rules:
- When the question asks for a count, use count_distinct on the semantic identifier of the target entity.
- Identify the target entity from the question's meaning, then select its corresponding identifier from the schema.
  Examples:
  - Number of orders -> distinct order_id.
  - Number of customers -> distinct customer_id.
  - Number of products -> distinct product_id.
  - Number of invoices -> distinct invoice_id.
- When joining tables with one-to-many relationships, ensure the count measures the requested entity rather than duplicated joined rows.
- Do not assume the primary key of the main table is always the identifier to count. Choose the identifier that represents the entity explicitly requested.
- If the semantic identifier cannot be determined from the supplied schema, do not invent one.

Answer type:
- Set answer_type to match the requested final value: number for amounts/counts/quantities, percent for percentages, string for one name, date for one date, boolean for true/false, list for an ordered sequence of names or numbers, and table for multiple named columns and rows.
- Choose the answer_type from the requested result shape, not from the SQL operation used to calculate it.
- Only `list` and `table` answers may contain multiple values or rows. For `number`, `percent`, `string`, `date`, and `boolean`, plan one final value and ensure the query returns one row with one requested value.
- For a scalar `percent`, compute the percentage as one metric (use the percentage aggregation with its numerator and denominator filters when counting). Do not return numerator and denominator as separate output columns.
- If a question requests grouped results with a grouping label and a measure, use `table`; use `list` only when the requested output is one ordered value per row.
- For list answers, select the one requested name or number as the output value for each row, and use order_by to preserve the order requested in the question or specified by the rulebook. If a metric is used only to rank names, keep the metric for sorting and return just the names.
- For table answers, include every requested named column and let the query result preserve a consistent column order; the answer will serialize these as `columns` and positional `rows`.

QueryPlan schema:
{json.dumps(QueryPlan.model_json_schema(), ensure_ascii=False)}

Database context:
{json.dumps(context, ensure_ascii=False, default=str)}"""
        return instructions


# Backward-compatible import for callers that used the old model-specific name.
QwenParser = LLMParser
