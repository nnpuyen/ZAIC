from __future__ import annotations

import json
import pytest
from unittest.mock import patch

from datamate.planning.qwen import QwenParser
from datamate.planning.schemas import Filter


def test_filter_schema_rejects_unsupported_boolean_operator() -> None:
    with pytest.raises(ValueError):
        Filter(column="active", operator="is_true")


def test_qwen_parser_validates_structured_plan() -> None:
    response = {
        "usage": {"prompt_tokens": 123, "completion_tokens": 17, "total_tokens": 140},
        "choices": [{
            "message": {
                "content": json.dumps({
                    "operation": "aggregation",
                    "table": "sales",
                    "metric": {"column": "amount", "aggregation": "sum"},
                })
            }
        }]
    }

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(response).encode()

    parser = QwenParser(base_url="http://qwen.test")
    with patch("datamate.planning.qwen.urlopen", return_value=Response()):
        plan = parser.parse(
            "tổng doanh thu", database=None, as_of_date="2026-03-04",
            context={"schema": {"sales": {"columns": ["amount"]}}},
        )
    assert plan.table == "sales"
    assert plan.metric.aggregation == "sum"
    assert parser.last_raw_plan == {
        "operation": "aggregation",
        "table": "sales",
        "metric": {"column": "amount", "aggregation": "sum"},
    }
    assert parser.last_token_usage == {
        "prompt_tokens": 123,
        "completion_tokens": 17,
        "total_tokens": 140,
    }


def test_qwen_parser_keeps_generated_plan_for_trace() -> None:
    raw_plan = {
        "operation": "aggregation",
        "table": "sales",
        "metric": {"column": "amount", "aggregation": "sum"},
        "joins": [{
            "table": "products",
            "left_column": "Order ID",
            "right_column": "order_id",
        }],
    }
    response = {
        "choices": [{"message": {"content": json.dumps(raw_plan)}}],
    }

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(response).encode()

    parser = QwenParser(base_url="http://qwen.test")
    with patch("datamate.planning.qwen.urlopen", return_value=Response()):
        parser.parse(
            "tổng doanh thu", database=None, as_of_date="2026-03-04",
            context={"schema": {"sales": {"columns": ["amount"]}}},
        )
    assert parser.last_raw_plan == raw_plan
    assert parser.last_response_content == json.dumps(raw_plan)


def test_qwen_parser_normalizes_not_in_alias() -> None:
    response = {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "table": "sales",
                    "metric": {"column": "amount", "aggregation": "sum"},
                    "filters": [{
                        "column": "status",
                        "operator": "not in",
                        "value": ["cancelled"],
                    }],
                })
            }
        }]
    }

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(response).encode()

    parser = QwenParser(base_url="http://qwen.test")
    with patch("datamate.planning.qwen.urlopen", return_value=Response()):
        plan = parser.parse(
            "tổng doanh thu", database=None, as_of_date="2026-03-04",
            context={"schema": {"sales": {"columns": ["amount", "status"]}}},
        )
    assert plan.filters[0].operator == "not_in"


def test_qwen_parser_wraps_single_filter_group() -> None:
    response = {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "table": "sales",
                    "metric": {"column": "amount", "aggregation": "sum"},
                    "filters": {
                        "logic": "and",
                        "conditions": [{
                            "column": "status",
                            "operator": "not in",
                            "value": ["cancelled"],
                        }],
                    },
                })
            }
        }]
    }

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(response).encode()

    parser = QwenParser(base_url="http://qwen.test")
    with patch("datamate.planning.qwen.urlopen", return_value=Response()):
        plan = parser.parse(
            "tổng doanh thu", database=None, as_of_date="2026-03-04",
            context={"schema": {"sales": {"columns": ["amount", "status"]}}},
        )
    assert len(plan.filters) == 1
    assert plan.filters[0].conditions[0].operator == "not_in"


def test_qwen_parser_supports_remote_provider_environment(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    parser = QwenParser()
    assert parser.base_url == "https://openrouter.ai/api/v1"
    assert parser.model == "qwen/qwen3.5-9b"
    assert parser.api_key == "test-key"


def test_qwen_parser_uses_hosted_gemma_for_huggingface(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "test-token")
    parser = QwenParser(provider="huggingface")
    assert parser.base_url == "https://router.huggingface.co/v1"
    assert parser.model == "google/gemma-3-12b-it"
    assert parser.api_key == "test-token"


def test_qwen_parser_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        QwenParser(provider="unknown")


def test_qwen_prompt_makes_rulebook_authoritative() -> None:
    prompt = QwenParser()._prompt(
        "2026-03-04",
        {
            "schema": {"tables": []},
            "rulebook_sections": {
                "revenue": "Số đơn chỉ tính đơn không huỷ, không trả."
            },
        },
    )

    assert "RULEBOOK COMPLIANCE IS MANDATORY" in prompt
    assert "authoritative business definition" in prompt
    assert "Câu hỏi nói khác thì làm theo câu hỏi; không nói thì áp luật mặc định ở đây." in prompt
    assert "Do not omit a condition" in prompt
    assert "counts the wrong entity" in prompt
    assert "compare the table row_count" in prompt
    assert "use count_distinct" in prompt
    assert "schema catalog lists every available table and column" in prompt
    assert "Filter values must also be grounded" in prompt
    assert "use an explicit in filter" in prompt
    assert "QUESTION NORMALIZATION IS THE FIRST STEP" in prompt
    assert "expand common Vietnamese business abbreviations" in prompt
    assert "Do not alter a name or code merely because it looks misspelled" in prompt
    assert '"rulebook"' not in prompt


def test_qwen_parser_supports_gemini_provider(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    parser = QwenParser(provider="gemini")
    assert parser.base_url == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert parser.model == "gemini-3.5-flash-lite"
    assert parser.api_key == "test-key"


def test_gemini_request_omits_local_chat_template_options() -> None:
    response = {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "operation": "aggregation",
                    "table": "sales",
                    "metric": {"column": "amount", "aggregation": "sum"},
                })
            }
        }]
    }

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(response).encode()

    requests = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        return Response()

    parser = QwenParser(provider="gemini", api_key="test-key")
    with patch("datamate.planning.qwen.urlopen", side_effect=fake_urlopen):
        parser.parse(
            "tổng doanh thu", database=None, as_of_date="2026-03-04",
            context={"schema": {"sales": {"columns": ["amount"]}}},
        )

    payload = json.loads(requests[0].data)
    assert "chat_template_kwargs" not in payload


def test_qwen_parser_delays_only_between_requests(monkeypatch) -> None:
    response = {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "operation": "aggregation",
                    "table": "sales",
                    "metric": {"column": "amount", "aggregation": "sum"},
                })
            }
        }]
    }

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(response).encode()

    sleeps = []
    clock = iter((100.0, 100.0, 100.1, 100.1))
    monkeypatch.setattr("datamate.planning.qwen.time.monotonic", lambda: next(clock))
    monkeypatch.setattr("datamate.planning.qwen.time.sleep", sleeps.append)
    parser = QwenParser(base_url="http://qwen.test", request_delay=1)
    with patch("datamate.planning.qwen.urlopen", return_value=Response()):
        for _ in range(2):
            parser.parse(
                "tổng doanh thu", database=None, as_of_date="2026-03-04",
                context={"schema": {"sales": {"columns": ["amount"]}}},
            )

    assert sleeps == [0.9]


def test_qwen_parser_retries_planner_error_and_succeeds() -> None:
    valid_body = {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "operation": "aggregation",
                    "table": "sales",
                    "metric": {"column": "amount", "aggregation": "sum"},
                })
            }
        }]
    }

    class Response:
        def __init__(self, body: object) -> None:
            self.body = body

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(self.body).encode()

    parser = QwenParser(base_url="http://qwen.test", request_delay=0)
    with patch(
        "datamate.planning.qwen.urlopen",
        side_effect=[
            Response({"choices": [{"message": {"content": "not json"}}]}),
            Response(valid_body),
        ],
    ) as mocked_urlopen:
        plan = parser.parse(
            "tổng doanh thu", database=None, as_of_date="2026-03-04",
            context={"schema": {"sales": {"columns": ["amount"]}}},
        )

    assert mocked_urlopen.call_count == 2
    assert plan.table == "sales"


def test_qwen_parser_stops_after_three_failed_attempts() -> None:
    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"choices": []}'

    parser = QwenParser(base_url="http://qwen.test", request_delay=0)
    with patch(
        "datamate.planning.qwen.urlopen", return_value=Response()
    ) as mocked_urlopen:
        with pytest.raises(ValueError, match="failed after 3 attempts"):
            parser.parse(
                "tổng doanh thu", database=None, as_of_date="2026-03-04",
                context={"schema": {"sales": {"columns": ["amount"]}}},
            )

    assert mocked_urlopen.call_count == 3
