from __future__ import annotations

import json

from datamate.planning.plan_extraction import extract_query_plan
from datamate.planning.schemas import QueryPlan


def test_extract_query_plan_from_nested_table_response() -> None:
    response = {
        "table": {
            "operation": "aggregation",
            "table": "chi_tiet_hoa_don",
            "metrics": [
                {
                    "column": "Thành tiền",
                    "aggregation": "sum",
                    "alias": "doanh thu",
                }
            ],
            "filters": [
                {
                    "logic": "and",
                    "conditions": [
                        {
                            "column": "Ngày giao dịch",
                            "operator": "between",
                            "value": ["2025-07-01", "2025-09-30"],
                            "table": "chi_tiet_hoa_don",
                        },
                        {
                            "column": "Tiến độ",
                            "operator": "neq",
                            "value": "đã hủy đơn",
                            "table": "chi_tiet_hoa_don",
                        },
                        {
                            "column": "Tiến độ",
                            "operator": "neq",
                            "value": "đã trả hàng",
                            "table": "chi_tiet_hoa_don",
                        },
                    ],
                }
            ],
            "answer_type": "number",
        }
    }

    plan = extract_query_plan(json.dumps(response, ensure_ascii=False))

    assert isinstance(plan, QueryPlan)
    assert plan.operation == "aggregation"
    assert plan.table == "chi_tiet_hoa_don"
    assert [metric.model_dump(exclude_none=True) for metric in plan.metrics] == [
        {
            "column": "Thành tiền",
            "aggregation": "sum",
            "alias": "doanh thu",
            "distinct": False,
            "is_percent_of_total": False,
        }
    ]
    assert plan.metric is None
    assert plan.answer_type == "number"
    assert len(plan.filters) == 1

    filter_group = plan.filters[0]
    assert filter_group.logic == "and"
    assert [
        (condition.column, condition.operator, condition.value, condition.table)
        for condition in filter_group.conditions
    ] == [
        (
            "Ngày giao dịch",
            "between",
            ["2025-07-01", "2025-09-30"],
            "chi_tiet_hoa_don",
        ),
        ("Tiến độ", "neq", "đã hủy đơn", "chi_tiet_hoa_don"),
        ("Tiến độ", "neq", "đã trả hàng", "chi_tiet_hoa_don"),
    ]
