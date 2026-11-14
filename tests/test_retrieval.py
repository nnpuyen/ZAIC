from datamate.planning.retrieval import retrieve_context


def test_retrieval_selects_relevant_tables_and_preserves_all_columns_and_sections() -> None:
    context = {
        "schema": {
            "tables": [
                {"table": "sales", "columns": [
                    {"name": "order_date", "type": "DATE"},
                    {"name": "amount", "type": "DOUBLE"},
                    {"name": "product", "type": "VARCHAR"},
                    {"name": "internal_note", "type": "VARCHAR"},
                ], "row_count": 100},
                {"table": "inventory", "columns": [
                    {"name": "stock", "type": "INTEGER"},
                ]},
            ]
        },
        "rulebook": "## 1. Revenue\nRevenue rules.\n## 2. Time\nTime rules.\n## 3. Ranking\nRanking rules.",
        "business_description": "Each sales row is one product line; order data repeats per row.",
        "calendar": "date,lunar_day,lunar_month,lunar_year\n2026-01-01,13,11,2025\n",
    }
    selected = retrieve_context("Doanh thu quý 1", "2026-03-04", context)
    assert selected["business_description"] == context["business_description"]
    assert selected["schema"]["tables"][0]["row_count"] == 100
    assert [table["table"] for table in selected["schema"]["tables"]] == [
        "sales", "inventory"
    ]
    assert {column["name"] for column in selected["schema"]["tables"][0]["columns"]} == {
        "order_date", "amount", "product", "internal_note"
    }
    assert selected["schema"]["tables"][1]["columns"] == [{"name": "stock", "type": "INTEGER"}]
    assert "rulebook" not in selected
    assert set(selected["rulebook_sections"]) == {"revenue", "time"}
    assert selected["calendar_evidence"] == []
    assert selected["time"]["start_date"] == "2026-01-01"


def test_retrieval_includes_only_lunar_calendar_evidence_for_tet() -> None:
    context = {
        "schema": {"tables": []},
        "rulebook": "## 2. Time\nTime rules.",
        "calendar": (
            "date,lunar_day,lunar_month,lunar_year,holiday\n"
            "2026-02-09,23,12,2025,\n"
            "2026-02-10,1,1,2026,\n"
            "2026-02-19,10,1,2026,\n"
        ),
    }
    selected = retrieve_context("Doanh thu dịp Tết năm 2026", "2026-03-04", context)
    assert [row["date"] for row in selected["calendar_evidence"]] == [
        "2026-02-09", "2026-02-19"
    ]
    assert "calendar_required" not in selected["time"]


def test_retrieval_does_not_send_calendar_for_gregorian_quarter() -> None:
    selected = retrieve_context(
        "Doanh thu quý 1",
        "2026-03-04",
        {
            "schema": {"sales": {"columns": ["date", "amount"]}},
            "rulebook": "",
            "calendar": "date,lunar_day,lunar_month,lunar_year\n2026-01-01,13,11,2025\n",
        },
    )
    assert selected["calendar_evidence"] == []
    assert "calendar_required" not in selected["time"]


def test_retrieval_includes_order_status_for_order_count() -> None:
    selected = retrieve_context(
        "Hôm nay bán được mấy đơn hàng?",
        "2025-08-12",
        {
            "schema": {
                "tables": [{
                    "table": "sales",
                    "columns": [
                        {"name": "Ngày", "type": "DATE"},
                        {"name": "Mã đơn hàng", "type": "VARCHAR"},
                        {"name": "Trạng thái", "type": "VARCHAR"},
                        {"name": "Thành tiền", "type": "DOUBLE"},
                    ],
                }],
            },
            "rulebook": (
                "## 1. Đơn hàng và doanh thu\n"
                "| Số đơn | chỉ tính đơn không huỷ, không trả |\n"
                "## 2. Thời gian\n"
                "| Hôm nay | trong ngày đó |"
            ),
            "calendar": "",
        },
    )

    assert set(selected["rulebook_sections"]) == {"revenue", "time"}
    assert {
        column["name"]
        for column in selected["schema"]["tables"][0]["columns"]
    } == {"Ngày", "Mã đơn hàng", "Trạng thái", "Thành tiền"}
