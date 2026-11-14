from __future__ import annotations

import json

import duckdb

from datamate.context import build_context
from datamate.ingestion.loader import BusinessDatabase, TableMetadata, load_business_data
from datamate.ingestion.profiler import profile_table
from datamate.pipeline import answer_questions
from datamate.planning.completion import complete_plan
from datamate.planning.retrieval import retrieve_context
from datamate.planning.schemas import Filter, Metric, QueryPlan


def _mapped_orders_dir(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "canonical_mapping.yml").write_text(
        "tables:\n"
        "  invoices:\n"
        "    canonical_name: orders\n"
        "    aliases: [danh_sach_hoa_don]\n"
        "    grain: one row per order\n"
        "    columns:\n"
        "      order_id:\n"
        "        aliases: [Mã bill]\n"
        "        role: identifier\n"
        "      order_date:\n"
        "        aliases: [Ngày giao dịch]\n"
        "        type: date\n"
        "        role: transaction_date\n"
        "      order_total_amount:\n"
        "        aliases: [TongTien]\n"
        "        type: decimal\n"
        "        role: measure\n"
        "        definition: Amount payable after discount\n"
        "      status:\n"
        "        aliases: [Trạng thái]\n"
        "        role: status\n",
        encoding="utf-8",
    )
    (data_dir / "invoices.csv").write_text(
        "Mã bill,Ngày giao dịch,TongTien,Trạng thái\n"
        "ORD-1,01/03/2026,10000,Hoàn thành\n"
        "ORD-2,02/03/2026,20000,Đã hủy\n"
        "ORD-3,03/03/2026,30000,Đã trả hàng\n",
        encoding="utf-8",
    )
    return data_dir


def test_profiler_exposes_canonical_roles_aliases_and_definitions(tmp_path) -> None:
    data_dir = _mapped_orders_dir(tmp_path)
    database = load_business_data(data_dir)
    try:
        metadata = database.tables["orders"]
        assert metadata.canonical_name == "orders"
        assert metadata.table_aliases == ["danh_sach_hoa_don"]
        assert metadata.grain == "one row per order"
        profile = profile_table(database, "orders")
        assert profile["aliases"] == ["danh_sach_hoa_don"]
        assert profile["grain"] == "one row per order"
        by_name = {column["name"]: column for column in profile["columns"]}
        assert by_name["order_total_amount"]["role"] == "measure"
        assert by_name["order_total_amount"]["aliases"] == ["TongTien"]
        assert by_name["order_total_amount"]["definition"] == "Amount payable after discount"
        assert by_name["order_date"]["role"] == "transaction_date"
        assert by_name["order_id"]["role"] == "identifier"
        assert by_name["status"]["role"] == "status"
        assert by_name["status"]["raw_name"] == "Trạng thái"
    finally:
        database.close()


def test_retrieval_keeps_canonical_semantics_and_catalog() -> None:
    context = {
        "schema": {
            "tables": [{
                "table": "orders",
                "canonical_name": "orders",
                "aliases": ["danh_sach_hoa_don"],
                "grain": "one row per order",
                "columns": [
                    {
                        "name": "order_total_amount",
                        "type": "INTEGER",
                        "canonical_name": "order_total_amount",
                        "aliases": ["TongTien"],
                        "role": "measure",
                        "definition": "Amount payable after discount",
                        "distinct_count": 80,
                        "sample_values": [10000],
                    },
                    {
                        "name": "order_date",
                        "type": "DATE",
                        "role": "transaction_date",
                        "aliases": ["Ngày giao dịch"],
                    },
                    {
                        "name": "status",
                        "type": "VARCHAR",
                        "role": "status",
                        "distinct_count": 3,
                        "sample_values": ["Hoàn thành", "Đã hủy", "Đã trả hàng"],
                    },
                ],
            }]
        }
    }
    retrieved = retrieve_context("tháng này", "2026-03-04", context)
    columns = {column["name"]: column for column in retrieved["schema"]["tables"][0]["columns"]}
    assert retrieved["schema"]["tables"][0]["grain"] == "one row per order"
    assert columns["order_total_amount"]["role"] == "measure"
    assert columns["order_total_amount"]["aliases"] == ["TongTien"]
    assert "sample_values" not in columns["order_total_amount"]
    canonical = retrieved["canonical"]["tables"][0]["columns"]
    assert any(column.get("role") == "measure" for column in canonical)


def test_complete_plan_rewrites_aliases_and_recovers_identifier() -> None:
    connection = duckdb.connect(":memory:")
    connection.execute(
        'CREATE TABLE "orders" ("order_id" VARCHAR, "order_date" DATE, '
        '"order_total_amount" INTEGER, "status" VARCHAR)'
    )
    connection.execute(
        "INSERT INTO orders VALUES "
        "('ORD-1', '2026-03-01', 10, 'Hoàn thành'), "
        "('ORD-2', '2026-03-02', 20, 'Đã hủy')"
    )
    database = BusinessDatabase(
        connection,
        {
            "orders": TableMetadata(
                "orders", "test", None, 2,
                ["order_id", "order_date", "order_total_amount", "status"],
            )
        },
    )
    context = {
        "schema": {
            "tables": [{
                "table": "orders",
                "aliases": ["invoices"],
                "columns": [
                    {"name": "order_id", "role": "identifier", "aliases": ["Mã bill"]},
                    {"name": "order_date", "role": "transaction_date", "aliases": ["Ngày giao dịch"]},
                    {
                        "name": "order_total_amount",
                        "role": "measure",
                        "aliases": ["TongTien"],
                        "definition": "Amount payable",
                    },
                    {
                        "name": "status",
                        "role": "status",
                        "sample_values": ["Hoàn thành", "Đã hủy", "Đã trả hàng"],
                    },
                ],
            }]
        }
    }
    plan = QueryPlan(
        table="invoices",
        metric=Metric(column="*", aggregation="count"),
        filters=[Filter(column="TongTien", operator="gt", value=0)],
    )
    result = complete_plan(plan, "số đơn tháng này", "2026-03-04", context, database)
    assert result.status == "completed"
    assert result.original_plan.metric.column == "*"
    assert result.plan.table == "orders"
    assert result.plan.metric.column == "order_id"
    assert result.plan.metric.aggregation == "count_distinct"
    assert result.plan.filters[0].column == "order_total_amount"
    date_filter = next(
        item for item in result.plan.filters if item.column == "order_date"
    )
    assert date_filter.operator == "between"
    assert date_filter.value == ["2026-03-01", "2026-03-04"]
    status_filter = next(
        item for item in result.plan.filters if item.column == "status"
    )
    assert status_filter.operator == "not_in"
    assert "Đã hủy" in status_filter.value
    database.close()


def test_complete_plan_expands_semantic_status_values() -> None:
    connection = duckdb.connect(":memory:")
    connection.execute('CREATE TABLE "orders" ("status" VARCHAR, "amount" INTEGER)')
    connection.execute("INSERT INTO orders VALUES ('Hoàn thành', 10), ('Đã hủy', 20)")
    database = BusinessDatabase(
        connection,
        {"orders": TableMetadata("orders", "test", None, 2, ["status", "amount"])},
    )
    context = {
        "schema": {
            "tables": [{
                "table": "orders",
                "columns": [{
                    "name": "status",
                    "role": "status",
                    "sample_values": ["Hoàn thành", "Đã hủy"],
                }, {"name": "amount", "role": "measure"}],
            }]
        }
    }
    plan = QueryPlan(
        table="orders",
        metric=Metric(column="amount", aggregation="sum"),
        filters=[Filter(column="status", operator="eq", value="cancelled")],
    )
    result = complete_plan(plan, "doanh thu đơn huỷ", "2026-03-04", context, database)
    assert result.plan.filters[0].value == "Đã hủy"
    database.close()


def test_status_matching_uses_complete_words_not_substrings() -> None:
    connection = duckdb.connect(":memory:")
    connection.execute('CREATE TABLE "orders" ("status" VARCHAR, "amount" INTEGER)')
    database = BusinessDatabase(
        connection,
        {"orders": TableMetadata("orders", "test", None, 2, ["status", "amount"])},
    )
    context = {
        "schema": {"tables": [{
            "table": "orders",
            "columns": [{
                "name": "status",
                "role": "status",
                "sample_values": ["Đổi trả", "Đã giao", "Đang vận chuyển"],
            }, {"name": "amount", "role": "measure"}],
        }]}
    }
    plan = QueryPlan(
        table="orders",
        metric=Metric(column="amount", aggregation="sum"),
        filters=[Filter(column="status", operator="not_in", value=["Đổi trả", "Huỷ"])],
    )
    result = complete_plan(plan, "tổng tiền bán được", "2026-05-25", context, database)
    assert result.plan.filters[0].value == ["Đổi trả", "Huỷ"]
    database.close()


def test_complete_plan_rejects_unknown_columns() -> None:
    connection = duckdb.connect(":memory:")
    connection.execute('CREATE TABLE "orders" ("amount" INTEGER)')
    database = BusinessDatabase(
        connection,
        {"orders": TableMetadata("orders", "test", None, 1, ["amount"])},
    )
    context = {"schema": {"tables": [{"table": "orders", "columns": [{"name": "amount"}]}]}}
    plan = QueryPlan(
        table="orders",
        metric=Metric(column="missing_column", aggregation="sum"),
    )
    result = complete_plan(plan, "doanh thu", "2026-03-04", context, database)
    assert result.status == "rejected"
    assert result.plan.metric.column == "missing_column"
    database.close()


def test_complete_plan_adds_product_join_for_missing_product_name() -> None:
    connection = duckdb.connect(":memory:")
    connection.execute('CREATE TABLE "purchase_lines" ("Mã hàng" VARCHAR, "Đơn giá" INTEGER)')
    connection.execute('CREATE TABLE "products" ("Mã hàng" VARCHAR, "Tên hàng" VARCHAR)')
    database = BusinessDatabase(
        connection,
        {
            "purchase_lines": TableMetadata(
                "purchase_lines", "test", None, 1, ["Mã hàng", "Đơn giá"]
            ),
            "products": TableMetadata(
                "products", "test", None, 1, ["Mã hàng", "Tên hàng"]
            ),
        },
    )
    context = {"schema": {"tables": [
        {"table": "purchase_lines", "columns": [{"name": "Mã hàng"}, {"name": "Đơn giá"}]},
        {"table": "products", "columns": [{"name": "Mã hàng"}, {"name": "Tên hàng"}]},
    ]}}
    plan = QueryPlan(
        table="purchase_lines",
        metric=Metric(column="Đơn giá", aggregation="avg"),
        filters=[Filter(column="Tên hàng", operator="eq", value="Mì Omachi sườn hầm")],
    )
    result = complete_plan(plan, "giá nhập Mì Omachi", "2026-08-05", context, database)
    assert result.status == "completed"
    assert result.plan.joins[0].table == "products"
    assert result.plan.joins[0].on_conditions[0].left_column == "Mã hàng"
    assert result.plan.filters[0].column == "Tên hàng"
    database.close()


def test_complete_plan_does_not_guess_among_multiple_measures() -> None:
    connection = duckdb.connect(":memory:")
    connection.execute('CREATE TABLE "orders" ("amount" INTEGER, "paid" INTEGER)')
    database = BusinessDatabase(
        connection,
        {"orders": TableMetadata("orders", "test", None, 1, ["amount", "paid"])},
    )
    context = {
        "schema": {
            "tables": [{
                "table": "orders",
                "columns": [
                    {"name": "amount", "role": "measure"},
                    {"name": "paid", "role": "measure"},
                ],
            }]
        }
    }
    plan = QueryPlan(table="orders", metric=Metric(column="*", aggregation="sum"))
    result = complete_plan(plan, "tổng tiền", "2026-03-04", context, database)
    assert result.status == "rejected"
    database.close()


def test_pipeline_preserves_original_plan_and_writes_completion(tmp_path) -> None:
    data_dir = tmp_path / "data" / "biz_001"
    source = _mapped_orders_dir(tmp_path / "mapped")
    data_dir.mkdir(parents=True)
    (data_dir / "canonical_mapping.yml").write_text(
        (source / "canonical_mapping.yml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (data_dir / "invoices.csv").write_text(
        (source / "invoices.csv").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        json.dumps({
            "id": "q1",
            "business_id": "biz_001",
            "question": "số đơn tháng này",
            "as_of_date": "2026-03-04",
        }) + "\n",
        encoding="utf-8",
    )

    class Parser:
        last_response_content = None
        last_raw_plan = {
            "table": "orders",
            "metric": {"column": "*", "aggregation": "count"},
        }

        def parse(self, question, database, as_of_date, context=None):
            return QueryPlan(
                table="orders",
                metric=Metric(column="*", aggregation="count"),
            )

    output = tmp_path / "output.jsonl"
    answer_questions(questions, tmp_path / "data", output, parser=Parser())
    evidence = tmp_path / "output_questions" / "q1" / "evidence"
    validated = json.loads((evidence / "06_validated_plan.json").read_text(encoding="utf-8"))
    completed = json.loads((evidence / "06_completed_plan.json").read_text(encoding="utf-8"))
    assert validated["metric"]["column"] == "*"
    assert completed["metric"]["column"] == "order_id"
    assert completed["metric"]["aggregation"] == "count_distinct"
    trace = json.loads((evidence / "07_query_trace.json").read_text(encoding="utf-8"))
    assert trace[0]["kind"] == "plan_completion"


def test_completion_resolves_mojibake_aliases_on_joined_columns() -> None:
    connection = duckdb.connect(":memory:")
    connection.execute('CREATE TABLE "orders" ("order_id" VARCHAR)')
    connection.execute('CREATE TABLE "products" ("Hoạt chất" VARCHAR)')
    database = BusinessDatabase(
        connection,
        {
            "orders": TableMetadata("orders", "test", None, 1, ["order_id"]),
            "products": TableMetadata("products", "test", None, 1, ["Hoạt chất"]),
        },
    )
    context = {"schema": {"tables": [
        {"table": "orders", "columns": [{"name": "order_id"}]},
        {"table": "products", "columns": [{"name": "Hoạt chất"}]},
    ]}}
    plan = QueryPlan(
        table="orders",
        metric=Metric(column="order_id", aggregation="count"),
        joins=[{"table": "products", "on_conditions": []}],
        filters=[Filter(column="Hoáº¡t cháº¥t", operator="eq", value="Natri")],
    )
    result = complete_plan(plan, "thuốc có hoạt chất", "2026-05-25", context, database)
    assert result.status == "completed"
    assert result.plan.filters[0].column == "Hoạt chất"
    database.close()


def test_completion_resolves_qualified_column_case_and_whitespace() -> None:
    connection = duckdb.connect(":memory:")
    connection.execute('CREATE TABLE "details" ("Số Lượng" INTEGER)')
    database = BusinessDatabase(
        connection,
        {"details": TableMetadata("details", "test", None, 1, ["Số Lượng"])},
    )
    context = {"schema": {"tables": [
        {"table": "details", "columns": [{"name": "Số Lượng"}]},
    ]}}
    plan = QueryPlan(
        table="details",
        metric=Metric(column="details.  SỐ   lượng ", aggregation="sum"),
    )

    result = complete_plan(plan, "tổng số lượng", "2026-05-25", context, database)

    assert result.status == "completed"
    assert result.plan.metric.column == "Số Lượng"
    database.close()


def test_count_wildcard_is_not_treated_as_unknown_column() -> None:
    connection = duckdb.connect(":memory:")
    connection.execute('CREATE TABLE "orders" ("order_id" VARCHAR)')
    database = BusinessDatabase(
        connection,
        {"orders": TableMetadata("orders", "test", None, 1, ["order_id"])},
    )
    plan = QueryPlan(table="orders", metric=Metric(column="*", aggregation="count"))
    from datamate.execution import compile_query

    sql, params = compile_query(plan, database)
    assert "COUNT(*)" in sql
    assert params == []
    database.close()


def test_completion_resolves_metric_on_joined_table() -> None:
    connection = duckdb.connect(":memory:")
    connection.execute('CREATE TABLE "orders" ("order_id" VARCHAR)')
    connection.execute('CREATE TABLE "lines" ("Đơn giá" INTEGER)')
    database = BusinessDatabase(
        connection,
        {
            "orders": TableMetadata("orders", "test", None, 1, ["order_id"]),
            "lines": TableMetadata("lines", "test", None, 1, ["Đơn giá"]),
        },
    )
    context = {"schema": {"tables": [
        {"table": "orders", "columns": [{"name": "order_id"}]},
        {"table": "lines", "columns": [{"name": "Đơn giá"}]},
    ]}}
    plan = QueryPlan(
        table="orders",
        metric=Metric(column="ÄÆ¡n giÃ¡", aggregation="sum"),
        joins=[{"table": "lines", "on_conditions": []}],
    )
    result = complete_plan(plan, "tổng tiền", "2026-05-25", context, database)
    assert result.status == "completed"
    assert result.plan.metric.column == "Đơn giá"
    database.close()
