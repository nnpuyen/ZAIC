from __future__ import annotations

import pandas as pd
import json
import pytest

from datamate.execution import compile_query, execute_plan, execute_plan_with_trace
from datamate.ingestion.loader import BusinessDatabase, TableMetadata
from datamate.ingestion.loader import load_business_data
from datamate.pipeline import _format_answer, answer_questions
from datamate.context import build_context
from datamate.ingestion.profiler import profile_table
from datamate.planning.retrieval import retrieve_context
from datamate.planning.dates import resolve_date_range
from datamate.planning.schemas import CTE, Filter, FilterGroup, Join, Metric, QueryPlan

import duckdb


def test_format_answer_uses_declared_answer_type_shape() -> None:
    scalar_plan = QueryPlan(table="orders")
    rows = [{"metric": 2.08}]

    assert _format_answer("number", rows, scalar_plan) == 2.08
    assert _format_answer("percent", rows, scalar_plan) == 2.08
    assert _format_answer("boolean", [{"metric": True}], scalar_plan) is True
    assert _format_answer("list", rows, scalar_plan) == rows
    assert _format_answer("table", rows, scalar_plan) == rows
    assert _format_answer("string", [{"value": "done"}], scalar_plan) == "done"
    assert _format_answer("string", [{"value": "done", "metric": 1}], scalar_plan) == "done"


def test_format_answer_preserves_multi_metric_rows() -> None:
    plan = QueryPlan(table="orders", metrics=[Metric(column="a", aggregation="sum")])
    rows = [{"metric": 2.08, "metric_1": 3.14}]

    assert _format_answer("percent", rows, plan) == rows


def test_format_answer_uses_source_column_semantic_for_number_precision() -> None:
    connection = duckdb.connect()
    connection.execute("CREATE TABLE orders (whole INTEGER, measured DOUBLE)")
    database = BusinessDatabase(connection=connection)

    integer_plan = QueryPlan(table="orders", metric=Metric(column="whole", aggregation="sum"))
    decimal_plan = QueryPlan(table="orders", metric=Metric(column="measured", aggregation="sum"))

    assert _format_answer("number", [{"metric": 12.75}], integer_plan, database) == 13
    assert _format_answer("number", [{"metric": 12.756}], decimal_plan, database) == 12.76
    connection.close()


def test_format_answer_defaults_computed_number_to_two_decimals() -> None:
    plan = QueryPlan(table="orders", metric=Metric(column="*", aggregation="ratio", denominator_column="x"))
    assert _format_answer("number", [{"metric": 2.756}], plan) == 2.76


def test_format_answer_keeps_count_based_percentages_decimal() -> None:
    plan = QueryPlan(
        table="orders",
        answer_type="percent",
        metric=Metric(column="order_id", aggregation="count_distinct"),
    )

    assert _format_answer("percent", [{"metric": 2.08}], plan) == 2.08


def test_format_answer_keeps_integer_average_for_integer_valued_source() -> None:
    connection = duckdb.connect()
    connection.execute("CREATE TABLE orders (unit_price INTEGER)")
    database = BusinessDatabase(connection=connection)
    plan = QueryPlan(
        table="orders",
        metric=Metric(column="unit_price", aggregation="avg"),
    )

    assert _format_answer("number", [{"metric": 200556.0}], plan, database) == 200556
    connection.close()


def test_date_resolution() -> None:
    assert resolve_date_range("tháng này", "2026-03-04") == ("2026-03-01", "2026-03-04")
    assert resolve_date_range("tháng 2/2026", "2026-03-04") == ("2026-02-01", "2026-02-28")
    assert resolve_date_range("quý 1", "2026-03-04") == ("2026-01-01", "2026-03-31")


def test_loader_infers_dates_and_comma_formatted_numbers(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "invoices.csv").write_text(
        "amount,date,status,code\n"
        '"35,000",01/04/2026,Hoàn thành,ORD000001\n',
        encoding="utf-8",
    )

    database = load_business_data(data_dir)
    columns = database.connection.execute(
        'SELECT column_name, column_type FROM (DESCRIBE "invoices")'
    ).fetchall()
    assert dict(columns) == {
        "amount": "BIGINT",
        "date": "DATE",
        "status": "VARCHAR",
        "code": "VARCHAR",
    }
    database.close()


def test_loader_lowercases_string_values(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "orders.csv").write_text(
        "order_id,status,note\n"
        "ORD-1, Hoàn Thành ,  Ghi Chú  \n",
        encoding="utf-8",
    )

    database = load_business_data(data_dir)
    assert database.connection.execute(
        'SELECT "order_id", "status", "note" FROM "orders"'
    ).fetchone() == ("ord-1", "hoàn thành", "ghi chú")
    database.close()


def test_loader_normalizes_mixed_date_formats(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "sales.csv").write_text(
        "sale_date,amount\n"
        "3 thg 3 2026,10\n"
        "04/03/2026,20\n"
        "46086,30\n",
        encoding="utf-8",
    )

    database = load_business_data(data_dir)
    columns = database.connection.execute(
        'SELECT column_name, column_type FROM (DESCRIBE "sales")'
    ).fetchall()
    assert dict(columns)["sale_date"] == "DATE"
    assert database.connection.execute(
        'SELECT "sale_date" FROM "sales" ORDER BY amount'
    ).fetchall() == [
        (pd.Timestamp("2026-03-03").date(),),
        (pd.Timestamp("2026-03-04").date(),),
        (pd.Timestamp("2026-03-05").date(),),
    ]
    database.close()


def test_loader_deduplicates_exact_rows_in_order_exports(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "order_lines.csv").write_text(
        "Order ID,amount,status\n"
        "ORD-1,80000,done\n"
        "ORD-1,80000,done\n"
        "ORD-2,10000,done\n",
        encoding="utf-8",
    )
    (data_dir / "payments.csv").write_text(
        "Order ID,amount\n"
        "ORD-1,80000\n"
        "ORD-1,80000\n",
        encoding="utf-8",
    )

    database = load_business_data(data_dir)
    assert database.tables["order_lines"].rows == 2
    assert database.tables["payments"].rows == 2
    database.close()


def test_loader_applies_canonical_mapping_before_type_inference(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "canonical_mapping.yml").write_text(
        "tables:\n"
        "  invoices:\n"
        "    canonical_name: orders\n"
        "    columns:\n"
        "      order_date:\n"
        "        aliases: [Ngày giao dịch]\n"
        "        type: date\n"
        "      order_total_amount:\n"
        "        aliases: [TongTien]\n"
        "        type: decimal\n",
        encoding="utf-8",
    )
    (data_dir / "invoices.csv").write_text(
        "TongTien,Ngày giao dịch\n"
        '"35,000",01/04/2026\n',
        encoding="utf-8",
    )

    database = load_business_data(data_dir)
    try:
        assert database.tables["orders"].columns == [
            "order_total_amount",
            "order_date",
        ]
        assert database.connection.execute(
            'SELECT "order_total_amount", "order_date" FROM "orders"'
        ).fetchall() == [(35000, pd.Timestamp("2026-04-01").date())]
        assert database.tables["orders"].raw_columns == ["TongTien", "Ngày giao dịch"]
        assert {
            item["canonical_name"] for item in database.tables["orders"].column_mappings
        } == {"order_date", "order_total_amount"}
    finally:
        database.close()


def test_compiler_binds_values_and_executes() -> None:
    connection = duckdb.connect(":memory:")
    frame = pd.DataFrame({"date": ["2026-03-01", "2026-03-02"], "amount": [10, 20]})
    connection.register("_frame", frame)
    connection.execute('CREATE TABLE "sales" AS SELECT * FROM _frame')
    database = BusinessDatabase(connection, {"sales": TableMetadata("sales", "test", None, 2, ["date", "amount"])})
    plan = QueryPlan(
        table="sales",
        metric=Metric(column="amount", aggregation="sum"),
        filters=[Filter(column="date", operator="between", value=["2026-03-01", "2026-03-02"])],
    )
    sql, params = compile_query(plan, database)
    assert "?" in sql
    assert params == ["2026-03-01", "2026-03-02"]
    assert execute_plan(plan, database) == [{"metric": 30}]


def test_compiler_supports_weighted_average() -> None:
    connection = duckdb.connect(":memory:")
    connection.execute('CREATE TABLE "sales" ("price" VARCHAR, "quantity" INTEGER)')
    connection.execute("INSERT INTO sales VALUES ('10,000', 1), ('20,000', 3)")
    database = BusinessDatabase(
        connection,
        {"sales": TableMetadata("sales", "test", None, 2, ["price", "quantity"])},
    )
    plan = QueryPlan(
        table="sales",
        metric=Metric(
            column="price", aggregation="weighted_avg", weight_column="quantity"
        ),
    )
    sql, _ = compile_query(plan, database)
    assert "SUM((TRY_CAST" in sql
    assert execute_plan(plan, database) == [{"metric": 17500.0}]
    database.close()


def test_compiler_qualifies_columns_when_joined_tables_share_names() -> None:
    connection = duckdb.connect(":memory:")
    connection.execute(
        'CREATE TABLE "order_lines" ("order_id" INTEGER, "amount" INTEGER, "name" VARCHAR)'
    )
    connection.execute(
        'CREATE TABLE "products" ("order_id" INTEGER, "amount" INTEGER, "name" VARCHAR)'
    )
    connection.execute("INSERT INTO order_lines VALUES (1, 10, 'line')")
    connection.execute("INSERT INTO products VALUES (1, 100, 'product')")
    database = BusinessDatabase(
        connection,
        {
            "order_lines": TableMetadata(
                "order_lines", "test", None, 1, ["order_id", "amount", "name"]
            ),
            "products": TableMetadata(
                "products", "test", None, 1, ["order_id", "amount", "name"]
            ),
        },
    )
    plan = QueryPlan(
        table="order_lines",
        joins=[
            Join(
                table="products",
                on_conditions=[
                    {"left_column": "order_id", "right_column": "order_id"}
                ],
            )
        ],
        metric=Metric(column="amount", aggregation="sum"),
        filters=[Filter(column="name", operator="eq", value="line")],
    )

    sql, _ = compile_query(plan, database)

    assert 'SUM(TRY_CAST(REPLACE(CAST("order_lines"."amount" AS VARCHAR)' in sql
    assert 'WHERE "order_lines"."name" = ?' in sql
    assert execute_plan(plan, database) == [{"metric": 10}]
    database.close()


def test_compiler_matches_column_names_case_and_whitespace_insensitively() -> None:
    connection = duckdb.connect(":memory:")
    connection.execute('CREATE TABLE "details" ("Số Lượng" INTEGER)')
    connection.execute('INSERT INTO "details" VALUES (77)')
    database = BusinessDatabase(
        connection,
        {"details": TableMetadata("details", "test", None, 1, ["Số Lượng"])},
    )
    plan = QueryPlan(
        table="details",
        metric=Metric(
            column="details.  SỐ   lượng ",
            aggregation="sum",
        ),
    )

    rows = execute_plan(plan, database)

    assert rows[0]["metric"] == 77
    database.close()


@pytest.mark.parametrize(
    ("operator", "value", "expected"),
    [
        ("eq", 10, 10),
        ("neq", 10, 20),
        ("gt", 10, 20),
        ("gte", 10, 30),
        ("lt", 20, 10),
        ("lte", 20, 30),
        ("in", [10], 10),
        ("not_in", [20], 10),
        ("contains", "0", 30),
        ("between", [10, 20], 30),
    ],
)
def test_compiler_supports_filter_operators(
    operator: str, value: object, expected: int
) -> None:
    connection = duckdb.connect(":memory:")
    connection.execute('CREATE TABLE "sales" ("amount" INTEGER)')
    connection.execute("INSERT INTO sales VALUES (10), (20)")
    database = BusinessDatabase(
        connection,
        {"sales": TableMetadata("sales", "test", None, 2, ["amount"])},
    )
    plan = QueryPlan(
        table="sales",
        metric=Metric(column="amount", aggregation="sum"),
        filters=[Filter(column="amount", operator=operator, value=value)],
    )
    assert execute_plan(plan, database) == [{"metric": expected}]
    database.close()


def test_grouped_order_threshold_returns_numeric_count() -> None:
    connection = duckdb.connect(":memory:")
    connection.execute(
        'CREATE TABLE "sales" ("order_id" VARCHAR, "amount" VARCHAR, "status" VARCHAR)'
    )
    connection.execute(
        "INSERT INTO sales VALUES "
        "('A', '30,000', 'done'), ('A', '30,000', 'done'), "
        "('B', '40,000', 'done'), ('C', '70,000', 'done'), "
        "('D', '100,000', 'cancelled')"
    )
    database = BusinessDatabase(
        connection,
        {"sales": TableMetadata("sales", "test", None, 5, ["order_id", "amount", "status"])},
    )
    plan = QueryPlan(
        operation="aggregation",
        table="sales",
        metric=Metric(column="order_id", aggregation="count_distinct"),
        filters=[Filter(column="status", operator="neq", value="cancelled")],
        having_filters=[Filter(column="amount", operator="gte", value=55000)],
        group_by=["order_id"],
        answer_type="number",
    )

    sql, _ = compile_query(plan, database)
    rows = execute_plan(plan, database)

    assert "HAVING" in sql
    assert rows == [{"metric": 2}]
    database.close()


def test_runtime_normalization_resolves_categorical_values_and_traces_changes() -> None:
    connection = duckdb.connect(":memory:")
    connection.execute('CREATE TABLE "orders" ("status" VARCHAR, "amount" INTEGER)')
    connection.execute("INSERT INTO orders VALUES ('Hoàn thành', 10), ('Huỷ', 20)")
    database = BusinessDatabase(
        connection,
        {"orders": TableMetadata("orders", "test", None, 2, ["status", "amount"])},
    )
    plan = QueryPlan(
        table="orders",
        metric=Metric(column="amount", aggregation="sum"),
        filters=[Filter(column="status", operator="eq", value="hoan thanh")],
    )

    rows, trace = execute_plan_with_trace(plan, database)

    assert rows == [{"metric": 10}]
    assert trace[0]["kind"] == "normalization"
    assert trace[0]["changes"][0]["normalized_value"] == "Hoàn thành"
    assert plan.filters[0].value == "hoan thanh"
    database.close()


def test_runtime_normalization_covers_nested_and_metric_conditions() -> None:
    connection = duckdb.connect(":memory:")
    connection.execute(
        'CREATE TABLE "orders" ("status" VARCHAR, "amount" INTEGER, "kind" VARCHAR)'
    )
    connection.execute(
        "INSERT INTO orders VALUES ('Hoàn thành', 10, 'Bán lẻ'), ('Huỷ', 20, 'Bán sỉ')"
    )
    database = BusinessDatabase(
        connection,
        {
            "orders": TableMetadata(
                "orders", "test", None, 2, ["status", "amount", "kind"]
            )
        },
    )
    metric = Metric(
        column="amount",
        aggregation="sum",
        filter=FilterGroup(
            conditions=[Filter(column="kind", operator="eq", value="ban le")]
        ),
        denominator_filter=Filter(
            column="status", operator="eq", value="hoan thanh"
        ),
    )
    plan = QueryPlan(
        table="orders",
        metric=metric,
        filters=[
            FilterGroup(
                conditions=[Filter(column="status", operator="eq", value="huy")]
            )
        ],
        having_filters=[Filter(column="kind", operator="eq", value="ban si")],
        qualify_filters=[Filter(column="status", operator="eq", value="hoan thanh")],
    )

    from datamate.execution import normalize_plan

    normalized, changes = normalize_plan(plan, database)

    assert len(changes) == 5
    assert normalized.filters[0].conditions[0].value == "Huỷ"
    assert normalized.metric.filter.conditions[0].value == "Bán lẻ"
    assert normalized.metric.denominator_filter.value == "Hoàn thành"
    assert normalized.having_filters[0].value == "Bán sỉ"
    assert normalized.qualify_filters[0].value == "Hoàn thành"
    database.close()


def test_compiler_supports_null_filter_operators() -> None:
    connection = duckdb.connect(":memory:")
    connection.execute('CREATE TABLE "sales" ("amount" INTEGER)')
    connection.execute("INSERT INTO sales VALUES (10), (NULL)")
    database = BusinessDatabase(
        connection,
        {"sales": TableMetadata("sales", "test", None, 2, ["amount"])},
    )
    for operator in ("is_null", "is_not_null"):
        plan = QueryPlan(
            table="sales",
            metric=Metric(column="amount", aggregation="count"),
            filters=[Filter(column="amount", operator=operator)],
        )
        assert execute_plan(plan, database) == [{"metric": 1}]
    database.close()


def test_compiler_supports_nested_filter_groups() -> None:
    connection = duckdb.connect(":memory:")
    connection.execute('CREATE TABLE "sales" ("status" VARCHAR, "amount" INTEGER)')
    connection.execute(
        "INSERT INTO sales VALUES ('paid', 10), ('pending', 20), ('cancelled', 30)"
    )
    database = BusinessDatabase(
        connection,
        {"sales": TableMetadata("sales", "test", None, 3, ["status", "amount"])},
    )
    plan = QueryPlan(
        table="sales",
        metric=Metric(column="amount", aggregation="sum"),
        filters=[
            FilterGroup(
                logic="or",
                conditions=[
                    Filter(column="status", operator="eq", value="paid"),
                    FilterGroup(
                        conditions=[
                            Filter(column="status", operator="eq", value="pending"),
                            Filter(column="amount", operator="gt", value=10),
                        ]
                    ),
                ],
            )
        ],
    )
    sql, params = compile_query(plan, database)
    assert " OR " in sql
    assert params == ["paid", "pending", 10]
    assert execute_plan(plan, database) == [{"metric": 30}]
    database.close()


def test_query_plan_accepts_join_operation() -> None:
    plan = QueryPlan(
        operation="join",
        table="sales",
        metric=Metric(column="amount", aggregation="sum"),
    )
    assert plan.operation == "join"


def test_join_plan_compiles_and_executes() -> None:
    connection = duckdb.connect(":memory:")
    connection.execute('CREATE TABLE "sales" ("product_id" INTEGER, "amount" INTEGER)')
    connection.execute('CREATE TABLE "products" ("id" INTEGER, "name" VARCHAR)')
    connection.execute('INSERT INTO "sales" VALUES (1, 10), (2, 20)')
    connection.execute("INSERT INTO products VALUES (1, 'A'), (3, 'C')")
    database = BusinessDatabase(
        connection,
        {
            "sales": TableMetadata("sales", "test", None, 2, ["product_id", "amount"]),
            "products": TableMetadata("products", "test", None, 2, ["id", "name"]),
        },
    )
    plan = QueryPlan(
        operation="join",
        table="sales",
        metric=Metric(column="amount", aggregation="sum"),
        joins=[Join(table="products", left_column="product_id", right_column="id")],
    )
    assert execute_plan(plan, database) == [{"metric": 10}]
    database.close()


def test_joined_filter_is_normalized_against_join_owner() -> None:
    connection = duckdb.connect(":memory:")
    connection.execute('CREATE TABLE "lines" ("product_id" INTEGER, "amount" INTEGER)')
    connection.execute('CREATE TABLE "products" ("id" INTEGER, "name" VARCHAR)')
    connection.execute("INSERT INTO lines VALUES (1, 10), (2, 20)")
    connection.execute("INSERT INTO products VALUES (1, 'A'), (2, 'B')")
    database = BusinessDatabase(
        connection,
        {
            "lines": TableMetadata("lines", "test", None, 2, ["product_id", "amount"]),
            "products": TableMetadata("products", "test", None, 2, ["id", "name"]),
        },
    )
    plan = QueryPlan(
        table="lines",
        joins=[Join(table="products", on_conditions=[{"left_column": "product_id", "right_column": "id"}])],
        metric=Metric(column="amount", aggregation="sum"),
        filters=[Filter(column="name", operator="eq", value="a")],
    )
    assert execute_plan(plan, database) == [{"metric": 10}]
    database.close()


def test_percentage_metric_compiles_filter_and_denominator() -> None:
    connection = duckdb.connect(":memory:")
    connection.execute('CREATE TABLE "orders" ("id" INTEGER, "status" VARCHAR)')
    connection.execute("INSERT INTO orders VALUES (1, 'done'), (2, 'cancelled'), (3, 'cancelled')")
    database = BusinessDatabase(
        connection,
        {"orders": TableMetadata("orders", "test", None, 3, ["id", "status"])},
    )
    plan = QueryPlan(
        table="orders",
        metric=Metric(
            column="id",
            aggregation="percentage",
            distinct=True,
            filter=Filter(column="status", operator="eq", value="cancelled"),
        ),
        answer_type="percent",
    )
    assert execute_plan(plan, database) == [{"metric": 66.67}]
    database.close()


def test_cte_can_be_joined_and_executed() -> None:
    connection = duckdb.connect(":memory:")
    connection.execute('CREATE TABLE "lines" ("product_id" INTEGER, "amount" INTEGER)')
    connection.execute('CREATE TABLE "products" ("id" INTEGER, "active" BOOLEAN)')
    connection.execute("INSERT INTO lines VALUES (1, 10), (2, 20)")
    connection.execute("INSERT INTO products VALUES (1, TRUE), (2, FALSE)")
    database = BusinessDatabase(
        connection,
        {
            "lines": TableMetadata("lines", "test", None, 2, ["product_id", "amount"]),
            "products": TableMetadata("products", "test", None, 2, ["id", "active"]),
        },
    )
    plan = QueryPlan(
        table="lines",
        ctes=[CTE(
            name="active_products",
            query=QueryPlan(
                table="products",
                select_columns=["id"],
                filters=[Filter(column="active", operator="eq", value=True)],
            ),
        )],
        joins=[Join(table="active_products", on_conditions=[{"left_column": "product_id", "right_column": "id"}])],
        metric=Metric(column="amount", aggregation="sum"),
    )
    sql, _ = compile_query(plan, database)
    assert sql.startswith('WITH "active_products" AS')
    assert execute_plan(plan, database) == [{"metric": 10}]
    database.close()


def test_execution_allows_filter_values_absent_from_column() -> None:
    connection = duckdb.connect(":memory:")
    connection.execute('CREATE TABLE "orders" ("status" VARCHAR, "amount" INTEGER)')
    connection.execute("INSERT INTO orders VALUES ('Đã giao', 10), ('Đổi trả', 5)")
    database = BusinessDatabase(
        connection,
        {"orders": TableMetadata("orders", "test", None, 2, ["status", "amount"])},
    )
    plan = QueryPlan(
        table="orders",
        metric=Metric(column="amount", aggregation="sum"),
        filters=[Filter(column="status", operator="neq", value="trả hàng")],
    )
    assert execute_plan(plan, database) == [{"metric": 15}]
    database.close()


def test_execution_accepts_iso_date_filter_for_date_column() -> None:
    connection = duckdb.connect(":memory:")
    connection.execute('CREATE TABLE "orders" ("date" DATE, "amount" INTEGER)')
    connection.execute("INSERT INTO orders VALUES ('2025-08-12', 10)")
    database = BusinessDatabase(
        connection,
        {"orders": TableMetadata("orders", "test", None, 1, ["date", "amount"])},
    )
    plan = QueryPlan(
        table="orders",
        metric=Metric(column="amount", aggregation="sum"),
        filters=[Filter(column="date", operator="eq", value="2025-08-12")],
    )
    assert execute_plan(plan, database) == [{"metric": 10}]
    database.close()


def test_profiler_includes_all_values_for_small_domains() -> None:
    connection = duckdb.connect(":memory:")
    connection.execute('CREATE TABLE "orders" ("status" VARCHAR, "amount" INTEGER)')
    connection.execute(
        "INSERT INTO orders VALUES ('Hoàn thành', 10), ('Đã hủy', 5), ('Đã trả hàng', 2)"
    )
    database = BusinessDatabase(
        connection,
        {"orders": TableMetadata("orders", "test", None, 3, ["status", "amount"])},
    )
    status = next(
        column for column in profile_table(database, "orders")["columns"]
        if column["name"] == "status"
    )
    assert status["sample_values"] == ["Hoàn thành", "Đã hủy", "Đã trả hàng"]
    database.close()


def test_retrieval_keeps_small_domain_profiles() -> None:
    context = {
        "schema": {
            "tables": [{
                "table": "orders",
                "columns": [
                    {"name": "status", "type": "VARCHAR", "distinct_count": 3,
                     "sample_values": ["Hoàn thành", "Đã hủy", "Đã trả hàng"]},
                    {"name": "amount", "type": "INTEGER"},
                ],
            }]
        }
    }
    retrieved = retrieve_context("tổng tiền", "2025-08-12", context)
    columns = retrieved["schema"]["tables"][0]["columns"]
    assert columns[0]["sample_values"] == ["Hoàn thành", "Đã hủy", "Đã trả hàng"]


def test_pipeline_writes_per_question_evidence(tmp_path) -> None:
    data_dir = tmp_path / "data" / "biz_001"
    data_dir.mkdir(parents=True)
    (data_dir / "sales.csv").write_text(
        "date,amount\n2026-03-01,10\n", encoding="utf-8"
    )
    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        json.dumps({
            "id": "q1",
            "business_id": "biz_001",
            "question": "total",
            "as_of_date": "2026-03-01",
        }) + "\n",
        encoding="utf-8",
    )

    class Parser:
        last_response_content = '{"operation":"aggregation"}'
        last_raw_plan = {
            "operation": "aggregation",
            "table": "sales",
            "metric": {"column": "amount", "aggregation": "sum"},
        }
        last_token_usage = {
            "prompt_tokens": 25,
            "completion_tokens": 8,
            "total_tokens": 33,
        }

        def parse(self, question, database, as_of_date, context=None):
            return QueryPlan(
                table="sales",
                metric=Metric(column="amount", aggregation="sum"),
            )

    output = tmp_path / "output.jsonl"
    answer_questions(questions, tmp_path / "data", output, parser=Parser())
    evidence = tmp_path / "output_questions" / "q1"
    assert (evidence / "answer.json").is_file()
    assert (evidence / "evidence" / "01_question.json").is_file()
    assert (evidence / "evidence" / "05_planner_plan.json").is_file()
    assert (evidence / "evidence" / "07_query_1_current.sql").is_file()
    assert json.loads((evidence / "answer.json").read_text())["answer"] == 10
    assert json.loads((evidence / "token_usage.json").read_text()) == {
        "question_id": "q1",
        "prompt_tokens": 25,
        "completion_tokens": 8,
        "total_tokens": 33,
    }
    assert json.loads((tmp_path / "token_usage.jsonl").read_text().splitlines()[0]) == {
        "id": "q1",
        "business_id": "biz_001",
        "usage": {
            "prompt_tokens": 25,
            "completion_tokens": 8,
            "total_tokens": 33,
        },
    }


def test_pipeline_writes_retrieved_context_to_question_evidence(tmp_path) -> None:
    data_dir = tmp_path / "data" / "biz_001"
    data_dir.mkdir(parents=True)
    (data_dir / "sales.csv").write_text(
        "date,amount\n2026-03-01,10\n", encoding="utf-8"
    )
    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        json.dumps({
            "id": "q1",
            "business_id": "biz_001",
            "question": "total",
            "as_of_date": "2026-03-01",
        }) + "\n",
        encoding="utf-8",
    )

    class Parser:
        last_response_content = None
        last_raw_plan = None
        last_retrieved_context = {"schema": {"tables": [{"table": "sales"}]}}

        def parse(self, question, database, as_of_date, context=None):
            return QueryPlan(
                table="sales",
                metric=Metric(column="amount", aggregation="sum"),
            )

    output = tmp_path / "output.jsonl"
    answer_questions(questions, tmp_path / "data", output, parser=Parser())
    evidence = tmp_path / "output_questions" / "q1"
    assert json.loads((evidence / "evidence" / "03_context.json").read_text()) == {
        "schema": {"tables": [{"table": "sales"}]}
    }


def test_context_auto_discovers_shared_rulebook_and_calendar(tmp_path) -> None:
    root = tmp_path / "dataset"
    data_dir = root / "data" / "biz_001"
    data_dir.mkdir(parents=True)
    (root / "rulebook.md").write_text("Revenue excludes cancelled orders.", encoding="utf-8")
    (root / "calendar.csv").write_text("lunar,solar\n1/1,2026-02-17\n", encoding="utf-8")
    (data_dir / "sales.csv").write_text("amount\n10\n", encoding="utf-8")

    database = load_business_data(data_dir)
    try:
        context = build_context(database, data_dir)
    finally:
        database.close()

    assert "rulebook" not in context
    assert context["rulebook_sections"] == {
        "general": "Revenue excludes cancelled orders."
    }
    assert context["calendar"] == "lunar,solar\n1/1,2026-02-17\n"
