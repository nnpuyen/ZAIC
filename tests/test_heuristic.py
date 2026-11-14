from __future__ import annotations

import duckdb
import pandas as pd

from datamate.ingestion.loader import BusinessDatabase, TableMetadata
from datamate.planning.heuristic import HeuristicParser


def _database() -> BusinessDatabase:
    connection = duckdb.connect(":memory:")
    frame = pd.DataFrame(
        {
            "Ngày tháng": ["2026-03-01", "2026-03-02", "2026-02-01"],
            "Nội dung": ["Coca", "Pepsi", "Coca"],
            "so_luong": [2, 5, 1],
            "Thanh tien": [20, 50, 10],
        }
    )
    connection.register("_frame", frame)
    connection.execute('CREATE TABLE "ban_hang" AS SELECT * FROM _frame')
    return BusinessDatabase(
        connection,
        {"ban_hang": TableMetadata("ban_hang", "test", None, 3, list(frame.columns))},
    )


def test_parser_builds_aggregation_and_product_filter() -> None:
    database = _database()
    plan = HeuristicParser().parse("Coca tháng 3/2026 bán được bao nhiêu sản phẩm?", database, "2026-03-04")
    assert plan.table == "ban_hang"
    assert plan.metric.column == "so_luong"
    assert plan.filters[0].value == ["2026-03-01", "2026-03-31"]
    assert any(item.value == "Coca" for item in plan.filters)
    database.close()


def test_parser_builds_ranking_and_comparison() -> None:
    database = _database()
    parser = HeuristicParser()
    ranking = parser.parse("Top 1 mặt hàng bán chạy nhất tháng này", database, "2026-03-04")
    assert ranking.operation == "ranking"
    assert ranking.limit == 1
    comparison = parser.parse("Doanh thu tháng này tăng bao nhiêu phần trăm so với tháng trước?", database, "2026-03-04")
    assert comparison.answer_type == "percent"
    assert comparison.comparison_filters
    database.close()
