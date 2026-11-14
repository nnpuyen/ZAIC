from __future__ import annotations

from typing import Any

from .loader import BusinessDatabase


def _json_value(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def profile_table(database: BusinessDatabase, table_name: str, sample_size: int = 5) -> dict[str, Any]:
    metadata = database.tables[table_name]
    quoted_table = '"' + table_name.replace('"', '""') + '"'
    columns = database.connection.execute(f"DESCRIBE SELECT * FROM {quoted_table}").fetchall()
    result: dict[str, Any] = {
        "table": table_name,
        "source": metadata.source,
        "sheet": metadata.sheet,
        "row_count": metadata.rows,
        "canonical_name": metadata.canonical_name,
        "aliases": metadata.table_aliases,
        "grain": metadata.grain,
        "columns": [],
    }
    mapping_by_name = {
        str(item.get("canonical_name")): item
        for item in metadata.column_mappings
    }
    for column_name, column_type, *_ in columns:
        column_name = str(column_name)
        quoted_column = '"' + column_name.replace('"', '""') + '"'
        distinct_count = int(database.connection.execute(
            f"SELECT COUNT(DISTINCT {quoted_column}) FROM {quoted_table}"
        ).fetchone()[0])
        item: dict[str, Any] = {
            "name": column_name,
            "type": str(column_type),
            "distinct_count": distinct_count,
        }
        # Include the complete domain for low-cardinality columns. For larger
        # domains, keep the sample bounded to the first five distinct values.
        profile_limit = (
            distinct_count
            if distinct_count <= 10
            else min(max(int(sample_size), 0), 5)
        )
        samples = database.connection.execute(
            f"SELECT DISTINCT {quoted_column} FROM {quoted_table} "
            f"WHERE {quoted_column} IS NOT NULL LIMIT {profile_limit}"
        ).fetchall()
        item["sample_values"] = [_json_value(row[0]) for row in samples]
        mapping = mapping_by_name.get(str(column_name), {})
        for key in ("aliases", "role", "definition"):
            if mapping.get(key) is not None:
                item[key] = mapping[key]
        result["columns"].append(item)
    return result


def profile_database(database: BusinessDatabase) -> dict[str, Any]:
    return {
        "tables": [profile_table(database, name) for name in database.tables],
        "errors": database.errors,
    }
