from __future__ import annotations

import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from datetime import timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import yaml

from ..numeric import (
    NUMBER_SQL_FUNCTION,
    has_number_format_marker,
    parse_measure_number,
)


SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xls", ".sqlite", ".sqlite3", ".db"}


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _safe_name(value: str) -> str:
    result = re.sub(r"\W+", "_", value.strip(), flags=re.UNICODE).strip("_")
    return result or "table"


def _read_csv(path: Path) -> pd.DataFrame:
    encodings = ("utf-8-sig", "utf-8", "cp1258", "cp1252")
    last_error: UnicodeError | None = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding, on_bad_lines="error")
        except UnicodeDecodeError as error:
            last_error = error
    raise ValueError(f"Cannot decode CSV {path}") from last_error


def _read_business_columns(root: Path) -> dict[str, list[str]]:
    """Read documented source headers, keyed by normalized source filename."""
    columns: dict[str, list[str]] = {}
    for readme in sorted(root.glob("README*")):
        content = ""
        for encoding in ("utf-8", "utf-8-sig", "cp1258", "cp1252"):
            try:
                content = readme.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        content = repair_mojibake(content)
        current_file: str | None = None
        for line in content.splitlines():
            section = re.search(r"^#{1,6}.*?`([^`]+)`", line)
            if section and re.search(r"\bcolumns?\b|\bcot\b", _fold_text(line)):
                current_file = _normalized_mapping_key(section.group(1))
                columns.setdefault(current_file, [])
                continue
            if re.match(r"^#{1,6}\s", line):
                current_file = None
                continue
            if current_file is None:
                continue
            bullet = re.match(r"\s*[-*+]\s+\*\*(.+?)\*\*\s*:", line)
            if bullet:
                name = bullet.group(1).strip()
                if name and name not in columns[current_file]:
                    columns[current_file].append(name)
    return columns


def _read_tabular(
    path: Path, documented_columns: list[str], sheet: str | None = None
) -> pd.DataFrame:
    """Find source headers after trimming sparse preamble and empty columns.

    A candidate is accepted only when its normalized value is documented for
    this source in the business README. Leading rows with values in fewer than
    half of the populated columns are discarded first. This lets exports with
    title/preamble rows use their later header row while keeping undocumented
    columns explicitly unnamed.
    """
    if path.suffix.lower() == ".csv":
        raw = None
        last_error: UnicodeError | None = None
        for encoding in ("utf-8-sig", "utf-8", "cp1258", "cp1252"):
            try:
                raw = pd.read_csv(path, encoding=encoding, header=None, on_bad_lines="error")
                break
            except UnicodeDecodeError as error:
                last_error = error
        if raw is None:
            raise ValueError(f"Cannot decode CSV {path}") from last_error
    else:
        raw = pd.read_excel(path, sheet_name=sheet or 0, header=None)

    # Remove columns that are blank in every row before measuring row density.
    # Whitespace-only cells count as blank alongside pandas' missing values.
    non_empty_by_column: list[pd.Series] = []
    active_column_positions: list[int] = []
    for column_index in range(raw.shape[1]):
        non_empty = raw.iloc[:, column_index].map(
            lambda value: pd.notna(value) and bool(str(value).strip())
        )
        non_empty_by_column.append(non_empty)
        if non_empty.any():
            active_column_positions.append(column_index)

    raw = raw.iloc[:, active_column_positions].copy()

    # Discard leading preamble rows until at least half of the remaining
    # columns have a value. This leaves the first qualifying row as the initial
    # header candidate while retaining the old per-column fallback below.
    if active_column_positions:
        row_value_counts = pd.concat(
            [non_empty_by_column[index] for index in active_column_positions], axis=1
        ).sum(axis=1)
        first_header_candidate = next(
            (
                row_index
                for row_index, value_count in enumerate(row_value_counts.tolist())
                if value_count * 2 >= len(active_column_positions)
            ),
            len(raw),
        )
        raw = raw.iloc[first_header_candidate:].copy()
    raw = raw.reset_index(drop=True)

    expected = {_normalized_mapping_key(name): name for name in documented_columns}
    expected.pop("", None)

    matched_headers: dict[int, str] = {}
    matched_header_rows: list[int] = []
    for column_position in range(raw.shape[1]):
        candidate_count = 0
        for index, value in raw.iloc[:, column_position].items():
            if pd.isna(value) or not str(value).strip():
                continue
            candidate_count += 1
            normalized = _normalized_mapping_key(str(value).strip())
            if normalized in expected:
                matched_headers[column_position] = expected[normalized]
                matched_header_rows.append(int(index))
                break
            if candidate_count == 2:
                break

    # Use the last documented match as the header row when the fallback finds
    # one. Otherwise, the first row remaining after sparse-preamble trimming is
    # the header. Keep its source labels even when this README has no column
    # documentation for the file.
    header_row = max(matched_header_rows, default=0)
    headers: list[str] = []
    for column_position in range(raw.shape[1]):
        header_name = matched_headers.get(column_position)
        if header_name is None and header_row < len(raw):
            value = raw.iat[header_row, column_position]
            if not pd.isna(value) and str(value).strip():
                header_name = str(value).strip()
        source_column_index = active_column_positions[column_position]
        headers.append(header_name or f"Unnamed: {source_column_index}")

    frame = raw.iloc[header_row + 1 :].copy()
    frame.columns = headers
    return frame.reset_index(drop=True)


def _read_sqlite(path: Path) -> list[tuple[str, pd.DataFrame]]:
    result: list[tuple[str, pd.DataFrame]] = []
    with sqlite3.connect(path) as connection:
        names = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name",
            connection,
        )["name"].tolist()
        for name in names:
            result.append(
                (
                    str(name),
                    pd.read_sql_query(
                        f"SELECT * FROM {_quote_identifier(str(name))}", connection
                    ),
                )
            )
    return result


DATE_COLUMN_HINTS = (
    "date", "datetime", "time", "created", "updated", "birth",
    "ngay", "thoi gian", "gio", "han dung", "hsd", "ap dung",
)

STATUS_COLUMN_HINTS = ("status", "trang thai", "tinh trang", "state")
MEASURE_COLUMN_HINTS = (
    "amount", "revenue", "sales", "total", "tong tien", "thanh tien",
    "doanh thu", "gia ban", "gia von", "don gia", "value", "so tien",
)
IDENTIFIER_COLUMN_HINTS = (
    "id", "code", "ma ", "ma_", "ma hoa don", "ma giao dich", "order",
)


def _fold_text(value: str) -> str:
    for _ in range(2):
        try:
            repaired = value.encode("latin1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            break
        if repaired == value:
            break
        value = repaired
    normalized = unicodedata.normalize("NFD", value.lower().replace("đ", "d"))
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def repair_mojibake(value: str) -> str:
    """Return readable text when a UTF-8 export was decoded as Latin-1."""
    for _ in range(2):
        try:
            repaired = value.encode("latin1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            break
        if repaired == value:
            break
        value = repaired
    return value


def _normalized_mapping_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _fold_text(str(value)))


def _is_date_column(name: str) -> bool:
    folded = _fold_text(name).replace("_", " ")
    return any(hint in folded for hint in DATE_COLUMN_HINTS)


def _infer_column_role(name: str) -> str | None:
    folded = _fold_text(name).replace("_", " ")
    if any(hint in folded for hint in STATUS_COLUMN_HINTS):
        return "status"
    if _is_date_column(name):
        return "transaction_date"
    if any(hint in folded for hint in MEASURE_COLUMN_HINTS):
        return "measure"
    if any(hint in folded for hint in IDENTIFIER_COLUMN_HINTS):
        return "identifier"
    return None


def _read_excel_sheet(path: Path, sheet: str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet, header=0)


def _deduplicate_export_rows(frame: pd.DataFrame, logical_name: str) -> pd.DataFrame:
    """Remove rows only when every cell in the row is exactly equal."""
    del logical_name
    return frame.drop_duplicates(keep="first").reset_index(drop=True)


def _parse_date_series(series: pd.Series) -> pd.Series | None:
    values = series.astype("string").str.strip()
    normalized = values.str.replace(
        r"(?i)(\d{1,2})\s+thg\s+(\d{1,2})\s+(\d{4})", r"\1/\2/\3", regex=True
    )
    normalized = normalized.str.replace(r"\s+", " ", regex=True).str.strip()

    numeric = pd.to_numeric(normalized, errors="coerce")
    excel_dates = numeric.between(20_000, 60_000).fillna(False)
    # Values read from Excel as datetimes become ISO strings above. Parse
    # year-first values before the locale-aware day-first fallback so dates
    # such as 2025-03-04 do not silently become 2025-04-03.
    iso_dates = normalized.str.match(
        r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:$|[ T])", na=False
    ) & ~excel_dates
    # Keep the intermediate container object-backed.  With newer pandas
    # versions, Excel/date parsing can produce higher-resolution timestamps
    # (for example datetime64[us]); assigning those into a datetime64[ns]
    # series raises instead of safely downcasting.
    parsed = pd.Series(pd.NaT, index=series.index, dtype="object")
    if excel_dates.any():
        parsed.loc[excel_dates] = pd.to_datetime(
            numeric.loc[excel_dates], unit="D", origin="1899-12-30", errors="coerce"
        )
    if iso_dates.any():
        try:
            parsed.loc[iso_dates] = pd.to_datetime(
                normalized.loc[iso_dates],
                yearfirst=True,
                dayfirst=False,
                format="mixed",
                errors="coerce",
            )
        except ValueError:
            parsed.loc[iso_dates] = pd.to_datetime(
                normalized.loc[iso_dates],
                yearfirst=True,
                dayfirst=False,
                errors="coerce",
            )
    remaining = ~(excel_dates | iso_dates)
    if remaining.any():
        try:
            parsed.loc[remaining] = pd.to_datetime(
                normalized.loc[remaining], dayfirst=True, format="mixed", errors="coerce"
            )
        except ValueError:
            parsed.loc[remaining] = pd.to_datetime(
                normalized.loc[remaining], dayfirst=True, errors="coerce"
            )
    non_null = values.notna() & values.ne("")
    if not non_null.any():
        return None
    parsed_non_null = parsed.loc[non_null].notna()
    # A date column can contain repeated report headers or a few malformed
    # cells. Keep the usable dates as timestamps and turn only those invalid
    # cells into NULL. If most of the column is not date-like, leave it as text
    # so a mislabeled free-text column is not silently erased.
    if not parsed_non_null.any() or parsed_non_null.mean() < 0.5:
        return None

    def as_datetime(value: Any) -> Any:
        if pd.isna(value):
            return None
        if hasattr(value, "to_pydatetime"):
            return value.to_pydatetime()
        return value

    datetimes = [as_datetime(value) for value in parsed]
    has_timezone = any(
        value is not None
        and value.tzinfo is not None
        and value.utcoffset() is not None
        for value in datetimes
    )
    if has_timezone:
        # Use UTC as the shared representation when a source mixes offsets or
        # combines offset-aware values with naive values. Naive values in such
        # a mixed column are interpreted as UTC because the source gives no
        # offset to apply.
        datetimes = [
            value.replace(tzinfo=timezone.utc)
            if value is not None and (value.tzinfo is None or value.utcoffset() is None)
            else value.astimezone(timezone.utc)
            if value is not None
            else None
            for value in datetimes
        ]

    return pd.Series(
        datetimes,
        index=series.index,
        dtype="object",
    )


def _infer_column_type(
    series: pd.Series, column_name: str, declared_type: str | None = None
) -> pd.Series:
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return series

    values = series.dropna().astype(str).str.strip()
    if values.empty:
        return series.astype("string")

    normalized_type = (declared_type or "").lower()
    if normalized_type in {"date", "datetime"} or (
        not normalized_type and _is_date_column(column_name)
    ):
        parsed_dates = _parse_date_series(series)
        if parsed_dates is not None:
            return parsed_dates

    numeric = pd.to_numeric(values.map(parse_measure_number), errors="coerce")
    all_numeric = numeric.notna().all()
    has_number_marker = values.map(has_number_format_marker).any()
    is_declared_numeric = normalized_type in {
        "decimal", "float", "number", "numeric", "integer", "int"
    }
    if all_numeric and (is_declared_numeric or has_number_marker):
        converted = pd.Series(pd.NA, index=series.index, dtype="Float64")
        converted.loc[values.index] = numeric
        return converted

    return series.map(lambda value: None if pd.isna(value) else str(value)).astype("string")


@dataclass
class TableMetadata:
    table: str
    source: str
    sheet: str | None
    rows: int
    columns: list[str]
    raw_columns: list[str] = field(default_factory=list)
    column_mappings: list[dict[str, Any]] = field(default_factory=list)
    canonical_name: str | None = None
    table_aliases: list[str] = field(default_factory=list)
    grain: str | None = None


@dataclass
class BusinessDatabase:
    connection: duckdb.DuckDBPyConnection
    tables: dict[str, TableMetadata] = field(default_factory=dict)
    errors: list[dict[str, str]] = field(default_factory=list)
    distinct_values: dict[tuple[str, str], list[Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.connection.create_function(
            NUMBER_SQL_FUNCTION,
            parse_measure_number,
            [duckdb.sqltypes.VARCHAR],
            duckdb.sqltypes.DOUBLE,
            null_handling="special",
        )

    def close(self) -> None:
        self.connection.close()


def _load_canonical_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Canonical mapping must be an object: {path}")
    mapping = value.get("tables", value)
    if not isinstance(mapping, dict):
        raise ValueError(f"Canonical mapping tables must be an object: {path}")
    return {"tables": mapping}


def _aliases(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise ValueError("Mapping aliases must be a string or list of strings")


def _table_mapping(
    mappings: dict[str, Any], logical_name: str
) -> tuple[str, dict[str, Any]] | None:
    normalized_name = _normalized_mapping_key(logical_name)
    for key, value in mappings.items():
        if not isinstance(value, dict):
            raise ValueError(f"Table mapping for {key!r} must be an object")
        aliases = _aliases(value.get("aliases"))
        if _normalized_mapping_key(str(key)) == normalized_name or any(
            _normalized_mapping_key(alias) == normalized_name for alias in aliases
        ):
            canonical_name = str(value.get("canonical_name", key))
            return canonical_name, value
    return None


def _apply_column_mapping(
    frame: pd.DataFrame,
    logical_name: str,
    mapping: tuple[str, dict[str, Any]] | None,
) -> tuple[pd.DataFrame, str, list[dict[str, Any]]]:
    if mapping is None:
        raw_columns = [str(column) for column in frame.columns]
        return frame, _safe_name(logical_name), [
            {
                "raw_name": column,
                "canonical_name": column,
                "role": _infer_column_role(column),
                "match_method": "unmapped",
                "confidence": 0.0,
            }
            for column in raw_columns
        ]

    canonical_table, table_config = mapping
    columns_config = table_config.get("columns", {})
    if not isinstance(columns_config, dict):
        raise ValueError(f"Column mappings for {logical_name!r} must be an object")
    normalized_headers: dict[str, str] = {}
    for column in frame.columns:
        raw_name = str(column).strip()
        key = _normalized_mapping_key(raw_name)
        if key and key in normalized_headers:
            raise ValueError(f"Duplicate normalized header {raw_name!r}")
        normalized_headers[key] = raw_name

    rename: dict[str, str] = {}
    mappings_output: list[dict[str, Any]] = []
    used_canonical: set[str] = set()
    for canonical_name, config in columns_config.items():
        if not isinstance(config, dict):
            raise ValueError(f"Column mapping for {canonical_name!r} must be an object")
        aliases = [str(canonical_name), *_aliases(config.get("aliases"))]
        matches = [
            header for header in normalized_headers.values()
            if any(_normalized_mapping_key(alias) == _normalized_mapping_key(header)
                   for alias in aliases)
        ]
        if len(matches) > 1:
            raise ValueError(
                f"Column mapping {canonical_name!r} matches multiple headers: {matches}"
            )
        if not matches:
            continue
        raw_name = matches[0]
        if canonical_name in used_canonical:
            raise ValueError(f"Canonical column collision: {canonical_name!r}")
        used_canonical.add(str(canonical_name))
        rename[raw_name] = str(canonical_name)
        mapping_item = {
            "raw_name": raw_name,
            "canonical_name": str(canonical_name),
            "type": config.get("type"),
            "role": config.get("role"),
            "match_method": "canonical_mapping",
            "confidence": 1.0,
        }
        # Preserve semantic mapping data through ingestion.  Profiling and
        # planning consume this lineage; dropping it here makes aliases and
        # definitions impossible to recover later.
        for semantic_key in ("aliases", "definition"):
            if config.get(semantic_key) is not None:
                mapping_item[semantic_key] = _aliases(config[semantic_key]) if semantic_key == "aliases" else config[semantic_key]
        mappings_output.append(mapping_item)

    ordered_mappings: list[dict[str, Any]] = []
    for raw_name in frame.columns:
        raw_name = str(raw_name)
        if raw_name in rename:
            ordered_mappings.append(
                next(
                    item for item in mappings_output if item["raw_name"] == raw_name
                )
            )
        else:
            ordered_mappings.append(
                {
                    "raw_name": raw_name,
                    "canonical_name": raw_name,
                    "role": _infer_column_role(raw_name),
                    "match_method": "unmapped",
                    "confidence": 0.0,
                }
            )
    renamed_columns = [rename.get(str(column), str(column)) for column in frame.columns]
    if len(renamed_columns) != len(set(renamed_columns)):
        raise ValueError(f"Canonical column collision in table {logical_name!r}")
    return frame.rename(columns=rename), _safe_name(canonical_table), ordered_mappings


def load_business_data(
    data_dir: str | Path,
    db_path: str = ":memory:",
    canonical_mapping_path: str | Path | None = None,
) -> BusinessDatabase:
    """Load every supported source in ``data_dir`` into uniquely named DuckDB tables."""
    root = Path(data_dir)
    if not root.is_dir():
        raise NotADirectoryError(root)

    connection = duckdb.connect(db_path)
    # Keep comparisons against timezone-aware source timestamps deterministic.
    connection.execute("SET TimeZone = 'UTC'")
    database = BusinessDatabase(connection)
    used_names: set[str] = set()
    mapping_path = (
        Path(canonical_mapping_path)
        if canonical_mapping_path is not None
        else root / "canonical_mapping.yml"
    )
    mapping = _load_canonical_mapping(mapping_path)
    documented_columns = _read_business_columns(root)

    def register(frame: pd.DataFrame, source: Path, logical_name: str, sheet: str | None) -> None:
        frame = frame.copy()
        if len(frame.columns) == 0:
            database.errors.append(
                {"source": str(source), "error": "No non-empty columns found"}
            )
            return
        frame.columns = [
            repair_mojibake(str(column).strip()) if str(column).strip() else f"column_{index}"
            for index, column in enumerate(frame.columns)
        ]
        for column in frame.select_dtypes(include=["object", "string"]).columns:
            frame[column] = frame[column].map(
                lambda value: (
                    repair_mojibake(value).strip().lower()
                    if isinstance(value, str)
                    else value
                )
            )
        frame = frame.loc[:, ~frame.columns.duplicated(keep="first")]
        table_mapping = _table_mapping(mapping.get("tables", {}), logical_name)
        frame, mapped_name, column_mappings = _apply_column_mapping(
            frame, logical_name, table_mapping
        )
        base = mapped_name
        name = base
        suffix = 2
        while name in used_names:
            name = f"{base}_{suffix}"
            suffix += 1
        used_names.add(name)
        frame = frame.loc[:, ~frame.columns.duplicated(keep="first")]
        frame = _deduplicate_export_rows(frame, logical_name)
        declared_types = {
            item["canonical_name"]: item.get("type")
            for item in column_mappings
            if item.get("match_method") == "canonical_mapping"
        }
        for column in frame.columns:
            frame[column] = _infer_column_type(
                frame[column], str(column), declared_types.get(str(column))
            )
        database.connection.register(f"_frame_{len(database.tables)}", frame)
        database.connection.execute(
            f"CREATE TABLE {_quote_identifier(name)} AS "
            f"SELECT * FROM _frame_{len(database.tables)}"
        )
        database.connection.unregister(f"_frame_{len(database.tables)}")
        table_config = table_mapping[1] if table_mapping is not None else {}
        database.tables[name] = TableMetadata(
            table=name,
            source=str(source),
            sheet=sheet,
            rows=len(frame),
            columns=[str(column) for column in frame.columns],
            raw_columns=[item["raw_name"] for item in column_mappings],
            column_mappings=column_mappings,
            canonical_name=(
                str(table_mapping[0]) if table_mapping is not None else name
            ),
            table_aliases=_aliases(table_config.get("aliases")),
            grain=(
                str(table_config["grain"])
                if table_config.get("grain") is not None
                else None
            ),
        )

    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            suffix = path.suffix.lower()
            if suffix == ".csv":
                expected = documented_columns.get(_normalized_mapping_key(path.name), [])
                register(_read_tabular(path, expected), path, path.stem, None)
            elif suffix in {".xlsx", ".xls"}:
                sheets = pd.ExcelFile(path).sheet_names
                expected = documented_columns.get(_normalized_mapping_key(path.name), [])
                for sheet in sheets:
                    register(
                        _read_tabular(path, expected, sheet),
                        path,
                        f"{path.stem}_{sheet}",
                        str(sheet),
                    )
            else:
                for table, frame in _read_sqlite(path):
                    register(frame, path, table, None)
        except (OSError, ValueError, UnicodeError, pd.errors.ParserError) as error:
            database.errors.append({"source": str(path), "error": str(error)})
    return database
