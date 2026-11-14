from __future__ import annotations

import argparse
import json
from pathlib import Path

from datamate.context import build_context
from datamate.ingestion import load_business_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest and profile one DataMate business directory")
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("schema.json"))
    parser.add_argument("--db", type=Path, default=Path(":memory:"))
    parser.add_argument("--rulebook", type=Path)
    parser.add_argument("--calendar", type=Path)
    args = parser.parse_args()
    database = load_business_data(args.data_dir, str(args.db))
    try:
        context = build_context(database, args.data_dir, args.rulebook, args.calendar)
        args.output.write_text(json.dumps(context, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    finally:
        database.close()


if __name__ == "__main__":
    main()
