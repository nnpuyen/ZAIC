from __future__ import annotations

from pathlib import Path
from typing import Any

from .ingestion.loader import BusinessDatabase
from .ingestion.profiler import profile_database
from .planning.retrieval import split_rulebook_sections


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    for encoding in ("utf-8", "utf-8-sig", "cp1258"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def _resolve_document(root: Path, path: str | Path | None, filename: str) -> Path | None:
    if path is not None:
        return Path(path)
    for candidate_root in (root.parent, root.parent.parent):
        candidate = candidate_root / filename
        if candidate.is_file():
            return candidate
    return None


def build_context(
    database: BusinessDatabase,
    business_dir: str | Path,
    rulebook_path: str | Path | None = None,
    calendar_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build bounded model context without loading raw rows into a prompt."""
    root = Path(business_dir)
    readmes = [_read_text(path) for path in sorted(root.glob("README*"))]
    resolved_rulebook = _resolve_document(root, rulebook_path, "rulebook.md")
    resolved_calendar = _resolve_document(root, calendar_path, "calendar.csv")
    rulebook = _read_text(resolved_rulebook) if resolved_rulebook else ""
    return {
        "schema": profile_database(database),
        "business_description": "\n\n".join(text for text in readmes if text),
        "rulebook_sections": split_rulebook_sections(rulebook),
        "calendar": _read_text(resolved_calendar) if resolved_calendar else "",
    }
