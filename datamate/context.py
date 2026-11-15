from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .ingestion.loader import BusinessDatabase
from .ingestion.profiler import profile_database
from .planning.retrieval import split_rulebook_sections


_SCHEMA_CACHE_NAME = ".datamate_schema_cache.json"
_SCHEMA_CACHE_VERSION = 4


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    for encoding in ("utf-8", "utf-8-sig", "cp1258"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def _context_fingerprint(root: Path) -> str | None:
    """Fingerprint files that affect the profile or business description."""
    try:
        files = []
        for path in sorted(
            root.iterdir(), key=lambda item: (item.name.casefold(), item.name)
        ):
            if not path.is_file() or path.name == _SCHEMA_CACHE_NAME or path.name.startswith(
                f"{_SCHEMA_CACHE_NAME}."
            ):
                continue
            metadata = path.stat()
            files.append((path.name, metadata.st_size, metadata.st_mtime_ns))
    except OSError:
        return None
    payload = json.dumps(files, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_context_cache(
    path: Path, fingerprint: str | None
) -> dict[str, Any] | None:
    if fingerprint is None:
        return None
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not isinstance(cached, dict)
        or cached.get("version") != _SCHEMA_CACHE_VERSION
        or cached.get("fingerprint") != fingerprint
    ):
        return None
    schema = cached.get("schema")
    if (
        not isinstance(schema, dict)
        or not isinstance(schema.get("tables"), list)
        or not isinstance(schema.get("errors"), list)
        or not isinstance(cached.get("business_description"), str)
    ):
        return None
    return {
        "schema": schema,
        "business_description": cached["business_description"],
    }


def _write_context_cache(
    path: Path,
    fingerprint: str | None,
    schema: dict[str, Any],
    business_description: str,
) -> None:
    if fingerprint is None:
        return
    cache = {
        "version": _SCHEMA_CACHE_VERSION,
        "fingerprint": fingerprint,
        "schema": schema,
        "business_description": business_description,
    }
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f"{_SCHEMA_CACHE_NAME}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = temporary.name
            json.dump(cache, temporary, ensure_ascii=False, indent=2, default=str)
            temporary.write("\n")
        os.replace(temporary_path, path)
    except OSError:
        # Profiling remains usable when the input directory is read-only.
        pass
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            try:
                os.unlink(temporary_path)
            except OSError:
                pass


def _build_business_description(root: Path) -> str:
    readmes = [_read_text(path) for path in sorted(root.glob("README*"))]
    return "\n\n".join(text for text in readmes if text)


def _profile_or_load_cached_context(
    database: BusinessDatabase, root: Path
) -> tuple[dict[str, Any], str]:
    cache_path = root / _SCHEMA_CACHE_NAME
    fingerprint = _context_fingerprint(root)
    cached = _read_context_cache(cache_path, fingerprint)
    if cached is not None:
        return cached["schema"], cached["business_description"]
    schema = profile_database(database)
    business_description = _build_business_description(root)
    _write_context_cache(cache_path, fingerprint, schema, business_description)
    return schema, business_description


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
    schema, business_description = _profile_or_load_cached_context(database, root)
    resolved_rulebook = _resolve_document(root, rulebook_path, "rulebook.md")
    resolved_calendar = _resolve_document(root, calendar_path, "calendar.csv")
    rulebook = _read_text(resolved_rulebook) if resolved_rulebook else ""
    return {
        "schema": schema,
        "business_description": business_description,
        "rulebook_sections": split_rulebook_sections(rulebook),
        "calendar": _read_text(resolved_calendar) if resolved_calendar else "",
    }
