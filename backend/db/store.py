"""Simple JSON-backed vault index.

This is NOT a vector store — it is a lightweight on-disk lookup table that
tracks every file in the vault, which workspace it belongs to, its mtime
(for change detection), and its display title.  The vector store handles
retrieval; this handles listing and bookkeeping.
"""
import json
import logging
import os
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_INDEX_PATH = _PROJECT_ROOT / "db" / "vault_index.json"

logger = logging.getLogger(__name__)

_index: dict[str, dict] = {}


def _ensure_dir() -> None:
    _INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_vault_index() -> dict[str, dict]:
    """Load the index from disk into the module-level ``_index`` dict."""
    global _index
    _ensure_dir()
    if _INDEX_PATH.exists():
        raw = _INDEX_PATH.read_text(encoding="utf-8")
        _index = json.loads(raw) if raw.strip() else {}
    else:
        _index = {}
    return _index


def save_vault_index(index: dict | None = None) -> None:
    """Persist the index to disk atomically.

    Writes to a temporary file in the same directory, then uses
    ``os.replace`` to swap it into place.  ``os.replace`` is atomic on both
    POSIX and Windows, so a crash or concurrent process can never leave
    ``vault_index.json`` truncated or half-written."""
    global _index
    if index is not None:
        _index = index
    _ensure_dir()
    payload = json.dumps(_index, indent=2, default=str)
    _INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(_INDEX_PATH.parent), suffix=".tmp", prefix=_INDEX_PATH.stem + "."
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, _INDEX_PATH)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def update_note(path: str, metadata: dict) -> None:
    """Upsert a note's metadata into the index and persist."""
    _index[path] = metadata
    save_vault_index()


def remove_note(path: str) -> None:
    """Delete a note from the index and persist."""
    _index.pop(path, None)
    save_vault_index()


def get_notes(workspace: str | None = None) -> list[dict]:
    """Return notes as ``[{"path": ..., ...metadata}]``, optionally filtered
    to a single workspace."""
    results = []
    for path, meta in _index.items():
        if workspace is not None and meta.get("workspace") != workspace:
            continue
        results.append({"path": path, **meta})
    return results


def rebuild_vault_index_from_disk(vault_path: str) -> int:
    """Walk ``vault_path`` and (re-)populate the index from the filesystem.

    Notes that no longer exist on disk are pruned. Returns the total note count.
    A single disk write batches the walk (per-note writes would serialize the
    JSON once per file).
    """
    from rag.ingester.vault import parse_frontmatter

    root = Path(vault_path).resolve()
    seen: set[str] = set()
    exts = {".md", ".markdown"}

    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in exts:
            continue
        rel = file_path.resolve().relative_to(root).as_posix()
        stat = file_path.stat()
        try:
            raw = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            logger.warning("skipping non-UTF-8 file: %s", rel)
            continue
        meta, _ = parse_frontmatter(raw)
        _index[rel] = {
            "workspace": str(meta.get("workspace") or "default"),
            "title": meta.get("title") or file_path.stem,
            "mtime": stat.st_mtime,
            "size": stat.st_size,
        }
        seen.add(rel)

    # Prune notes that no longer exist on disk (mutate in place, no per-file write).
    for path in list(_index):
        if path not in seen:
            _index.pop(path, None)

    save_vault_index()
    return len(seen)
