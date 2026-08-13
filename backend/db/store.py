"""Simple JSON-backed vault index.

This is NOT a vector store — it is a lightweight on-disk lookup table that
tracks every file in the vault, which workspace it belongs to, its mtime
(for change detection), and its display title.  The vector store handles
retrieval; this handles listing and bookkeeping.
"""
import json
from pathlib import Path
from typing import Optional

_INDEX_PATH = Path("./db/vault_index.json")

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


def save_vault_index(index: Optional[dict] = None) -> None:
    """Persist the index to disk. Uses the module ``_index`` dict when no
    argument is provided."""
    global _index
    if index is not None:
        _index = index
    _ensure_dir()
    _INDEX_PATH.write_text(json.dumps(_index, indent=2, default=str), encoding="utf-8")


def update_note(path: str, metadata: dict) -> None:
    """Upsert a note's metadata into the index and persist."""
    _index[path] = metadata
    save_vault_index()


def remove_note(path: str) -> None:
    """Delete a note from the index and persist."""
    _index.pop(path, None)
    save_vault_index()


def get_notes(workspace: Optional[str] = None) -> list[dict]:
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
    """
    from config import settings
    from rag.ingester.vault import parse_frontmatter

    root = Path(vault_path).resolve()
    seen: set[str] = set()
    exts = {".md", ".markdown"}

    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in exts:
            continue
        rel = file_path.resolve().relative_to(root).as_posix()
        stat = file_path.stat()
        raw = file_path.read_text(encoding="utf-8")
        meta, _ = parse_frontmatter(raw)
        update_note(
            rel,
            {
                "workspace": str(meta.get("workspace") or "default"),
                "title": meta.get("title") or file_path.stem,
                "mtime": stat.st_mtime,
                "size": stat.st_size,
            },
        )
        seen.add(rel)

    # Prune notes that no longer exist on disk
    for path in list(_index):
        if path not in seen:
            remove_note(path)

    return len(seen)
