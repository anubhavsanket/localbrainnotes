"""Watchdog-based incremental vault indexer.

Runs a background thread that watches the vault directory for file-system
events and updates the vector index on the fly.  A 500 ms debounce window
prevents double-index on rapid saves (e.g. autosave + manual save within
the same editor session).
"""
import threading
import time
from pathlib import Path
from typing import Optional

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from config import settings

_DEBOUNCE_MS = 500
_MARKDOWN_SUFFIXES = {".md", ".markdown"}


class _VaultHandler(FileSystemEventHandler):
    """Debounced handler that re-ingests changed markdown files."""

    def __init__(self, vault_path: str, workspace_filter: Optional[str] = None):
        super().__init__()
        self.vault_path = Path(vault_path).resolve()
        self.workspace_filter = workspace_filter
        self._lock = threading.Lock()
        self._pending: dict[str, float] = {}  # path_str → last-event timestamp

    def _is_markdown(self, path: str) -> bool:
        return Path(path).suffix.lower() in _MARKDOWN_SUFFIXES

    def _should_process(self, path: str) -> bool:
        with self._lock:
            now = time.monotonic()
            last = self._pending.get(path, 0.0)
            if now - last < _DEBOUNCE_MS / 1000.0:
                return False
            self._pending[path] = now
            return True

    def on_created(self, event: FileSystemEvent) -> None:
        self._handle(event, "created")

    def on_modified(self, event: FileSystemEvent) -> None:
        self._handle(event, "modified")

    def on_deleted(self, event: FileSystemEvent) -> None:
        self._handle(event, "deleted")

    def _handle(self, event: FileSystemEvent, kind: str) -> None:
        if event.is_directory:
            return
        path_str = str(Path(event.src_path).resolve())
        if not self._is_markdown(path_str):
            return
        if not self._should_process(path_str):
            return
        try:
            if kind == "deleted":
                self._delete_chunks(path_str)
            else:
                self._index_file(path_str)
        except Exception as exc:
            print(f"[watcher] {kind} {path_str}: {exc}")

    def _index_file(self, path_str: str) -> None:
        from rag.ingester.vault import ingest_file

        note_id = str(Path(path_str).resolve().relative_to(self.vault_path).as_posix())
        count = ingest_file(path_str, str(self.vault_path))
        print(f"[watcher] indexed {note_id}: {count} chunks")

    def _delete_chunks(self, path_str: str) -> None:
        from rag.vectorstore import vectorstore

        note_id = str(Path(path_str).resolve().relative_to(self.vault_path).as_posix())
        removed = vectorstore.delete_note_chunks(note_id)
        print(f"[watcher] deleted {note_id}: {removed} chunks removed")


def start_watcher(
    vault_path: Optional[str] = None,
    workspace_filter: Optional[str] = None,
) -> Observer:
    """Start a background thread watching ``vault_path`` for changes."""
    path = str(Path(vault_path or settings.VAULT_PATH).resolve())
    handler = _VaultHandler(path, workspace_filter=workspace_filter)
    observer = Observer()
    observer.schedule(handler, path, recursive=True)
    observer.daemon = True
    observer.start()
    print(f"[watcher] watching {path}")
    return observer


def stop_watcher(observer: Observer) -> None:
    observer.stop()
    observer.join(timeout=5)
