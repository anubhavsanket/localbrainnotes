"""Tests for the atomic persistence of the JSON vault index (db/store.py).

Regression guard: ``save_vault_index`` must never leave ``vault_index.json``
truncated or half-written, even when many writes race concurrently.  The
write-to-temp + ``os.replace`` pattern guarantees that any reader always sees
either the previous complete file or the new complete file.
"""
import json
import threading

import pytest


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    """Point the store module at a throwaway index file and reset its state."""
    import db.store as store_mod

    original_index = store_mod._index
    original_path = store_mod._INDEX_PATH
    store_mod._INDEX_PATH = tmp_path / "vault_index.json"
    store_mod._index = {}
    yield store_mod
    store_mod._INDEX_PATH = original_path
    store_mod._index = original_index


def test_save_then_load_roundtrip(isolated_store):
    isolated_store.update_note("a.md", {"workspace": "default", "title": "A"})
    reloaded = isolated_store.load_vault_index()
    assert reloaded["a.md"]["title"] == "A"


def test_concurrent_writes_never_corrupt_index(isolated_store):
    """Many threads hammering save_vault_index must always leave valid JSON.

    On Windows some ``os.replace`` calls may raise ``PermissionError`` because
    competing threads hold the file transiently; that's a race on the rename,
    not on data integrity.  The atomicity guarantee is that the file on disk
    is never truncated or half-written, so we just assert the on-disk file is
    valid JSON after the storm subsides.
    """
    errors = []

    def worker(i):
        try:
            isolated_store.update_note(
                f"note_{i}.md", {"workspace": "default", "title": f"Note {i}"}
            )
        except PermissionError:
            # Expected on Windows during concurrent file replaces.
            pass
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"unexpected errors during concurrent writes: {errors}"

    # The file on disk must be parseable JSON (no truncation / half-write).
    raw = isolated_store._INDEX_PATH.read_text(encoding="utf-8")
    data = json.loads(raw)  # raises if truncated/half-written
    assert isinstance(data, dict)
