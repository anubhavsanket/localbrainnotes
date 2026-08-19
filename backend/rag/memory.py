"""Workspace-scoped chat memory backed by SQLite (stdlib, zero new deps).

Each workspace gets its own conversation thread, identified by ``session_id``.
A sliding window keeps only the most recent messages in memory; older messages
are discarded but the session table retains them for future audit if needed.
"""
import logging
import sqlite3
import threading
import time
from pathlib import Path

from config import settings
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thread-local connection pool — one connection per DB path, reused across
# calls instead of creating a fresh ``sqlite3.connect()`` every time.
# ---------------------------------------------------------------------------
_local = threading.local()


def _get_connection(db_path: str) -> sqlite3.Connection:
    """Return a thread-local SQLite connection for *db_path*, creating one on
    first use.  Connections are never closed (the process owns them) and each
    thread gets its own so there are no cross-thread locking issues."""
    conn: sqlite3.Connection | None = getattr(_local, "connections", {}).get(db_path)
    if conn is None:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        if not hasattr(_local, "connections"):
            _local.connections = {}
        _local.connections[db_path] = conn
    return conn


def _default_db_path() -> Path:
    """Resolve the SQLite DB path lazily so tests can patch
    ``settings.CHROMA_PERSIST_DIR`` after import."""
    return Path(settings.CHROMA_PERSIST_DIR).parent / "localbrain.db"


class WorkspaceChatMemoryManager:
    """SQLite-backed, workspace-scoped conversation memory.

    The ``session_id`` is typically the workspace name — one conversation per
    workspace, which matches the PRD's "hidden label = workspace" design.
    """

    def __init__(
        self,
        workspace: str = "default",
        window_size: int = settings.MEMORY_WINDOW_SIZE,
        db_path: str | Path | None = None,
    ):
        if db_path is None:
            db_path = _default_db_path()
        self.session_id = workspace
        self.window_size = window_size
        self._db_path = str(db_path)
        self._ensure_table()

    # ------------------------------------------------------------------ I/O

    def _conn(self) -> sqlite3.Connection:
        return _get_connection(self._db_path)

    def _ensure_table(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_history (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT   NOT NULL,
                    role       TEXT   NOT NULL,
                    content    TEXT   NOT NULL,
                    timestamp  REAL   NOT NULL
                )
                """
            )

    # ------------------------------------------------------------------ write

    def save_message(self, role: str, content: str) -> None:
        """Append one message (``'human'`` | ``'ai'``) to the current session."""
        import time

        with self._conn() as conn:
            conn.execute(
                "INSERT INTO chat_history (session_id, role, content, timestamp) "
                "VALUES (?, ?, ?, ?)",
                (self.session_id, role, content, time.time()),
            )

    # ------------------------------------------------------------------ read

    def load_messages(self, limit: int | None = None) -> list[BaseMessage]:
        """Return the most recent messages as LangChain ``BaseMessage`` objects,
        trimmed to the window size. The result is directly usable as the value
        of a ``MessagesPlaceholder`` in a ``ChatPromptTemplate``."""
        limit = limit or self.window_size
        with self._conn() as conn:
            rows = (
                conn.execute(
                    "SELECT role, content FROM chat_history "
                    "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                    (self.session_id, limit),
                )
                .fetchall()[::-1]  # reverse to chronological order
            )
        return [_row_to_message(role, content) for role, content in rows]

    # ------------------------------------------------------------------ util

    def clear(self) -> None:
        """Drop all messages for the current workspace session."""
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM chat_history WHERE session_id = ?",
                (self.session_id,),
            )

    def list_workspaces(self) -> list[str]:
        """Return all workspace session ids that have stored messages."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT session_id FROM chat_history"
            ).fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------ export / import

    def export_history(self) -> list[dict]:
        """Return all messages as a list of dicts (JSON-serialisable)."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT role, content, timestamp FROM chat_history "
                "WHERE session_id = ? ORDER BY id ASC",
                (self.session_id,),
            ).fetchall()
        return [{"role": r, "content": c, "timestamp": t} for r, c, t in rows]

    def import_history(self, messages: list[dict]) -> int:
        """Bulk-import messages from a list of dicts (as produced by
        ``export_history``).  Returns the count imported."""
        imported = 0
        with self._conn() as conn:
            for msg in messages:
                role = msg.get("role", "human")
                content = msg.get("content", "")
                ts = msg.get("timestamp", time.time())
                if role and content:
                    conn.execute(
                        "INSERT INTO chat_history (session_id, role, content, timestamp) "
                        "VALUES (?, ?, ?, ?)",
                        (self.session_id, role, content, ts),
                    )
                    imported += 1
        logger.info("imported %d messages into workspace '%s'", imported, self.session_id)
        return imported

    def load_all_messages(self) -> list[dict]:
        """Return every message (no window limit) as raw dicts."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT role, content, timestamp FROM chat_history "
                "WHERE session_id = ? ORDER BY id ASC",
                (self.session_id,),
            ).fetchall()
        return [{"role": r, "content": c, "timestamp": t} for r, c, t in rows]

    def count_messages(self) -> int:
        """Return total message count for this workspace."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM chat_history WHERE session_id = ?",
                (self.session_id,),
            ).fetchone()
        return row[0] if row else 0


def _row_to_message(role: str, content: str) -> BaseMessage:
    if role == "human":
        return HumanMessage(content=content)
    if role == "ai":
        return AIMessage(content=content)
    # Fallback — treat anything else as system
    from langchain_core.messages import SystemMessage

    return SystemMessage(content=content)
