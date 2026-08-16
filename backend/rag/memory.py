"""Workspace-scoped chat memory backed by SQLite (stdlib, zero new deps).

Each workspace gets its own conversation thread, identified by ``session_id``.
A sliding window keeps only the most recent messages in memory; older messages
are discarded but the session table retains them for future audit if needed.
"""
import sqlite3
from pathlib import Path
from typing import Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from config import settings


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
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

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

    def load_messages(self, limit: Optional[int] = None) -> list[BaseMessage]:
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


def _row_to_message(role: str, content: str) -> BaseMessage:
    if role == "human":
        return HumanMessage(content=content)
    if role == "ai":
        return AIMessage(content=content)
    # Fallback — treat anything else as system
    from langchain_core.messages import SystemMessage

    return SystemMessage(content=content)
