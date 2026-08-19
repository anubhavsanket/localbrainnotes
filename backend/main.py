"""FastAPI application — the public surface of LocalBrain.

Routes cover vault ingestion, querying (sync + SSE streaming), workspace
management, conversation history, and a health probe.  A file-system watcher
is launched on startup to keep the vector index in sync with the vault on disk.
"""
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from config import settings
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)
from fastapi.staticfiles import StaticFiles
from models.schemas import (
    NoteWrite,
    QueryRequest,
    QueryResponse,
    WorkspaceResponse,
)
from starlette.requests import Request

# ---------------------------------------------------------------------------
# App + middleware
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the vault watcher on startup; stop it cleanly on shutdown."""
    from logging_config import setup_logging

    setup_logging()
    global _watcher
    try:
        from db.store import load_vault_index, rebuild_vault_index_from_disk
        from rag.watcher import start_watcher

        load_vault_index()  # load existing index
        rebuild_vault_index_from_disk(settings.VAULT_PATH)  # sync with disk
        _watcher = start_watcher(settings.VAULT_PATH)
    except Exception as exc:  # noqa: BLE001
        logger.warning("watcher failed to start: %s", exc)
    yield
    if _watcher is not None:
        from rag.watcher import stop_watcher

        stop_watcher(_watcher)
        _watcher = None


app = FastAPI(title="LocalBrain", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Watcher lifecycle
# ---------------------------------------------------------------------------

_watcher = None


# ---------------------------------------------------------------------------
# Vault management routes
# ---------------------------------------------------------------------------


@app.post("/api/vault/ingest")
def ingest_vault(
    workspace: str | None = Query(None, description="Scope ingestion to a workspace"),
):
    """Re-ingest the full vault directory (or a single workspace)."""
    from db.store import rebuild_vault_index_from_disk
    from rag.ingester.vault import ingest_vault as _ingest_vault

    count = _ingest_vault(settings.VAULT_PATH, workspace_filter=workspace)
    # Sync the workspace index so the UI knows about the new files
    rebuild_vault_index_from_disk(settings.VAULT_PATH)
    return {"status": "indexed", "chunks": count, "vault_path": settings.VAULT_PATH}


@app.post("/api/vault/ingest/file")
def ingest_single_file(path: str):
    """Ingest a single markdown file into the vector store.

    The file must live inside the vault (like every other note endpoint);
    arbitrary filesystem paths are rejected to prevent accidental or malicious
    ingestion of non-vault files.
    """
    from db.store import update_note
    from rag.ingester.vault import ingest_file

    # `path` is a filesystem path (repo-relative or absolute), not a
    # vault-relative note id. Validate it resides in the vault.
    file_path = _validate_path_within_vault(path)

    count = ingest_file(str(file_path), settings.VAULT_PATH)

    # Update vault index metadata for this file (best-effort, non-fatal).
    try:
        content = _read_utf8(file_path)
        rel, record = _note_record(file_path, content)
        update_note(rel, record)
    except Exception:  # noqa: BLE001, S110
        pass  # non-critical — vector ingest succeeded

    return {"status": "indexed", "chunks": count, "file": path}


@app.post("/api/vault/ingest/pdf")
def ingest_pdf(file_path: str, workspace: str = "default"):
    """Index a PDF file into the vector store.

    The file must live inside the vault; arbitrary filesystem paths are rejected
    to prevent accidental or malicious ingestion of non-vault files.
    """
    from rag.ingester.pdf import ingest_pdf as _ingest_pdf

    candidate = _validate_path_within_vault(file_path)

    count = _ingest_pdf(str(candidate), workspace)
    return {"status": "indexed", "chunks": count, "file": file_path}


@app.post("/api/vault/ingest/youtube")
def ingest_youtube(url: str, workspace: str = "default"):
    """Index a YouTube transcript into the vector store."""
    from rag.ingester.youtube import ingest_youtube as _ingest_youtube

    parsed = _validate_youtube_url(url)
    count = _ingest_youtube(parsed, workspace)
    return {"status": "indexed", "chunks": count, "url": url}


@app.get("/api/workspaces")
def list_workspaces():
    """List all known workspaces from the vault index."""
    from db.store import get_notes

    notes = get_notes()
    workspaces = sorted({n.get("workspace", "default") for n in notes})
    return WorkspaceResponse(workspaces=workspaces, count=len(workspaces))


@app.get("/api/notes")
def list_notes(workspace: str | None = Query(None)):
    """List notes in a workspace (or all notes)."""
    from db.store import get_notes

    notes = get_notes(workspace)
    return {"notes": notes, "count": len(notes)}


# ---------------------------------------------------------------------------
# Notes CRUD (file-level: read/write/delete markdown files in the vault)
# ---------------------------------------------------------------------------


def _resolve_vault_path(rel_path: str) -> Path:
    """Resolve a note's relative path inside the vault, refusing escapes.

    Unlike ``_validate_path_within_vault`` this accepts paths that do not yet
    exist on disk (needed for note creation) and treats ``rel_path`` as a
    vault-relative id rather than an absolute/fs path.
    """
    from config import settings

    root = Path(settings.VAULT_PATH).resolve()
    target = (root / rel_path).resolve()
    if not target.is_relative_to(root):
        raise HTTPException(status_code=400, detail=f"path escapes vault: {rel_path}")
    return target


def _read_utf8(file_path: Path) -> str:
    """Read a file as UTF-8, translating decode errors into a client-facing 400."""
    try:
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail=f"file is not valid UTF-8 text: {file_path.name}",
        ) from None


def _note_record(file_path: Path, content: str) -> tuple[str, dict]:
    """Build (relative_path, metadata) for a note purely from its content —
    no index writes, no second disk read. Mirrors the ingester's metadata."""
    from config import settings
    from rag.ingester.vault import parse_frontmatter

    root = Path(settings.VAULT_PATH).resolve()
    rel = file_path.resolve().relative_to(root).as_posix()
    meta, _ = parse_frontmatter(content)
    stat = file_path.stat()
    record = {
        "workspace": str(meta.get("workspace") or "default"),
        "title": meta.get("title") or file_path.stem,
        "mtime": stat.st_mtime,
        "size": stat.st_size,
    }
    return rel, record


def _validate_youtube_url(url: str) -> str:
    """Reject malformed or non-YouTube URLs before handing them to the ingester."""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in ("http", "https") or not host:
        raise HTTPException(status_code=400, detail=f"invalid URL: {url}")
    if not any(host == d or host.endswith(f".{d}") for d in ("youtube.com", "youtu.be")):
        raise HTTPException(status_code=400, detail=f"not a YouTube URL: {url}")
    return url


def _validate_path_within_vault(file_path: str) -> Path:
    """Validate that a filesystem path lives inside the configured vault.

    Returns the resolved Path object if valid; raises HTTPException(400) if the
    path attempts to escape the vault or does not point to an existing file.
    """
    from config import settings

    root = Path(settings.VAULT_PATH).resolve()
    candidate = Path(file_path).resolve()
    if not candidate.is_relative_to(root):
        raise HTTPException(status_code=400, detail=f"path escapes vault: {file_path}")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    return candidate


@app.get("/api/notes/{note_path:path}")
def read_note(note_path: str):
    """Read a note's markdown content and metadata from the vault."""
    file_path = _resolve_vault_path(note_path)
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"note not found: {note_path}")

    content = _read_utf8(file_path)
    rel, record = _note_record(file_path, content)
    return {"path": rel, **record, "content": content}


@app.post("/api/notes")
def create_note(note: NoteWrite):
    """Create a new note file in the vault (parent dirs auto-created)."""
    from db.store import update_note
    from rag.ingester.vault import ingest_file

    file_path = _resolve_vault_path(note.path)
    if file_path.exists():
        raise HTTPException(status_code=409, detail=f"note already exists: {note.path}")

    file_path.parent.mkdir(parents=True, exist_ok=True)
    body = note.content or f"# {file_path.stem}\n"
    file_path.write_text(body, encoding="utf-8")

    rel, record = _note_record(file_path, body)
    update_note(rel, record)

    # Immediately index, since the FS watcher applies only when running
    # (and even then a 500 ms debounce would briefly hide the note).
    try:
        ingest_file(str(file_path), str(settings.VAULT_PATH))
    except Exception as exc:  # noqa: BLE001
        # Keep the file + index entry (the note exists on disk; the watcher
        # will retry indexing), but surface the failure instead of hiding it.
        logger.warning("inline ingest failed for %s: %s", note.path, exc)
        return {"path": rel, **record, "content": body, "warning": f"index failed: {exc}"}

    return {"path": rel, **record, "content": body}


@app.put("/api/notes/{note_path:path}")
def update_note(note_path: str, note: NoteWrite):
    """Overwrite a note's content; re-indexes the vector store."""
    from rag.ingester.vault import ingest_file

    file_path = _resolve_vault_path(note_path)
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"note not found: {note_path}")

    file_path.write_text(note.content, encoding="utf-8")
    rel, record = _note_record(file_path, note.content)
    try:
        ingest_file(str(file_path), str(settings.VAULT_PATH))
    except Exception as exc:  # noqa: BLE001
        logger.warning("re-ingest failed for %s: %s", note_path, exc)
        return {"path": rel, **record, "content": note.content, "warning": f"index failed: {exc}"}

    return {"path": rel, **record, "content": note.content}


@app.delete("/api/notes/{note_path:path}")
def delete_note(note_path: str):
    """Delete a note file from the vault and drop its vector chunks.

    Ordering matters: the file is unlinked *first* so a failed unlink cannot
    leave a note that is listed but has already lost its chunks. Cleanup uses
    the normalized relative path so chunk ids always match."""
    from db.store import remove_note
    from rag.vectorstore import vectorstore

    file_path = _resolve_vault_path(note_path)
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"note not found: {note_path}")

    rel = file_path.resolve().relative_to(Path(settings.VAULT_PATH).resolve()).as_posix()
    file_path.unlink()

    try:
        vectorstore.delete_note_chunks(rel)
    except Exception as exc:  # noqa: BLE001
        logger.warning("chunk deletion failed for %s: %s", note_path, exc)
    remove_note(rel)
    return {"status": "deleted", "path": rel}


# ---------------------------------------------------------------------------
# Querying
# ---------------------------------------------------------------------------


_LLM_SETUP_ERRORS = (ConnectionError, ValueError)


def _build_agent_graph():
    """Build the agent graph, translating LLM setup failures (Ollama down,
    missing API key, unsupported provider) into a clear HTTP 503."""
    from rag.graph import get_agent_graph

    try:
        return get_agent_graph()
    except _LLM_SETUP_ERRORS as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/query", response_model=QueryResponse)
def query_notes(req: QueryRequest):
    """Run the full agent loop and return a structured answer."""
    from rag.memory import WorkspaceChatMemoryManager

    memory = WorkspaceChatMemoryManager(workspace=req.workspace)
    chat_history = memory.load_messages()

    graph = _build_agent_graph()
    input_state = {
        "question": req.question,
        "workspace": req.workspace,
        "chat_history": chat_history,
        "rewrite_count": 0,
    }

    start = time.time()
    result = graph.invoke(input_state, config={"configurable": {"thread_id": req.workspace}})
    elapsed = round(time.time() - start, 2)

    memory.save_message("human", req.question)
    memory.save_message("ai", result.get("answer", ""))

    return QueryResponse(
        answer=result.get("answer", ""),
        sources=result.get("sources", []),
        latency_seconds=elapsed,
    )


@app.post("/api/query/stream")
async def query_notes_stream(req: QueryRequest, request: Request):
    """SSE streaming variant — yields JSON lines as the agent runs.

    Token-level streaming comes from the ``generate`` node, which writes
    ``{"kind": "token"|"sources"}`` custom stream parts; they surface here via
    ``graph.astream(stream_mode=["custom", "values"])``.
    """
    from rag.memory import WorkspaceChatMemoryManager

    memory = WorkspaceChatMemoryManager(workspace=req.workspace)
    chat_history = memory.load_messages()

    graph = _build_agent_graph()
    input_state = {
        "question": req.question,
        "workspace": req.workspace,
        "chat_history": chat_history,
        "rewrite_count": 0,
    }

    async def event_generator():
        full_answer = ""
        async for mode, chunk in graph.astream(
            input_state,
            stream_mode=["custom", "values"],
            config={"configurable": {"thread_id": req.workspace}},
        ):
            if await request.is_disconnected():
                break
            if mode != "custom" or not isinstance(chunk, dict):
                continue
            kind = chunk.get("kind")
            if kind == "token":
                token = chunk.get("data", "")
                full_answer += token
                yield f"data: {json.dumps({'type': 'token', 'data': token})}\n\n"
            elif kind == "sources":
                yield f"data: {json.dumps({'type': 'source_documents', 'data': chunk.get('data', [])})}\n\n"

        memory.save_message("human", req.question)
        memory.save_message("ai", full_answer)
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Human-in-the-loop review (Phase F+)
# ---------------------------------------------------------------------------

import uuid

# In-memory store for pending reviews.  Keyed by query_id (UUID).
# Each entry holds the draft answer, sources, and the original state needed
# to resume or discard.  Entries expire after 30 minutes.
_pending_reviews: dict[str, dict] = {}
_REVIEW_TTL_SECS = 30 * 60


def _cleanup_stale_reviews() -> None:
    """Drop entries older than ``_REVIEW_TTL_SECS``."""
    now = time.time()
    stale = [k for k, v in _pending_reviews.items() if now - v["created_at"] > _REVIEW_TTL_SECS]
    for k in stale:
        _pending_reviews.pop(k, None)


@app.post("/api/query/preview")
def query_preview(req: QueryRequest):
    """Run the agent loop and return a draft answer for human review.

    The draft is NOT saved to conversation memory.  The caller must later
    call ``/api/query/{query_id}/approve`` to persist it, or
    ``/api/query/{query_id}/reject`` to regenerate with feedback.
    """
    from rag.memory import WorkspaceChatMemoryManager

    _cleanup_stale_reviews()

    memory = WorkspaceChatMemoryManager(workspace=req.workspace)
    chat_history = memory.load_messages()

    graph = _build_agent_graph()
    input_state = {
        "question": req.question,
        "workspace": req.workspace,
        "chat_history": chat_history,
        "rewrite_count": 0,
    }

    start = time.time()
    result = graph.invoke(input_state, config={"configurable": {"thread_id": f"preview-{uuid.uuid4().hex[:8]}"}})
    elapsed = round(time.time() - start, 2)

    query_id = uuid.uuid4().hex[:12]
    _pending_reviews[query_id] = {
        "question": req.question,
        "workspace": req.workspace,
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "latency_seconds": elapsed,
        "created_at": time.time(),
    }

    return {
        "query_id": query_id,
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "latency_seconds": elapsed,
    }


@app.post("/api/query/{query_id}/approve")
def approve_answer(query_id: str):
    """Approve a previewed answer — saves it to conversation memory and
    returns it as the final response."""
    review = _pending_reviews.pop(query_id, None)
    if review is None:
        raise HTTPException(status_code=404, detail="Review session expired or not found")

    from rag.memory import WorkspaceChatMemoryManager

    memory = WorkspaceChatMemoryManager(workspace=review["workspace"])
    memory.save_message("human", review["question"])
    memory.save_message("ai", review["answer"])

    return {
        "status": "approved",
        "answer": review["answer"],
        "sources": review["sources"],
        "latency_seconds": review["latency_seconds"],
    }


@app.post("/api/query/{query_id}/reject")
def reject_answer(query_id: str, feedback: str = ""):
    """Reject a previewed answer — re-runs the agent with the user's
    feedback appended to the question, then saves the new answer."""
    review = _pending_reviews.pop(query_id, None)
    if review is None:
        raise HTTPException(status_code=404, detail="Review session expired or not found")

    # Build a new question that includes the rejection feedback.
    if feedback.strip():
        new_question = (
            f"{review['question']}\n\n"
            f"[User feedback on previous answer: {feedback.strip()}]\n"
            f"Please regenerate your answer taking this feedback into account."
        )
    else:
        new_question = review["question"]

    from rag.memory import WorkspaceChatMemoryManager

    memory = WorkspaceChatMemoryManager(workspace=review["workspace"])
    chat_history = memory.load_messages()

    graph = _build_agent_graph()
    input_state = {
        "question": new_question,
        "workspace": review["workspace"],
        "chat_history": chat_history,
        "rewrite_count": 0,
    }

    start = time.time()
    result = graph.invoke(input_state, config={"configurable": {"thread_id": f"reject-{uuid.uuid4().hex[:8]}"}})
    elapsed = round(time.time() - start, 2)

    # Save both the original question and the new answer.
    memory.save_message("human", review["question"])
    memory.save_message("ai", result.get("answer", ""))

    return {
        "status": "regenerated",
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "latency_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


@app.get("/api/history")
def get_history(workspace: str = "default"):
    """Return recent conversation messages for a workspace."""
    from rag.memory import WorkspaceChatMemoryManager

    memory = WorkspaceChatMemoryManager(workspace=workspace)
    messages = memory.load_messages()
    return {
        "messages": [
            {"role": getattr(m, "type", "unknown"), "content": m.content}
            for m in messages
        ],
        "count": len(messages),
    }


@app.delete("/api/history")
def clear_history(workspace: str = "default"):
    """Clear conversation history for a workspace."""
    from rag.memory import WorkspaceChatMemoryManager

    memory = WorkspaceChatMemoryManager(workspace=workspace)
    memory.clear()
    return {"status": "cleared", "workspace": workspace}


@app.get("/api/history/export")
def export_history(workspace: str = "default"):
    """Export all conversation messages for a workspace as JSON."""
    from rag.memory import WorkspaceChatMemoryManager

    memory = WorkspaceChatMemoryManager(workspace=workspace)
    messages = memory.export_history()
    return {
        "workspace": workspace,
        "messages": messages,
        "count": len(messages),
    }


from pydantic import BaseModel


class HistoryImportRequest(BaseModel):
    messages: list[dict] = []


@app.post("/api/history/import")
def import_history(req: HistoryImportRequest, workspace: str = "default"):
    """Bulk-import conversation messages into a workspace."""
    from rag.memory import WorkspaceChatMemoryManager

    messages = req.messages
    if not messages:
        raise HTTPException(status_code=400, detail="No messages provided")
    memory = WorkspaceChatMemoryManager(workspace=workspace)
    imported = memory.import_history(messages)
    return {"status": "imported", "workspace": workspace, "count": imported}


@app.get("/api/history/summary")
def history_summary(workspace: str = "default"):
    """Return a summary of the conversation for the workspace."""
    from rag.memory import WorkspaceChatMemoryManager

    memory = WorkspaceChatMemoryManager(workspace=workspace)
    total = memory.count_messages()
    all_msgs = memory.load_all_messages()
    last_human = ""
    last_ai = ""
    for m in reversed(all_msgs):
        if m["role"] == "human" and not last_human:
            last_human = m["content"]
        elif m["role"] == "ai" and not last_ai:
            last_ai = m["content"]
        if last_human and last_ai:
            break
    return {
        "workspace": workspace,
        "total_messages": total,
        "last_question": last_human,
        "last_answer": last_ai,
    }


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health():
    """System health probe: vault path, note count, Ollama connectivity."""
    from db.store import get_notes
    from rag.llm_factory import _ollama_alive

    notes = get_notes()
    return {
        "status": "ok",
        "ollama_connected": _ollama_alive(settings.OLLAMA_BASE_URL),
        "vault_path": settings.VAULT_PATH,
        "notes_indexed": len(notes),
    }


# ---------------------------------------------------------------------------
# Static frontend mount (must come AFTER all API routes)
# ---------------------------------------------------------------------------

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if _FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")

if __name__ == "__main__":
    import os

    import uvicorn

    # The desktop shell (Tauri) disables auto-reload so killing the sidecar
    # process does not orphan a reloader subprocess on Windows.
    reload = os.environ.get("LOCALBRAIN_DISABLE_RELOAD", "").lower() not in {"1", "true", "yes"}
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=reload)
