"""FastAPI application — the public surface of LocalBrain.

Routes cover vault ingestion, querying (sync + SSE streaming), workspace
management, conversation history, and a health probe.  A file-system watcher
is launched on startup to keep the vector index in sync with the vault on disk.
"""
import asyncio
import json
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request

from config import settings
from models.schemas import (
    NoteCreate,
    NoteUpdate,
    QueryRequest,
    QueryResponse,
    WorkspaceResponse,
)

# ---------------------------------------------------------------------------
# App + middleware
# ---------------------------------------------------------------------------

app = FastAPI(title="LocalBrain", version="0.1.0")

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


@app.on_event("startup")
def _startup_watcher():
    global _watcher
    try:
        from rag.watcher import start_watcher
        from db.store import rebuild_vault_index_from_disk, load_vault_index

        load_vault_index()  # load existing index
        rebuild_vault_index_from_disk(settings.VAULT_PATH)  # sync with disk
        _watcher = start_watcher(settings.VAULT_PATH)
    except Exception as exc:
        print(f"[startup] watcher failed to start: {exc}")


@app.on_event("shutdown")
def _shutdown_watcher():
    global _watcher
    if _watcher is not None:
        from rag.watcher import stop_watcher

        stop_watcher(_watcher)
        _watcher = None


# ---------------------------------------------------------------------------
# Vault management routes
# ---------------------------------------------------------------------------


@app.post("/api/vault/ingest")
def ingest_vault(
    workspace: Optional[str] = Query(None, description="Scope ingestion to a workspace"),
):
    """Re-ingest the full vault directory (or a single workspace)."""
    from rag.ingester.vault import ingest_vault as _ingest_vault
    from db.store import rebuild_vault_index_from_disk

    count = _ingest_vault(settings.VAULT_PATH, workspace_filter=workspace)
    # Sync the workspace index so the UI knows about the new files
    rebuild_vault_index_from_disk(settings.VAULT_PATH)
    return {"status": "indexed", "chunks": count, "vault_path": settings.VAULT_PATH}


@app.post("/api/vault/ingest/file")
def ingest_single_file(
    path: str, workspace: Optional[str] = Query(None, description="Override workspace label"),
):
    """Ingest a single markdown file into the vector store."""
    from rag.ingester.vault import ingest_file
    from db.store import update_note
    from rag.ingester.vault import parse_frontmatter

    file_path = Path(path).resolve()
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    count = ingest_file(str(file_path), settings.VAULT_PATH)

    # Update vault index metadata for this file
    try:
        rel = file_path.resolve().relative_to(Path(settings.VAULT_PATH).resolve()).as_posix()
        raw = file_path.read_text(encoding="utf-8")
        meta, _ = parse_frontmatter(raw)
        stat = file_path.stat()
        update_note(rel, {
            "workspace": str(meta.get("workspace") or "default"),
            "title": meta.get("title") or file_path.stem,
            "mtime": stat.st_mtime,
            "size": stat.st_size,
        })
    except Exception:
        pass  # non-critical — vector ingest succeeded

    return {"status": "indexed", "chunks": count, "file": path}


@app.post("/api/vault/ingest/pdf")
def ingest_pdf(file_path: str, workspace: str = "default"):
    """Index a PDF file into the vector store."""
    from rag.ingester.pdf import ingest_pdf as _ingest_pdf

    count = _ingest_pdf(file_path, workspace)
    return {"status": "indexed", "chunks": count, "file": file_path}


@app.post("/api/vault/ingest/youtube")
def ingest_youtube(url: str, workspace: str = "default"):
    """Index a YouTube transcript into the vector store."""
    from rag.ingester.youtube import ingest_youtube as _ingest_youtube

    count = _ingest_youtube(url, workspace)
    return {"status": "indexed", "chunks": count, "url": url}


@app.get("/api/workspaces")
def list_workspaces():
    """List all known workspaces from the vault index."""
    from db.store import get_notes

    notes = get_notes()
    workspaces = sorted({n.get("workspace", "default") for n in notes})
    return WorkspaceResponse(workspaces=workspaces, count=len(workspaces))


@app.get("/api/notes")
def list_notes(workspace: Optional[str] = Query(None)):
    """List notes in a workspace (or all notes)."""
    from db.store import get_notes

    notes = get_notes(workspace)
    return {"notes": notes, "count": len(notes)}


# ---------------------------------------------------------------------------
# Querying
# ---------------------------------------------------------------------------


@app.post("/api/query", response_model=QueryResponse)
def query_notes(req: QueryRequest):
    """Run the full agent loop and return a structured answer."""
    from rag.graph import get_agent_graph
    from rag.llm_factory import get_llm
    from rag.memory import WorkspaceChatMemoryManager

    memory = WorkspaceChatMemoryManager(workspace=req.workspace)
    chat_history = memory.load_messages()

    graph = get_agent_graph()
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
    """SSE streaming variant — yields JSON lines as the agent runs."""
    from rag.graph import get_agent_graph
    from rag.memory import WorkspaceChatMemoryManager

    memory = WorkspaceChatMemoryManager(workspace=req.workspace)
    chat_history = memory.load_messages()

    graph = get_agent_graph()
    input_state = {
        "question": req.question,
        "workspace": req.workspace,
        "chat_history": chat_history,
        "rewrite_count": 0,
    }

    async def event_generator():
        full_answer = ""
        async for event in graph.astream_events(
            input_state,
            version="v2",
            config={"configurable": {"thread_id": req.workspace}},
        ):
            if await request.is_disconnected():
                break
            kind = event.get("event", "")
            if kind == "on_chat_model_stream":
                token = event.get("data", {}).get("chunk", None)
                if token and hasattr(token, "content"):
                    chunk = token.content
                    if chunk:
                        full_answer += chunk
                        yield f"data: {json.dumps({'type': 'token', 'data': chunk})}\n\n"
            elif kind == "on_custom_event":
                yield f"data: {json.dumps({'type': 'source_documents', 'data': event.get('data', {})})}\n\n"

        memory.save_message("human", req.question)
        memory.save_message("ai", full_answer)
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


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


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health():
    """System health probe: vault path, note count, Ollama connectivity."""
    from rag.vectorstore import vectorstore
    from rag.llm_factory import _ollama_alive

    return {
        "status": "ok",
        "ollama_connected": _ollama_alive(settings.OLLAMA_BASE_URL),
        "vault_path": settings.VAULT_PATH,
        "notes_indexed": vectorstore.collection.count(),
    }


# ---------------------------------------------------------------------------
# Static frontend mount (must come AFTER all API routes)
# ---------------------------------------------------------------------------

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if _FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
