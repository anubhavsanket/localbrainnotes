# 🧠 LocalBrain

**Local-first, Obsidian-compatible agentic RAG assistant over markdown vaults.**

LocalBrain indexes an Obsidian-style markdown vault using a LangGraph agent loop (router → retrieval → grading → rewrite → generate → reflect), backed by a unified ChromaDB vector store and optionally powered by a local Ollama LLM.

---

## Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                        FastAPI (main.py)                          │
│  POST /api/query   POST /api/vault/ingest   GET /api/health     │
└──────┬────────────────────────────┬──────────────────────────────┘
       │                            │
       ▼                            ▼
┌──────────────┐           ┌────────────────┐
│  LangGraph   │           │   Vault        │
│  Agent Loop  │◄──────────│   Ingester     │
│  (graph.py)  │           │   (vault.py)   │
└──────┬───────┘           └───────┬────────┘
       │                           │
       ▼                           ▼
┌──────────────────────────────────────────────────────────────────┐
│             Unified ChromaDB Vector Store                         │
│             (single collection, workspace = metadata field)       │
└──────────────────────────────────────────────────────────────────┘
```

### Agent loop (`graph.py`)

```
START → router
  ├── "fastpath"  → retrieve → generate (skips grad/rewrite/reflect for simple facts)
  ├── "retrieve"  → retrieve → filter_docs → grade_docs
  │     └── any relevant  → generate → reflect
  │         └── grounded  → END (return answer)
  │         └── ungrounded→ guard_answer (in-place repair, max GUARD_REPAIR_MAX)
  │              └── still ungrounded → query_rewrite → retrieve (cycle, max 3)
  │     └── none relevant → query_rewrite → retrieve (cycle, max 3)
  ├── "direct"    → generate (chit-chat, no context) → END
  └── "tool"      → tool_search → generate (live web search context) → END
```

### Workspace model

Workspaces are a **metadata field on each chunk** (not a separate collection).  
Query-time: `WHERE workspace = "work"`.  
This fixes the orphan-chunk bug in the affine-lite prototype.

---

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) running locally (or set an `OPENAI_API_KEY` in `.env`)

### Models

```bash
# Pull the LLM + embedding model, then create the project's named modelfile
# (an alias of phi4-mini that `config.py` expects by default).
ollama pull phi4-mini
ollama pull nomic-embed-text
ollama create phi4-mini-localbrain -f Modelfile

# Verify
ollama list
```

### Setup

```bash
# Clone
git clone <repo> && cd LocalBrainNotes

# Create venv
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate          # Windows

# Install deps
cd backend
pip install -r requirements.txt

# Copy env config
cp ../.env.example ../.env       # edit as needed
```

### Run

```bash
cd backend
python main.py                   # starts on http://localhost:8000
```

Open `http://localhost:8000` in a browser for the chat UI.

### Index your vault

```bash
curl -X POST http://localhost:8000/api/vault/ingest
```

Or set `VAULT_PATH` in `.env` to point at your Obsidian vault.

### Query

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What decisions were made?", "workspace": "work"}'
```

---

## Vault Format

Every markdown note can include a YAML frontmatter block. The `workspace` field becomes the workspace label used to scope queries.

```yaml
---
title: "Q3 Design Review"
workspace: work
tags: [design, meeting, q3]
created: 2026-08-01
---

# Q3 Design Review

## Decisions

- Key decision here.
```

| Field      | Type       | Default     | Purpose                              |
|------------|------------|-------------|--------------------------------------|
| `title`    | string     | filename    | Display name / citation in answers   |
| `workspace`| string     | `"default"` | Retrieval scope label                |
| `tags`     | list       | `[]`        | Metadata (future filtering)          |
| `created`  | date       | —           | Ordering / context                   |

Notes without frontmatter default to `workspace = "default"`.

---

## Ingestion Sources

| Source | Route | Description |
|--------|-------|-------------|
| Markdown vault | `POST /api/vault/ingest` | Walk vault, parse frontmatter, chunk by headings |
| Single markdown file | `POST /api/vault/ingest/file` | Index one file by path |
| PDF | `POST /api/vault/ingest/pdf` | PyMuPDF extraction + chunking (+ Tesseract OCR fallback for image-only PDFs) |
| YouTube | `POST /api/vault/ingest/youtube` | Transcript API + timestamped chunks |

## Notes (file CRUD)

Notes are real `.md` files in the vault; the API reads/writes them directly and
re-indexes on write so edits are immediately searchable.

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/api/notes` | List notes (optionally `?workspace=X`) |
| `GET` | `/api/notes/{path}` | Read note content + metadata |
| `POST` | `/api/notes` | Create a note file |
| `PUT` | `/api/notes/{path}` | Overwrite content + re-index |
| `DELETE` | `/api/notes/{path}` | Delete file + drop vector chunks |

Path safety: note paths are resolved inside `VAULT_PATH` and escapes are rejected.

---

## Configuration

All settings live in `backend/config.py` and are overridable via environment variables (see `.env.example`).

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | `ollama`, `openai`, `anthropic`, `groq`, `nvidia` |
| `LLM_TEMPERATURE` | `0.1` | Sampling temperature for generation |
| `EMBEDDING_PROVIDER` | `ollama` | `ollama`, `openai`, `huggingface` |
| `VAULT_PATH` | `./vaults/sample` | Root of Obsidian vault to index |
| `SEARCH_TYPE` | `mmr` | `similarity` or `mmr` (maximal marginal relevance) |
| `TOP_K` | `5` | Documents returned per retrieval step |
| `SIMILARITY_THRESHOLD` | `0.70` | Relevance cutoff for retrieved context |
| `FASTPATH_ENABLED` | `true` | Route simple factual queries straight to retrieve→generate |
| `GUARD_REPAIR_MAX` | `1` | Max in-place answer repairs by the groundedness guard |
| `CONTEXT_MAX_CHARS` | `6000` | Compress retrieved context above this budget (fastpath) |
| `WEB_SEARCH_ENABLED` | `true` | Allow the `tool` route (DuckDuckGo HTML, no API key) |
| `MEMORY_WINDOW_SIZE` | `10` | Messages kept per workspace in SQLite |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local model server |
| `EVAL_JUDGE_PROVIDER` | `ollama` | RAGAS judge backend (or `openai` for any `/v1` endpoint) |

---

## Testing

```bash
cd backend
pytest ../tests/ -v
```

### Eval benchmarks

```bash
cd backend
# RAGAS evaluation (agentic vs naive on golden_dataset.json)
python ../evals/run_eval.py            # --only=agentic|naive to run one pipeline
python ../evals/benchmark_speed.py     # keyword vs semantic retrieval timing
```

The judge defaults to the local Ollama model (`EVAL_JUDGE_PROVIDER=ollama`) for
consistent, dependency-free scoring. Point `EVAL_JUDGE_PROVIDER=openai` +
`EVAL_LLM_BASE_URL` at any `/v1` endpoint (OpenRouter, NVIDIA NIM, etc.) to use a
cloud judge. Metrics: `faithfulness`, `answer_relevancy`, `context_recall`,
`context_precision`.

> Note: the harness uses a `LocalAnswerRelevancy` subclass of RAGAS's
> `AnswerRelevancy` — local judges often mark every answer `noncommittal`,
> which zeroes the cosine score; the subclass replaces that flag with a
> deterministic regex gate so relevancy stays meaningful offline.

---

## Desktop App (Tauri + React)

`desktop/` is a native Tauri v2 shell around the FastAPI backend. The Rust
sidecar spawns `backend/main.py` on startup, waits for `/api/health`, opens the
native window, and kills the backend on exit. The React UI provides **Chat** and
**Notes** views.

```bash
cd desktop
npm install
powershell -ExecutionPolicy Bypass -File bootstrap-toolchain.ps1   # one-time Rust toolchain
npm run tauri dev      # dev window (Vite :1420 + Rust shell)
npm run tauri build    # production installer
```

See `desktop/README.md` for the full setup.

---

## Project Structure

```
LocalBrainNotes/
├── backend/
│   ├── main.py              FastAPI app (+ notes CRUD routes)
│   ├── config.py            pydantic-settings
│   ├── rag/
│   │   ├── graph.py         LangGraph agent wiring (fastpath / retrieve / tool routes)
│   │   ├── nodes.py         router / retrieve / filter / grade / rewrite / generate / reflect / guard_answer / tool_search
│   │   ├── state.py         AgentState schema
│   │   ├── llm_factory.py   multi-provider LLM factory
│   │   ├── vectorstore.py   unified ChromaDB (single collection, workspace filter)
│   │   ├── embedder.py      Ollama/OpenAI/HF embeddings with cache
│   │   ├── web_search.py    dependency-free DuckDuckGo search for the tool route
│   │   └── ingester/
│   │       ├── vault.py     frontmatter + markdown header splitter
│   │       ├── pdf.py       PyMuPDF + Tesseract OCR fallback
│   │       └── youtube.py   youtube-transcript-api
│   │   ├── watcher.py       watchdog incremental indexing
│   │   └── memory.py        workspace-scoped sqlite chat memory
│   ├── models/schemas.py    Pydantic request/response schemas
│   └── db/store.py          JSON-backed vault metadata index
├── frontend/index.html      minimal chat UI (no build step)
├── desktop/                 Tauri v2 + React native app (chat + notes editor)
├── evals/
│   ├── run_eval.py          RAGAS faithfulness / relevancy / recall / precision (agentic vs naive)
│   ├── benchmark_speed.py   keyword vs semantic retrieval timing
│   └── golden_dataset.json  20 vault-aware test questions
├── tests/                   pytest suite (ingester, vectorstore, graph, API, PDF, integration)
├── vaults/sample/           Obsidian vault for testing
├── .env.example
└── README.md
```

---

## Roadmap

**Done:**
- ✅ PDF OCR fallback (Tesseract for image-only PDFs)
- ✅ Tauri + React native desktop shell (chat + notes editor)
- ✅ Notes file CRUD API + JSON metadata index
- ✅ Web-search tool route (dependency-free)
- ✅ Groundedness guard + context compression + fast-path router
- ✅ Local Ollama-based RAGAS judge (consistent offline evals)

**Planned:**
- **Phase F+**: human-in-the-loop review via LangGraph interrupts
- **Phase G**: Obsidian plugin for in-vault chat
- **Phase H**: CRDT-based sync layer for multi-device vaults

---

## License

MIT
