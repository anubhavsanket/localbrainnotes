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
  ├── "retrieve"  → retrieve → grade_docs
  │     └── any relevant  → generate → reflect
  │         ├── grounded   → END (return answer)
  │         └── ungrounded → query_rewrite → retrieve (cycle, max 3)
  └── "direct"    → generate (no context) → END
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
| PDF | `POST /api/vault/ingest/pdf` | PyMuPDF extraction + chunking |
| YouTube | `POST /api/vault/ingest/youtube` | Transcript API + timestamped chunks |

---

## Configuration

All settings live in `backend/config.py` and are overridable via environment variables (see `.env.example`).

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | `ollama`, `openai`, `anthropic`, `groq`, `nvidia` |
| `EMBEDDING_PROVIDER` | `ollama` | `ollama`, `openai`, `huggingface` |
| `VAULT_PATH` | `./vaults/sample` | Root of Obsidian vault to index |
| `SEARCH_TYPE` | `mmr` | `similarity` or `mmr` (maximal marginal relevance) |
| `TOP_K` | `5` | Documents returned per retrieval step |
| `MEMORY_WINDOW_SIZE` | `10` | Messages kept per workspace in SQLite |

---

## Testing

```bash
cd backend
pytest ../tests/ -v
```

### Eval benchmarks

```bash
cd backend
# RAGAS evaluation (requires OPENAI_API_KEY for the judge model)
python ../evals/run_eval.py

# Retrieval speed benchmark
python ../evals/benchmark_speed.py
```

---

## Project Structure

```
LocalBrainNotes/
├── backend/
│   ├── main.py              FastAPI app
│   ├── config.py            pydantic-settings
│   ├── rag/
│   │   ├── graph.py         LangGraph agent wiring
│   │   ├── nodes.py         router / retrieve / grade / rewrite / generate / reflect
│   │   ├── state.py         AgentState schema
│   │   ├── llm_factory.py   multi-provider LLM factory
│   │   ├── vectorstore.py   unified ChromaDB (single collection, workspace filter)
│   │   ├── embedder.py      Ollama/OpenAI/HF embeddings with cache
│   │   ├── ingester/
│   │   │   ├── vault.py     frontmatter + markdown header splitter
│   │   │   ├── pdf.py       PyMuPDF extraction
│   │   │   └── youtube.py   youtube-transcript-api
│   │   ├── watcher.py       watchdog incremental indexing
│   │   └── memory.py        workspace-scoped sqlite chat memory
│   ├── models/schemas.py    Pydantic request/response schemas
│   └── db/store.py          JSON-backed vault metadata index
├── frontend/index.html      minimal chat UI (no build step)
├── evals/
│   ├── run_eval.py          RAGAS faithfulness / relevance / recall / precision
│   ├── benchmark_speed.py   keyword vs semantic retrieval timing
│   └── golden_dataset.json  20 vault-aware test questions
├── tests/                   pytest suite (ingester, vectorstore, graph, API)
├── vaults/sample/           Obsidian vault for testing
├── .env.example
└── README.md
```

---

## Roadmap

- **Phase F+**: human-in-the-loop review via LangGraph interrupts
- **Phase G**: Obsidian plugin for in-vault chat
- **Phase H**: CRDT-based sync layer for multi-device vaults
- **Phase I**: PDF OCR pipeline (anydoc + Tesseract for image-only PDFs)
- **Phase J**: Tauri+React native desktop frontend

---

## License

MIT
