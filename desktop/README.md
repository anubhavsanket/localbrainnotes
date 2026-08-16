# LocalBrain Desktop Shell

Tauri v2 + React wrapper around the existing FastAPI backend. The Rust shell
spawns `backend/main.py` (uvicorn on `127.0.0.1:8000`) as a local sidecar on
startup, opens the native window hosting the React chat UI, and kills the
backend on exit.

```
desktop/
├── bootstrap-toolchain.ps1   # installs Rust + MSVC Build Tools (run once)
├── package.json              # React 18 + Vite + Tauri CLI
├── src/                      # React chat UI (mirrors frontend/index.html)
│   ├── App.tsx               # chat state machine, SSE streaming, history
│   ├── api.ts                # typed client for /api/* (VITE_BACKEND_URL)
│   ├── Markdown.tsx          # dependency-free markdown renderer (XSS-safe)
│   └── styles.css
└── src-tauri/                # Rust shell
    ├── Cargo.toml
    ├── tauri.conf.json       # window config, devUrl :1420, bundle icons
    ├── capabilities/default.json
    └── src/lib.rs            # spawns the FastAPI backend, waits for /api/health
```

## Prerequisites

- Node 18+ (`node --version` — verified: v22)
- Rust toolchain **with the MSVC toolchain** + Visual Studio Build Tools
  (C++ workload) — **not yet installed on this machine**
- WebView2 runtime (present on stock Windows 10/11 — verified present)
- The repo's Python venv at `.venv/` with `pip install -r requirements.txt`
  done (the shell reuses it to run the backend)

## One-time setup

```powershell
powershell -ExecutionPolicy Bypass -File desktop\bootstrap-toolchain.ps1
```

This installs rustup (stable, MSVC) and the VS Build Tools C++ workload.
**Restart your terminal afterwards** so `cargo` is on PATH.

## Run (dev)

```powershell
cd desktop
npm install
npm run tauri dev
```

`tauri dev` starts Vite at `http://localhost:1420` (the webview source in dev),
compiles the Rust shell, launches the native window, and the shell spawns the
FastAPI backend automatically. First Rust build takes 15–30 min.

The React app talks to the backend at `http://127.0.0.1:8000` by default.
Override with `VITE_BACKEND_URL` (e.g. a `.env.local` file).

## Build a production installer

```powershell
cd desktop
npm run tauri build
```

## Architecture notes

- **Sidecar strategy**: Python can't be statically linked into the Tauri
  binary, so the shell shells out to the venv python. This keeps the full RAG
  stack (agent loop, Ollama, ChromaDB, watcher) unchanged and testable.
- The shell waits up to 30 s for `/api/health` before showing the window, so
  the UI's health dot turns green without a manual retry.
- Browser-only dev (no Tauri) works too: run `npm run dev` in `desktop/` and
  `python backend/main.py` in a second terminal — CORS on the backend is
  `allow_origins=["*"]`.
