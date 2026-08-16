// Typed client for the LocalBrain FastAPI backend.
//
// In the Tauri shell the backend is spawned locally by Rust and served at
// 127.0.0.1:8000. In browser-dev the Vite proxy can forward /api → :8000;
// both paths are covered by the BACKEND_URL constant below.

const DEFAULT_BACKEND = "http://127.0.0.1:8000";

export const BACKEND_URL: string = import.meta.env.VITE_BACKEND_URL ?? DEFAULT_BACKEND;

export interface QueryResponse {
  answer: string;
  sources: string[];
  latency_seconds: number;
}

export interface WorkspaceResponse {
  workspaces: string[];
  count: number;
}

export interface NoteMeta {
  workspace: string;
  path: string;
  title: string;
  mtime: number;
  size: number;
}

export interface NoteFull extends NoteMeta {
  content: string;
}

export interface NotesResponse {
  notes: NoteMeta[];
  count: number;
}

export interface HistoryMessage {
  role: string;
  content: string;
}

export interface HistoryResponse {
  count: number;
  messages: HistoryMessage[];
}

export async function health(): Promise<{ status: string; ollama_connected: boolean }> {
  return request("/api/health");
}

export async function listWorkspaces(): Promise<WorkspaceResponse> {
  return request("/api/workspaces");
}

export async function listNotes(workspace?: string): Promise<NotesResponse> {
  const qs = workspace ? `?workspace=${encodeURIComponent(workspace)}` : "";
  return request(`/api/notes${qs}`);
}

export async function readNote(path: string): Promise<NoteFull> {
  return request(`/api/notes/${encodeURI(path)}`);
}

export async function createNote(path: string, content: string, workspace = "default"): Promise<NoteFull> {
  return request("/api/notes", {
    method: "POST",
    body: JSON.stringify({ path, content, workspace }),
  });
}

export async function updateNote(path: string, content: string, workspace = "default"): Promise<NoteFull> {
  return request(`/api/notes/${encodeURI(path)}`, {
    method: "PUT",
    body: JSON.stringify({ path, content, workspace }),
  });
}

export async function deleteNote(path: string): Promise<{ status: string; path: string }> {
  return request(`/api/notes/${encodeURI(path)}`, { method: "DELETE" });
}

/** Encode a vault-relative path for use inside a URL segment. */
function encodeURI(path: string): string {
  return path.split("/").map(encodeURIComponent).join("/");
}

export async function getHistory(workspace: string): Promise<HistoryResponse> {
  return request(`/api/history?workspace=${encodeURIComponent(workspace)}`);
}

export async function clearHistory(workspace: string): Promise<{ status: string }> {
  return request(`/api/history?workspace=${encodeURIComponent(workspace)}`, {
    method: "DELETE",
  });
}

export async function ingestVault(): Promise<{ status: string; chunks: number }> {
  return request("/api/vault/ingest", { method: "POST" });
}

/** POST a question and consume the SSE token stream. */
export async function streamQuestion(
  question: string,
  workspace: string,
  callbacks: {
    onToken: (chunk: string) => void;
    onSources: (sources: string[]) => void;
  }
): Promise<void> {
  const res = await fetch(`${BACKEND_URL}/api/query/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, workspace }),
  });

  if (!res.ok) {
    let errText = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body.detail) errText = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* ignore */
    }
    throw new Error(errText);
  }

  const reader = res.body?.getReader();
  if (!reader) throw new Error("Streaming not supported by this response");

  const decoder = new TextDecoder();
  let buffer = "";

  const processLines = (lines: string[]) => {
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data: ")) continue;
      try {
        const msg = JSON.parse(trimmed.slice(6));
        if (msg.type === "token") callbacks.onToken(msg.data);
        else if (msg.type === "source_documents") callbacks.onSources(msg.data);
      } catch {
        /* partial line */
      }
    }
  };

  for (;;) {
    const { done, value } = await reader.read();
    if (done) {
      if (buffer.trim()) processLines(buffer.split("\n"));
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    processLines(lines);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BACKEND_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}
