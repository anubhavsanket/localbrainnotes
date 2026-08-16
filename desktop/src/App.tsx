import { useCallback, useEffect, useRef, useState } from "react";
import {
  health,
  listWorkspaces,
  streamQuestion,
  getHistory,
  clearHistory,
  ingestVault,
} from "./api";
import { Markdown } from "./Markdown";
import NotesView from "./NotesView";

interface Message {
  role: "user" | "ai" | "system";
  content: string;
  sources?: string[];
}

export default function App() {
  const [workspaces, setWorkspaces] = useState<string[]>(["default"]);
  const [workspace, setWorkspace] = useState("default");
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [backendOk, setBackendOk] = useState<boolean | null>(null);
  const [indexing, setIndexing] = useState(false);
  const [view, setView] = useState<"chat" | "notes">("chat");
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    health()
      .then((h) => setBackendOk(h.status === "ok"))
      .catch(() => setBackendOk(false));
    listWorkspaces()
      .then((w) => {
        const list = w.workspaces.includes("default") ? w.workspaces : [...w.workspaces, "default"];
        setWorkspaces(list);
        setWorkspace(list[0] ?? "default");
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (!backendOk) return;
    getHistory(workspace)
      .then((h) => {
        const msgs: Message[] = h.messages
          .filter((m) => m.role === "human" || m.role === "ai")
          .map((m) => ({ role: m.role === "human" ? "user" : "ai", content: m.content }));
        setMessages((prev) => [...prev.filter((m) => m.role === "system"), ...msgs]);
      })
      .catch(() => {});
  }, [backendOk, workspace]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const send = useCallback(async () => {
    const text = input.trim();
    if (!text || sending || !backendOk) return;
    setSending(true);
    setInput("");

    const aiMsg: Message = { role: "ai", content: "" };
    setMessages((prev) => [...prev, { role: "user", content: text }, aiMsg]);

    let full = "";
    try {
      await streamQuestion(text, workspace, {
        onToken: (chunk) => {
          full += chunk;
          setMessages((prev) => {
            const next = [...prev];
            next[next.length - 1] = { ...next[next.length - 1], content: full };
            return next;
          });
        },
        onSources: (sources) => {
          setMessages((prev) => {
            const next = [...prev];
            next[next.length - 1] = { ...next[next.length - 1], sources };
            return next;
          });
        },
      });
    } catch (err) {
      setMessages((prev) => {
        const next = [...prev];
        next[next.length - 1] = {
          role: "ai",
          content: `Error: ${(err as Error).message}`,
        };
        return next;
      });
    }
    setSending(false);
  }, [input, sending, backendOk, workspace]);

  const onClear = async () => {
    await clearHistory(workspace);
    setMessages([{ role: "system", content: "History cleared." }]);
  };

  const onReindex = async () => {
    setIndexing(true);
    try {
      const res = await ingestVault();
      setMessages((prev) => [...prev, { role: "system", content: `Indexed ${res.chunks} chunks.` }]);
    } catch (err) {
      setMessages((prev) => [...prev, { role: "system", content: `Index failed: ${(err as Error).message}` }]);
    }
    setIndexing(false);
  };

  return (
    <div className="app">
      <header>
        <div className="brand">
          <h1>🧠 LocalBrain</h1>
          <nav className="view-switch">
            <button
              className={view === "chat" ? "active" : ""}
              onClick={() => setView("chat")}
            >
              Chat
            </button>
            <button
              className={view === "notes" ? "active" : ""}
              onClick={() => setView("notes")}
            >
              Notes
            </button>
          </nav>
        </div>
        <div className="controls">
          <span className={`dot ${backendOk ? "ok" : "err"}`} title={backendOk ? "backend up" : "backend down"} />
          <select value={workspace} onChange={(e) => setWorkspace(e.target.value)}>
            {workspaces.map((w) => (
              <option key={w} value={w}>
                {w}
              </option>
            ))}
          </select>
          {view === "chat" && (
            <>
              <button onClick={onClear}>Clear</button>
              <button onClick={onReindex} disabled={indexing}>
                {indexing ? "Indexing…" : "Re-index"}
              </button>
            </>
          )}
        </div>
      </header>

      {view === "notes" ? (
        <NotesView workspace={workspace} />
      ) : (
        <>
        <main className="chat">
        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            <div className="label">{m.role === "user" ? "You" : m.role === "ai" ? "LocalBrain" : ""}</div>
            {m.role === "ai" && m.content ? <Markdown text={m.content} /> : null}
            {m.role === "user" ? <div className="plain">{m.content}</div> : null}
            {m.role === "system" ? <div className="plain">{m.content}</div> : null}
            {m.sources && m.sources.length ? (
              <details className="sources">
                <summary>Sources ({m.sources.length})</summary>
                {m.sources.map((s, j) => (
                  <span key={j} className="tag">
                    {s}
                  </span>
                ))}
              </details>
            ) : null}
          </div>
        ))}
        {!backendOk && (
          <div className="msg system">
            <div className="plain">⚠ Backend unreachable at 127.0.0.1:8000 — start the Rust shell's sidecar or run backend/main.py.</div>
          </div>
        )}
        <div ref={chatEndRef} />
        </main>

        <footer className="input-bar">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
            placeholder="Ask a question…"
            rows={1}
          />
          <button onClick={send} disabled={sending || !backendOk}>
            {sending ? "…" : "Send"}
          </button>
        </footer>
        </>
      )}
    </div>
  );
}
