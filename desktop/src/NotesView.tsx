import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createNote, deleteNote, listNotes, readNote, updateNote, NoteMeta } from "./api";
import { Markdown } from "./Markdown";

interface NotesViewProps {
  workspace: string;
}

type NoteGroup = { workspace: string; notes: NoteMeta[] };

export default function NotesView({ workspace }: NotesViewProps) {
  const [groups, setGroups] = useState<NoteGroup[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [content, setContent] = useState("");
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [preview, setPreview] = useState(false);
  const [query, setQuery] = useState("");
  const [newTitle, setNewTitle] = useState("");
  const [err, setErr] = useState("");
  const [savedFlash, setSavedFlash] = useState(false);
  const saveTimer = useRef<number | null>(null);

  const refresh = useCallback(async () => {
    try {
      const res = await listNotes();
      const byWs = new Map<string, NoteMeta[]>();
      for (const n of res.notes) {
        const list = byWs.get(n.workspace) ?? [];
        list.push(n);
        byWs.set(n.workspace, list);
      }
      const sorted: NoteGroup[] = [...byWs.entries()]
        .map(([name, notes]) => ({ workspace: name, notes: notes.slice().sort((a, b) => a.title.localeCompare(b.title)) }))
        .sort((a, b) => a.workspace.localeCompare(b.workspace));
      setGroups(sorted);
    } catch (e) {
      setErr((e as Error).message);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const open = useCallback(async (path: string) => {
    try {
      const note = await readNote(path);
      setSelected(path);
      setContent(note.content);
      setDirty(false);
      setPreview(false);
      setErr("");
    } catch (e) {
      setErr((e as Error).message);
    }
  }, []);

  const save = useCallback(
    async (path: string, body: string) => {
      setSaving(true);
      try {
        const saved = await updateNote(path, body, workspace);
        setContent(saved.content);
        setSavedFlash(true);
        setDirty(false);
        if (saveTimer.current) window.clearTimeout(saveTimer.current);
        saveTimer.current = window.setTimeout(() => setSavedFlash(false), 1500);
        await refresh();
      } catch (e) {
        setErr((e as Error).message);
      }
      setSaving(false);
    },
    [workspace, refresh]
  );

  const onCreate = async () => {
    const title = newTitle.trim() || `untitled-${Date.now()}`;
    const safeTitle = title.replace(/[\\/:*?"<>|]/g, "-");
    const path = `${workspace === "default" ? "" : workspace + "/"}${safeTitle}.md`;
    try {
      const note = await createNote(path, `# ${safeTitle}\n\n`, workspace);
      setNewTitle("");
      setGroups([]);
      await refresh();
      await open(note.path);
    } catch (e) {
      setErr((e as Error).message);
    }
  };

  const onDelete = async () => {
    if (!selected) return;
    if (!window.confirm(`Delete note "${selected}"? This removes the file and its index.`)) return;
    try {
      await deleteNote(selected);
      setSelected(null);
      setContent("");
      setDirty(false);
      await refresh();
    } catch (e) {
      setErr((e as Error).message);
    }
  };

  const visibleGroups = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return groups;
    return groups
      .map((g) => ({ ...g, notes: g.notes.filter((n) => (n.title + " " + n.path).toLowerCase().includes(q)) }))
      .filter((g) => g.notes.length > 0);
  }, [groups, query]);

  return (
    <div className="notes-view">
      <aside className="notes-sidebar">
        <div className="notes-sidebar-top">
          <input
            className="notes-search"
            placeholder="Search notes…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <input
            className="notes-new"
            placeholder="New note title…"
            value={newTitle}
            onChange={(e) => setNewTitle(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") onCreate();
            }}
          />
          <button className="notes-new-btn" onClick={onCreate} title="Create note">
            +
          </button>
        </div>
        <div className="notes-list">
          {visibleGroups.map((g) => (
            <div key={g.workspace} className="notes-group">
              <div className="notes-group-title">{g.workspace}</div>
              {g.notes.map((n) => (
                <button
                  key={n.path}
                  className={`notes-item${selected === n.path ? " active" : ""}`}
                  onClick={() => open(n.path)}
                  title={n.path}
                >
                  <span className="notes-item-title">{n.title}</span>
                </button>
              ))}
            </div>
          ))}
          {visibleGroups.length === 0 && <div className="notes-empty">No notes yet.</div>}
        </div>
      </aside>

      <section className="notes-editor">
        {selected ? (
          <>
            <div className="notes-toolbar">
              <span className="notes-path" title={selected}>
                {selected}
              </span>
              <div className="notes-actions">
                <button
                  className={preview ? "active" : ""}
                  onClick={() => setPreview((p) => !p)}
                >
                  {preview ? "Edit" : "Preview"}
                </button>
                <button className="delete" onClick={onDelete} title="Delete note">
                  Delete
                </button>
                <button
                  className="save"
                  onClick={() => save(selected, content)}
                  disabled={saving || !dirty}
                >
                  {saving ? "Saving…" : savedFlash ? "Saved" : "Save"}
                </button>
              </div>
            </div>
            {preview ? (
              <div className="notes-preview">
                <Markdown text={content} />
              </div>
            ) : (
              <textarea
                className="notes-textarea"
                value={content}
                onChange={(e) => {
                  setContent(e.target.value);
                  setDirty(true);
                }}
                spellCheck={false}
              />
            )}
          </>
        ) : (
          <div className="notes-placeholder">Select a note from the sidebar to start editing.</div>
        )}
        {err && <div className="notes-error">{err}</div>}
      </section>
    </div>
  );
}