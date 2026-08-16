// Minimal, dependency-free Markdown renderer (ported from frontend/index.html).
// Renders the safe markdown the LLM produces; HTML is escaped first (XSS-safe).

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function inlineMd(s: string): string {
  return escapeHtml(s)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>")
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" rel="noopener">$1</a>');
}

export function renderMarkdown(md: string): string {
  if (!md) return "";
  const lines = md.split("\n");
  const out: string[] = [];
  let listType: "ul" | "ol" | null = null;
  let inCode = false;
  let codeBuf: string[] = [];
  let table: string[][] = [];

  const closeList = () => {
    if (listType) {
      out.push(`</${listType}>`);
      listType = null;
    }
  };
  const flushTable = () => {
    if (!table.length) return;
    let h = "<table>";
    table.forEach((row, r) => {
      h += "<tr>";
      row.forEach((cell) => {
        const tag = r === 0 ? "th" : "td";
        h += `<${tag}>${inlineMd(cell.trim())}</${tag}>`;
      });
      h += "</tr>";
    });
    out.push(h + "</table>");
    table = [];
  };

  for (const line0 of lines) {
    const line = line0.trimEnd();

    if (line.trim().startsWith("```")) {
      closeList();
      flushTable();
      if (inCode) {
        out.push("<pre><code>" + escapeHtml(codeBuf.join("\n")) + "</code></pre>");
        codeBuf = [];
        inCode = false;
      } else {
        inCode = true;
      }
      continue;
    }
    if (inCode) {
      codeBuf.push(line);
      continue;
    }

    const h = /^(#{1,4})\s+(.*)$/.exec(line);
    if (h) {
      closeList();
      flushTable();
      const level = h[1].length;
      out.push(`<h${level}>${inlineMd(h[2])}</h${level}>`);
      continue;
    }

    if (line.startsWith("|") && line.endsWith("|")) {
      closeList();
      const cells = line.slice(1, -1).split("|");
      const isSep = cells.every((c) => /^:?-{3,}:?$/.test(c.trim()));
      if (isSep) continue;
      table.push(cells);
      continue;
    }
    flushTable();

    if (line.startsWith("> ")) {
      closeList();
      out.push(`<blockquote>${inlineMd(line.slice(2))}</blockquote>`);
      continue;
    }

    let m = /^[-*]\s+(.*)$/.exec(line.trim());
    if (m) {
      if (listType !== "ul") {
        closeList();
        out.push("<ul>");
        listType = "ul";
      }
      out.push(`<li>${inlineMd(m[1])}</li>`);
      continue;
    }
    m = /^\d+\.\s+(.*)$/.exec(line.trim());
    if (m) {
      if (listType !== "ol") {
        closeList();
        out.push("<ol>");
        listType = "ol";
      }
      out.push(`<li>${inlineMd(m[1])}</li>`);
      continue;
    }
    closeList();

    if (!line.trim()) continue;
    out.push(`<p>${inlineMd(line)}</p>`);
  }
  closeList();
  flushTable();
  if (inCode) out.push("<pre><code>" + escapeHtml(codeBuf.join("\n")) + "</code></pre>");
  return out.join("\n");
}

export function Markdown({ text }: { text: string }) {
  return (
    <div className="md" dangerouslySetInnerHTML={{ __html: renderMarkdown(text) }} />
  );
}
