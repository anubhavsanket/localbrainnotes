"""Markdown vault ingestion.

Reads an Obsidian-style vault directory, parses YAML frontmatter, splits on
markdown headers, and indexes chunks into the unified vector store. The
``workspace`` YAML field becomes the metadata label used at query time to scope
retrieval to a workspace.
"""
from pathlib import Path
from typing import Optional

import yaml
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from config import settings
from rag import embedder as _embedder
from rag.vectorstore import stable_chunk_id, vectorstore

DEFAULT_WORKSPACE = "default"

HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

MARKDOWN_EXTENSIONS = {".md", ".markdown"}


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Split a leading ``---``-delimited YAML block from the markdown body.

    Returns ``({}, content)`` unchanged when there is no frontmatter or the
    block is malformed — the note still gets ingested with default metadata.
    """
    if not content.startswith("---"):
        return {}, content

    lines = content.splitlines()
    closing = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            closing = i
            break
    if closing is None:
        return {}, content

    frontmatter_text = "\n".join(lines[1:closing])
    body = "\n".join(lines[closing + 1:])
    try:
        meta = yaml.safe_load(frontmatter_text) or {}
    except yaml.YAMLError:
        return {}, content
    if not isinstance(meta, dict):
        return {}, content
    return meta, body


def _heading_path(metadata: dict) -> list[str]:
    """Rebuild the ordered heading ancestry from the splitter's output."""
    return [metadata[key] for key in ("h1", "h2", "h3") if key in metadata]


def chunk_vault_file(file_path: str, base_path: str) -> list[dict]:
    """Parse + chunk one vault file.

    Returns chunks shaped ``{"id", "text", "metadata"}`` *without* embeddings,
    so the text pipeline is unit-testable without an embedding backend.
    """
    path = Path(file_path)
    try:
        rel_path = path.resolve().relative_to(Path(base_path).resolve()).as_posix()
    except ValueError:
        rel_path = path.as_posix()

    raw = path.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(raw)

    title = meta.get("title") or path.stem
    workspace = str(meta.get("workspace") or DEFAULT_WORKSPACE)
    # ChromaDB rejects empty *lists* in metadata ("must be non-empty"), but
    # accepts an empty string. Encode tagless notes as '' so they index fine.
    # A YAML scalar like `tags: project-alpha` (string, not list) is coerced
    # to a single-element list so downstream tag handling stays predictable.
    tags = meta.get("tags") or ""
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()] if tags.strip() else ""
    elif isinstance(tags, list):
        tags = tags if tags else ""
    else:
        tags = str(tags) if tags else ""
    created = meta.get("created")

    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)
    sections = header_splitter.split_text(body)

    # Oversized header sections get recursively split into roughly equal chunks.
    sub_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )

    text_parts: list[str] = []
    heading_paths: list[list[str]] = []
    for section in sections:
        content = section.page_content
        if not content.strip():
            continue
        headings = _heading_path(section.metadata)
        if len(content) > settings.CHUNK_SIZE:
            for sub in sub_splitter.split_text(content):
                text_parts.append(sub)
                heading_paths.append(headings)
        else:
            text_parts.append(content)
            heading_paths.append(headings)

    chunks = []
    for idx, (text, headings) in enumerate(zip(text_parts, heading_paths)):
        chunks.append(
            {
                "id": stable_chunk_id(rel_path, idx),
                "text": text,
                "metadata": {
                    "note_id": rel_path,
                    "path": rel_path,
                    "workspace": workspace,
                    "title": title,
                    "tags": tags,
                    "created": str(created) if created is not None else None,
                    "heading_path": headings,
                    "chunk_index": idx,
                    "source_type": "markdown",
                },
            }
        )
    return chunks


def _with_embeddings(chunks: list[dict]) -> list[dict]:
    """Attach an ``embedding`` to each chunk dict (batched)."""
    texts = [c["text"] for c in chunks]
    embeddings = _embedder.embed_documents(texts)
    return [{**c, "embedding": emb} for c, emb in zip(chunks, embeddings)]


def ingest_file(file_path: str, base_path: str) -> int:
    """Delete any stale chunks for a note, then embed + upsert its new chunks.
    Returns the number of chunks indexed (0 if the file produced none)."""
    chunks = chunk_vault_file(file_path, base_path)
    if not chunks:
        return 0
    note_id = chunks[0]["metadata"]["note_id"]
    workspace = chunks[0]["metadata"]["workspace"]
    vectorstore.delete_note_chunks(note_id)  # avoid orphan chunks on edit
    embedded = _with_embeddings(chunks)
    return vectorstore.upsert_chunks(embedded, workspace=workspace)


def ingest_vault(vault_path: Optional[str] = None, workspace_filter: Optional[str] = None) -> int:
    """Index every markdown file under ``vault_path``.

    When ``workspace_filter`` is set, only notes whose frontmatter declares that
    workspace are indexed. Returns the total number of chunks indexed.
    """
    root = Path(vault_path or settings.VAULT_PATH)
    total = 0
    if not root.is_dir():
        raise FileNotFoundError(f"Vault path does not exist: {root}")

    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in MARKDOWN_EXTENSIONS:
            continue

        if workspace_filter is not None:
            meta, _ = parse_frontmatter(file_path.read_text(encoding="utf-8"))
            if str(meta.get("workspace") or DEFAULT_WORKSPACE) != workspace_filter:
                continue

        total += ingest_file(str(file_path), str(root))
    return total
