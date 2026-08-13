"""PDF ingestion via PyMuPDF (``fitz``). Text is extracted page by page and
chunked with the shared RecursiveCharacterTextSplitter. Image-heavy PDFs that
yield no text should be OCR'd upstream (anydoc/OCR) before being passed here.
"""
from pathlib import Path

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from rag import embedder as _embedder
from rag.vectorstore import stable_chunk_id, vectorstore


def _extract_text(file_path: str) -> str:
    doc = fitz.open(file_path)
    try:
        pages = []
        for page in doc:
            text = page.get_text()
            if text.strip():
                pages.append(text)
    finally:
        doc.close()
    return "\n\n".join(pages)


def chunk_pdf(file_path: str, workspace: str = "default") -> list[dict]:
    """Parse a PDF and return chunk dicts (no embeddings yet)."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    text = _extract_text(str(path))
    if not text.strip():
        raise ValueError(f"No extractable text in {file_path} — OCR the PDF first")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    doc = Document(page_content=text, metadata={"source": str(path)})
    pieces = splitter.split_documents([doc])

    chunks = []
    for idx, piece in enumerate(pieces):
        chunks.append(
            {
                "id": stable_chunk_id(path.name, idx),
                "text": piece.page_content,
                "metadata": {
                    "note_id": path.name,
                    "path": str(path),
                    "workspace": workspace,
                    "title": path.stem,
                    "chunk_index": idx,
                    "source_type": "pdf",
                },
            }
        )
    return chunks


def ingest_pdf(file_path: str, workspace: str = "default") -> int:
    """Index a PDF into the vector store. Returns chunk count."""
    chunks = chunk_pdf(file_path, workspace)
    note_id = chunks[0]["metadata"]["note_id"]
    vectorstore.delete_note_chunks(note_id)
    texts = [c["text"] for c in chunks]
    embeddings = _embedder.embed_documents(texts)
    embedded = [{**c, "embedding": emb} for c, emb in zip(chunks, embeddings)]
    return vectorstore.upsert_chunks(embedded, workspace=workspace)
