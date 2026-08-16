"""PDF ingestion via PyMuPDF (``fitz``) with an automatic OCR fallback.

Strategy:
  1. Try fast text extraction page-by-page (works for born-digital PDFs).
  2. Any page that yields no text is rendered to an image and OCR'd
     (Tesseract via ``pytesseract``). OCR is opt-in at the system level:
     you must install:
       - ``pip install pytesseract``
       - Tesseract binary: https://github.com/tesseract-ocr/tesseract
         (Windows: `winget install UB-Mannheim.TesseractOCR`)
     If Tesseract is unavailable, the fallback raises a clear message so the
     caller can still surface the original error instead of failing silently.
  3. The merged text (embedded text + OCR'd pages) is chunked with the
     shared RecursiveCharacterTextSplitter, keeping the same metadata shape
     as the pure-text path (``source_type="pdf"``).

Each chunk records ``extraction="text" | "ocr" | "mixed"`` so downstream code
(or tests) can tell which pipeline produced it.
"""
import io
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

try:
    from PIL import Image

    _PIL_AVAILABLE = True
except ImportError:  # pragma: no cover - Pillow is a hard dep in requirements
    _PIL_AVAILABLE = False

try:
    import pytesseract

    _TESSERACT_AVAILABLE = True
except ImportError:
    pytesseract = None  # type: ignore[assignment]
    _TESSERACT_AVAILABLE = False

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from rag import embedder as _embedder
from rag.vectorstore import stable_chunk_id, vectorstore

# Pages with fewer characters than this are treated as image-only.
_MIN_TEXT_CHARS = 8

# Render pages at this DPI for OCR (balances speed vs accuracy).
_OCR_DPI = 200


class OcrNotConfigured(ValueError):
    """Raised when an image-only PDF is found but Tesseract isn't installed."""


def _ocr_available() -> bool:
    """True when the Tesseract binary is on PATH (checked lazily)."""
    if not _TESSERACT_AVAILABLE or not _PIL_AVAILABLE:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _ocr_page(page: fitz.Page) -> str:
    """Render a single page to an image and OCR it with Tesseract."""
    if not _ocr_available():
        raise OcrNotConfigured(
            "Image-only PDF detected. Enable OCR by installing: "
            "`pip install pytesseract` and the Tesseract binary "
            "(Windows: `winget install UB-Mannheim.TesseractOCR`)."
        )
    pix = page.get_pixmap(dpi=_OCR_DPI)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img)


def _extract_text_with_ocr(file_path: str) -> str:
    """Extract text per page; OCR only the pages that yield none.

    Returns the page text (embedded or OCR'd) plus the extraction label used
    to set per-chunk ``source_type``.
    """
    doc = fitz.open(file_path)
    try:
        pages: list[str] = []
        labels: list[str] = []
        for page in doc:
            text = page.get_text()
            if text.strip() and len(text.strip()) >= _MIN_TEXT_CHARS:
                pages.append(text.strip())
                labels.append("text")
            else:
                ocr_text = _ocr_page(page)
                pages.append(ocr_text.strip() if ocr_text.strip() else "")
                labels.append("ocr")
    finally:
        doc.close()

    extraction_label = (
        "text"
        if "ocr" not in labels
        else ("ocr" if "text" not in labels else "mixed")
    )
    return "\n\n".join(pages), extraction_label


def _extract_text(file_path: str) -> tuple[str, str]:
    """Extract all pages' text, OCR-ing image-only pages.

    Returns ``(full_text, extraction_label)``.
    """
    return _extract_text_with_ocr(file_path)


def chunk_pdf(file_path: str, workspace: str = "default") -> list[dict]:
    """Parse a PDF and return chunk dicts (no embeddings yet).

    Raises ``ValueError`` if the PDF has no text at all — either because it is
    truly empty or because OCR is not configured (see ``OcrNotConfigured``).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    text, extraction_label = _extract_text(str(path))
    if not text.strip():
        raise ValueError(
            f"No extractable text in {file_path} — the pages produced neither "
            "embedded text nor OCR output (was Tesseract installed?)."
        )

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
                    "extraction": extraction_label,
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