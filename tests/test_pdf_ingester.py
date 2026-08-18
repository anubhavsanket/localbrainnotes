"""Tests for PDF ingestion, including the image-only-OCR fallback path."""
import io
import sys
from pathlib import Path

import fitz  # PyMuPDF

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import pytest
import rag.ingester.pdf as pdf_mod


def _make_text_pdf(path: Path, text: str) -> Path:
    """Create a small born-digital PDF containing real text."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=11)
    doc.save(str(path))
    doc.close()
    return path


def _make_image_only_pdf(path: Path) -> Path:
    """Create a PDF containing an image with absolutely no embedded text."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (400, 120), "white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 40), "Invoice Total $1,234", fill="black")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    doc = fitz.open()
    page = doc.new_page(width=400, height=120)
    page.insert_image(page.rect, stream=img_bytes.getvalue())
    doc.save(str(path))
    doc.close()
    return path


class TestPdfTextPath:
    def test_chunks_text_pdf(self, tmp_path, fake_embeddings):
        pdf = _make_text_pdf(
            tmp_path / "notes.pdf",
            "LocalBrain indexes Obsidian vaults. This is a test document "
            "that contains enough text to be extracted by PyMuPDF.\n\n"
            "The vault uses frontmatter to label workspaces, and the agent "
            "loop routes questions through retrieval before answering.",
        )
        chunks = pdf_mod.chunk_pdf(str(pdf), workspace="work")
        assert len(chunks) >= 1
        assert all(c["metadata"]["source_type"] == "pdf" for c in chunks)
        assert all(c["metadata"]["extraction"] == "text" for c in chunks)
        assert "Obsidian" in chunks[0]["text"]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            pdf_mod.chunk_pdf(str(tmp_path / "missing.pdf"))


class TestPdfOcrFallback:
    def test_image_only_pdf_raises_clear_error_when_ocr_unconfigured(
        self, tmp_path, monkeypatch
    ):
        """Without Tesseract installed we must NOT silently produce bad chunks."""
        pdf = _make_image_only_pdf(tmp_path / "scan.pdf")
        monkeypatch.setattr(pdf_mod, "_ocr_available", lambda: False)
        with pytest.raises(pdf_mod.OcrNotConfigured, match="pytesseract"):
            pdf_mod.chunk_pdf(str(pdf))

    def test_image_only_pdf_uses_ocr_when_available(
        self, tmp_path, monkeypatch
    ):
        pdf = _make_image_only_pdf(tmp_path / "scan.pdf")

        class _FakeTesseract:
            @staticmethod
            def image_to_string(img):
                return "Invoice Total $1,234\n"

        monkeypatch.setattr(pdf_mod, "_ocr_available", lambda: True)
        monkeypatch.setattr(pdf_mod, "pytesseract", _FakeTesseract)
        chunks = pdf_mod.chunk_pdf(str(pdf))
        joined = " ".join(c["text"] for c in chunks)
        assert "Invoice" in joined
        assert all(c["metadata"]["extraction"] == "ocr" for c in chunks)

    def test_ocr_not_configured_exception_has_actionable_message(self):
        err = pdf_mod.OcrNotConfigured(
            "pip install pytesseract and the Tesseract binary"
        )
        assert "pytesseract" in str(err)