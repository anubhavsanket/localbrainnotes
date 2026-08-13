"""Tests for the vault ingester."""
import pytest


class TestFrontmatter:
    def test_parses_valid_frontmatter(self):
        from rag.ingester.vault import parse_frontmatter

        raw = "---\ntitle: Test\nworkspace: work\ntags: [a, b]\n---\n# Body"
        meta, body = parse_frontmatter(raw)
        assert meta["title"] == "Test"
        assert meta["workspace"] == "work"
        assert meta["tags"] == ["a", "b"]
        assert "# Body" in body

    def test_returns_empty_when_no_frontmatter(self):
        from rag.ingester.vault import parse_frontmatter

        raw = "# Just a note\nNo frontmatter here."
        meta, body = parse_frontmatter(raw)
        assert meta == {}
        assert body == raw

    def test_handles_malformed_yaml(self):
        from rag.ingester.vault import parse_frontmatter

        raw = "---\n[[invalid yaml\n---\n# Body"
        meta, body = parse_frontmatter(raw)
        assert meta == {}

    def test_handles_frontmatter_with_no_closing(self):
        from rag.ingester.vault import parse_frontmatter

        raw = "---\ntitle: Test\n# No closing"
        meta, body = parse_frontmatter(raw)
        assert meta == {}


class TestChunkVaultFile:
    def test_chunks_with_frontmatter(self, sample_vault_path):
        from rag.ingester.vault import chunk_vault_file

        file_path = f"{sample_vault_path}/workspace/note1.md"
        chunks = chunk_vault_file(file_path, sample_vault_path)

        assert len(chunks) >= 2
        for c in chunks:
            assert "id" in c
            assert "text" in c
            assert "metadata" in c
            assert c["metadata"]["workspace"] == "work"
            assert c["metadata"]["source_type"] == "markdown"

    def test_stable_ids_are_deterministic(self, sample_vault_path):
        from rag.ingester.vault import chunk_vault_file
        from rag.vectorstore import stable_chunk_id

        c1 = chunk_vault_file(f"{sample_vault_path}/workspace/note1.md", sample_vault_path)
        c2 = chunk_vault_file(f"{sample_vault_path}/workspace/note1.md", sample_vault_path)
        assert [ch["id"] for ch in c1] == [ch["id"] for ch in c2]

    def test_no_chunks_for_empty_file(self, tmp_path):
        from rag.ingester.vault import chunk_vault_file

        empty = tmp_path / "empty.md"
        empty.write_text("")
        chunks = chunk_vault_file(str(empty), str(tmp_path))
        assert chunks == []


class TestIngestVault:
    def test_counts_all_notes(self, sample_vault_path, fake_embeddings):
        from rag.ingester.vault import ingest_vault as _ingest_vault

        count = _ingest_vault(sample_vault_path)
        assert count > 0
