"""Tests for the vault ingester."""


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
        meta, _ = parse_frontmatter(raw)
        assert meta == {}

    def test_handles_frontmatter_with_no_closing(self):
        from rag.ingester.vault import parse_frontmatter

        raw = "---\ntitle: Test\n# No closing"
        meta, _ = parse_frontmatter(raw)
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

        c1 = chunk_vault_file(f"{sample_vault_path}/workspace/note1.md", sample_vault_path)
        c2 = chunk_vault_file(f"{sample_vault_path}/workspace/note1.md", sample_vault_path)
        assert [ch["id"] for ch in c1] == [ch["id"] for ch in c2]

    def test_no_chunks_for_empty_file(self, tmp_path):
        from rag.ingester.vault import chunk_vault_file

        empty = tmp_path / "empty.md"
        empty.write_text("")
        chunks = chunk_vault_file(str(empty), str(tmp_path))
        assert chunks == []

    def test_tagless_note_encodes_empty_tags_as_string(self, tmp_path):
        """ChromaDB rejects empty *list* metadata; tagless notes must index
        with tags encoded as '' (regression for the ingest failure)."""
        from rag.ingester.vault import chunk_vault_file

        note = tmp_path / "tagless.md"
        note.write_text("# Tagless\n\nNo frontmatter at all.")
        chunks = chunk_vault_file(str(note), str(tmp_path))

        assert len(chunks) == 1
        meta = chunks[0]["metadata"]
        assert meta["tags"] == ""  # not []
        assert not isinstance(meta["tags"], list)
        assert meta["workspace"] == "default"

    def test_upsert_accepts_tagged_and_tagless(self, tmp_path, fake_embeddings, in_memory_client):
        """The full upsert path (as used by ingest_file) must tolerate notes
        with and without frontmatter tags — previously tagless notes raised
        "Expected metadata list value for key 'tags' to be non-empty"."""
        import rag.vectorstore as vs_mod

        original_store = vs_mod.vectorstore
        vs_mod.vectorstore = vs_mod.LocalBrainVectorStore(client=in_memory_client)

        try:
            from rag.ingester.vault import ingest_file

            tagged = tmp_path / "tagged.md"
            tagged.write_text("---\ntitle: T\ntags: [a]\n---\n# T\n\nContent.")
            tagless = tmp_path / "tagless.md"
            tagless.write_text("# No tags\n\nContent.")

            assert ingest_file(str(tagged), str(tmp_path)) > 0
            assert ingest_file(str(tagless), str(tmp_path)) > 0
        finally:
            vs_mod.vectorstore = original_store

    def test_string_tags_coerced_to_list(self, tmp_path):
        """Regression: `tags: project-alpha` (YAML scalar) must NOT be stored
        and iterated as an individual-character string."""
        from rag.ingester.vault import chunk_vault_file

        note = tmp_path / "strtags.md"
        note.write_text("---\nworkspace: work\ntags: project-alpha\n---\n# S\n\nBody.")
        chunks = chunk_vault_file(str(note), str(tmp_path))

        assert chunks, "expected at least one chunk"
        tags = chunks[0]["metadata"]["tags"]
        assert isinstance(tags, list), f"tags should be a list, got {type(tags)}"
        assert tags == ["project-alpha"]


class TestIngestVault:
    def test_counts_all_notes(self, sample_vault_path, fake_embeddings):
        from rag.ingester.vault import ingest_vault as _ingest_vault

        count = _ingest_vault(sample_vault_path)
        assert count > 0
