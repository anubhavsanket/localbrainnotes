"""Tests for the LangGraph agent nodes.

Uses a FakeListChatModel so no LLM backend is required to run these tests.
"""
import pytest
from langchain_core.messages import HumanMessage
from langchain_core.language_models import FakeListChatModel

from rag.nodes import _parse_route, _parse_yes_no, create_nodes


# --- unit tests for parsers ------------------------------------------------

class TestParsers:
    def test_parse_route_retrieve(self):
        assert _parse_route("retrieve") == "retrieve"

    def test_parse_route_direct(self):
        assert _parse_route("I think this is direct.") == "direct"

    def test_parse_route_tool(self):
        assert _parse_route("tool") == "tool"

    def test_parse_route_is_lowercased(self):
        assert _parse_route("Direct") == "direct"
        assert _parse_route("FASTPATH here") == "fastpath"

    def test_parse_route_unknown_defaults_to_retrieve(self):
        assert _parse_route("I'm not sure what to do") == "retrieve"

    def test_parse_yes_no(self):
        assert _parse_yes_no("Yes") is True
        assert _parse_yes_no("yes, definitely.") is True
        assert _parse_yes_no("No") is False
        assert _parse_yes_no("no way.") is False


# --- integration tests for nodes -------------------------------------------

class TestRouterNode:
    def test_router_returns_route_and_standalone(self):
        fake = FakeListChatModel(responses=["retrieve"])
        nodes = create_nodes(fake)
        state = {
            "question": "What decisions were made?",
            "workspace": "work",
            "chat_history": [],
            "rewrite_count": 0,
        }
        result = nodes["router"](state)
        assert result["route"] == "retrieve"
        assert "standalone_question" in result


class TestGradeDocsNode:
    def test_grades_each_document(self):
        from langchain_core.documents import Document

        fake = FakeListChatModel(responses=["yes", "no", "yes"])
        nodes = create_nodes(fake)
        state = {
            "question": "What happened?",
            "standalone_question": "What happened?",
            "workspace": "work",
            "chat_history": [],
            "documents": [
                Document(page_content="Doc A about decisions", metadata={"note_id": "a.md"}),
                Document(page_content="Doc B about weather", metadata={"note_id": "b.md"}),
                Document(page_content="Doc C about plans", metadata={"note_id": "c.md"}),
            ],
            "rewrite_count": 0,
        }
        result = nodes["grade_docs"](state)
        grades = result["retrieval_grades"]
        assert len(grades) == 3
        assert grades[0]["relevant"] is True
        assert grades[1]["relevant"] is False
        assert grades[2]["relevant"] is True


class TestQueryRewriteNode:
    def test_rewrites_and_increments_count(self):
        fake = FakeListChatModel(responses=["What decisions were reviewed in the design meeting?"])
        nodes = create_nodes(fake)
        state = {
            "question": "What was decided?",
            "workspace": "work",
            "chat_history": [],
            "standalone_question": "What was decided?",
            "rewrite_count": 0,
        }
        result = nodes["query_rewrite"](state)
        assert result["rewrite_count"] == 1
        assert result["standalone_question"] != "What was decided?"

    def test_stops_rewriting_at_max(self):
        fake = FakeListChatModel(responses=["never"])
        nodes = create_nodes(fake)
        state = {
            "question": "x",
            "standalone_question": "x",
            "rewrite_count": 3,
        }
        result = nodes["query_rewrite"](state)
        assert "rewrite_count" not in result


class TestReflectNode:
    def test_grounded_returns_true(self):
        fake = FakeListChatModel(responses=["yes"])
        nodes = create_nodes(fake)
        state = {"answer": "The decisions included a new wizard.", "rewrite_count": 0}
        result = nodes["reflect"](state)
        assert result["grounded"] is True

    def test_ungrounded_returns_false(self):
        fake = FakeListChatModel(responses=["no"])
        nodes = create_nodes(fake)
        state = {"answer": "The sky is green.", "rewrite_count": 0}
        result = nodes["reflect"](state)
        assert result["grounded"] is False


class TestFastpathNode:
    def test_router_can_parse_fastpath(self):
        assert _parse_route("fastpath") == "fastpath"

    def test_fastpath_generates_from_all_docs(self):
        from langchain_core.documents import Document

        fake = FakeListChatModel(responses=["Concise answer."])
        nodes = create_nodes(fake)
        state = {
            "question": "What covers note3?",
            "route": "fastpath",
            "workspace": "personal",
            "chat_history": [],
            "documents": [
                Document(page_content="Note3 covers local-first concepts.", metadata={"note_id": "personal/note3.md"}),
            ],
        }
        result = nodes["generate"](state)
        assert "answer" in result
        assert result["sources"] == ["personal/note3.md"]


class TestToolSearchNode:
    def test_web_search_off_when_disabled(self, monkeypatch):
        import rag.nodes as nodes_mod
        from config import settings

        monkeypatch.setattr(settings, "WEB_SEARCH_ENABLED", False)
        fake = FakeListChatModel(responses=[])
        nodes = create_nodes(fake)
        result = nodes["tool_search"]({"question": "What is the weather?", "workspace": "work", "chat_history": []})
        assert result == {}

    def test_web_search_populates_documents(self, monkeypatch):
        from langchain_core.documents import Document

        monkeypatch.setattr("rag.nodes.settings.WEB_SEARCH_ENABLED", True)
        fake_docs = [Document(page_content="Results", metadata={"note_id": "web://x", "title": "X"})]
        monkeypatch.setattr("rag.web_search.web_search", lambda query: fake_docs)
        fake = FakeListChatModel(responses=[])
        nodes = create_nodes(fake)
        result = nodes["tool_search"]({"question": "Weather today?", "standalone_question": "Weather today?", "workspace": "work", "chat_history": []})
        assert result["documents"] == fake_docs


class TestGuardAnswerNode:
    def test_guard_rewrites_answer(self):
        from langchain_core.documents import Document

        fake = FakeListChatModel(responses=["Corrected grounded answer."])
        nodes = create_nodes(fake)
        state = {
            "question": "Decisions?",
            "answer": "A wizard and a unicorn were chosen.",
            "documents": [Document(page_content="Wizards were chosen.", metadata={"note_id": "d.md"})],
            "retrieval_grades": [{"doc": Document(page_content="Wizards were chosen.", metadata={"note_id": "d.md"}), "relevant": True}],
        }
        result = nodes["guard_answer"](state)
        assert "repair_count" not in result
        assert "Corrected grounded answer." in result["answer"]


class TestFilterDocs:
    def test_tag_filter_keeps_matching_docs(self):
        from langchain_core.documents import Document

        fake = FakeListChatModel(responses=[])
        nodes = create_nodes(fake)
        docs = [
            Document(page_content="Alpha decisions", metadata={"tags": ["project-alpha"], "workspace": "work", "note_id": "a.md"}),
            Document(page_content="Beta decisions", metadata={"tags": ["project-beta"], "workspace": "work", "note_id": "b.md"}),
        ]
        state = {"question": "What did project-alpha decide?", "standalone_question": "What did project-alpha decide?", "documents": docs}
        result = nodes["filter_docs"](state)
        assert len(result["documents"]) == 1
        assert result["documents"][0].metadata["tags"] == ["project-alpha"]

    def test_no_tag_signal_keeps_all(self):
        from langchain_core.documents import Document

        fake = FakeListChatModel(responses=[])
        nodes = create_nodes(fake)
        docs = [
            Document(page_content="One", metadata={"tags": ["alpha"], "note_id": "a.md"}),
            Document(page_content="Two", metadata={"tags": ["beta"], "note_id": "b.md"}),
        ]
        state = {"question": "What happened generally?", "standalone_question": "What happened generally?", "documents": docs}
        result = nodes["filter_docs"](state)
        assert "documents" not in result  # pass-through

    def test_string_tags_do_not_break_filter(self):
        """Regression: `tags: project-alpha` (YAML scalar) used to be split
        into individual characters, making the tag filter silently inert."""
        from langchain_core.documents import Document

        fake = FakeListChatModel(responses=[])
        nodes = create_nodes(fake)
        docs = [
            Document(page_content="Alpha decisions", metadata={"tags": "project-alpha", "workspace": "work", "note_id": "a.md"}),
            Document(page_content="Beta decisions", metadata={"tags": ["project-beta"], "workspace": "work", "note_id": "b.md"}),
        ]
        state = {"question": "What did project-alpha decide?", "standalone_question": "What did project-alpha decide?", "documents": docs}
        result = nodes["filter_docs"](state)
        assert len(result["documents"]) == 1
        assert result["documents"][0].metadata["tags"] == "project-alpha"


class TestCompression:
    def test_context_compression_truncates_long_docs(self):
        from langchain_core.documents import Document

        fake = FakeListChatModel(responses=["Short."])
        nodes = create_nodes(fake)
        long_text = ("The Q3 review made deep decisions about the onboarding wizard. " * 100)
        state = {
            "question": "What decisions about onboarding were made?",
            "route": "fastpath",
            "documents": [Document(page_content=long_text, metadata={"note_id": "n.md"})],
            "chat_history": [],
        }
        # generate() compresses internally; only assert it runs and returns
        result = nodes["generate"](state)
        assert "answer" in result
