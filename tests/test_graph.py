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
