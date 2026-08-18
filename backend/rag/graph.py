"""LangGraph agent graph.

``build_graph()`` produces a compiled ``StateGraph`` wired as:

    START → router → [conditional]
      "retrieve" → retrieve → filter_docs → grade_docs → [conditional]
        any relevant → generate → reflect → [conditional]
          grounded → END
          not grounded → guard_answer → END (rewrite in place, once)
        none relevant → query_rewrite → retrieve (cycle, max 3)
      "fastpath" → retrieve → generate → END  (skip grade/rewrite/reflect)
      "direct"   → generate (no context) → END
      "tool"     → tool_search → generate → END

The graph is parameterized by an LLM.  ``get_agent_graph()`` is the module-
level convenience function that builds it once with the default LLM from
settings.
"""
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from rag.llm_factory import get_llm
from rag.nodes import create_nodes
from rag.state import AgentState, REWRITE_MAX

_GENERATE_ROUTES = {"direct", "fastpath", "tool"}  # skip reflection on these


def build_graph(llm: BaseChatModel) -> StateGraph:
    """Wire the agent nodes into a compiled LangGraph with checkpointing."""
    nodes = create_nodes(llm)
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("router", nodes["router"])
    graph.add_node("retrieve", nodes["retrieve"])
    graph.add_node("filter_docs", nodes["filter_docs"])
    graph.add_node("grade_docs", nodes["grade_docs"])
    graph.add_node("query_rewrite", nodes["query_rewrite"])
    graph.add_node("generate", nodes["generate"])
    graph.add_node("reflect", nodes["reflect"])
    graph.add_node("guard_answer", nodes["guard_answer"])
    graph.add_node("tool_search", nodes["tool_search"])

    # Edges
    graph.add_edge(START, "router")

    graph.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "retrieve": "retrieve",
            "fastpath": "retrieve",
            "direct": "generate",
            "tool": "tool_search",
        },
    )

    graph.add_edge("tool_search", "generate")

    # retrieve → filter (tag-aware) → grade only in the slow path;
    # fastpath jumps straight to generate
    graph.add_conditional_edges(
        "retrieve",
        lambda state: "generate" if state.get("route") == "fastpath" else "filter_docs",
        {
            "generate": "generate",
            "filter_docs": "filter_docs",
        },
    )

    graph.add_edge("filter_docs", "grade_docs")

    graph.add_conditional_edges(
        "grade_docs",
        _after_grade,
        {
            "rewrite": "query_rewrite",
            "generate": "generate",
        },
    )

    graph.add_edge("query_rewrite", "retrieve")  # cycle back to retrieve

    # generate → skip reflection on direct/fastpath/tool routes
    graph.add_conditional_edges(
        "generate",
        lambda state: (
            "end" if state.get("route") in _GENERATE_ROUTES else "reflect"
        ),
        {
            "reflect": "reflect",
            "end": END,
        },
    )

    graph.add_conditional_edges(
        "reflect",
        _after_reflect,
        {
            "guard": "guard_answer",
            "end": END,
        },
    )

    graph.add_edge("guard_answer", END)

    return graph.compile(checkpointer=MemorySaver())


# ---------------------------------------------------------------------------
# Conditional edge helpers
# ---------------------------------------------------------------------------

def _after_grade(state: AgentState) -> str:
    """If at least one document is relevant, proceed to generate. Otherwise
    rewrite the query (with cycle guard)."""
    relevant = any(g.get("relevant") for g in state.get("retrieval_grades", []))
    if relevant:
        return "generate"
    if state.get("rewrite_count", 0) >= _REWRITE_MAX:
        return "generate"  # give up rewriting, generate with whatever we have
    return "rewrite"


def _after_reflect(state: AgentState) -> str:
    """If the answer is grounded, return it. Otherwise let guard_answer do one
    in-place repair; guard_answer always terminates, so no re-entry happens."""
    if state.get("grounded") is True:
        return "end"
    if not state.get("documents") and not state.get("retrieval_grades"):
        # Chit-chat/direct answer with no context to repair against: return.
        return "end"
    return "guard"


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_agent_graph = None


def get_agent_graph(llm: Optional[BaseChatModel] = None):
    """Return a compiled agent graph (rebuilt when ``llm`` is provided)."""
    global _agent_graph
    if llm is not None:
        _agent_graph = build_graph(llm)
    if _agent_graph is None:
        _agent_graph = build_graph(get_llm())
    return _agent_graph