"""LangGraph agent graph.

``build_graph()`` produces a compiled ``StateGraph`` wired as:

    START → router → [retrieve | direct]
        retrieve → grade_docs → [any relevant → generate → reflect → [grounded → END | rewrite]]
        direct → generate (no context) → END

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
from rag.state import AgentState

_REWRITE_MAX = 3


def build_graph(llm: BaseChatModel) -> StateGraph:
    """Wire the agent nodes into a compiled LangGraph with checkpointing."""
    nodes = create_nodes(llm)
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("router", nodes["router"])
    graph.add_node("retrieve", nodes["retrieve"])
    graph.add_node("grade_docs", nodes["grade_docs"])
    graph.add_node("query_rewrite", nodes["query_rewrite"])
    graph.add_node("generate", nodes["generate"])
    graph.add_node("reflect", nodes["reflect"])

    # Edges
    graph.add_edge(START, "router")

    graph.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "retrieve": "retrieve",
            "direct": "generate",
            "tool": "generate",  # deferred — treat as generate for now
        },
    )

    graph.add_edge("retrieve", "grade_docs")

    graph.add_conditional_edges(
        "grade_docs",
        _after_grade,
        {
            "rewrite": "query_rewrite",
            "generate": "generate",
        },
    )

    graph.add_edge("query_rewrite", "retrieve")  # cycle back to retrieve

    graph.add_edge("generate", "reflect")

    graph.add_conditional_edges(
        "reflect",
        _after_reflect,
        {
            "rewrite": "query_rewrite",
            "end": END,
        },
    )

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
    """If the answer is grounded, return it. Otherwise rewrite and try once
    more (up to the rewrite limit)."""
    if state.get("grounded") is True:
        return "end"
    if state.get("rewrite_count", 0) >= _REWRITE_MAX:
        return "end"
    return "rewrite"


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
