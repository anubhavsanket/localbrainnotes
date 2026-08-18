"""LangGraph agent state schema.

Every node in the graph reads from and writes back to a shared ``AgentState``
dictionary. ``answer`` is the *only* field the caller ever consumes; the rest
are internal bookkeeping the agent uses to route, grade, rewrite, and verify.
"""
from typing import TypedDict

from langchain_core.documents import Document

# Query-rewrite cycle guard (shared by nodes.py and graph.py).
REWRITE_MAX = 3


class AgentState(TypedDict, total=False):
    """State flowing through the agent loop."""

    # --- inbound (from caller / API layer) ---------------------------------
    question: str  # original user question
    workspace: str  # workspace scope for retrieval
    chat_history: list  # LangChain BaseMessage objects for conversation context

    # --- internal ----------------------------------------------------------
    standalone_question: str  # after condense / rewrite
    route: str  # "retrieve" | "direct" | "tool" | "fastpath"
    documents: list[Document]  # retrieved docs (vault or web)
    retrieval_grades: list[dict]  # [{"doc": Document, "relevant": bool}, ...]
    rewrite_count: int  # cycle guard (max 3)
    repair_count: int  # groundedness-guard cycle guard (max 1)

    # --- outbound (to caller) ----------------------------------------------
    answer: str  # final or draft
    sources: list[str]  # deduplicated source note_ids
    grounded: bool | None  # reflection verdict (None until reflect runs)
