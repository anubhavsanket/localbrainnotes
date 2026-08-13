"""LangGraph node functions.

Each node is a pure ``(AgentState → dict)`` function.  ``create_nodes(llm)``
returns a dict of such functions with the LLM captured in their closures, so
the graph can be built once and the LLM swapped out without rebuilding.
"""
import re
from typing import Any, Callable

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from rag.state import AgentState

_REWRITE_MAX = 3

# ---------------------------------------------------------------------------
# Prompt templates (kept module-level so they are easy to A/B test)
# ---------------------------------------------------------------------------

_ROUTER_SYSTEM = """\
You are a routing assistant for a knowledge retrieval system.

Given the user's conversation history and their latest question, decide which
category the question falls into:

  (A) retrieve — the question needs information from a document vault / knowledge base.
  (B) direct — the question is casual chit-chat, small talk, or can be answered
      from the conversation history alone without needing external documents.
  (C) tool   — the question requires a live web search or external API call
      (NOT implemented yet — use only when the user explicitly asks for
      current events, live stock prices, weather, etc.).

Reply with EXACTLY one word: retrieve, direct, or tool."""

_CONDENSE_SYSTEM = """\
Given the following conversation and a follow-up question, rephrase the
follow-up as a standalone question in its original language. If there is no
prior conversation, just repeat the question as-is.

Chat History:
{chat_history}

Follow-up: {question}

Standalone question:"""

_GRADE_SYSTEM = """\
You are a relevance grader. You will be given a document and a question.

Answer EXACTLY "yes" if the document contains information relevant to answering
the question. Answer EXACTLY "no" if it does not.

Document:
{document}

Question: {question}"""

_REWRITE_SYSTEM = """\
You are a question rewriter. The current search did not find useful results
for the user's question. Rewrite the question to be broader or more specific
so a document search is more likely to succeed. Reply with ONLY the rewritten
question, nothing else.

Question: {question}"""

_ANSWER_SYSTEM = """\
You are an assistant for question-answering tasks. Use the following pieces of
retrieved context to answer the question. If you don't know the answer, just
say that you don't know — do not try to make up an answer. Keep the answer
concise (three sentences maximum).

Context:
{context}"""

_REFLECT_SYSTEM = """\
You are a response quality checker. Given a question and an answer, determine
whether the answer is fully grounded in the provided context.

Answer EXACTLY "yes" if the answer is supported by the context or if no
external context was needed (chit-chat). Answer "no" if the answer makes claims
that are not in the context and could be hallucinated.

Answer: {answer}"""

# ---------------------------------------------------------------------------
# Helper: parse a single-word yes/no or category from LLM text
# ---------------------------------------------------------------------------

_YES_RE = re.compile(r"\b(yes)\b", re.IGNORECASE)
_NO_RE = re.compile(r"\b(no)\b", re.IGNORECASE)
_ROUTE_RE = re.compile(r"\b(retrieve|direct|tool)\b", re.IGNORECASE)


def _parse_route(text: str) -> str:
    m = _ROUTE_RE.search(text)
    return m.group(1) if m else "retrieve"  # default: try retrieval


def _parse_yes_no(text: str) -> bool:
    return bool(_YES_RE.search(text))


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------

def create_nodes(llm: BaseChatModel) -> dict[str, Callable[[AgentState], dict[str, Any]]]:
    """Return a dict of node functions with ``llm`` bound in each closure."""
    parser = StrOutputParser()
    condense_chain = ChatPromptTemplate.from_template(_CONDENSE_SYSTEM) | llm | parser
    grade_chain = ChatPromptTemplate.from_template(_GRADE_SYSTEM) | llm | parser
    rewrite_chain = ChatPromptTemplate.from_template(_REWRITE_SYSTEM) | llm | parser
    answer_chain = ChatPromptTemplate.from_messages([
        ("system", _ANSWER_SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]) | llm | parser
    reflect_chain = ChatPromptTemplate.from_template(_REFLECT_SYSTEM) | llm | parser

    def _format_chat_history(messages: list) -> str:
        parts = []
        for msg in messages:
            role = "Human" if getattr(msg, "type", "") == "human" else "Assistant"
            parts.append(f"{role}: {msg.content}")
        return "\n".join(parts)

    def _format_docs(docs: list[Document]) -> str:
        parts = []
        for doc in docs:
            note_id = doc.metadata.get("note_id", "unknown")
            parts.append(f"[{note_id}]:\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    # ---- node: router -----------------------------------------------------
    def router(state: AgentState) -> dict:
        question = state["question"]
        chat_history = _format_chat_history(state.get("chat_history", []))
        route_text = llm.invoke([
            SystemMessage(content=_ROUTER_SYSTEM),
            HumanMessage(content=f"Chat History:\n{chat_history}\n\nQuestion: {question}"),
        ]).content
        route = _parse_route(route_text)

        # Build standalone question (for downstream retrieval even when routing)
        if chat_history.strip():
            standalone = condense_chain.invoke({"chat_history": chat_history, "question": question})
        else:
            standalone = question

        return {
            "route": route,
            "standalone_question": standalone,
            "chat_history": state.get("chat_history", []),
        }

    # ---- node: retrieve ---------------------------------------------------
    def retrieve(state: AgentState) -> dict:
        # Defer import to avoid circular dep — vectorstore.py → embedder → ...
        from rag.vectorstore import vectorstore

        search_type = None  # uses settings default
        retriever = vectorstore.get_retriever(
            workspace=state.get("workspace"),
            filter={"workspace": state["workspace"]} if state.get("workspace") else None,
        )
        docs = retriever.invoke(state["standalone_question"])
        return {"documents": docs}

    # ---- node: grade_docs -------------------------------------------------
    def grade_docs(state: AgentState) -> dict:
        grades = []
        for doc in state["documents"]:
            response = grade_chain.invoke({
                "document": doc.page_content,
                "question": state["standalone_question"],
            })
            grades.append({"doc": doc, "relevant": _parse_yes_no(response)})
        return {"retrieval_grades": grades}

    # ---- node: query_rewrite ----------------------------------------------
    def query_rewrite(state: AgentState) -> dict:
        count = state.get("rewrite_count", 0)
        if count >= _REWRITE_MAX:
            return {}  # cycle guard — stop rewriting
        rewritten = rewrite_chain.invoke({"question": state["standalone_question"]})
        return {
            "standalone_question": rewritten,
            "rewrite_count": count + 1,
        }

    # ---- node: generate ---------------------------------------------------
    def generate(state: AgentState) -> dict:
        # Build context from relevant docs only (empty list if none)
        relevant = [
            g["doc"] for g in state.get("retrieval_grades", [])
            if g.get("relevant", True)  # include all when called without grading
        ]
        if not relevant:
            relevant = state.get("documents", [])

        context = _format_docs(relevant) if relevant else ""
        answer = answer_chain.invoke({
            "context": context,
            "question": state["question"],
            "chat_history": state.get("chat_history", []),
        })

        sources = list(dict.fromkeys(
            doc.metadata.get("note_id", "unknown") for doc in relevant
        ))
        return {"answer": answer, "sources": sources}

    # ---- node: reflect ----------------------------------------------------
    def reflect(state: AgentState) -> dict:
        response = reflect_chain.invoke({
            "answer": state.get("answer", ""),
        })
        return {"grounded": _parse_yes_no(response)}

    return {
        "router": router,
        "retrieve": retrieve,
        "grade_docs": grade_docs,
        "query_rewrite": query_rewrite,
        "generate": generate,
        "reflect": reflect,
    }
