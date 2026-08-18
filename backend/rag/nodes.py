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

from config import settings
from rag.state import AgentState, REWRITE_MAX


def _as_tags(md_tags: Any) -> list[str]:
    """Normalize a metadata ``tags`` value to a list of strings.

    Older chunks may carry a YAML scalar string (``tags: project-alpha``) or an
    empty string instead of a list — both would break ``", ".join(...)`` and the
    tag filter. This coerces every shape to a clean list.
    """
    if not md_tags:
        return []
    if isinstance(md_tags, str):
        return [t.strip() for t in md_tags.split(",") if t.strip()]
    if isinstance(md_tags, (list, tuple, set)):
        return [str(t).strip() for t in md_tags if str(t).strip()]
    return [str(md_tags).strip()]

# ---------------------------------------------------------------------------
# Prompt templates (kept module-level so they are easy to A/B test)
# ---------------------------------------------------------------------------

_ROUTER_SYSTEM = """\
You are a routing assistant for a knowledge retrieval system.

Given the user's conversation history and their latest question, decide which
category the question falls into:

  (A) retrieve — the question needs information from a document vault /
      knowledge base and may need multi-hop reasoning or follow-ups.
  (B) fastpath — the question is a simple, direct factual lookup that a single
      vault search can answer (e.g. "what does note3 cover?", "what tag is on
      the sync note?"). Use this when NO multi-hop reasoning, rewrites, or
      reflection are needed.
  (C) direct — the question is casual chit-chat, small talk, general knowledge
      not tied to the vault, or can be answered from conversation history alone.
  (D) tool   — the question requires a live web search or external API call
      (current events, live stock prices, weather, today's date, etc.).

Reply with exactly ONE word and nothing else: retrieve, fastpath, direct, or tool."""

_CONDENSE_SYSTEM = """\
Given the following conversation and a follow-up question, rephrase the
follow-up as a standalone question in its original language. If there is no
prior conversation, just repeat the question as-is.

Chat History:
{chat_history}

Follow-up: {question}

Standalone question:"""

_GRADE_SYSTEM = """\
You are a relevance grader. A document is RELEVANT if it contains information
that helps answer the question, even if it does not use the exact wording of
the question. Respond with ONLY a single word: yes or no.

Question: {question}

Document:
{document}"""

_REWRITE_SYSTEM = """\
You are a question rewriter. The current search did not find useful results
for the user's question. Rewrite the question to be broader or more specific
so a document search is more likely to succeed. Reply with ONLY the rewritten
question, nothing else.

Question: {question}"""

_ANSWER_SYSTEM = """\
You are a concise assistant in a local-first Markdown vault app. Use the
retrieved context below to answer the question. Follow these rules strictly:

1. Answer directly — no preamble, no meta-commentary. Do not reference
   retrieval, tools, reflection, or the reasoning process. Never write
   "Based on the context provided", "According to your notes", "In summary",
   or similar phrasings.
2. Base every claim on the context. Never invent note names, file paths,
   wikilinks, dates, numbers, or people.
3. If you attribute a claim to a source, use ONLY the bracketed note name as it
   literally appears in the context (e.g. [note1.md]). If no source is shown,
   do not cite one.
4. If the context has no relevant information, respond with exactly:
   "I don't have enough context to answer this."
5. Keep the answer concise — three sentences maximum.

Context:
{context}"""

_REFLECT_SYSTEM = """\
You are a response quality checker. Given a question, retrieved context, and an answer, determine
whether the answer is fully grounded in the provided context and directly answers the question.

Answer EXACTLY "yes" if the answer is supported by the context or if no
external context was needed (casual chit-chat). Answer "no" if the answer makes claims
that are not in the context and could be hallucinated. Respond with ONLY a
single word: yes or no.

Question: {question}

Context:
{context}

Answer: {answer}"""

_GUARD_SYSTEM = """\
You are a strict groundedness guard. Compare the Answer against the provided
Context. Rewrite the Answer so that EVERY factual claim is explicitly supported
by the Context:

1. Delete or correct any claim that is NOT in the Context — do not keep it.
2. Keep all claims that ARE supported, preserving their wording where possible.
3. Do not add new facts, examples, or dates that are not in the Context.
4. Keep the answer concise (three sentences maximum).
5. If the Answer is already fully grounded, return it unchanged.
6. Output ONLY the rewritten answer text — no extra commentary, no preamble.

Question: {question}

Context:
{context}

Answer:
{answer}"""

_DIRECT_SYSTEM = """\
You are a friendly assistant for a local-first Markdown note app. The user is
chatting with you directly (no vault context was retrieved). Follow these rules:

1. Answer the question directly and concisely (three sentences maximum).
2. Answer chit-chat and general-knowledge questions honestly. If something is
   a current event / live fact you cannot verify, say you do not have that
   real-time information.
3. Never mention retrieval, tools, the vault, or the reasoning process.
4. If the question asks about the user's notes, say you don't have access to
   the vault right now and suggest re-asking with their workspace selected."""

# ---------------------------------------------------------------------------
# Helper: parse a single-word yes/no or category from LLM text
# ---------------------------------------------------------------------------

_YES_RE = re.compile(r"\b(yes)\b", re.IGNORECASE)
_NO_RE = re.compile(r"\b(no)\b", re.IGNORECASE)
_ROUTE_RE = re.compile(r"\b(retrieve|fastpath|direct|tool)\b", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Helper: strip agent chatter patterns from generated text
# ---------------------------------------------------------------------------

_AGENT_CHATTER_PATTERNS = [
    r"^\s*Based on the context provided[^\n]*\n?",
    r"^\s*According to the context[^\n]*\n?",
    r"^\s*As per the context[^\n]*\n?",
    r"^\s*Based on the retrieved context[^\n]*\n?",
    r"^\s*I do not have enough context to answer this question[^\n]*\n?",
    r"^\s*Sure, here is the answer[^\n]*\n?",
    r"^\s*Here is what I found[^\n]*\n?",
    r"^\s*The context suggests[^\n]*\n?",
    r"^\s*In summary[^\n]*\n?",
    r"^\s*In conclusion[^\n]*\n?",
    r"^\s*Based on the information provided[^\n]*\n?",
    r"^\s*Looking at the context[^\n]*\n?",
    r"^\s*From the context[^\n]*\n?",
]


def _strip_agent_chatter(text: str) -> str:
    """Remove common small-model phrasing habits that RAGAS penalizes."""
    result = text
    for pattern in _AGENT_CHATTER_PATTERNS:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE | re.MULTILINE)
    # Remove excessive blank lines
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def _parse_route(text: str) -> str:
    m = _ROUTE_RE.search(text)
    if m:
        # Normalize to lowercase — the LLM may return "Direct" or "FASTPATH",
        # but the graph edge keys are lowercase.
        return m.group(1).lower()
    return "retrieve"  # default: try retrieval


def _parse_yes_no(text: str) -> bool:
    """Decide yes/no from the model's answer.

    Trusts the first word when it is an unambiguous yes/no, and only falls back
    to a whole-text scan otherwise (guards against trailing ramble flipping
    the verdict).
    """
    stripped = (text or "").strip()
    if not stripped:
        return False
    first = stripped.split()[0].lower().strip(".,!?;:\"'()")
    if first in ("yes", "no"):
        return first == "yes"
    return bool(_YES_RE.search(stripped))


def _stream_writer() -> Any | None:
    """Best-effort access to LangGraph's stream writer.

    Returns ``None`` when the node is running outside a streaming run (e.g. a
    plain ``.invoke()`` or a direct unit-test call), so callers can always
    safely guard on ``writer is not None``.
    """
    try:
        from langgraph.config import get_stream_writer

        return get_stream_writer()
    except Exception:
        return None


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
    direct_chain = ChatPromptTemplate.from_messages([
        ("system", _DIRECT_SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]) | llm | parser
    reflect_chain = ChatPromptTemplate.from_template(_REFLECT_SYSTEM) | llm | parser
    guard_chain = ChatPromptTemplate.from_template(_GUARD_SYSTEM) | llm | parser

    def _format_chat_history(messages: list) -> str:
        parts = []
        for msg in messages:
            role = "Human" if getattr(msg, "type", "") == "human" else "Assistant"
            parts.append(f"{role}: {msg.content}")
        return "\n".join(parts)

    def _format_docs(docs: list[Document]) -> str:
        """Render retrieved documents as an LLM-friendly context block.

        Each chunk is anchored with its source path + metadata so the small
        model can attribute claims instead of inventing connections.
        """
        parts = []
        for doc in docs:
            md = doc.metadata
            note_id = md.get("note_id", "unknown")
            title = md.get("title")
            heading_path = md.get("heading_path") or []
            workspace = md.get("workspace") or "default"
            tags = _as_tags(md.get("tags"))
            anchor = f"[{note_id}]"
            extras = []
            if title:
                extras.append(f"Title: {title}")
            if workspace and workspace != "default":
                extras.append(f"Workspace: {workspace}")
            if heading_path:
                extras.append(f"Section: {' > '.join(heading_path)}")
            if tags:
                extras.append(f"Tags: {', '.join(tags)}")
            if extras:
                anchor += " (" + "; ".join(extras) + ")"
            parts.append(f"{anchor}:\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    def _compress_context(docs: list[Document], question: str) -> list[Document]:
        """Trim over-long retrieved chunks to the most question-relevant
        sentences (a deterministic, LLM-free extractive compressor)."""
        budget = settings.CONTEXT_MAX_CHARS
        total = sum(len(d.page_content) for d in docs)
        if total <= budget or not docs:
            return docs
        # Question-bearing tokens (words ≥ 3 chars) used to score sentences.
        q_words = {w.lower() for w in re.findall(r"[A-Za-z0-9_]{3,}", question)}

        def _score(sentence: str) -> int:
            words = {w.lower() for w in re.findall(r"[A-Za-z0-9_]{3,}", sentence)}
            return len(words & q_words)

        per_doc_budget = max(int(budget / len(docs)), 200)  # at least 200 chars each
        compressed = []
        for doc in docs:
            text = doc.page_content
            if len(text) <= per_doc_budget:
                compressed.append(doc)
                continue
            sentences = re.split(r"(?<=[.!?])\s+", text)
            scored = sorted((( _score(s), s) for s in sentences), reverse=True)
            kept: list[str] = []
            used = 0
            for _score_val, sent in scored:
                if used + len(sent) > per_doc_budget:
                    break
                kept.append(sent)
                used += len(sent)
            if not kept:  # single huge sentence — hard-truncate
                kept = [text[:per_doc_budget].rsplit(" ", 1)[0]]
            compressed.append(
                Document(
                    page_content=" ".join(kept),
                    metadata={**doc.metadata, "compressed": True},
                )
            )
        return compressed

    def _relevant_docs(state: AgentState) -> list[Document]:
        """The documents that survived relevance grading (or all docs when no
        grading has run yet)."""
        if "retrieval_grades" in state:
            return [
                g["doc"] for g in state["retrieval_grades"]
                if g.get("relevant", False)
            ]
        return state.get("documents", [])

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

        retriever = vectorstore.get_retriever(
            workspace=state.get("workspace"),
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
        if count >= REWRITE_MAX:
            return {}  # cycle guard — stop rewriting
        rewritten = rewrite_chain.invoke({"question": state["standalone_question"]})
        return {
            "standalone_question": rewritten,
            "rewrite_count": count + 1,
        }

    # ---- node: generate ---------------------------------------------------
    def generate(state: AgentState) -> dict:
        route = state.get("route", "retrieve")

        # Build context from relevant docs (retrieve/fastpath/tool) only.
        relevant = _relevant_docs(state)
        if route == "fastpath":
            # Fast path still gets all retrieved docs but skips grading — so
            # compress to keep the model focused and under the context budget.
            relevant = _compress_context(relevant, state["question"])

        # "direct" / chit-chat always uses the no-context prompt; a "tool"
        # query with an empty web result set degrades to it too.
        if route == "direct" or (route == "tool" and not relevant):
            chain = direct_chain
        else:
            chain = answer_chain

        context = _format_docs(relevant) if relevant else ""
        sources = list(dict.fromkeys(
            doc.metadata.get("note_id", "unknown") for doc in relevant
        ))

        writer = _stream_writer()

        # Stream the answer through the LLM and forward each token as a custom
        # stream event. The SSE endpoint reads these via astream(stream_mode=[
        # "custom", ...]) — LangGraph's astream_events v2 does NOT surface token
        # chunks for nodes that call their chains with .invoke().
        parts: list[str] = []
        for chunk in chain.stream({
            "context": context,
            "question": state["question"],
            "chat_history": state.get("chat_history", []),
        }):
            if chunk:
                parts.append(chunk)
                if writer is not None:
                    writer({"kind": "token", "data": chunk})

        if writer is not None:
            writer({"kind": "sources", "data": sources})

        answer = _strip_agent_chatter("".join(parts))
        return {"answer": answer, "sources": sources}

    # ---- node: reflect ----------------------------------------------------
    def reflect(state: AgentState) -> dict:
        context = _format_docs(_relevant_docs(state))
        response = reflect_chain.invoke({
            "question": state.get("question", ""),
            "context": context,
            "answer": state.get("answer", ""),
        })
        return {"grounded": _parse_yes_no(response)}

    # ---- node: guard_answer -----------------------------------------------
    def guard_answer(state: AgentState) -> dict:
        """Grounding guard — rewrite the answer to drop claims not supported
        by the retrieved context (run only when reflection flagged the draft)."""
        context = _format_docs(_relevant_docs(state))
        answer = _strip_agent_chatter(guard_chain.invoke({
            "question": state.get("question", ""),
            "context": context,
            "answer": state.get("answer", ""),
        }))
        return {
            "answer": answer,
            "repair_count": state.get("repair_count", 0) + 1,
        }

    # ---- node: tool_search ------------------------------------------------
    def tool_search(state: AgentState) -> dict:
        """Live web search for the 'tool' route (graceful offline fallback)."""
        from rag.web_search import web_search

        query = state.get("standalone_question") or state.get("question", "")
        if not settings.WEB_SEARCH_ENABLED:
            return {}
        docs = web_search(query)
        return {"documents": docs}

    # ---- node: filter_docs ------------------------------------------------
    def filter_docs(state: AgentState) -> dict:
        """Metadata/tag-aware pre-filter before grading.

        If the question explicitly names a tag, workspace, title or path
        element found in the retrieved metadata vocabulary, drop chunks that
        do NOT carry it. When there is no tag signal at all, keep every doc
        (a conservative filter that never hurts recall).
        """
        question = state.get("standalone_question") or state.get("question", "")
        docs = state.get("documents", [])
        if not docs:
            return {}

        vocab: dict[str, list] = {"tags": [], "workspace": [], "title": [], "path": []}
        for d in docs:
            md = d.metadata
            vocab["tags"] += _as_tags(md.get("tags"))
            vocab["workspace"].append(str(md.get("workspace") or "default"))
            title = md.get("title")
            vocab["title"].append(str(title) if title else "")
            vocab["path"].append(str((md.get("path") or md.get("note_id", ""))))

        q_lower = question.lower()
        signals: list[str] = []
        for kw in sorted(set(t.lower() for t in vocab["tags"])):
            if kw and kw in q_lower and len(kw) >= 3:
                signals.append(kw)

        if not signals:
            return {}  # no tag signal → pass all documents through

        kept = [
            d for d in docs
            if any(s in str(d.metadata.get("tags", [])).lower() for s in signals)
        ]
        return {"documents": kept} if kept else {"documents": docs}  # keep all on 0 matches

    # ---- redirect helper --------------------------------------------------
    return {
        "router": router,
        "retrieve": retrieve,
        "filter_docs": filter_docs,
        "grade_docs": grade_docs,
        "query_rewrite": query_rewrite,
        "generate": generate,
        "reflect": reflect,
        "guard_answer": guard_answer,
        "tool_search": tool_search,
    }
