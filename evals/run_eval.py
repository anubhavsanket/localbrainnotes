"""RAGAS evaluation harness.

Runs the golden dataset through the agentic RAG pipeline and a naive
retrieve→generate baseline, then scores faithfulness, answer_relevancy,
context_recall, and context_precision with RAGAS.

The RAGAS *judge* LLM is OpenAI-compatible and configurable:

  - ``EVAL_LLM_BASE_URL`` + ``EVAL_LLM_API_KEY`` + ``EVAL_LLM_MODEL``
    (e.g. OpenRouter https://openrouter.ai/api/v1 or NVIDIA NIM
    https://integrate.api.nvidia.com/v1)
  - or plain ``OPENAI_API_KEY`` (fallback)

Embeddings for ``answer_relevancy`` come from the local Ollama
``nomic-embed-text`` model (offline-first).
"""
import json
import os
import re
import sys
import types
from pathlib import Path

# Ensure backend/ is importable when run from repo root
_backend = str(Path(__file__).resolve().parent.parent / "backend")
if _backend not in sys.path:
    sys.path.insert(0, _backend)


def _parse_only() -> str | None:
    """Return 'agentic'/'naive' when ``--only=X`` is passed, else None."""
    for arg in sys.argv[1:]:
        if arg.startswith("--only="):
            val = arg.split("=", 1)[1].strip().lower()
            if val in ("agentic", "naive"):
                return val
    return None


def _install_vertexai_shim() -> None:
    """RAGAS imports ``langchain_community.chat_models.vertexai``, which was
    removed from langchain-community >=0.4. ChatVertexAI is only referenced for
    membership in MULTIPLE_COMPLETION_SUPPORTED, so a placeholder class is safe.
    """
    if "langchain_community.chat_models.vertexai" in sys.modules:
        return
    mod = types.ModuleType("langchain_community.chat_models.vertexai")
    mod.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules["langchain_community.chat_models.vertexai"] = mod


_install_vertexai_shim()

from langchain_ollama import ChatOllama  # noqa: E402
from ragas.metrics._answer_relevance import AnswerRelevancy  # noqa: E402

from config import settings  # noqa: E402

DATA_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.json"

_METRICS = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]


# ---------------------------------------------------------------------------
# Judge wiring (OpenAI-compatible endpoint or plain OpenAI)
# ---------------------------------------------------------------------------


def _strip_markdown_fences(text: str) -> str:
    """Extract the JSON object from judge output.

    Small local models (e.g. phi4-mini) rarely return bare JSON. They add
    ```json ... ``` fences and/or a human prefix like ``Output: {...}``.
    RAGAS calls ``model_validate_json`` directly on the raw text, so any of
    those decorations crash answer_relevancy and friends → 0.0 everywhere.
    This extracts the first balanced ``{...}`` object (or a bare JSON array),
    falling back to the stripped text when nothing JSON-ish is found."""
    import re

    if not text:
        return text

    # 1) If it's already pure JSON (starts with { or [), just strip whitespace.
    stripped = text.strip()
    if stripped.startswith(("{", "[")):
        return stripped

    # 2) Grab everything between the first '{' and the LAST '}' (balanced-ish),
    #    which also skips ```json fences and "Output: " prefixes.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1].strip()

    # 3) Fallback: remove ``` markers if no braces at all.
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    return m.group(1).strip() if m else stripped


class FenceStrippingChatOllama(ChatOllama):
    """ChatOllama that strips markdown code fences from every generation.

    RAGAS calls ``model_validate_json`` directly on the judge's raw text, and
    small local models (phi4-mini) wrap their JSON in ```json ... ``` fences —
    which crashes the parser and silently yields 0.0 for every sample. This
    subclass removes the fences at the source, keeping the model a proper
    LangChain chat model so RAGAS's ``is_langchain_llm`` branch works.
    """

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        for gen in result.generations:
            gen.text = _strip_markdown_fences(gen.text)
        return result

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        result = await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        for gen in result.generations:
            gen.text = _strip_markdown_fences(gen.text)
        return result


# ---------------------------------------------------------------------------
# Local answer-relevancy gate
# ---------------------------------------------------------------------------
# RAGAS's stock AnswerRelevancy multiplies the cosine score by 0 whenever the
# *judge* marks the generated questions as noncommittal. The local phi4-mini
# judge flags virtually every answer as noncommittal=1 (even "State-based and
# operation-based."), zeroing relevancy for correct answers. We keep RAGAS's
# question-generation + cosine similarity, but replace the judge's
# noncommittal verdict with a deterministic regex over the final answer text.

_EVASIVE_RE = re.compile(
    r"\b(?:"
    r"(?:don'?t|do not|doesn'?t|does not) (?:know|have|understand)|"
    r"i'?m not sure|i am not sure|not sure|"
    r"no context|not enough context|"
    r"unable to (?:answer|determine)|cannot (?:answer|say)|"
    r"can'?t (?:answer|say)"
    r")\b",
    re.IGNORECASE,
)


def _looks_evasive(answer: str) -> bool:
    """Heuristic: an answer is evasive if it declines/hedges rather than
    committing to a factual response."""
    return bool(_EVASIVE_RE.search(answer or ""))


class LocalAnswerRelevancy(AnswerRelevancy):
    """AnswerRelevancy with a local, deterministic noncommittal gate.

    Mirrors RAGAS's ``_calculate_score`` but replaces the judge-supplied
    ``noncommittal`` flag with ``_looks_evasive(response)`` so the metric is
    robust to the local judge's misclassification.
    """

    def _calculate_score(self, answers, row):
        import numpy as np

        question = row["user_input"]
        gen_questions = [a.question for a in answers]
        if all(q == "" for q in gen_questions):
            return float("nan")
        cosine_sim = self.calculate_similarity(question, gen_questions)
        evasive = _looks_evasive(row.get("response", ""))
        return float(np.mean(cosine_sim)) * (0 if evasive else 1)


def _build_judge_llm():
    provider = settings.EVAL_JUDGE_PROVIDER or os.getenv("EVAL_JUDGE_PROVIDER", "ollama")
    model = settings.EVAL_LLM_MODEL

    if provider == "ollama":
        # Local, offline-first judge: deterministic (temp 0), free, no rate
        # limits. Consistent run-to-run comparisons.
        return FenceStrippingChatOllama(
            model=model,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0.0,
            num_predict=1024,
        )

    from langchain_openai import ChatOpenAI

    base_url = settings.EVAL_LLM_BASE_URL or os.getenv("EVAL_LLM_BASE_URL")
    api_key = settings.EVAL_LLM_API_KEY or os.getenv("EVAL_LLM_API_KEY")

    if base_url:
        if not api_key:
            raise ValueError(
                "EVAL_LLM_API_KEY must be set when EVAL_LLM_BASE_URL is configured."
            )
        return ChatOpenAI(
            model=model, api_key=api_key, base_url=base_url, temperature=0.0, timeout=120
        )

    api_key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "No judge LLM configured. Set EVAL_LLM_BASE_URL + EVAL_LLM_API_KEY "
            "(any OpenAI-compatible endpoint) or OPENAI_API_KEY in .env"
        )
    return ChatOpenAI(model=model, api_key=api_key, temperature=0.0, timeout=120)


def _build_judge_embeddings():
    from langchain_ollama import OllamaEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    # Scored with the local Ollama embedder (offline-first).
    return LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=settings.EMBEDDING_MODEL, base_url=settings.OLLAMA_BASE_URL)
    )


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------


def _run_agentic(graph, question: str) -> tuple[str, list[str]]:
    input_state = {
        "question": question,
        # No workspace scope: the golden questions span note1/note2 ("work")
        # and note3 ("personal"), so retrieval crosses workspaces.
        "workspace": None,
        "chat_history": [],
        "rewrite_count": 0,
    }
    result = graph.invoke(input_state, config={"configurable": {"thread_id": "eval"}})
    contexts = [doc.page_content for doc in result.get("documents", [])]
    return result.get("answer", ""), contexts


def _run_naive(llm, question: str) -> tuple[str, list[str]]:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    from rag.vectorstore import vectorstore
    from rag.nodes import _GRADE_SYSTEM, _ANSWER_SYSTEM, _parse_yes_no

    # Cross-workspace retrieval, mirroring the agentic eval setup.
    retriever = vectorstore.get_retriever(workspace=None, filter=None)
    docs = retriever.invoke(question)

    grade_chain = ChatPromptTemplate.from_template(_GRADE_SYSTEM) | llm | StrOutputParser()
    grades = [
        _parse_yes_no(grade_chain.invoke({"document": d.page_content, "question": question}))
        for d in docs
    ]
    relevant = [docs[i] for i, g in enumerate(grades) if g]
    context_text = "\n\n".join(d.page_content for d in relevant) if relevant else ""

    answer_chain = (
        ChatPromptTemplate.from_messages([
            ("system", _ANSWER_SYSTEM),
            ("human", "Question: {question}\nContext: {context}"),
        ])
        | llm
        | StrOutputParser()
    )
    ans = answer_chain.invoke({"question": question, "context": context_text})
    # Mirrors the agentic side: contexts = retrieved document text.
    return ans, [d.page_content for d in docs]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_evaluation(data_path: str | Path = DATA_PATH):
    try:
        from ragas import evaluate
        from ragas.metrics._faithfulness import faithfulness
        from ragas.metrics._context_recall import context_recall
        from ragas.metrics._context_precision import context_precision
        from ragas.run_config import RunConfig
        from datasets import Dataset
    except ImportError as e:
        print(f"Missing eval dependencies: {e}\n  pip install 'ragas>=0.4' datasets")
        return

    # Ollama can serve a few concurrent judge calls; with a generous per-job
    # timeout (600s) parallelism speeds up the run without false TimeoutErrors.
    max_workers = 3 if settings.EVAL_JUDGE_PROVIDER == "ollama" else 2
    timeout = 600 if settings.EVAL_JUDGE_PROVIDER == "ollama" else 120
    run_config = RunConfig(max_workers=max_workers, timeout=timeout, max_retries=5)

    # Use the local-gate relevancy metric: robust to the judge mislabeling
    # every answer as noncommittal (which zeroes the stock metric).
    metrics = [faithfulness, LocalAnswerRelevancy(), context_recall, context_precision]

    with open(data_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    # Generation LLM (the local model powering the pipelines).
    from rag.graph import get_agent_graph
    from rag.llm_factory import get_llm

    graph = get_agent_graph()
    llm = get_llm()

    questions = [item["question"] for item in benchmark_data]
    ground_truths = [item["ground_truth"] for item in benchmark_data]

    answers_ag, contexts_ag = [], []
    answers_nv, contexts_nv = [], []

    CACHE_GEN = Path(__file__).parent / "_gen_cache.json"
    if CACHE_GEN.exists():
        print(f"Loading cached pipeline outputs from {CACHE_GEN}...")
        cached = json.loads(CACHE_GEN.read_text(encoding="utf-8"))
        answers_ag = cached["answers_ag"]
        contexts_ag = cached["contexts_ag"]
        answers_nv = cached["answers_nv"]
        contexts_nv = cached["contexts_nv"]
    else:
        print(f"Running {len(questions)} questions through both pipelines...")
        for i, q in enumerate(questions, 1):
            ans, ctxs = _run_agentic(graph, q)
            answers_ag.append(ans)
            contexts_ag.append(ctxs)
            print(f"  [{i}/{len(questions)}] agentic done", flush=True)

        print("Generating naive baseline...")
        for i, q in enumerate(questions, 1):
            ans, ctxs = _run_naive(llm, q)
            answers_nv.append(ans)
            contexts_nv.append(ctxs)
            print(f"  [{i}/{len(questions)}] naive done", flush=True)

        CACHE_GEN.write_text(
            json.dumps(
                {
                    "answers_ag": answers_ag,
                    "contexts_ag": contexts_ag,
                    "answers_nv": answers_nv,
                    "contexts_nv": contexts_nv,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    dataset = Dataset.from_dict({
        "question": questions,
        "answer_ag": answers_ag,
        "answer_nv": answers_nv,
        "contexts_ag": contexts_ag,
        "contexts_nv": contexts_nv,
        "ground_truth": ground_truths,
    })

    judge = _build_judge_llm()
    embeddings = _build_judge_embeddings()
    judge_desc = (
        f"ollama {settings.EVAL_LLM_MODEL} @ {settings.OLLAMA_BASE_URL}"
        if settings.EVAL_JUDGE_PROVIDER == "ollama"
        else f"{settings.EVAL_LLM_MODEL} @ {settings.EVAL_LLM_BASE_URL or 'openai'}"
    )
    print(f"\nJudge LLM:   {judge_desc}")
    print(f"Embeddings:  ollama {settings.EMBEDDING_MODEL} @ {settings.OLLAMA_BASE_URL}")

    results: dict = {
        "judge": settings.EVAL_LLM_MODEL,
        "judge_provider": settings.EVAL_JUDGE_PROVIDER,
        "per_sample": {},
        "summary": {},
    }
    # Load any previously-saved half so partial runs don't waste work.
    if RESULTS_PATH.exists():
        existing = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        results["per_sample"] = existing.get("per_sample", {})
        results["summary"] = existing.get("summary", {})

    only = _parse_only()
    pipelines = [("agentic", "answer_ag", "contexts_ag"),
                 ("naive", "answer_nv", "contexts_nv")]
    if only:
        pipelines = [p for p in pipelines if p[0] == only]

    for label, answer_col, contexts_col in pipelines:
        print(f"\nEvaluating {label} with RAGAS...", flush=True)
        eval_result = evaluate(
            dataset,
            metrics=metrics,
            llm=judge,
            embeddings=embeddings,
            column_map={"response": answer_col, "retrieved_contexts": contexts_col},
            run_config=run_config,
            raise_exceptions=False,
        )
        df = eval_result.to_pandas()
        results["per_sample"][label] = df.to_dict(orient="records")
        results["summary"][label] = {
            m: round(float(df[m].mean()), 4) for m in _METRICS
        }
        print(f"\n{label.upper()} — mean scores:")
        for m in _METRICS:
            print(f"  {m:>20}: {results['summary'][label][m]}")

        # Save after each pipeline so an interrupted run keeps completed work.
        RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        print(f"  (partial results saved to {RESULTS_PATH})")

    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    run_evaluation()