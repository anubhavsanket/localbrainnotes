"""RAGAS evaluation harness.

Runs the golden dataset through the agentic RAG pipeline and scores
faithfulness, answer_relevance, context_recall, and context_precision.

Requires OPENAI_API_KEY because RAGAS uses OpenAI models as evaluators.
"""
import asyncio
import json
import sys
from pathlib import Path

# Ensure backend/ is importable when run from repo root
_backend = str(Path(__file__).resolve().parent.parent / "backend")
if _backend not in sys.path:
    sys.path.insert(0, _backend)

from rag.graph import get_agent_graph
from rag.memory import WorkspaceChatMemoryManager
from rag.vectorstore import vectorstore
from config import settings

DATA_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.json"


async def run_evaluation(data_path: str | Path = DATA_PATH):
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevance,
            context_recall,
            context_precision,
        )
        from datasets import Dataset
    except ImportError as e:
        print(f"Missing eval dependencies: {e}\n  pip install ragas datasets")
        return

    if not settings.OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY is required for RAGAS evaluation.")
        print("RAGAS uses OpenAI models as judges. Set OPENAI_API_KEY in your .env.")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    # --- Agentic pipeline -------------------------------------------------------
    graph = get_agent_graph()
    questions = [item["question"] for item in benchmark_data]
    ground_truths = [[item["ground_truth"]] for item in benchmark_data]

    answers_ag, contexts_ag = [], []
    answers_nv, contexts_nv = [], []

    print(f"Running queries for {len(questions)} items...")
    for q in questions:
        # Agentic pipeline
        input_state = {
            "question": q,
            "workspace": "default",
            "chat_history": [],
            "rewrite_count": 0,
        }
        result = graph.invoke(input_state, config={"configurable": {"thread_id": "eval"}})
        answers_ag.append(result.get("answer", ""))
        contexts_ag.append([
            doc.page_content for doc in result.get("documents", [])
        ])

        # Naive retrieval: just retrieve (no agent loop) then generate
        from rag.vectorstore import vectorstore
        from rag.nodes import create_nodes, _GRADE_SYSTEM, _ANSWER_SYSTEM
        from langchain_core.language_models import BaseChatModel
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_core.output_parsers import StrOutputParser

        # Use the same LLM as the agent
        from rag.llm_factory import get_llm
        llm = get_llm()

        # Simple retrieve + grade + generate (no router, no reflect, no rewrite)
        from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
        from config import settings

        # Retrieve with workspace filter
        retriever = vectorstore.get_retriever(
            workspace="default",
            filter={"workspace": "default"},
        )
        docs = retriever.invoke(q)
        # Grade docs
        grade_chain = (
            ChatPromptTemplate.from_template(_GRADE_SYSTEM)
            | llm
            | StrOutputParser()
        )
        grades = []
        for doc in docs:
            response = grade_chain.invoke({
                "document": doc.page_content,
                "question": q,
            })
            grades.append(_parse_yes_no(response))

        # Build context from relevant docs only
        relevant_docs = [docs[i] for i, g in enumerate(grades) if g]
        if relevant_docs:
            context_text = "\n\n".join(doc.page_content for doc in relevant_docs)
        else:
            context_text = ""

        # Generate answer from context
        answer_chain = (
            ChatPromptTemplate.from_messages([
                ("system", _ANSWER_SYSTEM),
                ("human", "Question: {question}\nContext: {context}"),
            ])
            | llm
            | StrOutputParser()
        )
        ans = answer_chain.invoke({
            "question": q,
            "context": context_text,
        })
        answers_nv.append(ans)
        # For naive, we use the union of all docs' metadata note_ids as sources
        nv_sources = list(dict.fromkeys(
            doc.metadata.get("note_id", "unknown") for doc in docs
        ))
        contexts_nv.append(nv_sources)

    # Build dataset comparing agentic vs naive
    dataset = Dataset.from_dict({
        "question": questions,
        "answer_ag": answers_ag,
        "answer_nv": answers_nv,
        "contexts_ag": contexts_ag,
        "contexts_nv": contexts_nv,
        "ground_truth": ground_truths,
    })

    print("Evaluating metrics...")
    eval_result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevance, context_recall, context_precision],
    )

    print("\nEvaluation Results:")
    print(eval_result)

    # The result may be a dict (ragas >= 0.2) or an EvaluationResult object.
    # For json.dump compatibility, convert via pandas.
    try:
        results_df = eval_result.to_pandas()
        results_dict = results_df.to_dict(orient="records")
    except AttributeError:
        results_dict = eval_result if isinstance(eval_result, dict) else {"raw": str(eval_result)}

    RESULTS_PATH.write_text(json.dumps(results_dict, indent=2, default=str), encoding="utf-8")
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(run_evaluation())
