"""Retrieval speed benchmark: naive keyword scan vs. ChromaDB semantic search.

Creates a synthetic corpus, ingests it, and times both keyword and semantic
retrieval on a battery of queries.
"""
import statistics
import sys
import time
import uuid
from pathlib import Path

_backend = str(Path(__file__).resolve().parent.parent / "backend")
if _backend not in sys.path:
    sys.path.insert(0, _backend)

import chromadb
from config import settings
from rag.embedder import embed, embed_documents


def _generate_synthetic_data(n: int) -> list[dict]:
    """Create n text chunks with embedded metadata for benchmarking."""
    topics = [
        "science", "history", "technology", "art", "music",
        "cooking", "travel", "sports", "finance", "nature",
    ]
    chunks = []
    for i in range(n):
        topic = topics[i % len(topics)]
        text = f"Document {i}: This is a piece of text about {topic}. It contains information regarding {uuid.uuid4()}. " * 5
        chunks.append({
            "id": f"synth_{i}",
            "text": text,
            "metadata": {"source": "synthetic", "index": i, "topic": topic},
        })
    return chunks


def _keyword_search_baseline(query: str, chunks: list[dict], k: int = 5) -> list[str]:
    """Linear-scan keyword baseline."""
    words = query.lower().split()
    scored = [
        (sum(1 for w in words if w in c["text"].lower()), c["id"])
        for c in chunks
    ]
    scored.sort(key=lambda t: t[0], reverse=True)
    return [cid for _, cid in scored[:k]]


def run_benchmark(n_chunks: int = 2000, n_queries: int = 10):
    print(f"--- Retrieval Speed Benchmark ({n_chunks} chunks) ---")

    # 1. Synthetic corpus
    chunks = _generate_synthetic_data(n_chunks)
    print(f"Embedding {n_chunks} chunks...")
    t0 = time.perf_counter()
    embeddings = embed_documents([c["text"] for c in chunks])
    print(f"Embedding completed in {time.perf_counter() - t0:.2f}s")

    # 2. Ingest into an in-memory ChromaDB (no persistence)
    client = chromadb.Client()
    col = client.create_collection(
        name="benchmark",
        metadata={"hnsw:space": "cosine"},
    )
    col.upsert(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )

    queries = [
        "information about technology and science",
        "details regarding nature and travel",
        "cooking recipes and music trends",
        "finance and sports updates",
        "history of art and science",
    ] * ((n_queries // 5) + 1)
    queries = queries[:n_queries]

    # 3. Keyword baseline
    keyword_times = []
    for q in queries:
        t0 = time.perf_counter()
        _keyword_search_baseline(q, chunks, k=settings.TOP_K)
        keyword_times.append(time.perf_counter() - t0)
    avg_keyword = statistics.mean(keyword_times)

    # 4. ChromaDB semantic search (includes embedding time)
    chroma_times = []
    for q in queries:
        t0 = time.perf_counter()
        q_emb = embed(q)
        col.query(query_embeddings=[q_emb], n_results=settings.TOP_K)
        chroma_times.append(time.perf_counter() - t0)
    avg_chroma = statistics.mean(chroma_times)

    # 5. ChromaDB semantic search (embedding excluded — pre-computed)
    chroma_pre_times = []
    for q in queries:
        q_emb = embed(q)
        t0 = time.perf_counter()
        col.query(query_embeddings=[q_emb], n_results=settings.TOP_K)
        chroma_pre_times.append(time.perf_counter() - t0)
    avg_chroma_pre = statistics.mean(chroma_pre_times)

    # 6. Report
    improvement = (avg_keyword - avg_chroma) / avg_keyword * 100

    print("\n" + "=" * 50)
    print(f"RESULTS FOR {n_chunks} CHUNKS")
    print("=" * 50)
    print(f"Keyword Baseline Avg:              {avg_keyword * 1000:.2f} ms")
    print(f"ChromaDB (with embedding):         {avg_chroma * 1000:.2f} ms")
    print(f"ChromaDB (embedding excluded):     {avg_chroma_pre * 1000:.2f} ms")

    if improvement > 0:
        print(f"Semantic vs Keyword improvement:   {improvement:.1f}%")
    else:
        print(f"ChromaDB is {abs(improvement):.1f}% slower (embedding API latency dominates)")
    print("=" * 50)

    client.delete_collection("benchmark")


if __name__ == "__main__":
    run_benchmark()
