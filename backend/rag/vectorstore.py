"""Unified ChromaDB vector store.

Single collection for every workspace; workspaces are a *metadata field* on
each chunk rather than a separate collection per workspace. This fixes the
orphan-chunk bug in the affine-lite prototype (re-indexing a note used to
leave stale vectors behind) and enables cross-workspace search for free.
"""
import hashlib

import chromadb
from config import settings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from rag.embedder import embed

HNSW_METADATA = {
    "hnsw:space": "cosine",
    "hnsw:construction_ef": 200,
    "hnsw:search_ef": 100,
    "hnsw:M": 32,
}


def stable_chunk_id(relative_path: str, idx: int) -> str:
    """Deterministic chunk id — editing a note reuses ids for unchanged chunks
    and changes ids only for chunks that actually moved."""
    return hashlib.sha256(f"{relative_path}::chunk_{idx}".encode()).hexdigest()


def _mmr_indices(query_embedding, doc_embeddings, k: int, lambda_mult: float = 0.7) -> list[int]:
    """Maximal Marginal Relevance: greedily pick candidates that are relevant
    to the query *and* diverse relative to what is already selected."""
    import numpy as np

    query = np.asarray(query_embedding, dtype="float32")
    docs = np.asarray(doc_embeddings, dtype="float32")
    q_norm = query / (np.linalg.norm(query) + 1e-12)
    d_norm = docs / (np.linalg.norm(docs, axis=1, keepdims=True) + 1e-12)
    sim = d_norm @ q_norm  # relevance of each candidate

    selected: list[int] = []
    remaining = set(range(len(d_norm)))
    while selected.__len__() < k and remaining:
        scores = []
        for i in remaining:
            if selected:
                diversity = 1.0 - float(np.max(d_norm[selected] @ d_norm[i]))
            else:
                diversity = 0.0
            scores.append((i, lambda_mult * float(sim[i]) + (1 - lambda_mult) * diversity))
        best_i, _ = max(scores, key=lambda t: t[1])
        selected.append(best_i)
        remaining.remove(best_i)
    return selected


class LocalBrainVectorStore:
    """Thin wrapper over a persistent ChromaDB client.

    ``chunks`` are dicts shaped like ``{"id", "text", "embedding", "metadata"}``
    as produced by the ingesters.
    """

    def __init__(self, client: chromadb.ClientAPI | None = None):
        self._client = client or chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)

    @property
    def collection(self):
        return self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata=HNSW_METADATA,
        )

    # -- writes --------------------------------------------------------------

    def upsert_chunks(self, chunks: list[dict], workspace: str = "default") -> int:
        col = self.collection
        col.upsert(
            ids=[c["id"] for c in chunks],
            embeddings=[c["embedding"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[{**c["metadata"], "workspace": workspace} for c in chunks],
        )
        return len(chunks)

    def delete_note_chunks(self, note_id: str) -> int:
        """Remove every chunk belonging to a note (id scheme is note-agnostic,
        so we filter on the note_id metadata field instead)."""
        col = self.collection
        ids = col.get(where={"note_id": note_id})["ids"]
        if ids:
            col.delete(ids=ids)
        return len(ids)

    def delete_chunks_by_id(self, ids: list[str]) -> int:
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    # -- reads ---------------------------------------------------------------

    def query_similar(
        self,
        embedding: list[float],
        n: int = settings.TOP_K,
        workspace: str | None = None,
        filter: dict | None = None,
        include=("documents", "metadatas", "distances"),
    ) -> dict:
        """Query the single collection. ``workspace=None`` searches every
        workspace; otherwise results are scoped with a ``where`` filter."""
        where = None
        if workspace is not None:
            where = {"workspace": workspace}
        if filter is not None:
            where = filter if where is None else {"$and": [where, filter]}

        return self.collection.query(
            query_embeddings=[embedding],
            n_results=n,
            where=where,
            include=list(include),
        )

    def get_retriever(
        self,
        search_type: str | None = None,
        workspace: str | None = None,
        k: int | None = None,
        filter: dict | None = None,
    ) -> "WorkspaceRetriever":
        """Build a LangChain retriever bound to a workspace scope."""
        return WorkspaceRetriever(
            store=self,
            workspace=workspace,
            search_type=search_type or settings.SEARCH_TYPE,
            k=k or settings.TOP_K,
            metadata_filter=filter,
        )


class WorkspaceRetriever(BaseRetriever):
    """LangChain ``BaseRetriever`` that runs semantic (or MMR) search scoped to
    a workspace. Returns ``Document`` objects carrying the note metadata."""

    store: LocalBrainVectorStore
    workspace: str | None = None
    search_type: str = "mmr"
    k: int = settings.TOP_K
    fetch_k: int = settings.MMR_FETCH_K
    metadata_filter: dict | None = None

    def _get_relevant_documents(self, query: str) -> list[Document]:
        query_embedding = embed(query)

        if self.search_type == "mmr":
            # Fetch min(fetch_k, total_count) candidates for MMR re-ranking
            # rather than pulling the entire collection (which scales poorly
            # on large vaults).
            collection_count = self.store.collection.count()
            fetch_n = min(self.fetch_k, collection_count) if collection_count else self.k
            fetch_n = max(fetch_n, self.k)  # never fetch fewer than k
            raw = self.store.query_similar(
                query_embedding,
                n=fetch_n,
                workspace=self.workspace,
                filter=self.metadata_filter,
                include=("documents", "metadatas", "embeddings"),
            )
            docs, metas, embeds = raw["documents"][0], raw["metadatas"][0], raw["embeddings"][0]
            if len(docs) <= self.k:
                chosen = list(range(len(docs)))
            else:
                chosen = _mmr_indices(query_embedding, embeds, self.k)
        else:  # similarity
            raw = self.store.query_similar(
                query_embedding,
                n=self.k,
                workspace=self.workspace,
                filter=self.metadata_filter,
            )
            docs, metas = raw["documents"][0], raw["metadatas"][0]
            chosen = list(range(len(docs)))

        return [
            Document(page_content=docs[i], metadata=dict(metas[i]))
            for i in chosen
            if docs[i] is not None
        ]

    async def _aget_relevant_documents(self, query: str) -> list[Document]:
        import asyncio

        return await asyncio.to_thread(self._get_relevant_documents, query)


# Module-level singleton so the API, graph and watcher share one client.
vectorstore = LocalBrainVectorStore()
