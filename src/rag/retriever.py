"""Retriever for RAG pipeline."""

from typing import List, Dict
from .embedder import Embedder
from .vector_store import VectorStore


class Retriever:
    """Retrieves relevant documents for a query."""

    def __init__(self, embedder: Embedder, vector_store: VectorStore, top_k: int = 5, score_threshold: float = 0.3):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold

    def retrieve(self, query: str) -> List[Dict]:
        """Retrieve documents for a query."""
        query_embedding = self.embedder.embed_text(query)
        results = self.vector_store.search(query_embedding.tolist(), top_k=self.top_k)
        return [r for r in results if r["score"] >= self.score_threshold]

    def retrieve_with_rerank(self, query: str, initial_k: int = 10) -> List[Dict]:
        """Retrieve with simple reranking."""
        query_embedding = self.embedder.embed_text(query)
        results = self.vector_store.search(query_embedding.tolist(), top_k=initial_k)

        if not results:
            return []

        query_terms = set(query.lower().split())
        scored_results = []

        for result in results:
            text = result["text"].lower()
            term_overlap = sum(1 for term in query_terms if term in text)
            rerank_score = result["score"] + (term_overlap * 0.05)
            scored_results.append({**result, "rerank_score": rerank_score})

        scored_results.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_results[:self.top_k]

    def get_context_string(self, results: List[Dict], max_length: int = 4000) -> str:
        """Get context string from results."""
        context_parts = []
        current_length = 0

        for i, result in enumerate(results):
            text = result["text"]
            source = result.get("metadata", {}).get("source", "Unknown")
            language = result.get("metadata", {}).get("language", "en")

            header = f"[Source: {source}, Language: {language}]\n"
            entry = f"{header}{text}\n"

            if current_length + len(entry) > max_length:
                break

            context_parts.append(entry)
            current_length += len(entry)

        return "\n".join(context_parts)
