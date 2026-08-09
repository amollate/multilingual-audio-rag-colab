"""Vector store using ChromaDB for Colab."""

from typing import List, Dict, Optional
import hashlib
import chromadb
from chromadb.config import Settings
from pathlib import Path


class VectorStore:
    """ChromaDB vector store for document embeddings."""

    def __init__(self, persist_directory: str = "./data/processed/chroma_db"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self.client.get_or_create_collection(
            name="audio_transcripts",
            metadata={"hnsw:space": "cosine"}
        )

    def _generate_doc_id(self, source: str, chunk_id: int, text: str) -> str:
        """Generate deterministic document ID."""
        prefix = Path(source).stem if source else "doc"
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
        return f"{prefix}_{chunk_id}_{text_hash}"

    def add_documents(self, documents: List[Dict]) -> int:
        """Add documents to vector store."""
        ids = []
        embeddings = []
        metadatas = []
        documents_text = []

        for i, doc in enumerate(documents):
            source = doc.get("metadata", {}).get("source", "")
            chunk_id = doc.get("chunk_id", i)
            doc_id = self._generate_doc_id(source, chunk_id, doc.get("text", ""))
            ids.append(doc_id)
            embeddings.append(doc.get("embedding", []))
            metadatas.append({
                "source": source,
                "language": doc.get("metadata", {}).get("language", "en"),
                "chunk_id": chunk_id,
                "chunk_size": doc.get("chunk_size", 0),
            })
            documents_text.append(doc["text"])

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents_text
        )

        return len(ids)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Search for similar documents."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        documents = []
        if results and results.get("documents"):
            for i, doc_text in enumerate(results["documents"][0]):
                distance = results["distances"][0][i] if results.get("distances") else 0.0
                score = 1 - distance

                documents.append({
                    "text": doc_text,
                    "score": score,
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {}
                })

        return documents

    def get_count(self) -> int:
        """Get total document count."""
        return self.collection.count()

    def list_documents(self, limit: int = 100) -> List[Dict]:
        """List documents in store."""
        results = self.collection.get(limit=limit, include=["documents", "metadatas"])
        documents = []
        for i in range(len(results.get("ids", []))):
            documents.append({
                "id": results["ids"][i],
                "text": results["documents"][i] if results.get("documents") else "",
                "metadata": results["metadatas"][i] if results.get("metadatas") else {}
            })
        return documents

    def delete_collection(self):
        """Delete all documents."""
        self.client.delete_collection("audio_transcripts")
        self.collection = self.client.get_or_create_collection(
            name="audio_transcripts",
            metadata={"hnsw:space": "cosine"}
        )
