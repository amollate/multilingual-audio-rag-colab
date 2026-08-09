"""RAG pipeline for Colab environment."""

import os
from typing import List, Dict, Optional
from ..ingestion.transcript_processor import TranscriptProcessor
from ..rag.chunker import Chunker
from ..rag.embedder import Embedder
from ..rag.vector_store import VectorStore
from ..rag.retriever import Retriever
from ..llm.cloud_llm import get_llm_provider
from ..llm.prompt_templates import get_system_prompt, get_qa_prompt


class RAGPipeline:
    """RAG pipeline optimized for Colab GPU."""

    def __init__(self, config):
        self.config = config
        self.processor = TranscriptProcessor()
        self.chunker = Chunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        self.embedder = Embedder(
            model_name=config.embedding_model,
            device=config.device
        )
        self.vector_store = VectorStore(persist_directory=str(config.vector_db_path))
        self.retriever = Retriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            top_k=config.top_k,
            score_threshold=config.score_threshold
        )
        self.llm = get_llm_provider(config)

    def ingest_transcripts(self, transcript_dir: str) -> Dict:
        """Ingest transcripts from directory."""
        transcripts = self.processor.load_all_transcripts(transcript_dir)
        print(f"Loaded {len(transcripts)} transcripts")

        if not transcripts:
            return {"status": "error", "message": "No transcripts found"}

        chunks = self.chunker.chunk_transcripts(transcripts)
        print(f"Created {len(chunks)} chunks")

        embedded_docs = self.embedder.embed_documents(chunks, batch_size=self.config.embedding_batch_size)
        added_count = self.vector_store.add_documents(embedded_docs)

        return {
            "status": "success",
            "transcripts_processed": len(transcripts),
            "chunks_created": len(chunks),
            "vectors_added": added_count,
            "total_documents": self.vector_store.get_count()
        }

    def query(self, question: str, language: str = "en") -> Dict:
        """Query the RAG system."""
        results = self.retriever.retrieve_with_rerank(question)
        context = self.retriever.get_context_string(results)

        system_prompt = get_system_prompt(language)
        prompt = get_qa_prompt(context, question)

        answer = self.llm.generate(prompt, system_prompt=system_prompt)

        sources = list(set([r.get("metadata", {}).get("source", "") for r in results]))

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": len(results)
        }

    def get_status(self) -> Dict:
        """Get system status."""
        return {
            "gpu_available": self.config.gpu_available,
            "gpu_info": self.config.gpu_info,
            "llm_provider": self.config.llm_provider,
            "llm_model": self.llm.get_model_name(),
            "vector_store_count": self.vector_store.get_count(),
            "embedding_model": self.config.embedding_model,
            "device": self.config.device,
        }
