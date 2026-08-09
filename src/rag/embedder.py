"""GPU-aware embedding model for Colab."""

from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import torch


class Embedder:
    """Multilingual embedding model with GPU acceleration."""

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", device: str = "cuda"):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.normalize = True

        print(f"Loading embedding model: {model_name}")
        print(f"Device: {self.device}")

        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        self.model = SentenceTransformer(model_name, device=self.device)
        print("Embedding model loaded.")

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text."""
        embedding = self.model.encode(text, normalize_embeddings=self.normalize)
        return embedding

    def embed_batch(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """Embed multiple texts with batching."""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=self.normalize,
            batch_size=batch_size,
            show_progress_bar=show_progress
        )
        return embeddings

    def embed_documents(self, documents: List[Dict], batch_size: int = 32) -> List[Dict]:
        """Embed documents and add embeddings to them."""
        texts = [doc["text"] for doc in documents]
        embeddings = self.embed_batch(texts, batch_size=batch_size)

        for doc, embedding in zip(documents, embeddings):
            doc["embedding"] = embedding.tolist()

        return documents

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()

    def cleanup(self):
        """Free GPU memory."""
        if self.device == "cuda":
            torch.cuda.empty_cache()
