"""Configuration for Colab environment with auto-detection."""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ColabConfig:
    """Auto-detecting configuration for Google Colab."""

    def __init__(self, drive_mount_path: str = "/content/drive/MyDrive/multilingual-audio-rag"):
        self.drive_mount_path = Path(drive_mount_path)
        self.data_dir = self.drive_mount_path / "data"
        self.audio_dir = self.data_dir / "audio"
        self.transcript_dir = self.data_dir / "transcripts"
        self.processed_dir = self.data_dir / "processed"
        self.vector_db_path = self.processed_dir / "chroma_db"

        # Auto-detect GPU
        self.gpu_available = self._detect_gpu()
        self.gpu_info = self._get_gpu_info() if self.gpu_available else {}

        # Auto-configure based on hardware
        self._configure_for_hardware()

        # LLM configuration
        self.llm_provider = os.getenv("LLM_PROVIDER", "openai")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.huggingface_api_token = os.getenv("HUGGINGFACE_API_TOKEN", "")

        # Model configurations
        self.embedding_model = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        self.llm_model = self._select_best_llm_model()

        # Chunking configuration
        self.chunk_size = 500
        self.chunk_overlap = 50

        # Retrieval configuration
        self.top_k = 5
        self.score_threshold = 0.3

        logger.info(f"Config initialized - GPU: {self.gpu_available}, Model: {self.llm_model}")

    def _detect_gpu(self) -> bool:
        """Check if GPU is available in Colab."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information."""
        try:
            import torch
            info = {
                "available": True,
                "name": torch.cuda.get_device_name(0),
                "memory_total": torch.cuda.get_device_properties(0).total_memory / 1024**3,
                "memory_allocated": torch.cuda.memory_allocated(0) / 1024**3,
            }
            return info
        except Exception as e:
            logger.warning(f"Could not get GPU info: {e}")
            return {"available": False, "error": str(e)}

    def _configure_for_hardware(self):
        """Adjust configuration based on available hardware."""
        if self.gpu_available:
            self.device = "cuda"
            # Use larger batch sizes on GPU
            self.embedding_batch_size = 64
            # Use float16 on GPU for speed
            self.embedding_precision = "float16"
        else:
            self.device = "cpu"
            self.embedding_batch_size = 16
            self.embedding_precision = "float32"
            logger.warning("No GPU detected. Running on CPU will be slower.")

    def _select_best_llm_model(self) -> str:
        """Select the best available LLM based on API keys and hardware."""
        # Priority: OpenAI > Anthropic > HuggingFace > Local
        if self.openai_api_key:
            return "gpt-4o-mini"
        elif self.anthropic_api_key:
            return "claude-3-5-haiku-20240620"
        elif self.huggingface_api_token:
            return "HuggingFaceH4/zephyr-7b-beta"
        else:
            # Fallback to a small local model if GPU is available
            if self.gpu_available:
                return "microsoft/Phi-3-mini-4k-instruct"
            else:
                logger.warning("No LLM API keys found. Using smallest local model.")
                return "HuggingFaceH4/zephyr-7b-beta"

    def setup_directories(self):
        """Create necessary directories."""
        for dir_path in [self.data_dir, self.audio_dir, self.transcript_dir, self.processed_dir, self.vector_db_path]:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        return {
            "gpu_available": self.gpu_available,
            "gpu_info": self.gpu_info,
            "device": self.device,
            "embedding_model": self.embedding_model,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "score_threshold": self.score_threshold,
            "data_dir": str(self.data_dir),
            "vector_db_path": str(self.vector_db_path),
        }

    def save(self, path: Optional[str] = None):
        """Save configuration to JSON."""
        path = path or str(self.processed_dir / "config.json")
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ColabConfig":
        """Load configuration from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
