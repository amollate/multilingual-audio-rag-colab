"""GPU and environment precheck utilities."""

import os
import sys
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


def check_gpu() -> Dict[str, Any]:
    """Check GPU availability and information."""
    info = {
        "available": False,
        "name": None,
        "memory_total_gb": None,
        "memory_allocated_gb": None,
        "cuda_version": None,
    }

    try:
        import torch
        if torch.cuda.is_available():
            info["available"] = True
            info["name"] = torch.cuda.get_device_name(0)
            info["memory_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
            info["memory_allocated_gb"] = round(torch.cuda.memory_allocated(0) / 1024**3, 1)
            info["cuda_version"] = torch.version.cuda
    except ImportError:
        logger.warning("PyTorch not installed. Cannot check GPU.")
    except Exception as e:
        logger.error(f"Error checking GPU: {e}")

    return info


def check_models() -> Dict[str, Any]:
    """Check available models and providers."""
    status = {
        "openai_available": bool(os.getenv("OPENAI_API_KEY")),
        "anthropic_available": bool(os.getenv("ANTHROPIC_API_KEY")),
        "huggingface_available": bool(os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_TOKEN")),
        "torch_available": False,
        "transformers_available": False,
        "chromadb_available": False,
    }

    try:
        import torch
        status["torch_available"] = True
    except ImportError:
        pass

    try:
        import transformers
        status["transformers_available"] = True
    except ImportError:
        pass

    try:
        import chromadb
        status["chromadb_available"] = True
    except ImportError:
        pass

    return status


def get_recommended_config(gpu_info: Dict[str, Any], model_status: Dict[str, Any]) -> Dict[str, Any]:
    """Get recommended configuration based on available resources."""
    config = {
        "device": "cuda" if gpu_info["available"] else "cpu",
        "embedding_batch_size": 64 if gpu_info["available"] else 16,
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "use_gpu_embeddings": gpu_info["available"],
    }

    # Adjust based on available API keys
    if not model_status["openai_available"] and model_status["anthropic_available"]:
        config["llm_provider"] = "anthropic"
        config["llm_model"] = "claude-3-5-haiku-20240620"
    elif not model_status["openai_available"] and not model_status["anthropic_available"]:
        config["llm_provider"] = "huggingface"
        if gpu_info["available"]:
            config["llm_model"] = "microsoft/Phi-3-mini-4k-instruct"
        else:
            config["llm_model"] = "HuggingFaceH4/zephyr-7b-beta"

    # Adjust batch size based on GPU memory
    if gpu_info.get("memory_total_gb"):
        if gpu_info["memory_total_gb"] >= 16:
            config["embedding_batch_size"] = 128
        elif gpu_info["memory_total_gb"] >= 8:
            config["embedding_batch_size"] = 64
        else:
            config["embedding_batch_size"] = 32

    return config


def precheck() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Run complete precheck and return status."""
    print("\\n" + "="*60)
    print("🔍 PRE-CHECK: Analyzing Environment")
    print("="*60)

    gpu_info = check_gpu()
    print(f"\\n🖥️  GPU Status:")
    print(f"   Available: {gpu_info['available']}")
    if gpu_info["available"]:
        print(f"   Name: {gpu_info['name']}")
        print(f"   Memory: {gpu_info['memory_total_gb']} GB")
        print(f"   CUDA: {gpu_info['cuda_version']}")
    else:
        print("   ⚠️  No GPU detected. CPU mode will be slower.")

    model_status = check_models()
    print(f"\\n🤖 Model Status:")
    print(f"   OpenAI: {'✅' if model_status['openai_available'] else '❌'}")
    print(f"   Anthropic: {'✅' if model_status['anthropic_available'] else '❌'}")
    print(f"   HuggingFace: {'✅' if model_status['huggingface_available'] else '❌'}")
    print(f"   PyTorch: {'✅' if model_status['torch_available'] else '❌'}")
    print(f"   Transformers: {'✅' if model_status['transformers_available'] else '❌'}")
    print(f"   ChromaDB: {'✅' if model_status['chromadb_available'] else '❌'}")

    recommended = get_recommended_config(gpu_info, model_status)
    print(f"\\n💡 Recommended Configuration:")
    print(f"   Device: {recommended['device']}")
    print(f"   Embedding Batch Size: {recommended['embedding_batch_size']}")
    print(f"   LLM Provider: {recommended['llm_provider']}")
    print(f"   LLM Model: {recommended['llm_model']}")

    print("="*60 + "\\n")

    return gpu_info, model_status, recommended
