"""LLM providers for Colab environment.

Supports:
- OpenAI (GPT-4o, GPT-4o-mini)
- Anthropic (Claude 3.5 Sonnet, Haiku)
- HuggingFace models (local or API)
- Google Colab-specific optimizations
"""

import os
import time
import logging
from typing import Dict, List, Optional, Generator
from abc import ABC, abstractmethod
import requests
import json

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, max_tokens: int = 1024) -> str:
        """Generate response from prompt."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get current model name."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def generate(self, prompt: str, system_prompt: Optional[str] = None, max_tokens: int = 1024) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.1
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[OpenAI Error: {str(e)}]"

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=self.headers,
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False

    def get_model_name(self) -> str:
        return self.model


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API provider."""

    def __init__(self, api_key: str, model: str = "claude-3-5-haiku-20240620"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.anthropic.com/v1"
        self.headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }

    def generate(self, prompt: str, system_prompt: Optional[str] = None, max_tokens: int = 1024) -> str:
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = requests.post(
                f"{self.base_url}/messages",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            return data["content"][0]["text"]
        except Exception as e:
            return f"[Anthropic Error: {str(e)}]"

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            response = requests.get(
                f"{self.base_url}/messages",
                headers=self.headers,
                timeout=5
            )
            return response.status_code in [200, 404]  # 404 is OK, means endpoint exists
        except Exception:
            return False

    def get_model_name(self) -> str:
        return self.model


class HuggingFaceProvider(BaseLLMProvider):
    """HuggingFace model provider - works with API or local."""

    def __init__(self, model: str = "HuggingFaceH4/zephyr-7b-beta", api_token: Optional[str] = None):
        self.model = model
        self.api_token = api_token or os.getenv("HF_TOKEN", "")
        self.api_url = f"https://api-inference.huggingface.co/models/{model}"
        self.headers = {"Authorization": f"Bearer {self.api_token}"} if self.api_token else {}

    def generate(self, prompt: str, system_prompt: Optional[str] = None, max_tokens: int = 1024) -> str:
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt

        payload = {
            "inputs": full_prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": 0.1,
                "return_full_text": False
            }
        }

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                return data[0].get("generated_text", "")
            elif isinstance(data, dict):
                return data.get("generated_text", str(data))
            return str(data)
        except Exception as e:
            return f"[HuggingFace Error: {str(e)}]"

    def is_available(self) -> bool:
        try:
            response = requests.get(self.api_url, headers=self.headers, timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def get_model_name(self) -> str:
        return self.model


def get_llm_provider(config) -> BaseLLMProvider:
    """Factory function to get the appropriate LLM provider based on config."""
    provider_type = getattr(config, 'llm_provider', 'openai')

    if provider_type == 'openai' and getattr(config, 'openai_api_key', ''):
        return OpenAIProvider(api_key=config.openai_api_key, model=getattr(config, 'llm_model', 'gpt-4o-mini'))
    elif provider_type == 'anthropic' and getattr(config, 'anthropic_api_key', ''):
        return AnthropicProvider(api_key=config.anthropic_api_key, model=getattr(config, 'llm_model', 'claude-3-5-haiku-20240620'))
    elif provider_type == 'huggingface':
        return HuggingFaceProvider(
            model=getattr(config, 'llm_model', 'HuggingFaceH4/zephyr-7b-beta'),
            api_token=getattr(config, 'huggingface_api_token', '')
        )
    else:
        # Fallback to HuggingFace
        return HuggingFaceProvider(
            model=getattr(config, 'llm_model', 'HuggingFaceH4/zephyr-7b-beta'),
            api_token=getattr(config, 'huggingface_api_token', '')
        )
