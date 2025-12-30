"""
Ollama LLM Adapter for local model inference.
"""
import httpx
import json
from typing import Optional, Dict, Any, List
from pydantic import BaseModel


class OllamaConfig(BaseModel):
    """Configuration for Ollama LLM."""
    base_url: str = "http://localhost:11434"
    model: str = "mistral:latest"  # Changed from llama3.1:8b (use --ollama-model to override)
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 120


class OllamaLLM:
    """Adapter for Ollama local LLM inference."""

    def __init__(self, config: Optional[OllamaConfig] = None):
        self.config = config or OllamaConfig()
        self.client = httpx.Client(timeout=self.config.timeout)

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Generate text from prompt using Ollama."""
        url = f"{self.config.base_url}/api/generate"

        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature or self.config.temperature,
                "num_predict": max_tokens or self.config.max_tokens,
            }
        }

        if system:
            payload["system"] = system

        if stop:
            payload["options"]["stop"] = stop

        try:
            response = self.client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except httpx.HTTPError as e:
            raise RuntimeError(f"Ollama API error: {e}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Chat completion using Ollama chat endpoint."""
        url = f"{self.config.base_url}/api/chat"

        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature or self.config.temperature,
                "num_predict": max_tokens or self.config.max_tokens,
            }
        }

        try:
            response = self.client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("message", {}).get("content", "").strip()
        except httpx.HTTPError as e:
            raise RuntimeError(f"Ollama chat API error: {e}")

    def embed(self, text: str) -> List[float]:
        """Generate embeddings (note: Ollama embeddings require specific models)."""
        url = f"{self.config.base_url}/api/embeddings"

        payload = {
            "model": self.config.model,
            "prompt": text,
        }

        try:
            response = self.client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("embedding", [])
        except httpx.HTTPError as e:
            # Fallback: some models don't support embeddings
            raise RuntimeError(f"Ollama embeddings error: {e}")

    def is_available(self) -> bool:
        """Check if Ollama service is available."""
        try:
            response = self.client.get(f"{self.config.base_url}/api/tags")
            return response.status_code == 200
        except:
            return False

    def __del__(self):
        """Cleanup HTTP client."""
        if hasattr(self, 'client'):
            self.client.close()
