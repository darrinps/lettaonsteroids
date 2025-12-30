"""
OpenAI LLM Adapter for cloud-based model inference.
"""
import os
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
from openai import OpenAI


class OpenAIConfig(BaseModel):
    """Configuration for OpenAI LLM."""
    api_key: Optional[str] = None
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 60


class OpenAILLM:
    """Adapter for OpenAI cloud LLM inference."""

    def __init__(self, config: Optional[OpenAIConfig] = None):
        """Initialize OpenAI LLM adapter."""
        self.config = config or OpenAIConfig()

        # Get API key from config or environment
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key in config."
            )

        self.client = OpenAI(api_key=api_key, timeout=self.config.timeout)

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Generate text from prompt using OpenAI."""
        messages = []

        if system:
            messages.append({"role": "system", "content": system})

        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=temperature or self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens,
                stop=stop,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Chat completion using OpenAI chat endpoint."""
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=temperature or self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"OpenAI chat API error: {e}")

    def embed(self, text: str) -> List[float]:
        """Generate embeddings using OpenAI embeddings API."""
        try:
            # Use text-embedding-3-small by default (cheaper and faster)
            response = self.client.embeddings.create(
                model=os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-3-small"),
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            raise RuntimeError(f"OpenAI embeddings error: {e}")

    def is_available(self) -> bool:
        """Check if OpenAI API is available."""
        try:
            # Simple test call
            self.client.models.list()
            return True
        except:
            return False
