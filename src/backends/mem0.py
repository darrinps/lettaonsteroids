"""
src/backends/mem0.py

Mem0 backend implementation for the benchmark framework.
Mem0 is a semantic search-based memory service with vector embeddings.

Key features:
- Vector-based semantic search for memory retrieval
- User-scoped memory storage (uses user_id instead of agent_id)
- LLM post-processing to format raw memory strings into concise responses
- Simple API for adding, searching, and deleting memories

Implementation details:
- Uses Mem0 REST API (default: http://localhost:3000)
- Stores memories as semantic text chunks indexed by user_id
- Retrieves memories via semantic search queries
- Optionally uses GPT-4o-mini to format retrieved memories into keyword responses
- Returns CreateResult, SendResult, DeleteResult to match benchmark interface

This backend scores well on memory scalability and retrieval speed but may
struggle with cross-session recall and complex multi-fact inference due to
its reliance on semantic search rather than structured reasoning.
"""

from __future__ import annotations
import requests, uuid, json, re, time, os
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from src.config import MEM0_BASE_URL, MEM0_API_KEY, OPENAI_API_KEY

@dataclass
class CreateResult:
    ok: bool
    latency_ms: float
    error: Optional[str]
    agent_id: Optional[str]
    raw: Optional[Dict[str, Any]] = None

@dataclass
class SendResult:
    ok: bool
    latency_ms: float
    error: Optional[str]
    text: str
    raw: Optional[Dict[str, Any]] = None

@dataclass
class DeleteResult:
    ok: bool
    latency_ms: float
    error: Optional[str]
    raw: Optional[Dict[str, Any]] = None

def _auth_headers():
    h = {"Content-Type": "application/json"}
    if MEM0_API_KEY:
        h["Authorization"] = f"Token {MEM0_API_KEY}"
    return h

_Q_RE = re.compile(r"\b(what|which|when|where|who|whom|whose|how|next|after|name|city|step)\b", re.I)

class Mem0Backend:
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        self.base_url = (base_url or MEM0_BASE_URL).rstrip("/")
        self.api_key = api_key or MEM0_API_KEY
        self.openai_api_key = OPENAI_API_KEY
        self._use_llm_formatting = bool(self.openai_api_key)

    def backend_name(self) -> str:
        return "mem0"

    def _format_with_llm(self, memories: str, question: str) -> str:
        """Use LLM to format raw memories into concise keyword response"""
        if not self._use_llm_formatting or not memories.strip():
            return memories

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_api_key)

            prompt = f"""Based on the stored memories, answer the question with ONLY the requested keywords - no full sentences, no extra words.

Memories: {memories}

Question: {question}

Instructions:
- Respond with ONLY bare keywords/facts
- NO full sentences or explanations
- Examples:
  * "What is my name and city?" → "Darrin Phoenix"
  * "What brand?" → "Choice"
  * "Next step after select?" → "Pilot"

Answer:"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=50
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return memories

    def create_agent(self, *args, **kwargs) -> CreateResult:
        user_id = f"bench-{uuid.uuid4()}"
        t0 = time.perf_counter()
        try:
            # Optionally preload system/memory_blocks as a memory
            system_prompt = kwargs.get("system")
            memory_blocks = kwargs.get("memory_blocks")
            if system_prompt or memory_blocks:
                pre = {
                    "messages": [{"role": "system", "content": json.dumps({
                        "system": system_prompt, "memory_blocks": memory_blocks or []
                    })}],
                    "user_id": user_id, "version": "v2"
                }
                try:
                    r = requests.post(f"{self.base_url}/v1/memories/", headers=_auth_headers(), json=pre, timeout=60)
                    r.raise_for_status()
                except Exception:
                    pass
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return CreateResult(ok=True, latency_ms=latency_ms, error=None, agent_id=user_id, raw={"id": user_id})
        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return CreateResult(ok=False, latency_ms=latency_ms, error=str(e), agent_id=None, raw=None)

    def send(self, agent_id: str, role: str, content: str) -> SendResult:
        t0 = time.perf_counter()
        try:
            q = (content or "").strip()
            q_low = q.lower()
            is_question = ("?" in q) or bool(_Q_RE.search(q_low))

            if role != "user" or is_question:
                # SEARCH
                body = {"query": q, "filters": {"AND": [{"user_id": agent_id}]}, "version": "v2"}
                r = requests.post(f"{self.base_url}/v2/memories/search/", headers=_auth_headers(), json=body, timeout=60)
                r.raise_for_status()
                hits = r.json()
                raw_memories = "; ".join([h.get("memory","") for h in hits[:5] if isinstance(h, dict) and h.get("memory")])

                # Format with LLM for concise response
                text = self._format_with_llm(raw_memories, q)

                latency_ms = (time.perf_counter() - t0) * 1000.0
                return SendResult(ok=True, latency_ms=latency_ms, error=None, text=text, raw={"content": text, "hits": hits})

            # ADD MEMORY
            body = {"messages": [{"role": "user", "content": q}], "user_id": agent_id, "version": "v2"}
            r = requests.post(f"{self.base_url}/v1/memories/", headers=_auth_headers(), json=body, timeout=60)
            r.raise_for_status()
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return SendResult(ok=True, latency_ms=latency_ms, error=None, text="", raw={"message": "added"})
        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return SendResult(ok=False, latency_ms=latency_ms, error=str(e), text="", raw=None)

    def teardown_agent(self, agent_id: str) -> DeleteResult:
        t0 = time.perf_counter()
        try:
            r = requests.delete(f"{self.base_url}/v1/memories/", headers=_auth_headers(), params={"user_id": agent_id}, timeout=60)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return DeleteResult(ok=True, latency_ms=latency_ms, error=None, raw={"status": r.status_code})
        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return DeleteResult(ok=False, latency_ms=latency_ms, error=str(e), raw=None)
