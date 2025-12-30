"""
src/backends/base.py

Abstract base class and shared data structures for memory backend implementations.
Defines:
- StepResult: Standard return type for backend operations
- BaseMemoryBackend: Abstract interface that all backends must implement

Required methods for backend implementations:
- create_agent(): Initialize a new agent/session with memory capabilities
- send(): Send a message to the agent and get a response
- upsert_memory(): Manually insert or update memory items
- delete_memory(): Remove memory items by label
- teardown_agent(): Clean up and destroy an agent session

All backends (Letta, Mem0, etc.) must inherit from BaseMemoryBackend and
implement these methods with consistent StepResult return types.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
import time

@dataclass
class StepResult:
    ok: bool
    latency_ms: float
    payload: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

class BaseMemoryBackend:
    name: str = "base"

    def create_agent(self, model: str, embedding: Optional[str], system_prompt: str, memory_blocks: List[Dict[str, str]]) -> StepResult:
        raise NotImplementedError

    def send(self, agent_id: str, role: str, content: str) -> StepResult:
        raise NotImplementedError

    def upsert_memory(self, agent_id: str, items: List[Dict[str, str]]) -> StepResult:
        raise NotImplementedError

    def delete_memory(self, agent_id: str, labels: List[str]) -> StepResult:
        raise NotImplementedError

    def teardown_agent(self, agent_id: str) -> StepResult:
        raise NotImplementedError

    def _timeit(self, fn, *args, **kwargs) -> StepResult:
        t0 = time.perf_counter()
        try:
            data = fn(*args, **kwargs)
            return StepResult(ok=True, latency_ms=(time.perf_counter()-t0)*1000.0, payload=data)
        except Exception as e:
            return StepResult(ok=False, latency_ms=(time.perf_counter()-t0)*1000.0, error=str(e), payload={})
