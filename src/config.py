"""
src/config.py

Centralized configuration management for the benchmark application.
Loads environment variables from .env file and provides configuration constants for:
- API keys (OpenAI, Letta, Mem0)
- Backend server URLs
- Model names (LLM and embedding models)
- Benchmark defaults (turns, sessions, noise ratio)

All values can be overridden via environment variables or .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LETTA_BASE_URL = os.getenv("LETTA_BASE_URL", "http://localhost:8283")
MEM0_BASE_URL = os.getenv("MEM0_BASE_URL", "http://localhost:3000")
MEM0_API_KEY = os.getenv("MEM0_API_KEY", "")

DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "openai/gpt-4o-mini")
DEFAULT_EMBEDDING_MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL", "openai/text-embedding-3-small")

BENCH_DEFAULT_TURNS = int(os.getenv("BENCH_DEFAULT_TURNS", "12"))
BENCH_SESSIONS = int(os.getenv("BENCH_SESSIONS", "3"))
BENCH_NOISE_RATIO = float(os.getenv("BENCH_NOISE_RATIO", "0.25"))
