"""
Sanity check script to verify all imports work correctly.
Run with: poetry run python test_imports.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing imports...")
print()

try:
    from src.adapters.llm_ollama import OllamaLLM, OllamaConfig
    print("[OK] src.adapters.llm_ollama")
except Exception as e:
    print(f"[FAIL] src.adapters.llm_ollama: {e}")
    sys.exit(1)

try:
    from src.retrieval.index import HybridIndex, Document, SearchResult
    print("[OK] src.retrieval.index")
except Exception as e:
    print(f"[FAIL] src.retrieval.index: {e}")
    sys.exit(1)

try:
    from src.retrieval.rerank import CrossEncoderReranker
    print("[OK] src.retrieval.rerank")
except Exception as e:
    print(f"[FAIL] src.retrieval.rerank: {e}")
    sys.exit(1)

try:
    from src.controller.policies import SelfRAGPolicy, RetrievalDecision, CritiqueResult
    print("[OK] src.controller.policies")
except Exception as e:
    print(f"[FAIL] src.controller.policies: {e}")
    sys.exit(1)

try:
    from src.controller.tools import RetrievalTool, MemoryTool, ToolRegistry
    print("[OK] src.controller.tools")
except Exception as e:
    print(f"[FAIL] src.controller.tools: {e}")
    sys.exit(1)

try:
    from src.core.memory import BaselineMemory, AugmentedMemory, MemoryMode
    print("[OK] src.core.memory")
except Exception as e:
    print(f"[FAIL] src.core.memory: {e}")
    sys.exit(1)

try:
    from src.core.letta_controller import LettaController, ChatResponse
    print("[OK] src.core.letta_controller")
except Exception as e:
    print(f"[FAIL] src.core.letta_controller: {e}")
    sys.exit(1)

try:
    from src.cli import ingest
    print("[OK] src.cli.ingest")
except Exception as e:
    print(f"[FAIL] src.cli.ingest: {e}")
    sys.exit(1)

try:
    from src.cli import chat
    print("[OK] src.cli.chat")
except Exception as e:
    print(f"[FAIL] src.cli.chat: {e}")
    sys.exit(1)

try:
    from src.eval import eval
    print("[OK] src.eval.eval")
except Exception as e:
    print(f"[FAIL] src.eval.eval: {e}")
    sys.exit(1)

print()
print("=" * 50)
print("All imports successful!")
print("=" * 50)
print()
print("Next steps:")
print("1. Ensure Ollama is running: ollama serve")
print("2. Pull model: ollama pull llama3.1:8b")
print("3. Build index: poetry run python -m src.cli.ingest build")
print("4. Start chat: poetry run python -m src.cli.chat run")
