# Letta + Mem0 Grounded Memory Architecture

## System Overview
Two cooperating layers power this repo: `/src` contains the CLI benchmark harness and HTTP backends for Letta and Mem0, while `/letta_hybrid_poc` houses the hybrid-retrieval stack, Self-RAG policies, and optional UI/API. Together they stage identical conversations through multiple memory strategies so grounding quality, latency, and critique behaviors can be compared apples-to-apples.

### Request Lifecycle
1. **Driver** (benchmark CLI, dashboard, or UI) issues a scenario turn and selects a memory mode.
2. **LettaController** (`letta_hybrid_poc/src/core/letta_controller.py`) checks the active mode (Baseline, Mem0, Mem0 Enhanced, Letta hybrid).
3. **Self-RAG gate** (`controller/policies.py`) decides whether retrieval is needed before any backend call; evaluations can force retrieval for fairness.
4. **Memory abstraction** (`core/memory.py`) routes to the correct data surface:
   - Baseline keyword overlap
   - Mem0 semantic store
   - EnhancedMem0 wrapper (Mem0 + rerank + policies)
   - Letta AugmentedMemory (hybrid BM25/FAISS + rerank)
5. **RetrievalTool + HybridIndex** (`controller/tools.py`, `retrieval/index.py`) gather BM25 and FAISS hits, optionally reranking with the shared cross-encoder.
6. **LLM adapters** (`adapters/llm_openai.py`, `adapters/llm_ollama.py`) craft the final response using retrieved snippets and recent chat context.
7. **Self-RAG critique** optionally validates the answer, correcting hallucinations or skipping critique when confidence is high.
8. **Benchmark logger/UI** (`src/benchmark.py`, `dashboard.py`, `letta_hybrid_poc/src/eval`) records latency, accuracy, similarity, and visualizes side-by-side results.

```
User / Benchmark / UI
          ↓
    LettaController
          ↓
   Self-RAG gate
          ↓
  Memory surface
 (Baseline | Mem0 |
  Mem0+ | Letta)
          ↓
 Hybrid Retrieval
  BM25 + FAISS
          ↓
Cross-Encoder Rerank
          ↓
  LLM Generation
          ↓
 Self-RAG Critique
          ↓
 Metrics + Display
```

## Technology Stack (10–30 word contributions)
| Technology | Key files | Interacts with | Contribution |
|------------|-----------|----------------|--------------|
| Letta backend & controller | `src/backends/letta.py`, `letta_hybrid_poc/src/core/letta_controller.py` | Self-RAG policy, retrieval tool, LLM adapters | Provides structured agents, default memory blocks, and deterministic tool loops so Letta can insert/search memories and emit concise benchmark-friendly answers. |
| Mem0 backend & Enhanced wrapper | `src/backends/mem0.py`, `letta_hybrid_poc/src/core/memory.py` | Self-RAG policy, reranker, benchmark harness | Stores user-scoped semantic vectors and, when wrapped by EnhancedMem0, shares rerankers and policies to mirror Letta’s retrieval stack for fair tests. |
| Hybrid Retrieval Index | `letta_hybrid_poc/src/retrieval/index.py` | RetrievalTool, AugmentedMemory, Letta mode | Fuses BM25 keywords with FAISS embeddings, giving Letta hybrid recall that captures literal requirements and semantic paraphrases in one ranked list. |
| Cross-Encoder Reranker | `letta_hybrid_poc/src/retrieval/rerank.py` | RetrievalTool, EnhancedMem0, AugmentedMemory | Re-scores top hybrid hits via ms-marco cross encoder to boost precision before answers are generated or critiqued, minimizing noisy grounding. |
| Self-RAG Policy Engine | `letta_hybrid_poc/src/controller/policies.py` | LettaController, EnhancedMem0, AugmentedMemory | Applies heuristic plus LLM decisions to gate retrieval, critique answers, and correct hallucinations, ensuring both backends only trust supported facts. |
| Memory abstractions & tools | `letta_hybrid_poc/src/controller/tools.py`, `letta_hybrid_poc/src/core/memory.py` | Hybrid index, reranker, controllers | Standardize Baseline, Mem0, Enhanced, and Letta flows, cache retrieval, and feed Self-RAG the conversation snippets needed for consistent decisions. |
| LLM adapters | `letta_hybrid_poc/src/adapters/llm_openai.py`, `letta_hybrid_poc/src/adapters/llm_ollama.py` | Controllers, Self-RAG prompts, critique | Wrap OpenAI or Ollama completion APIs so every backend shares the same generation surface, simplifying comparisons and policy prompts. |
| Benchmark, eval, and dashboard | `src/benchmark.py`, `letta_hybrid_poc/src/eval/eval.py`, `dashboard.py` | Letta/Mem0 backends, UI, CSV/JSON outputs | Drives automated scenarios, calculates accuracy and similarity, and streams metrics to the dashboard so architectural trade-offs stay observable in real time. |

## Interaction Details
- **Shared critique/safety:** Self-RAG policies are injected into both Letta AugmentedMemory and EnhancedMem0, meaning retrieval skips, critiques, and corrections follow identical heuristics regardless of backend.
- **Identical LLM surfaces:** Controllers feed the same prompts, retrieved snippets, and temperature settings to OpenAI or Ollama adapters, so differences are attributable to retrieval/memory rather than generation.
- **Parallel metrics:** The root benchmark (`src/benchmark.py`) and the hybrid POC evaluator (`letta_hybrid_poc/src/eval/eval.py`) call the same controller APIs, ensuring CSV/JSON outputs can be compared to streaming UI panels without drift.
- **Swappable retrieval:** LettaController can switch modes mid-run, allowing researchers to replay a session against Baseline, Mem0, Enhanced, or Letta without re-ingesting documents.
- **Grounding consistency:** RetrievalTool’s smart rerank logic is shared by both Letta and EnhancedMem0, while the benchmark-level Letta backend can fall back to local shims to keep grading deterministic even if remote tool loops falter.

## Letta & Mem0 Summary (65 words)
Letta orchestrates agentic conversations with structured memory blocks, hybrid retrieval, adaptive reranking, and Self-RAG gating so every reply cites the strongest available evidence while staying terse. Mem0 supplies semantic vector storage that, when wrapped with the same policies, offers a parallel pipeline using its own API surface. Comparing them side-by-side isolates retrieval choices while sharing LLM generation, critique logic, and benchmarking telemetry.
