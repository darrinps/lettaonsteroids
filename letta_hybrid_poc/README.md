# Letta Hybrid POC

A production-ready demonstration of **Letta + Hybrid Retrieval (BM25 + FAISS) + Cross-Encoder Reranking + Self-RAG**, with **fair comparison** against Mem0 semantic search.

**Key Achievement:** Letta provides 100% recall with quality control, explainability, and critique - at only **+32.7% latency overhead** (3.7s vs 2.8s baseline).

**🆕 NEW: Four-Way Fair Comparison** - Now includes:
1. **Baseline** (keyword matching)
2. **Mem0 Basic** (vector search only)
3. **Mem0 Enhanced** (vector + same enhancements as Letta)
4. **Letta** (hybrid + all enhancements)

See [FAIR_COMPARISON.md](FAIR_COMPARISON.md) for why this matters!

Note: The test queries favor keyword matching, which is why Baseline and Letta (which includes BM25) may often get perfect scores, while pure semantic search (Mem0) will struggle. This is "real world" though.

---

## 👋 New Here?

**✅ Windows Users:** This works natively on Windows - no Ubuntu/WSL needed! (Unless you want to use Ollama instead of OpenAI)

**Building a UI?** → You don't need to install anything. Jump to [Path A: UI Development](#path-a-just-want-to-build-a-ui-no-installation-needed)

**Running evaluations or tests?** → You'll need Python and Poetry. See [Path B: Backend Setup](#path-b-run-the-evaluation-system-backend-setup---2-minutes)

---

## Table of Contents

- [Features](#features)
- [Key Technologies Explained](#key-technologies-explained)
- [Performance Results](#performance-results)
- [Interpreting Your Results](#interpreting-your-results)
- [Quick Start (2 Minutes)](#quick-start-2-minutes)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [API Server](#api-server)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Development](#development)
- [Changelog](#changelog)
- [License](#license)

## Features

### Core Capabilities

- **Dual LLM Support**: OpenAI (cloud, faster, **recommended - works on Windows natively**) or Ollama (local, free, requires WSL2 on Windows)
- **Baseline Mode**: Simple keyword-based retrieval (Mem0-style)
- **Augmented Mode**: Advanced RAG pipeline with:
  - Hybrid retrieval (BM25 sparse + FAISS dense)
  - Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
  - Self-RAG policies (should_retrieve + critique_and_correct)
  - Smart optimizations (conditional critique, adaptive reranking)
- **Real-Time API**: FastAPI + SSE for streaming results to any UI
- **Evaluation Suite**: Automated comparison with latency and recall metrics

### Optimizations

✅ **Heuristic Retrieval Decision** - Skips retrieval for simple queries (disabled during evaluation for accurate recall measurement)
✅ **Conditional Critique** - Only critiques long/uncertain answers
✅ **Smart Reranking** - Skips when BM25 and FAISS already agree (Letta-specific)
✅ **Lightweight Models** - Fast cross-encoder (22M params vs 110M)

**Note:** During evaluation, `force_retrieve=True` is used to ensure all systems retrieve documents for every query, allowing accurate recall measurement. In production/chat mode, the heuristic retrieval decision is active and saves time for simple conversational queries.

## Key Technologies Explained

### Letta
An agentic framework for building stateful LLM applications with long-term memory. Manages conversation context, retrieval decisions, and multi-step reasoning. Acts as the orchestration layer that decides when to retrieve information and how to process it.

### Mem0
A semantic memory library providing vector-based search using embeddings. Stores documents in a vector database (Qdrant) and retrieves them based on semantic similarity rather than keyword matching. Excels at finding conceptually related content even when exact terms differ.

### Hybrid Retrieval
Combines keyword search (BM25) with semantic search (FAISS vectors). BM25 finds exact term matches while FAISS captures semantic meaning. Results from both are merged and deduplicated, leveraging strengths of both approaches for more comprehensive document retrieval.

### Cross-Encoder Reranking
A transformer model that re-scores retrieved documents by evaluating query-document pairs together. More accurate than initial retrieval but computationally expensive, so it's applied only to top candidates (e.g., top-8) to refine ranking without reviewing all documents.

### Self-RAG
Quality control technique that makes LLMs self-assess their responses. Includes two components: (1) deciding whether to retrieve information, and (2) critiquing answers against retrieved context to detect hallucinations and unsupported claims. Improves reliability and factual accuracy.

## Performance Results

### Final Optimized Results

| Metric | Baseline | Augmented | Overhead |
|--------|----------|-----------|----------|
| **Avg Latency** | 2,788 ms | **3,700 ms** | **+32.7%** |
| **Avg Recall@5** | 100% | **100%** | - |

**Before Optimization:** 11.3s augmented (174.6% overhead)
**After Optimization:** 3.7s augmented (32.7% overhead)
**Improvement:** **67% faster** while maintaining 100% quality

### What You Get for +900ms

✅ **Quality Control**: Self-RAG critique validates answers against context
✅ **Explainability**: Full audit trail of retrieval decisions
✅ **Error Detection**: Catches hallucinations and unsupported claims
✅ **Better Retrieval**: Hybrid search finds more relevant documents
✅ **Consistency**: Always returns exactly top-k results

### Optimization Journey

**Original (Unoptimized):**
- Reranker: BAAI/bge-reranker-base (110M params)
- Reranking: Top-12 candidates
- Critique: Always on
- Retrieval decision: Always via LLM
- **Result:** 11.3s (too slow)

**Final (Optimized):**
- Reranker: ms-marco-MiniLM-L-6-v2 (22M params) - 2s saved
- Reranking: Top-8, skipped when sources agree - 1s saved
- Critique: Only for long/uncertain answers - 1.5s saved
- Retrieval decision: Heuristic for question words - 1.5s saved
- **Result:** 3.7s (practical!)

## Interpreting Your Results

When you run the evaluation, you'll see metrics for all four systems. Here's how to understand what they mean:

### Understanding the Four Systems

1. **Baseline (Keyword Matching)**
   - Simple token overlap scoring
   - Fast but limited semantic understanding
   - Often performs well when queries use exact keywords from documents

2. **Mem0 Basic (Vector Search Only)**
   - Pure semantic search using Mem0 library
   - OpenAI embeddings with Qdrant vector database
   - No reranking or quality controls

3. **Mem0 Enhanced (Vector + Optimizations)**
   - Same Mem0 vector search as Basic
   - **+ Cross-encoder reranking** (reorders results for relevance)
   - **+ Self-RAG critique** (validates answer quality)
   - **+ Heuristic retrieval decision** (skips retrieval for simple queries)

4. **Letta (Hybrid + All Optimizations)**
   - Hybrid retrieval combining BM25 (keyword) + FAISS (semantic)
   - **+ Cross-encoder reranking**
   - **+ Self-RAG critique**
   - **+ Heuristic retrieval decision**
   - **+ Adaptive reranking** (skips reranking when BM25/FAISS agree)

### Why we don't use adaptive reranking for Mem0
  - Adaptive reranking requires comparing two retrieval methods (BM25 vs FAISS)
  - Mem0 Enhanced only has one retrieval method (vector search)
  - There's nothing to compare or "agree" on with just one method

### Why Results Look the Way They Do

#### Why Does Baseline Often Get High Recall?

The test queries are designed to be **realistic** and often contain exact keywords from the documents (e.g., "What is Cedar policy language used for?" when the document title is "Cedar Policy Language"). This is how real users search! Baseline excels at keyword matching, so it performs well on these queries.

**This is expected behavior** - keyword search works great when users know the exact terms.

#### Why Doesn't Mem0 Enhanced Have Better Recall Than Mem0 Basic?

**Short answer:** They retrieve the **same documents**, so recall is identical.

**Here's why:**
- **Recall@k measures:** "Of the top-k documents retrieved, how many are relevant?"
- Both Mem0 Basic and Mem0 Enhanced use the **same Mem0 vector search** to retrieve documents
- **Reranking changes the ORDER of documents, not WHICH documents are retrieved**
- Since Recall@k only cares about "are relevant docs in the top-k?" (not their order), it stays the same

**What Mem0 Enhanced IS better at:**
- ✅ **Better ranking** - Most relevant documents appear first
- ✅ **Better answers** - Self-RAG critique catches hallucinations
- ✅ **Quality control** - Validates answers against retrieved context

#### Why Is Mem0 Enhanced Slower Than Mem0 Basic?

Mem0 Enhanced adds two expensive operations:

1. **Cross-encoder reranking** - Runs a separate model on each retrieved document (~200-500ms)
2. **Self-RAG critique** - Makes an LLM call to evaluate answer quality (~500-1000ms)

**Result:** Higher latency but better answer quality and relevance

#### What's the Point of the Four-Way Comparison?

The key insight comes from comparing **Mem0 Enhanced vs Letta**:
- Both have identical enhancements (reranking, critique, heuristic decision)
- The **only difference** is retrieval: Mem0 vector vs Letta hybrid (BM25+FAISS)
- This isolates whether hybrid retrieval finds better documents than pure vector search

**The comparison reveals:**
- **Baseline vs Mem0 Basic:** Keyword vs semantic search (raw comparison)
- **Mem0 Basic vs Mem0 Enhanced:** Impact of reranking + critique on vector search
- **Mem0 Enhanced vs Letta:** Pure vector vs hybrid retrieval (apples-to-apples!)
- **Baseline vs Letta:** Full journey from simple keyword to production-ready hybrid RAG

### Reading the 3D Graph

The dashboard shows a 3D bar chart with:

**Left side (Latency):**
- Shows milliseconds (1000, 2000, 3000, 4000, 5000)
- Taller bars = slower response time
- Expected order: Baseline < Mem0 Basic < Mem0 Enhanced ≈ Letta

**Right side (Recall):**
- Shows recall percentage (0.25, 0.50, 0.75, 1.0)
- Taller bars = found more relevant documents
- Perfect recall = 1.0 (bar reaches the top)

**Colors:**
- 🔵 Blue = Baseline
- 🟠 Orange = Mem0 Basic
- 🟣 Pink = Mem0 Enhanced
- 🟢 Green = Letta

### Common Scenarios and What They Mean

**Scenario 1: Baseline Gets 100% Recall**
- ✅ Expected! Test queries use exact keywords from documents
- This is "real world" - users often search with specific terms
- Shows keyword matching works when users know the vocabulary

**Scenario 2: Mem0 Basic Has Lower Recall**
- ✅ Expected! Vector search may miss keyword-specific queries
- Trade-off: Better for semantic similarity, weaker for exact term matching
- Hybrid (Letta) combines both strengths

**Scenario 3: Mem0 Enhanced and Mem0 Basic Have Same Recall**
- ✅ Expected! They retrieve the same documents, just reordered
- Mem0 Enhanced provides better ranking and answer quality
- Recall only measures "are relevant docs present?", not "are they ranked well?"

**Scenario 4: Letta Has Similar Latency to Mem0 Enhanced**
- ✅ Expected! Both use the same enhancements (reranking + critique)
- Letta's adaptive reranking optimization sometimes makes it faster
- The real difference is in retrieval quality, not speed

### Key Takeaways

1. **Recall measures retrieval, not ranking** - Same documents = same recall
2. **Enhancements add latency but improve quality** - Worth it for production
3. **Hybrid retrieval (Letta) combines strengths** - Keywords + semantics
4. **Test queries favor keyword matching** - Realistic user behavior
5. **Compare Mem0 Enhanced vs Letta** - This is the fairest comparison

See [FAIR_COMPARISON.md](FAIR_COMPARISON.md) for more details on why we test four systems!

## Quick Start

Choose your path:

### Path A: Just Want to Build a UI? (No Installation Needed!)

**The API server is already running or someone else runs it for you.**

1. Open `example_client.html` in your browser
2. Or connect from your app:
   ```javascript
   const eventSource = new EventSource('http://localhost:8000/api/eval/stream');
   eventSource.onmessage = (event) => {
     const data = JSON.parse(event.data);
     // Update your UI with real-time results
   };
   ```

**That's it!** No Poetry, no Python, no dependencies. Just HTTP and EventSource.

See [API Server](#api-server) section for full API documentation.

---

### Path B: Run the Evaluation System (Backend Setup)

**For running tests, evaluations, or starting the API server yourself.**

#### Prerequisites

**✅ Works natively on Windows - No Ubuntu/WSL needed!**

1. **Python 3.11+** - Check with `python --version`
2. **OpenAI API Key** - Get from https://platform.openai.com/api-keys
3. **Poetry** - See installation below if you don't have it

**Note:** Ubuntu/WSL is only needed if you choose to use Ollama (local LLM) instead of OpenAI. The default setup uses OpenAI and runs perfectly on Windows.

#### First-Time Setup: Install Poetry (Windows)

**Check if Poetry is already installed:**
```powershell
poetry --version
```

**If not installed, run in PowerShell:**
```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

**Add Poetry to PATH:**
1. Press `Win + R`, type `sysdm.cpl`, press Enter
2. Click "Advanced" tab → "Environment Variables"
3. Find "Path" under "User variables" → Click "Edit"
4. Click "New" and add: `%APPDATA%\Python\Scripts`
5. Click "OK" on all dialogs
6. **IMPORTANT:** Close and reopen PowerShell/Terminal
7. Verify: `poetry --version`

**Alternative (if above fails):**
```powershell
# Using pipx (recommended)
python -m pip install --user pipx
python -m pipx ensurepath
pipx install poetry
```

#### Project Setup (Windows PowerShell)

```powershell
# 1. Navigate to project
cd C:\AI_Work\letta_on_steroids\letta_hybrid_poc

# 2. Set up OpenAI API key
Copy-Item .env.example .env
notepad .env  # Add: OPENAI_API_KEY=sk-your-actual-key-here

# 3. Install dependencies
poetry install

# ⚠️ GOTCHA: If dependencies don't install or you get errors:
# Run: poetry lock
# Then: poetry install

# 4. Verify imports work
poetry run python test_imports.py

# 5. Test OpenAI connection
poetry run python test_openai.py

# 6. Build search index
poetry run python -m src.cli.ingest build

# 7. Verify index
poetry run python -m src.cli.ingest info

# 8. Run evaluation (see the results!)
poetry run python run_eval.py

# 9. Or try interactive chat
poetry run python -m src.cli.chat run --provider openai
```

Done! You've just run a full evaluation comparing baseline vs augmented modes.

**To start the API server for UIs:**
```powershell
.\start_server.bat
# Server runs at http://localhost:8000
# UIs can now connect (see Path A)
```

#### Setup for Linux/macOS

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Add to PATH (add to ~/.bashrc or ~/.zshrc)
export PATH="$HOME/.local/bin:$PATH"

# Then follow same steps as Windows (use 'cp' instead of 'Copy-Item')
cd letta_hybrid_poc
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=sk-your-key-here
poetry install
poetry run python -m src.cli.ingest build
poetry run python run_eval.py
```

---

### Path C: Interactive UI Dashboard

**To use the visually appealing dashboard to run and monitor evaluations.**

1.  **Start the API Server:**
    ```bash
    start_server.bat
    # Server runs at http://localhost:8000
    ```
    *Ensure the server is running before proceeding.*

2.  **Open the Dashboard:**
    Open `dashboard.html` in your web browser.

3.  **Control Evaluations:**
    Use the "Start Evaluation" and "Stop Evaluation" buttons within the dashboard to control the evaluation process. Results will be displayed incrementally and visualized in 3D.

---

## Architecture

```
User Query
    ↓
Self-RAG: should_retrieve? (heuristic + LLM fallback)
    ↓
Hybrid Retrieval (BM25 + FAISS) → Top-8 candidates
    ↓
Smart Reranking (skip if sources agree) → Top-5
    ↓
LLM Generation (OpenAI gpt-4o-mini or Ollama)
    ↓
Conditional Critique (only if needed)
    ↓
Final Answer + Metadata
```

### Component Breakdown

| Component | Purpose | Optimization |
|-----------|---------|--------------|
| **BM25** | Sparse keyword matching | Fast lexical search |
| **FAISS** | Dense semantic search | CPU-optimized IndexFlatL2 |
| **Hybrid** | Combine both signals | Weighted fusion |
| **Reranker** | Precision improvement | Lightweight 22M model, top-8 only |
| **Self-RAG** | Quality control | Heuristic + conditional critique |

## Installation

**👉 For complete installation instructions, see [Path B: Backend Setup](#path-b-run-the-evaluation-system-backend-setup) in the Quick Start section above.**

### Quick Summary

**UI Developers (Frontend Only):**
- ✅ Just a browser or HTTP client
- ❌ No Python, Poetry, or dependencies needed
- 👉 See [Path A: UI Development](#path-a-just-want-to-build-a-ui-no-installation-needed)

**Backend Developers / Running Evaluations:**
- ✅ Python 3.11+
- ✅ Poetry 1.7+ (installation instructions in Path B)
- ✅ OpenAI API Key (from https://platform.openai.com/api-keys) - **Recommended, works on Windows natively**
- ⚠️ Ollama (optional, only if you want local/free LLM - requires WSL2 on Windows)
- 👉 See [Path B: Backend Setup](#path-b-run-the-evaluation-system-backend-setup)

### Optional: Ollama Setup (Local Execution)

**⚠️ Only needed if you want local/offline execution instead of OpenAI.**

**The default setup uses OpenAI and does NOT require Ollama or Ubuntu/WSL.**

If you want to use Ollama instead of OpenAI:

**Windows (Native - Experimental):**
```powershell
winget install Ollama.Ollama
ollama serve
ollama pull mistral:latest
```

**Windows (WSL2/Ubuntu - Recommended for Ollama):**
```bash
# First, install WSL2 with Ubuntu if you haven't:
wsl --install

# Then inside WSL2 Ubuntu terminal:
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve
ollama pull mistral:latest
```

**Linux/macOS:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve
ollama pull mistral:latest
```

## Usage

### 1. Run Evaluation (Recommended First Step)

```bash
# Full three-way evaluation with OpenAI
poetry run python run_eval.py

# Results saved to eval_results_final.json
```

This runs 8 test queries through **three systems** (Baseline, Mem0, and Letta), showing:
- Three-way latency comparison
- Recall@5 metrics for each system
- Individual query results
- Aggregate statistics

**📖 First time using Mem0?** See [MEM0_SETUP.md](MEM0_SETUP.md) for installation and setup.

**Note:** If Mem0 isn't configured, evaluation will gracefully run with just Baseline vs Letta.

#### What Each Evaluation Query Tests

These 8 queries validate different aspects of the system's retrieval and reasoning capabilities:

**1. "What is Cedar policy language used for?"**
Tests multi-document synthesis across 3 technical documents. Proves the system can combine information from multiple sources into one coherent answer.

**2. "How do I ensure my website is ADA compliant?"**
Tests cross-document retrieval for compliance topics spanning 3 documents. Shows the system finds all relevant legal/regulatory information, not just one source.

**3. "What are the different types of dermal fillers?"**
Tests precision retrieval from a single medical document. Validates the system finds the right needle in the haystack without retrieving irrelevant information.

**4. "How long does Botox treatment last?"**
Tests factual lookup for specific medical information. Proves the system can extract precise facts (duration, numbers) from documents accurately.

**5. "What is the difference between BM25 and FAISS?"**
Tests comparative retrieval across 2 technical documents. Shows the system retrieves both concepts and helps compare them, not just one or the other.

**6. "What are WCAG 2.1 Level AA contrast requirements?"**
Tests retrieval of highly specific technical specifications. Validates the system handles precise regulatory requirements without confusing similar standards.

**7. "How does CoolSculpting work?"**
Tests medical procedure explanation retrieval. Proves the system can find and explain complex processes from specialized medical content.

**8. "What laser types are used for hair removal?"**
Tests technical detail extraction from medical equipment documentation. Shows the system retrieves specific technical specifications from domain-specific content.

**Coverage Summary:**
- **Domain Diversity:** Technical (3), Medical (4), Compliance (1)
- **Retrieval Complexity:** Single-doc (5), Multi-doc (3)
- **Query Types:** Factual lookup, comparison, explanation, specification

### 2. Interactive Chat

**Augmented Mode (Recommended):**
```bash
poetry run python -m src.cli.chat run --provider openai
```

**Baseline Mode:**
```bash
poetry run python -m src.cli.chat run --provider openai --mode baseline
```

**With Ollama (Local):**
```bash
poetry run python -m src.cli.chat run --provider ollama --ollama-model mistral:latest
```

**Chat Commands:**
- `quit` - Exit chat
- `/reset` - Clear conversation history
- `/mode` - Switch between baseline and augmented modes

### 3. Single Query

```bash
poetry run python -m src.cli.chat query "What is Cedar policy language?" --provider openai
```

### 4. API Server (Real-Time Streaming)

```bash
# Start server
start_server.bat

# Or manually
poetry run uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000
```

Then:
- **Web UI**: Open `example_client.html` in browser
- **Python Client**: `poetry run python example_client.py`
- **API Docs**: http://localhost:8000/docs
- **Your UI**: See [API_README.md](API_README.md) for integration guide

The API streams evaluation results in real-time via Server-Sent Events. Perfect for building dashboards or monitoring tools.

### 5. Index Management

```bash
# Rebuild index
poetry run python -m src.cli.ingest build

# Check index status
poetry run python -m src.cli.ingest info

# Custom corpus
poetry run python -m src.cli.ingest build --corpus-path path/to/corpus.json
```

## API Server

The project includes a production-ready FastAPI server for streaming evaluation results to any UI.

**Two Separate Roles:**

### Role 1: Backend Developer (Runs the Server)

**Requires:** Python, Poetry, OpenAI API Key

```bash
# Start server
start_server.bat

# Or manually
poetry run uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000
```

Server runs at `http://localhost:8000`

### Role 2: UI Developer (Consumes the API)

**Requires:** Nothing! Just HTTP.

```bash
# Open example UI (no installation needed!)
# Just open example_client.html in any browser
# Click "Start Evaluation" - works immediately
```

Or build your own UI - see examples below.

### API Endpoints

```
POST   /api/eval/start     - Start evaluation
GET    /api/eval/stream    - Stream results via SSE (real-time)
GET    /api/eval/status    - Get current status
GET    /api/eval/results   - Get completed results
POST   /api/eval/stop      - Stop running evaluation
```

### Example: JavaScript Client

```javascript
// Start evaluation
await fetch('http://localhost:8000/api/eval/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ provider: 'openai', top_k: 5 })
});

// Stream results
const eventSource = new EventSource('http://localhost:8000/api/eval/stream');
eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.type === 'query_completed') {
        console.log(`${data.query}: ${data.augmented.latency_ms}ms`);
    }
};
```

### Full API Documentation

See [API_README.md](API_README.md) for:
- Detailed endpoint documentation
- React/Vue/Python examples
- Production deployment guide
- Troubleshooting tips

## Configuration

### LLM Provider

**OpenAI (Default):**
```bash
poetry run python -m src.cli.chat run --provider openai --openai-model gpt-4o-mini
```

**Ollama:**
```bash
poetry run python -m src.cli.chat run --provider ollama --ollama-model mistral:latest
```

### Embedding Model

Edit embedding model in `src/cli/ingest.py`:

```python
# CPU-friendly options
"sentence-transformers/all-MiniLM-L6-v2"      # 384 dim, fast
"sentence-transformers/all-mpnet-base-v2"    # 768 dim, accurate
"BAAI/bge-small-en-v1.5"                      # 384 dim, balanced
```

### Reranker Model

Current: `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M params, fast)

Alternatives in `src/retrieval/rerank.py`:
```python
"BAAI/bge-reranker-base"                      # 110M params, slower, better
"BAAI/bge-reranker-large"                     # Best quality
"cross-encoder/ms-marco-MiniLM-L-6-v2"       # Fastest (current)
```

### Optimization Tuning

**Conditional Critique** (`src/core/letta_controller.py`):
```python
def _needs_critique(self, answer, retrieved_docs):
    word_count = len(answer.split())
    if word_count < 50:  # Adjust threshold
        return False
    # ...
```

**Reranking Candidates** (`src/core/memory.py`):
```python
retrieval_k=8  # Change to 6 (faster) or 10 (better recall)
```

**Smart Reranking** (`src/controller/tools.py`):
```python
if 2 <= bm25_count <= 3:  # Adjust balance threshold
    return False  # Skip reranking
```

## Project Structure

```
letta_hybrid_poc/
├── README.md                    # This file
├── API_README.md                # API documentation
├── CHANGELOG.md                 # Version history
├── pyproject.toml               # Dependencies
├── start_server.bat             # API server startup
├── run_eval.py                  # Evaluation runner
├── example_client.html          # Web UI example
├── example_client.py            # Python client example
│
├── data/
│   ├── raw/corpus.json          # Sample corpus (15 docs)
│   └── index/                   # Built index artifacts
│
├── src/
│   ├── adapters/
│   │   ├── llm_ollama.py        # Ollama LLM adapter
│   │   └── llm_openai.py        # OpenAI LLM adapter
│   │
│   ├── core/
│   │   ├── memory.py            # Baseline & Augmented memory
│   │   └── letta_controller.py # Main controller
│   │
│   ├── retrieval/
│   │   ├── index.py             # Hybrid index (BM25 + FAISS)
│   │   └── rerank.py            # Cross-encoder reranking
│   │
│   ├── controller/
│   │   ├── policies.py          # Self-RAG policies
│   │   └── tools.py             # Retrieval & memory tools
│   │
│   ├── cli/
│   │   ├── ingest.py            # Index building CLI
│   │   └── chat.py              # Interactive chat CLI
│   │
│   ├── eval/
│   │   └── eval.py              # Evaluation script
│   │
│   └── api/
│       ├── server.py            # FastAPI server
│       └── eval_runner.py       # Async evaluation runner
```

## Development

### Understanding the Tests

The project includes several tests to verify everything works correctly. Here's what each one checks:

**1. Import Test (`test_imports.py`)**
Verifies all required software packages are installed. This catches setup problems before you try to run the system. If this fails, your Python dependencies aren't installed correctly.

**2. OpenAI Connection Test (`test_openai.py`)**
Confirms your API key works and can communicate with OpenAI servers. If this passes, you can ask questions and get answers. Failures mean your API key is missing or invalid.

**3. Index Check (`ingest info`)**
Confirms the search database has been built from your documents. Without this index, the system can't find relevant information to answer questions. Build it with `ingest build`.

**4. Full Evaluation (`run_eval.py`)**
Compares two versions of the system using 8 test questions. Shows which approach is faster and more accurate. Results prove the advanced version finds better answers without being too slow.

**5. Test Suite (`run_tests.bat`)**
Runs tests 1-3 in sequence to verify your complete setup. Pass all three and you're ready to chat or run evaluations. One convenient command for troubleshooting.

### Run All Tests

```powershell
.\run_tests.bat
```

Or run individually:
```powershell
# Test 1: Verify packages
poetry run python test_imports.py

# Test 2: Verify OpenAI connection
poetry run python test_openai.py

# Test 3: Verify index exists
poetry run python -m src.cli.ingest info

# Test 4: Run full evaluation
poetry run python run_eval.py
```

### Add New Documents

1. Edit `data/raw/corpus.json`:
```json
{
  "id": "doc_016",
  "title": "Your Document Title",
  "content": "Your document content..."
}
```

2. Rebuild index:
```bash
poetry run python -m src.cli.ingest build
```

### Add Custom Evaluation Queries

Edit `src/eval/eval.py`:
```python
EVAL_QUERIES = [
    {
        "query": "Your custom query?",
        "relevant_docs": ["doc_001", "doc_002"],
        "category": "custom"
    },
    # ...
]
```

### GPU Support (Optional)

```bash
# Remove CPU PyTorch
poetry remove torch

# Add CUDA version (check your CUDA version)
poetry add torch --source https://download.pytorch.org/whl/cu118

# Use GPU
poetry run python -m src.cli.ingest build --device cuda
```

Expected speedup: 3-5x on indexing, 2-3x on retrieval+reranking.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and changes.

**Latest (v0.2.0):**
- Added OpenAI LLM support (recommended)
- Optimized augmented mode (11.3s → 3.7s)
- Added FastAPI + SSE streaming API
- Improved Self-RAG with heuristics
- Lightweight reranker

## Troubleshooting

### "poetry install" Doesn't Install New Dependencies

```
Package not found or dependencies not updating
```

**Problem:** Lock file needs updating when dependencies change

**Fix:**
```powershell
# Update lock file first
poetry lock

# Then install
poetry install
```

**When this happens:**
- After pulling changes that modify `pyproject.toml`
- After adding new dependencies manually
- After switching branches with different dependencies

### "poetry is not recognized" (Windows)

```
The term 'poetry' is not recognized as the name of a cmdlet, function, script file, or operable program.
```

**Problem:** Poetry not installed or not in PATH

**Fix:**
1. Install Poetry (see [Path B: Backend Setup](#path-b-run-the-evaluation-system-backend-setup))
2. Add `%APPDATA%\Python\Scripts` to PATH
3. **Close and reopen** PowerShell/Terminal (required!)
4. Verify: `poetry --version`

**Alternative install methods:**
```powershell
# Method 1: pipx (recommended)
python -m pip install --user pipx
python -m pipx ensurepath
pipx install poetry

# Method 2: pip (not recommended but works)
pip install --user poetry
```

### Python Not Found

```
'python' is not recognized...
```

**Fix:**
```powershell
# Install Python 3.11+
winget install Python.Python.3.11

# Or download from python.org
```

### OpenAI API Error

```
Error: Cannot connect to OpenAI
```

**Fix:**
```powershell
# Check .env file exists
dir .env

# Verify format: OPENAI_API_KEY=sk-...
notepad .env

# Test connection
poetry run python test_openai.py
```

### Ollama Connection Failed

```
Error: Cannot connect to Ollama at http://localhost:11434
```

**Fix:**
```bash
# Check if running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve

# Verify model
ollama list
```

### Index Not Found

```
Error: No index found at data/index
```

**Fix:**
```bash
poetry run python -m src.cli.ingest build
```

### Slow Performance

Expected latencies (CPU, OpenAI gpt-4o-mini):
- Baseline: ~2-3 seconds
- Augmented: ~3-4 seconds

**If slower:**
1. Check internet connection (OpenAI API)
2. Use GPU for embeddings (3-5x speedup)
3. Reduce top_k parameter
4. Try lighter embedding model

### Mem0 Initialization Error

```
Warning: Could not initialize Mem0: Failed to initialize Mem0: no such column: prev_value
```

**Problem:** Qdrant database schema incompatibility

**Fix:**
```powershell
# Delete the Qdrant database
Remove-Item -Recurse -Force data\qdrant

# Rerun evaluation (database rebuilds automatically)
poetry run python run_eval.py
```

**Note:** First run after deletion takes 30-60 seconds to rebuild the vector database.

### API Server Port Conflict

```bash
# Windows - find and kill process on port 8000
netstat -ano | findstr :8000
taskkill /PID <pid> /F
```

## Production Deployment

### AWS Migration Path

| Local Component | AWS Equivalent |
|----------------|----------------|
| FAISS (CPU) | OpenSearch with k-NN plugin |
| BM25 ranking | OpenSearch BM25 scoring |
| Cross-encoder | SageMaker endpoint (PyTorch) |
| OpenAI LLM | Bedrock (Claude) or keep OpenAI |
| Local disk cache | ElastiCache or S3 |
| FastAPI server | ECS Fargate or Lambda (function URL) |
| Policy controller | Lambda functions |

### Docker Example

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install poetry && poetry install --no-dev
EXPOSE 8000
CMD ["poetry", "run", "uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

## License

MIT

## Citation

```bibtex
@software{letta_hybrid_poc,
  title = {Letta Hybrid POC: Self-RAG with Optimized Hybrid Retrieval},
  year = {2025},
  version = {0.2.0}
}
```

## References

- [Self-RAG Paper](https://arxiv.org/abs/2310.11511)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25)
- [MS MARCO Reranker](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)
- [OpenAI API](https://platform.openai.com/docs/api-reference)
- [Ollama](https://ollama.ai/)
- [Letta Framework](https://github.com/cpacker/MemGPT)

## Acknowledgments

- **OpenAI** for GPT-4o-mini API
- **Ollama** for local LLM inference
- **HuggingFace** for embedding and reranking models
- **FAISS** for efficient vector search
- **Sentence Transformers** for embedding generation
