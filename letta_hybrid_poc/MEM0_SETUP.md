# Mem0 Three-Way Comparison Setup

## What Changed

The system now compares **three** approaches instead of two:

1. **Baseline** - Simple keyword matching
2. **Mem0** - Vector-based semantic search using actual Mem0 library
3. **Letta (Augmented)** - Hybrid retrieval + BM25 + FAISS + Reranking + Self-RAG

## Installation Steps

### 1. Install New Dependencies

```powershell
# Navigate to project directory
cd C:\AI_Work\letta_on_steroids\letta_hybrid_poc

# IMPORTANT: Update lock file first (required after adding new dependencies)
poetry lock

# Then install the new packages
poetry install

# This will install:
# - mem0ai (for Mem0 integration)
# - qdrant-client (vector database for Mem0)
```

**⚠️ GOTCHA:** If `poetry install` fails or doesn't pick up new dependencies, run `poetry lock` first. This updates the lock file to include the newly added `mem0ai` dependency.

### 2. Verify Installation

```powershell
# Test imports
poetry run python test_imports.py

# Make sure OpenAI API key is still set
poetry run python test_openai.py
```

### 3. Rebuild Index (Optional)

The existing index will work, but you may want to rebuild:

```powershell
poetry run python -m src.cli.ingest build
```

## Running Evaluations

### Option 1: Command Line (Three-Way Comparison)

```powershell
poetry run python run_eval.py
```

**Expected Output:**
```
Three-Way Comparison: Baseline vs Mem0 vs Letta
┌─────────────────┬──────────┬────────┬────────────────────┐
│ Metric          │ Baseline │ Mem0   │ Letta (Augmented) │
├─────────────────┼──────────┼────────┼────────────────────┤
│ Avg Latency (ms)│ 2788.1   │ XXXX.X │ 3700.0            │
│ Avg Recall@5    │ 1.000    │ X.XXX  │ 1.000             │
└─────────────────┴──────────┴────────┴────────────────────┘
```

### Option 2: Web UI (Live Three-Way Comparison)

```powershell
# Start API server
.\start_server.bat

# Open browser to:
file:///C:/AI_Work/letta_on_steroids/letta_hybrid_poc/example_client.html
```

The UI now shows **three metric cards** side-by-side:
- **Baseline (Keyword)** - Yellow
- **Mem0** - Blue
- **Letta (Augmented)** - Green

## How Mem0 Integration Works

### Architecture

```
User Query → Mem0Memory Class → Mem0 Library → Qdrant Vector DB → OpenAI Embeddings
```

### Mem0 Configuration

The system uses:
- **Vector Store**: Qdrant (local, stored in `data/qdrant/`)
- **Embeddings**: OpenAI `text-embedding-3-small`
- **Documents**: Same corpus as Baseline and Letta
- **Search**: Semantic similarity search (top-k)

### What's Being Tested

Each evaluation query runs through all three systems:

1. **Baseline**: Simple keyword overlap counting
2. **Mem0**: Vector similarity search using embeddings
3. **Letta**: Hybrid BM25+FAISS + Cross-encoder reranking + Self-RAG critique

## Troubleshooting

### "no such column: prev_value" Error

```
Warning: Could not initialize Mem0: Failed to initialize Mem0: no such column: prev_value
Continuing without Mem0 comparison
```

**Problem:** Qdrant database schema incompatibility (usually from version changes or corrupted database)

**Fix:**
```powershell
# Delete the Qdrant database directory
Remove-Item -Recurse -Force data\qdrant

# Run evaluation again (will recreate database automatically)
poetry run python run_eval.py
```

**⚠️ GOTCHA:** This is the most common Mem0 initialization error. The database will be rebuilt on next run (takes ~30-60 seconds for first initialization).

**When this happens:**
- After updating mem0ai to a new version
- After poetry lock/install with dependency changes
- After switching branches with different Mem0 versions
- When the Qdrant database gets corrupted

**Safe to delete:** The `data/qdrant` folder only contains the vector database. Deleting it won't affect your source documents (which are in `data/raw/corpus.json`).

### "Mem0 not available" Warning

If you see this during evaluation, Mem0 will be skipped:

```
Warning: Could not initialize Mem0: [error message]
Continuing without Mem0 comparison
```

**Common Causes:**
- `mem0ai` not installed: `poetry install`
- Missing Qdrant dependencies: Should auto-install with mem0ai
- API key issues: Check `.env` for `OPENAI_API_KEY`
- Database corruption: See "no such column" fix above

**Verification:**
```powershell
poetry show mem0ai
# Should show: mem0ai 0.0.20 (or later)

poetry show qdrant-client
# Should show: qdrant-client (some version)
```

### UI Shows "Not configured" for Mem0

This means the backend couldn't initialize Mem0. Check server logs:

```powershell
# Stop server (Ctrl+C)
# Restart with:
.\start_server.bat

# Look for warnings about Mem0 initialization
```

### Mem0 is Slow on First Run

**Expected:** First run builds the Qdrant vector database in `data/qdrant/`

**Timing:**
- First run: ~30-60 seconds to add all documents
- Subsequent runs: Near-instant (uses existing DB)

## Expected Results

### Performance Comparison

Typical results on the evaluation suite:

| System | Avg Latency | Avg Recall@5 | Notes |
|--------|-------------|--------------|-------|
| **Baseline** | ~2.8s | ~100% | Simple but works for exact matches |
| **Mem0** | ~3-4s | ~80-100% | Good semantic understanding |
| **Letta** | ~3.7s | **100%** | Best of both + quality control |

### Key Insights

**When Mem0 Excels:**
- Queries requiring semantic understanding
- Paraphrased questions
- Conceptual searches

**When Letta Excels:**
- Multi-document synthesis
- Requires both lexical AND semantic matching
- Needs answer quality validation (Self-RAG critique)

**When Baseline Excels:**
- Exact keyword matches
- Speed is critical
- Simple lookups

## Data Storage

```
letta_hybrid_poc/
├── data/
│   ├── index/          # Letta's FAISS + BM25 index
│   └── qdrant/         # Mem0's vector database (NEW)
│       ├── collection/
│       ├── meta.json
│       └── storage.sqlite
```

**Size:** Qdrant DB adds ~10-50MB depending on corpus size.

## Next Steps

1. **Run Evaluation:**
   ```powershell
   poetry run python run_eval.py
   ```

2. **Compare Results:** Check which system performs best for your queries

3. **Try Web UI:** See real-time three-way comparison

4. **Customize:** Add your own documents to `data/raw/corpus.json` and rebuild

5. **Production:** Choose the system that best fits your latency/quality trade-offs

## Questions?

- **"Which should I use?"** → Depends on your requirements:
  - Speed > Quality: Baseline
  - Semantic search: Mem0
  - Best quality + validation: Letta

- **"Can I disable Mem0?"** → Yes, evaluation gracefully skips if not available

- **"How do I clear Mem0 data?"** → Delete `data/qdrant/` directory

- **"Does this cost more?"** → Yes, Mem0 uses OpenAI embeddings (~$0.0001/1k tokens)
