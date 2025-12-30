# Letta vs Mem0 Benchmark App

A comprehensive benchmarking suite that compares Letta and Mem0 memory backends on key metrics including recall accuracy, memory governance, task continuity, and noise resilience. Mem0 here is a stand in for AWS AgentCore Memory

Conceptually, Mem0 is designed to solve the exact same problem as AWS AgentCore Memory, making it an excellent stand-in for a local test.
AWS AgentCore Memory is a fully managed service that provides both short-term (session) and long-term (persistent) memory. Its key feature is that it automatically and asynchronously runs "extraction strategies" to pull out facts, summaries, and user preferences from conversations to store in the long-term memory.
Mem0 is an open-source, self-hostable memory layer that does the same thing. It uses a "two-phase memory pipeline" to analyze conversations, extract salient facts, and store them in a queryable way

## Features

- **Accurate Grading**: Strict evaluation requiring both correct keywords AND concise responses
- **Semantic Similarity**: Optional OpenAI embedding-based similarity metrics
- **Multiple Test Scenarios** (9 total):
  - Cross-session recall (name/location memory)
  - Memory editing governance (updating preferences)
  - Task continuity (multi-step process tracking)
  - Degradation resilience (handling noisy inputs)
  - Temporal conflict resolution (recency handling)
  - Multi-fact inference (reasoning across facts)
  - Personality consistency (maintaining persona)
  - Conditional memory updates (business logic)
  - Memory scalability (50+ facts performance)
- **Configurable**: Customize backends, sessions, and noise levels

## Quick Start

### Using the Automated Scripts

**Windows:**
```cmd
run_benchmarks.bat [backend] [sessions] [noise]
```

**Linux/Mac:**
```bash
chmod +x run_benchmarks.sh
./run_benchmarks.sh [backend] [sessions] [noise]
```

**Parameters:**
- `backend`: `letta`, `mem0`, or `both` (default: `both`)
- `sessions`: Number of test sessions (default: `1`)
- `noise`: Noise ratio 0.0-1.0 (default: `0.0`)

**Examples:**
```bash
# Run both backends with default settings
./run_benchmarks.sh

# Test only Letta with 3 sessions
./run_benchmarks.sh letta 3 0.0

# Test both backends with noise
./run_benchmarks.sh both 1 0.25
```

### Manual Setup

**Windows PowerShell:**
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install openai  # Required for similarity metrics

# Set environment variables (create .env file)
$env:OPENAI_API_KEY="your-key-here"
$env:LETTA_BASE_URL="http://localhost:8283"
$env:MEM0_BASE_URL="http://localhost:3000"

# Run benchmarks
python -m src.benchmark --backend both --sessions 1 --noise 0.0 --out out\results.json --csv out\results.csv
```

**Linux/Mac/WSL:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install openai  # Required for similarity metrics

# Set environment variables (or use .env file)
export OPENAI_API_KEY="your-key-here"
export LETTA_BASE_URL="http://localhost:8283"
export MEM0_BASE_URL="http://localhost:3000"

# Run benchmarks
python -m src.benchmark --backend both --sessions 1 --noise 0.0 --out out/results.json --csv out/results.csv
```

## Prerequisites

1. **Python 3.9+** installed
2. **OpenAI API Key** (for similarity metrics) - set as `OPENAI_API_KEY` environment variable
3. **Backend Servers Running**:
   - **Letta**: Running at `http://localhost:8283` (or set `LETTA_BASE_URL`)
   - **Mem0**: Running at `http://localhost:3000` (or set `MEM0_BASE_URL`)

## Configuration

Create a `.env` file in the root directory:

```bash
# Required for similarity metrics
OPENAI_API_KEY=your-openai-api-key-here

# Backend URLs (defaults shown)
LETTA_BASE_URL=http://localhost:8283
MEM0_BASE_URL=http://localhost:3000

# Optional: Mem0 API key if using authentication
MEM0_API_KEY=

# Optional: Override model defaults
BENCH_DEFAULT_MODEL=openai/gpt-4o-mini
BENCH_DEFAULT_EMBEDDING=openai/text-embedding-3-small
```

## Understanding Results

### ⚠️ Important: Latency Comparison Note

**Latency metrics are NOT directly comparable** in this benchmark:
- **Letta**: Runs locally (localhost) with ~1-5ms network latency
- **Mem0**: Uses cloud API (api.mem0.ai) with ~100-400ms network latency per call

**Impact**: Mem0's latency includes significant network overhead. For example:
- Memory Scalability test (50+ API calls): ~5,000-20,000ms of pure network delay
- Other tests (3-4 API calls): ~300-600ms of network delay

**Mem0 would be 40-60% faster if run locally via Docker.**

✅ **Accuracy and Similarity metrics remain fair** - network latency doesn't affect response quality.

See [TEST_DOCUMENTATION.md](TEST_DOCUMENTATION.md#-important-latency-comparison-caveat) for detailed network overhead estimates per test.

---

### Output Files

- **`out/results.csv`**: Summary table with mean latency, accuracy, and similarity per scenario
- **`out/results.json`**: Detailed results including individual responses and grading

### Metrics Explained

**Accuracy Scoring:**
- **1.0** = Perfect: Correct keywords AND concise response (≤ word limit)
- **0.5** = Partial: Correct keywords but too verbose
- **0.0** = Failed: Missing required keywords

**Similarity Scoring:**
- Based on OpenAI embeddings (cosine similarity)
- Range: 0.0 (completely different) to 1.0 (identical meaning)
- Used as tiebreaker when accuracy = 0.0 and similarity ≥ 0.75 → gives 0.5 partial credit

### Example Results

```
backend                  scenario  mean_latency_ms  accuracy  mean_similarity
  letta      cross_session_recall      4864.648167       1.0         0.833355
  letta    degradation_resilience      6202.116300       1.0         0.885112
  letta memory_editing_governance      3780.381667       1.0         0.685483
  letta           task_continuity      3371.604133       1.0         0.880822
   mem0      cross_session_recall      3335.680667       0.0         0.614321
   mem0    degradation_resilience      3561.004850       0.0         0.000000
   mem0 memory_editing_governance     10185.221433       1.0         0.667638
   mem0           task_continuity     10567.665567       0.5         0.863086
```

## Test Scenarios

### 1. Cross-Session Recall
Tests if the system can remember name and location across multiple interactions.
- **Input**: "My name is Darrin Smith and I live in Phoenix."
- **Query**: "What is my name and city?"
- **Expected**: "Darrin Phoenix" (≤6 words)
- **Evaluates**: Basic memory persistence across conversation turns

### 2. Memory Editing Governance
Tests if the system can update/replace existing memories correctly.
- **Input**: "My favorite hotel brand is Hilton." → "Actually, replace that: my favorite hotel brand is Choice Hotels."
- **Query**: "What brand do I prefer now?"
- **Expected**: "Choice" (≤3 words)
- **Evaluates**: Memory update logic and conflict resolution

### 3. Task Continuity
Tests if the system can track multi-step processes.
- **Input**: "My rollout has 3 steps: Select tech, Pilot property, Full deploy."
- **Query**: "What's the next step after Select tech?"
- **Expected**: "Pilot" (≤5 words)
- **Evaluates**: Sequential task tracking and process memory

### 4. Degradation Resilience
Tests robustness with noisy/ambiguous inputs.
- **Input**: "Remember: Phoenix and Choice."
- **Query**: "What city and brand do I prefer?"
- **Expected**: "Phoenix Choice" (≤3 words)
- **Evaluates**: Graceful handling of incomplete/terse input

### 5. Temporal Conflict Resolution
Tests how the system handles conflicting information based on recency.
- **Input**: "My office is in San Francisco." → "My office moved to Austin."
- **Query**: "Where is my office?"
- **Expected**: "Austin" (≤3 words)
- **Evaluates**: Recency bias and temporal reasoning

### 6. Multi-Fact Inference
Tests reasoning across multiple stored facts to answer a complex query.
- **Input**: "I'm vegetarian, allergic to peanuts, love Italian food." → "I'm traveling to Rome."
- **Query**: "Recommend a dish."
- **Expected**: Answer reflecting all constraints (≤10 words)
- **Evaluates**: Multi-constraint reasoning and contextual inference

### 7. Personality Consistency
Tests if the system maintains a consistent personality/tone across interactions.
- **Input**: System configured with formal/professional persona
- **Interactions**: Multiple queries expecting consistent tone
- **Expected**: All responses maintain formal style
- **Evaluates**: Persona memory and behavioral consistency

### 8. Conditional Memory Updates
Tests if the system respects business logic rules when storing memories.
- **Input**: "Store my preference as Marriott only if I book 5+ nights."
- **Interaction**: User books 3 nights
- **Query**: "What's my preference?"
- **Expected**: No preference stored (≤5 words)
- **Evaluates**: Conditional logic and rule-based memory

### 9. Memory Scalability
Tests performance with large numbers of facts (50+ items).
- **Input**: 50 preference facts stored sequentially
- **Query**: Retrieve specific facts from early, middle, and late in sequence
- **Expected**: Accurate retrieval despite scale (≤5 words each)
- **Evaluates**: Scalability, retrieval speed, and accuracy under load

## Architecture

```
letta_mem_bench_full/
├── src/
│   ├── benchmark.py           # Main benchmark runner
│   ├── backends/
│   │   ├── letta.py          # Letta backend with memory blocks
│   │   └── mem0.py           # Mem0 backend with LLM formatting
│   ├── scenarios.py          # Test scenario definitions
│   └── config.py             # Environment configuration
├── out/                      # Output directory for results
├── run_benchmarks.sh         # Linux/Mac automated runner
├── run_benchmarks.bat        # Windows automated runner
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Key Improvements Made

### Benchmark Framework
1. ✅ **Strict Grading**: Added conciseness checks with word limits
2. ✅ **Fixed Accuracy Calculation**: "create" steps now excluded from accuracy averages
3. ✅ **Stricter Similarity Fallback**: Raised threshold from 0.30 → 0.75, gives 0.5 instead of 1.0

### Letta Backend
1. ✅ **Default Memory Blocks**: Added "human" and "persona" blocks for storage
2. ✅ **Enhanced System Prompt**: Clear instructions with examples for concise responses
3. ✅ **Result**: 100% accuracy across all scenarios

### Mem0 Backend
1. ✅ **Interface Adaptation**: Converted to match benchmark's expected API
2. ✅ **LLM Post-Processing**: Added GPT-4o-mini to format raw memories into concise responses
3. ✅ **Better Error Handling**: Graceful degradation when services unavailable

### Runner Scripts
1. ✅ **Argument Parsing**: Configurable backend, sessions, and noise
2. ✅ **Dependency Management**: Auto-installs openai package
3. ✅ **Health Checks**: Warns if backend servers unreachable
4. ✅ **Error Handling**: Proper exit codes and error messages

## Troubleshooting

**"Cannot reach Letta/Mem0 server"**
- Ensure the backend server is running
- Check the URL configuration in `.env`
- Verify firewall/network settings

**"OPENAI_API_KEY not set"**
- Set the environment variable: `export OPENAI_API_KEY=your-key`
- Or add to `.env` file
- Similarity metrics will be disabled without it (accuracy still works)

**"ModuleNotFoundError: No module named 'openai'"**
- Run: `pip install openai`
- Or: `./run_benchmarks.sh` (auto-installs)

**Tests show 0% accuracy**
- Verify backend servers are properly configured
- Check that memory blocks are being created (Letta)
- Review `out/results.json` for detailed error messages

## Contributing

When making changes:
1. Test both Letta and Mem0 backends
2. Run with multiple sessions: `./run_benchmarks.sh both 3 0.0`
3. Verify accuracy metrics make sense (not inflated by lenient grading)
4. Update this README with any new features

## License

See project root for license information.
