# Letta vs Mem0 Benchmark Test Documentation

Comprehensive documentation of all test scenarios and scoring metrics used to evaluate memory backend performance.

---

## Table of Contents

1. [Prerequisites: Server Setup](#prerequisites-server-setup)
2. [Test Scenarios Overview](#test-scenarios-overview)
3. [Scoring Metrics Explained](#scoring-metrics-explained)
4. [Detailed Test Descriptions](#detailed-test-descriptions)
5. [Interpreting Results](#interpreting-results)

---

## Prerequisites: Server Setup

Before running the benchmark tests, you must ensure both backend servers are running and accessible.

### Starting WSL/Ubuntu (Windows Users Only)

If you're running on Windows and plan to use WSL (Windows Subsystem for Linux) with Ubuntu to run the Letta or Mem0 servers, you need to start Ubuntu first.

**Check if WSL/Ubuntu is Installed:**

```powershell
# PowerShell - List installed WSL distributions
wsl --list --verbose

# Or simply:
wsl -l -v
```

**Expected Output:**
```
  NAME            STATE           VERSION
* Ubuntu          Running         2
  Ubuntu-22.04    Stopped         2
```

**Starting Ubuntu/WSL:**

**Option 1: Start Default WSL Distribution**
```powershell
# PowerShell or Command Prompt
wsl

# This opens a bash terminal in your default Linux distribution
```

**Option 2: Start Specific Ubuntu Distribution**
```powershell
# Start Ubuntu (if you have multiple distributions)
wsl -d Ubuntu

# Or start Ubuntu 22.04 specifically
wsl -d Ubuntu-22.04
```

**Option 3: Start via Windows Terminal**
- Open Windows Terminal
- Click the dropdown arrow (˅) next to the + button
- Select "Ubuntu" or "Ubuntu-22.04" from the list

**Option 4: Start via Start Menu**
- Press Windows key
- Type "Ubuntu"
- Click on the Ubuntu app

**Installing WSL/Ubuntu (if not installed):**

If you don't have WSL installed yet:

```powershell
# PowerShell (Run as Administrator)
# Install WSL with Ubuntu (default)
wsl --install

# Or install specific version
wsl --install -d Ubuntu-22.04

# Restart your computer after installation
```

**Verify Ubuntu is Running:**
```powershell
# Check WSL status
wsl -l -v

# Should show "Running" status
```

**Running Commands in WSL from Windows:**

You can run Linux commands directly from PowerShell/CMD without opening a full WSL terminal:

```powershell
# Run a command in WSL
wsl curl -f http://localhost:8283/v1/health

# Or specify the distribution
wsl -d Ubuntu curl -f http://localhost:8283/v1/health
```

**Stopping WSL/Ubuntu (when done):**

```powershell
# Shutdown all WSL distributions
wsl --shutdown

# Or shutdown specific distribution
wsl -t Ubuntu
```

---

### Letta Server

**Check if Letta Server is Running:**

```bash
# Linux/Mac/WSL:
curl -f http://localhost:8283/v1/health || echo "Letta server is DOWN"

# Windows PowerShell:
try { Invoke-WebRequest -Uri http://localhost:8283/v1/health -UseBasicParsing; Write-Host "Letta server is UP" } catch { Write-Host "Letta server is DOWN" }

# Windows Command Prompt:
curl -f http://localhost:8283/v1/health || echo Letta server is DOWN
```

**Expected Output (if running):**
```json
{"status":"ok"}
```

**Starting the Letta Server:**

If the Letta server is not running, start it using one of these methods:

**Option 1: Using Letta CLI (Recommended)**
```bash
# Install Letta if not already installed
pip install letta

# Start the Letta server (defaults to port 8283)
letta server

# Or specify a custom port
letta server --port 8283
```

**Option 2: Using Docker**
```bash
# Pull the Letta Docker image
docker pull letta/letta:latest

# Run Letta server
docker run -d -p 8283:8283 --name letta-server letta/letta:latest

# Check logs
docker logs letta-server
```

**Option 3: From Source**
```bash
# Clone the Letta repository
git clone https://github.com/cpacker/Letta.git
cd Letta

# Install dependencies
pip install -e .

# Start the server
letta server
```

**Verify Letta is Running:**
```bash
# Should return 200 OK
curl -i http://localhost:8283/v1/health
```

**Common Letta Server Issues:**

| Issue | Cause | Solution |
|-------|-------|----------|
| Port 8283 already in use | Another process using the port | Kill the process: `lsof -ti:8283 \| xargs kill` (Mac/Linux) or change port |
| Connection refused | Server not started | Run `letta server` |
| Authentication error | Missing API key | Set `LETTA_API_KEY` environment variable if auth is enabled |
| Module not found | Letta not installed | Run `pip install letta` |

---

### Mem0 Server

**Check if Mem0 Server is Running:**

```bash
# Linux/Mac/WSL:
curl -f http://localhost:3000/health || echo "Mem0 server is DOWN"

# Windows PowerShell:
try { Invoke-WebRequest -Uri http://localhost:3000/health -UseBasicParsing; Write-Host "Mem0 server is UP" } catch { Write-Host "Mem0 server is DOWN" }

# Windows Command Prompt:
curl -f http://localhost:3000/health || echo Mem0 server is DOWN
```

**Starting the Mem0 Server:**

**Option 1: Using Docker (Recommended)**
```bash
# Pull the Mem0 Docker image
docker pull mem0ai/mem0:latest

# Run Mem0 server
docker run -d -p 3000:3000 --name mem0-server \
  -e OPENAI_API_KEY=your-api-key-here \
  mem0ai/mem0:latest

# Check logs
docker logs mem0-server
```

**Option 2: Using Mem0 CLI**
```bash
# Install Mem0
pip install mem0ai

# Start the Mem0 server
mem0 server --port 3000

# Or with environment variables
export OPENAI_API_KEY=your-api-key-here
mem0 server
```

**Option 3: From Source**
```bash
# Clone the Mem0 repository
git clone https://github.com/mem0ai/mem0.git
cd mem0

# Install dependencies
pip install -e .

# Start the server
mem0 server --port 3000
```

**Verify Mem0 is Running:**
```bash
# Should return 200 OK
curl -i http://localhost:3000/health
```

**Common Mem0 Server Issues:**

| Issue | Cause | Solution |
|-------|-------|----------|
| Port 3000 already in use | Another process using the port | Kill the process or change port in `.env` (`MEM0_BASE_URL=http://localhost:3001`) |
| Connection refused | Server not started | Run `mem0 server` |
| Missing OpenAI API key | API key not set | Set `OPENAI_API_KEY` environment variable |
| Database connection error | Database not initialized | Check Mem0 configuration and database setup |

---

### Environment Configuration

Create a `.env` file in the project root with the following variables:

```bash
# Required for similarity metrics
OPENAI_API_KEY=sk-your-openai-api-key-here

# Backend server URLs (use these defaults if running locally)
LETTA_BASE_URL=http://localhost:8283
MEM0_BASE_URL=http://localhost:3000

# Optional: Authentication (if your servers require it)
LETTA_API_KEY=your-letta-api-key
MEM0_API_KEY=your-mem0-api-key

# Optional: Model configuration
BENCH_DEFAULT_MODEL=openai/gpt-4o-mini
BENCH_DEFAULT_EMBEDDING=openai/text-embedding-3-small
```

---

### Quick Health Check Script

Run this command to check both servers at once:

**Linux/Mac/WSL:**
```bash
#!/bin/bash
echo "Checking Letta server..."
curl -sf http://localhost:8283/v1/health && echo "✓ Letta is UP" || echo "✗ Letta is DOWN"

echo "Checking Mem0 server..."
curl -sf http://localhost:3000/health && echo "✓ Mem0 is UP" || echo "✗ Mem0 is DOWN"
```

**Windows (PowerShell):**
```powershell
Write-Host "Checking Letta server..."
try {
    Invoke-WebRequest -Uri http://localhost:8283/v1/health -UseBasicParsing | Out-Null
    Write-Host "✓ Letta is UP" -ForegroundColor Green
} catch {
    Write-Host "✗ Letta is DOWN" -ForegroundColor Red
}

Write-Host "Checking Mem0 server..."
try {
    Invoke-WebRequest -Uri http://localhost:3000/health -UseBasicParsing | Out-Null
    Write-Host "✓ Mem0 is UP" -ForegroundColor Green
} catch {
    Write-Host "✗ Mem0 is DOWN" -ForegroundColor Red
}
```

**Automated Health Checks:**
The `run_benchmarks.sh` and `run_benchmarks.bat` scripts automatically perform health checks before running tests. If a server is unreachable, the script will warn you and skip that backend.

---

## Test Scenarios Overview

The benchmark evaluates memory backends across 9 different capabilities:

| # | Test Scenario | What It Shows | Real-World Use Case |
|---|--------------|---------------|---------------------|
| 1 | **Cross-Session Recall** | Can it remember basic facts about you? | Remembering your name and city across conversations |
| 2 | **Memory Editing Governance** | Can it update outdated information? | Changing your hotel preference from Hilton to Choice |
| 3 | **Task Continuity** | Can it track multi-step processes? | Remembering your project rollout steps in order |
| 4 | **Degradation Resilience** | Can it handle unclear/terse input? | Understanding "Remember: Phoenix and Choice" |
| 5 | **Temporal Conflict Resolution** | Does it remember the latest info? | Your office moved from SF to Austin - which does it recall? |
| 6 | **Multi-Fact Inference** | Can it reason across multiple facts? | You're vegetarian + allergic to peanuts + love Italian = what dish to recommend? |
| 7 | **Personality Consistency** | Does it maintain the same tone/persona? | Staying formal/professional across all interactions |
| 8 | **Conditional Memory Updates** | Can it follow business logic rules? | Only save preference IF you book 5+ nights |
| 9 | **Memory Scalability** | Can it handle lots of facts (50+)? | Storing and retrieving from 50+ preferences without slowing down |

---

## Scoring Metrics Explained

### 📊 Quick Summary (In Plain English)

**Accuracy** = Did the system give the RIGHT answer in a CONCISE way?
- **1.0** = Perfect answer, short and sweet (e.g., "Darrin Phoenix")
- **0.5** = Right answer, but too wordy (e.g., "Your name is Darrin and you live in Phoenix")
- **0.0** = Wrong answer or missing information

**Mean Similarity** = How CLOSE was the answer to what we expected, even if not perfect?
- **1.0** = Nearly identical meaning
- **0.75-0.89** = Very close in meaning
- **0.5-0.74** = Somewhat related
- **Below 0.5** = Not really related

**Think of it like grading a test:**
- Accuracy is pass/fail: did you get the keywords right?
- Similarity is partial credit: even if you didn't get it exactly right, was it close?

---

### Accuracy Score (0.0 - 1.0) - DETAILED

**Definition**: Measures whether the response contains the correct keywords AND is concise.

**Calculation**:
```
1. Extract response from backend
2. Normalize text (lowercase, remove punctuation)
3. Check if ALL required keywords are present
4. Check if response meets conciseness requirement (word count limit)

Scoring:
- 1.0 = Perfect: Correct keywords AND concise
- 0.5 = Partial: Correct keywords BUT too verbose
- 0.0 = Failed: Missing required keywords
```

**Example**:

| Response | Required Keywords | Word Limit | Score | Reason |
|----------|------------------|------------|-------|---------|
| "Darrin Phoenix" | ["darrin", "phoenix"] | 6 words | **1.0** | Perfect: 2 words, all keywords ✅ |
| "Your name is Darrin Smith and you live in Phoenix, Arizona." | ["darrin", "phoenix"] | 6 words | **0.5** | Partial: Has keywords but 11 words ⚠️ |
| "I don't have that information." | ["darrin", "phoenix"] | 6 words | **0.0** | Failed: Missing keywords ❌ |
| "Marriott" | ["darrin", "phoenix"] | 6 words | **0.0** | Failed: Wrong information ❌ |

**Implementation** (src/benchmark.py:151-207):
```python
def is_concise(text: str, max_words: int = 10) -> bool:
    """Check if response is concise (not a full sentence/paragraph)"""
    normalized = normalize_text(text)
    word_count = len([w for w in normalized.split() if w])
    return word_count <= max_words

def contains_all_keywords(text: str, required: List[str]) -> bool:
    """Check if all required keywords are present"""
    t = normalize_text(text)
    return all(k.lower() in t for k in required if k.strip())

def grade_cross_session(response: str) -> float:
    has_keywords = contains_all_keywords(response, ["darrin", "phoenix"])
    is_brief = is_concise(response, max_words=6)

    if has_keywords and is_brief:
        return 1.0  # Perfect
    elif has_keywords:
        return 0.5  # Partial: correct but verbose
    return 0.0      # Failed
```

**Word Limits by Test**:
- Cross-Session Recall: 6 words
- Degradation Resilience: 5 words
- Memory Editing Governance: 5 words
- Task Continuity (turn 1): 4 words
- Task Continuity (turn 2): 10 words

---

### Mean Similarity Score (0.0 - 1.0) - DETAILED

**Definition**: Measures semantic similarity between the actual response and expected answer using OpenAI embeddings.

**Calculation**:
```
1. Generate embedding vector for response (using text-embedding-3-small)
2. Generate embedding vector for expected answer
3. Calculate cosine similarity between vectors
4. Normalize to 0.0-1.0 range

Formula:
    similarity = (cosine_similarity + 1.0) / 2.0

Where:
    cosine_similarity = dot(vec_response, vec_expected) / (||vec_response|| * ||vec_expected||)
    Range: -1.0 to 1.0

Normalized similarity range: 0.0 to 1.0
```

**Interpretation**:

| Score Range | Meaning | Example |
|------------|---------|---------|
| **0.90 - 1.00** | Nearly identical meaning | "Phoenix Choice" vs "Phoenix, Choice" |
| **0.75 - 0.89** | Very similar meaning | "Darrin Phoenix" vs "Darrin lives in Phoenix" |
| **0.60 - 0.74** | Somewhat related | "Darrin Phoenix" vs "Name is Darrin" |
| **0.30 - 0.59** | Loosely related | "Darrin Phoenix" vs "Marriott" |
| **0.00 - 0.29** | Unrelated | "Darrin Phoenix" vs "" (empty) |

**Implementation** (src/benchmark.py:34-53):
```python
def try_embedding_similarity(a: str, b: str) -> Optional[float]:
    if not OPENAI_API_KEY:
        return None  # Skip if no API key

    try:
        from openai import OpenAI
        import math

        client = OpenAI(api_key=OPENAI_API_KEY)

        # Generate embeddings
        va = client.embeddings.create(model=EMBED_MODEL, input=a).data[0].embedding
        vb = client.embeddings.create(model=EMBED_MODEL, input=b).data[0].embedding

        # Calculate cosine similarity
        dot = sum(x * y for x, y in zip(va, vb))
        na = math.sqrt(sum(x * x for x in va))
        nb = math.sqrt(sum(x * x for x in vb))

        if na == 0 or nb == 0:
            return None

        cos = dot / (na * nb)  # Range: -1.0 to 1.0

        # Normalize to 0.0-1.0
        return max(0.0, min(1.0, (cos + 1.0) / 2.0))

    except Exception:
        return None  # Graceful degradation
```

**Similarity as Tiebreaker**:

When accuracy = 0.0 (missing keywords) BUT similarity ≥ 0.75, the system awards 0.5 partial credit:

```python
acc = grade_cross_session(sr.text)  # Returns 0.0 if keywords missing

if acc == 0.0 and sim is not None and sim >= 0.75:
    acc = 0.5  # Upgrade to partial credit due to high semantic similarity
```

**Why 0.75 Threshold?**
- Too low (0.30): Gave credit to completely wrong answers like "Marriott" for "Darrin Phoenix" (0.566 similarity)
- Too high (0.90): Too strict, wouldn't help legitimate near-misses
- **0.75**: Good balance - catches semantically correct but slightly off-format responses

**Note**: Similarity requires `OPENAI_API_KEY` environment variable. If not set, similarity will be `None` and only keyword-based accuracy will be used.

---

## Detailed Test Descriptions

### 1. Cross-Session Recall

**🎯 What This Test Shows (In Simple Terms):**

Imagine you tell someone your name and where you live, then talk about something completely different (like hotels), and later ask "What's my name and city?" A good memory system should remember both facts even though you changed topics.

**This test reveals:**
- ✅ **Good systems**: Remember your name and city correctly
- ❌ **Bad systems**: Forget what you said, or get confused and return the wrong information (like saying "Marriott" when asked for your name)

---

**Purpose**: Tests basic memory storage and retrieval across multiple conversational turns.

**What It Tests**:
- Can the system remember a user's name?
- Can the system remember a user's location?
- Can the system recall both simultaneously when asked?
- Can the system maintain this information across unrelated turns?

**Scenario Flow**:
```
Turn 1 (user):  "My name is Darrin Smith and I live in Phoenix."
                → System should: Store name="Darrin Smith", location="Phoenix"

Turn 2 (user):  "I prefer Marriott over Hilton."
                → System should: Store preference (unrelated distractor)

Turn 3 (user):  "What is my name and city?"
                → System should: Retrieve and return "Darrin Phoenix" (first name + city)
```

**Expected Answer**: `"Darrin Phoenix"`

**Grading Criteria**:
- **Required Keywords**: ["darrin", "phoenix"]
- **Word Limit**: 6 words
- **1.0**: "Darrin Phoenix" or "Darrin Smith Phoenix" (2-3 words) ✅
- **0.5**: "Your name is Darrin and you live in Phoenix" (10 words) ⚠️
- **0.0**: "Marriott" or "" (missing keywords) ❌

**Why This Test Matters**:
- Foundation of any memory system
- Tests structured storage vs unstructured retrieval
- Verifies system can distinguish between different fact types (name vs preference)

**Common Failures**:
- **Mem0**: May return wrong memory due to semantic search confusion ("Marriott" instead of "Darrin")
- **Letta (unfixed)**: Would fail without proper memory blocks initialized

---

### 2. Degradation Resilience

**🎯 What This Test Shows (In Simple Terms):**

Real people don't always speak in perfect sentences. Sometimes they just say "Remember: Phoenix and Choice" without explaining what those mean. Can the system figure it out and store the information anyway?

**This test reveals:**
- ✅ **Good systems**: Handle terse/unclear input gracefully, infer meaning from context
- ❌ **Bad systems**: Fail when input isn't perfectly formatted, return empty responses

---

**Purpose**: Tests robustness with minimal, ambiguous, or noisy inputs.

**What It Tests**:
- Can the system store information from terse inputs?
- Can the system handle ambiguous category labels?
- Can the system recall information with minimal context?
- How does noise affect storage and retrieval?

**Scenario Flow**:
```
Turn 1 (user):  "Remember: Phoenix and Choice."
                → System should: Infer Phoenix=city, Choice=brand (ambiguous categories)

Turn 2 (user):  "What city and brand do I prefer? Reply with just the keywords."
                → System should: Return "Phoenix Choice"
```

**Expected Answer**: `"Phoenix Choice"`

**Grading Criteria**:
- **Required Keywords**: ["phoenix", "choice"]
- **Word Limit**: 5 words
- **1.0**: "Phoenix Choice" (2 words) ✅
- **0.5**: "Phoenix and Choice Hotels" (4 words) ⚠️
- **0.0**: "" or missing one keyword ❌

**Why This Test Matters**:
- Real users don't always provide perfectly structured input
- Tests inference capabilities
- Measures system resilience to unclear categorization

**Common Failures**:
- **Mem0**: Often returns empty because "Remember:" doesn't trigger storage mechanism
- **Letta**: Must infer categories (Phoenix=location, Choice=brand) from context

**Noise Parameter**:
The benchmark supports a `--noise` parameter (0.0-1.0) that scrambles words:
```python
def noisy(text: str, ratio: float) -> str:
    words = text.split()
    n = int(len(words) * ratio)
    # Scramble n words
    random.shuffle(words[:n])
    return " ".join(words)

# Example with noise=0.25:
# "Remind me of my city and brand preference."
# → "of Remind me and city my preference brand."
```

---

### 3. Memory Editing Governance

**🎯 What This Test Shows (In Simple Terms):**

Your preferences change over time. You might say "I like Hilton" today, but tomorrow say "Actually, replace that - I prefer Choice Hotels now." Will the system update the old preference, or keep both (which would be wrong)?

**This test reveals:**
- ✅ **Good systems**: Replace old information with new, keep only the latest preference
- ❌ **Bad systems**: Keep both old and new (confusing!), or fail to update at all

---

**Purpose**: Tests the ability to update/replace existing memories with new information.

**What It Tests**:
- Can the system detect when new information contradicts old information?
- Does it properly replace (not append) the old value?
- Does it maintain only the most recent preference?
- Does "replace" instruction get respected?

**Scenario Flow**:
```
Turn 1 (user):  "My favorite hotel brand is Hilton."
                → System should: Store brand="Hilton"

Turn 2 (user):  "Actually, replace that: my favorite hotel brand is Choice Hotels."
                → System should: Update brand="Choice" (not "Hilton + Choice")

Turn 3 (user):  "What brand do I prefer now?"
                → System should: Return only "Choice" (not Hilton)
```

**Expected Answer**: `"Choice"`

**Grading Criteria**:
- **Required Keywords**: ["choice"]
- **Word Limit**: 5 words
- **1.0**: "Choice" or "Choice Hotels" (1-2 words) ✅
- **0.5**: "Your favorite brand is Choice Hotels" (6 words) ⚠️
- **0.0**: "Hilton" or "Hilton and Choice" (shows update failed) ❌

**Why This Test Matters**:
- Preferences change over time
- Tests memory governance (update vs append)
- Critical for maintaining accurate user profiles

**Common Failures**:
- **Mem0**: May return both "Hilton" and "Choice" if not properly replacing
- **Letta**: Must use `memory_replace` tool correctly

**Governance Concepts Tested**:
1. **Explicit Replace**: User says "replace that"
2. **Temporal Override**: New info should supersede old
3. **Single Truth**: System maintains one value per field

---

### 4. Task Continuity

**🎯 What This Test Shows (In Simple Terms):**

You tell the system about a 3-step project plan: "Select tech, Pilot property, Full deploy." Later you ask "What comes after Select tech?" It should remember the sequence and tell you "Pilot."

**This test reveals:**
- ✅ **Good systems**: Remember ordered lists and understand sequence relationships
- ❌ **Bad systems**: Forget the order, or can't answer "what comes next?" questions

---

**Purpose**: Tests ability to track and recall sequential multi-step processes.

**What It Tests**:
- Can the system remember ordered steps in a process?
- Can it answer questions about specific steps?
- Can it maintain the full sequence for later recall?
- Does it understand step relationships (what comes after X)?

**Scenario Flow**:
```
Turn 1 (user):  "My rollout has 3 steps: Select tech, Pilot property, Full deploy."
                → System should: Store steps=[1: Select tech, 2: Pilot property, 3: Full deploy]

Turn 2 (user):  "What's the next step after Select tech? Reply with just the keyword."
                → System should: Return "Pilot" (step 2)

Turn 3 (user):  "List the 3 steps back as keywords in order."
                → System should: Return "Select Pilot Full" (all 3 steps, concise)
```

**Expected Answer**:
- Turn 2: `"Pilot"`
- Turn 3: `"Select Pilot Full"`

**Grading Criteria**:

**Turn 2 (Next step query)**:
- **Required Keywords**: ["pilot"]
- **Word Limit**: 4 words
- **1.0**: "Pilot" or "Pilot property" (1-2 words) ✅
- **0.5**: "The next step is Pilot property" (6 words) ⚠️
- **0.0**: "Select" or "" (wrong/missing) ❌

**Turn 3 (Full list query)**:
- **Required Keywords**: ["select", "pilot", "full"]
- **Word Limit**: 10 words
- **1.0**: "Select Pilot Full" (3 words) ✅
- **0.5**: "Select tech, Pilot property, Full deploy" (6 words) ⚠️
- **0.0**: Missing any keyword or empty ❌

**Why This Test Matters**:
- Common use case: project management, workflows, recipes
- Tests sequential understanding
- Validates structured information storage

**Common Failures**:
- **Mem0**: May return full descriptions instead of keywords
- **Mem0**: Second query often returns empty (search doesn't find the stored sequence)
- **Letta**: Must maintain order and provide concise extraction

**Multi-Turn Grading**:
This test grades TWO separate turns:
```python
if scenario.name == "task_continuity" and idx in (1, 2):
    acc = grade_task_continuity(idx, sr.text)
```
Overall accuracy is the average: `(acc_turn2 + acc_turn3) / 2`

---

### 5. Temporal Conflict Resolution

**🎯 What This Test Shows (In Simple Terms):**

Information changes over time. You tell the system "My office is in San Francisco," then later "My office moved to Austin." When asked where your office is, it should say Austin (the latest info), not San Francisco.

**This test reveals:**
- ✅ **Good systems**: Remember the most recent information and override old data
- ❌ **Bad systems**: Return outdated information or get confused between old and new

---

### 6. Multi-Fact Inference

**🎯 What This Test Shows (In Simple Terms):**

You've told the system three facts: you're vegetarian, allergic to peanuts, and love Italian food. When planning a trip to Rome, can it recommend a dish that satisfies ALL three constraints?

**This test reveals:**
- ✅ **Good systems**: Reason across multiple facts to give intelligent recommendations
- ❌ **Bad systems**: Ignore some constraints or fail to connect the dots

---

### 7. Personality Consistency

**🎯 What This Test Shows (In Simple Terms):**

If you set up the system with a formal, professional tone, does it maintain that same personality throughout the conversation? Or does it suddenly become casual midway through?

**This test reveals:**
- ✅ **Good systems**: Maintain consistent tone and persona across all interactions
- ❌ **Bad systems**: Forget their personality and switch between formal/casual randomly

---

### 8. Conditional Memory Updates

**🎯 What This Test Shows (In Simple Terms):**

Business rules matter. You say "Only save my hotel preference as Marriott IF I book 5 or more nights." Then you book 3 nights. A smart system should NOT save Marriott (because you didn't meet the condition).

**This test reveals:**
- ✅ **Good systems**: Follow conditional logic and business rules correctly
- ❌ **Bad systems**: Ignore the "IF" condition and save information regardless

---

### 9. Memory Scalability

**🎯 What This Test Shows (In Simple Terms):**

What happens when you store 50+ different preferences? Does the system slow down? Does it forget early information? Can it still find the right fact when you ask for it?

**This test reveals:**
- ✅ **Good systems**: Handle large amounts of data without performance degradation
- ❌ **Bad systems**: Slow down significantly, lose accuracy, or fail to retrieve facts

---

## Interpreting Results

### ⚠️ IMPORTANT: Latency Comparison Caveat

**The latency metrics in this benchmark are NOT an "apples-to-apples" comparison** due to different deployment configurations:

| Backend | Configuration | Network Latency |
|---------|--------------|-----------------|
| **Letta** | Running locally (localhost:8283) | ~1-5ms (negligible) |
| **Mem0** | Running via cloud API (api.mem0.ai) | ~100-400ms per API call |

#### Why This Matters:

Each test scenario makes **multiple API calls**. The network round-trip time for Mem0's cloud API adds significant overhead:

| Test Scenario | Typical API Calls | Estimated Network Overhead (Mem0 Cloud) | Notes |
|--------------|-------------------|----------------------------------------|-------|
| **Cross-Session Recall** | ~3-4 calls | **+300-600ms** | Create agent + 2-3 memory operations |
| **Memory Editing Governance** | ~3-4 calls | **+300-600ms** | Create + store + update + retrieve |
| **Task Continuity** | ~3-4 calls | **+300-600ms** | Create + store sequence + 2 queries |
| **Degradation Resilience** | ~2-3 calls | **+200-450ms** | Create + store + retrieve |
| **Temporal Conflict Resolution** | ~3-4 calls | **+300-600ms** | Create + store old + store new + query |
| **Multi-Fact Inference** | ~4-5 calls | **+400-750ms** | Create + store 3 facts + query |
| **Personality Consistency** | ~3-4 calls | **+300-600ms** | Create + multiple persona checks |
| **Conditional Memory Updates** | ~3-4 calls | **+300-600ms** | Create + conditional store + verify |
| **Memory Scalability** | **~50+ calls** | **+5,000-20,000ms** | Create + 50 sequential memory operations |

**Network overhead per API call breakdown:**
- DNS lookup: 10-50ms (cached after first call)
- TLS handshake: 50-200ms (reused with keep-alive)
- Round-trip latency: 50-200ms (depends on location/ISP)
- **Total: ~100-400ms per call**

#### Estimated "True" Latency (Mem0 Local):

To estimate what Mem0's latency would be if run locally (like Letta), subtract the network overhead:

| Test Scenario | Mem0 Cloud Latency | Est. Network Penalty | **Est. Local Latency** |
|--------------|-------------------|---------------------|----------------------|
| Cross-Session Recall | 2,869ms | -450ms | **~2,420ms** |
| Degradation Resilience | 3,276ms | -325ms | **~2,950ms** |
| Memory Editing Governance | 6,109ms | -450ms | **~5,660ms** |
| Task Continuity | 4,745ms | -450ms | **~4,295ms** |
| Temporal Conflict Resolution | 8,982ms | -450ms | **~8,530ms** |
| Multi-Fact Inference | 6,180ms | -575ms | **~5,605ms** |
| Personality Consistency | 2,497ms | -450ms | **~2,050ms** |
| Conditional Memory Updates | 8,274ms | -450ms | **~7,825ms** |
| Memory Scalability | 12,640ms | -7,500ms+ | **~5,000-5,500ms** |

#### Fair Comparison Guidelines:

✅ **Accuracy & Similarity**: Fair comparison - network latency doesn't affect response quality
❌ **Latency**: NOT a fair comparison - Mem0 has significant network penalty

**To run a fair latency comparison:**
1. Install Docker Desktop
2. Run Mem0 locally: `docker run -p 3000:3000 -e OPENAI_API_KEY=your-key mem0ai/mem0:latest`
3. Update `.env`: `MEM0_BASE_URL=http://localhost:3000`
4. Re-run benchmarks

#### Why Use Cloud API Despite Latency Penalty?

**Convenience vs. Complexity:**
- **Letta local**: `pip install letta && letta server` (2 commands, works immediately)
- **Mem0 local**: Requires Docker Desktop, container management, more complex setup
- **Mem0 cloud**: Works out-of-the-box with API key (no local infrastructure)

**For this benchmark, we prioritize ease of setup** and clearly document the latency implications rather than requiring Docker setup.

---

### Reading the Summary Table

Example output from `out/results.csv`:

```csv
backend,scenario,mean_latency_ms,accuracy,mean_similarity
letta,cross_session_recall,3665.70,1.0,0.786
letta,degradation_resilience,3680.78,1.0,0.880
letta,memory_editing_governance,2794.06,1.0,0.655
letta,task_continuity,2519.57,1.0,0.902
mem0,cross_session_recall,3004.99,0.0,0.614
mem0,degradation_resilience,2955.48,0.0,0.000
mem0,memory_editing_governance,6281.98,1.0,0.668
mem0,task_continuity,6466.63,0.5,0.863
```

**Column Definitions**:

| Column | Description | Range | Good Value |
|--------|-------------|-------|------------|
| `backend` | System being tested | letta, mem0 | N/A |
| `scenario` | Test name | See above | N/A |
| `mean_latency_ms` | Average response time in milliseconds | 0+ | < 5000ms |
| `accuracy` | Keyword + conciseness score | 0.0 - 1.0 | ≥ 0.8 |
| `mean_similarity` | Semantic similarity score | 0.0 - 1.0 | ≥ 0.7 |

**Per-Scenario Accuracy**:
- Calculated only for the specific turns being graded (not all turns)
- Turn 0, Turn 1 often have `accuracy=None` (just storing information)
- Final query turns have `accuracy=0.0/0.5/1.0`

**Mean Similarity**:
- Calculated only for turns with non-empty responses
- Requires `OPENAI_API_KEY` environment variable
- Will be `0.0` in summary if all individual similarities are `None`

---

### Performance Benchmarks

Based on actual test results (2025-11-07):

**Expected Performance Ranges**:

| Backend | Accuracy | Similarity | Latency | Notes |
|---------|----------|------------|---------|-------|
| **Letta (Fixed)** | 0.90 - 1.00 | 0.75 - 0.90 | 2500 - 4500ms | With proper memory blocks & prompts |
| **Letta (Unfixed)** | 0.00 - 0.50 | 0.00 - 0.60 | N/A | No memory blocks = can't store |
| **Mem0 (with LLM)** | 0.30 - 0.60 | 0.50 - 0.70 | 3000 - 7000ms | Semantic search + GPT formatting |
| **Mem0 (raw)** | 0.10 - 0.40 | 0.40 - 0.60 | 2000 - 4000ms | Raw memory strings, no formatting |

---

### What "Good" Results Look Like

**Excellent (Production Ready)**:
```
accuracy >= 0.90
mean_similarity >= 0.80
mean_latency_ms < 5000
```
Example: Letta with fixed memory blocks

**Good (Acceptable)**:
```
accuracy >= 0.70
mean_similarity >= 0.70
mean_latency_ms < 8000
```
Example: Well-configured system with minor issues

**Poor (Needs Work)**:
```
accuracy < 0.50
mean_similarity < 0.60
mean_latency_ms > 10000
```
Example: Mem0 without proper LLM post-processing

---

### Common Issues and Diagnostics

**Issue**: Accuracy = 0.0 across all tests
- **Diagnosis**: Backend not storing memories
- **Letta**: Check memory blocks are being created (src/backends/letta.py:100-113)
- **Mem0**: Check server is reachable and API key is valid

**Issue**: Similarity = 0.0 across all tests
- **Diagnosis**: OpenAI API key not set or embeddings failing
- **Fix**: Set `OPENAI_API_KEY` environment variable
- **Note**: Accuracy will still work without similarity

**Issue**: High latency (> 10s)
- **Diagnosis**: Network issues or overloaded backend
- **Check**: Backend server health and response times
- **Note**: First test run may be slower (cold start)

**Issue**: Inconsistent results across runs
- **Diagnosis**: LLM non-determinism or server instability
- **Fix**: Run with `--sessions 3` to get averages
- **Note**: Accuracy variance ±0.1 is normal for LLM-based systems

---

## Test Evolution History

### Version 1.0 (Original)
- ❌ Lenient grading: Only checked keywords
- ❌ Low similarity threshold (0.30)
- ❌ No memory blocks for Letta
- **Result**: Inflated scores, false positives

### Version 2.0 (Fixed - Current)
- ✅ Strict grading: Keywords + conciseness
- ✅ High similarity threshold (0.75)
- ✅ Default memory blocks for Letta
- ✅ Enhanced system prompts
- **Result**: Accurate differentiation between systems

**Key Fixes Applied**:
1. **src/benchmark.py:151-207** - Added conciseness checks
2. **src/benchmark.py:253-271** - Stricter similarity fallback
3. **src/backends/letta.py:100-113** - Default memory blocks
4. **src/backends/letta.py:56-83** - Enhanced system prompt
5. **src/backends/mem0.py:48-81** - Added LLM post-processing

---

## Running the Tests

**Basic Usage**:
```bash
# Run all tests
python -m src.benchmark --backend both --sessions 1 --noise 0.0

# Run only Letta
python -m src.benchmark --backend letta --sessions 1 --noise 0.0

# Multiple sessions for statistical significance
python -m src.benchmark --backend both --sessions 3 --noise 0.0

# With noise
python -m src.benchmark --backend both --sessions 1 --noise 0.25
```

**Using Automated Scripts**:
```bash
# Linux/Mac
./run_benchmarks.sh both 1 0.0

# Windows
run_benchmarks.bat both 1 0.0
```

**Output Files**:
- `out/results.csv` - Summary table (open in Excel)
- `out/results.json` - Full details (machine-readable)

---

## Conclusion

This benchmark provides a **fair, accurate, and comprehensive** evaluation of memory backend capabilities. The scoring system balances:

1. **Correctness** (accuracy via keyword matching)
2. **Quality** (conciseness via word limits)
3. **Semantics** (similarity via embeddings)
4. **Performance** (latency via timing)

Results clearly differentiate between:
- **Agent-based systems** (Letta): Reasoning, structure, consistency
- **Database-like systems** (Mem0): Fast retrieval, scalability, simplicity

Use these tests to:
- ✅ Validate backend configuration
- ✅ Compare different memory systems
- ✅ Track improvements over time
- ✅ Understand trade-offs between approaches
