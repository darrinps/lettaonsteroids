# Implementation Guide for Additional Tests

Step-by-step guide to add the 5 new test scenarios to the benchmark framework.

## Step 1: Add Test Scenarios to `src/benchmark.py`

Add these scenario classes after the existing ones (after line 116):

```python
class TemporalConflictResolution(Scenario):
    name = "temporal_conflict_resolution"
    def build_session(self, noise: float = 0.0) -> List[Turn]:
        return [
            Turn("user", "I work at Microsoft."),
            Turn("user", "I work at Google."),
            Turn("user", "I work at Amazon."),
            Turn("user", "Where do I work? Reply with just the company name."),
        ]
    def expected_answer(self) -> str:
        return "Amazon"

class MultiFactInference(Scenario):
    name = "multi_fact_inference"
    def build_session(self, noise: float = 0.0) -> List[Turn]:
        return [
            Turn("user", "My budget for hotels is $200 per night."),
            Turn("user", "Marriott hotels average $180 per night."),
            Turn("user", "Hilton hotels average $220 per night."),
            Turn("user", "Which hotel brand fits my budget? Reply with just the brand name."),
        ]
    def expected_answer(self) -> str:
        return "Marriott"

class PersonalityConsistency(Scenario):
    name = "personality_consistency"
    def build_session(self, noise: float = 0.0) -> List[Turn]:
        return [
            Turn("user", "My name is Alex."),
            Turn("user", "What is my name?"),
            Turn("user", "I prefer casual conversation style."),
            Turn("user", "What is my name?"),
        ]
    def expected_answer(self) -> str:
        return "Alex"

class ConditionalMemoryUpdates(Scenario):
    name = "conditional_memory_updates"
    def build_session(self, noise: float = 0.0) -> List[Turn]:
        return [
            Turn("user", "My loyalty tier is Silver."),
            Turn("user", "I have earned 1000 points."),
            Turn("user", "Upgrade my tier to Gold only if I have over 500 points."),
            Turn("user", "What is my loyalty tier? Reply with just the tier name."),
            Turn("user", "I spent 800 points."),
            Turn("user", "Downgrade my tier to Bronze if I have under 500 points."),
            Turn("user", "What is my loyalty tier? Reply with just the tier name."),
        ]
    def expected_answer(self) -> str:
        return "Gold Bronze"  # Both answers concatenated for multi-turn grading

class MemoryScalability(Scenario):
    name = "memory_scalability"
    def build_session(self, noise: float = 0.0) -> List[Turn]:
        # Generate 100 diverse facts
        facts = [
            "My favorite city is Phoenix.",
            "John's email is john@example.com.",
            "Meeting on Monday at 2pm.",
            "Project alpha deadline is June 15.",
            "Sarah's phone number is 555-0123.",
            "My car is a Tesla Model 3.",
            "Favorite food is pizza.",
            "Wife's birthday is March 10.",
            "Office address is 123 Main St.",
            "Pet dog named Max.",
            # Add 90 more varied facts - hotels, meetings, contacts, preferences, dates
            "Preferred airline is Delta.",
            "Gym membership expires December 2024.",
            "Doctor appointment Thursday 10am.",
            "Password hint is first pet name.",
            "Favorite color is blue.",
            # ... continue with diverse facts to reach 100
        ]

        # Pad to 100 facts if needed
        while len(facts) < 100:
            facts.append(f"Random fact number {len(facts) + 1}.")

        turns = [Turn("user", fact) for fact in facts]
        turns.extend([
            Turn("user", "What is John's email?"),
            Turn("user", "What is my favorite city?"),
            Turn("user", "When is my Monday meeting?"),
        ])
        return turns

    def expected_answer(self) -> str:
        return "john@example.com Phoenix Monday 2pm"  # All expected answers
```

## Step 2: Add Grading Functions to `src/benchmark.py`

Add these after the existing grading functions (after line 207):

```python
def grade_temporal_conflict(response: str) -> float:
    """Grade temporal conflict resolution"""
    has_amazon = contains_all_keywords(response, ["amazon"])
    is_brief = is_concise(response, max_words=3)

    if has_amazon and is_brief:
        return 1.0
    elif has_amazon:
        return 0.5
    return 0.0

def grade_multi_fact_inference(response: str) -> float:
    """Grade multi-fact inference"""
    has_marriott = contains_all_keywords(response, ["marriott"])
    is_brief = is_concise(response, max_words=3)

    if has_marriott and is_brief:
        return 1.0
    elif has_marriott:
        return 0.5
    return 0.0

def grade_personality_consistency(response: str, turn_idx: int) -> float:
    """Grade personality consistency"""
    if turn_idx not in [1, 3]:  # Only grade name query turns
        return None

    has_name = contains_all_keywords(response, ["alex"])
    formal_markers = ["sir", "ma'am", "mister", "ms", "your name is", "your name:", "name:"]
    has_formality = any(marker in response.lower() for marker in formal_markers)

    if has_name and has_formality:
        return 1.0  # Perfect: name + personality
    elif has_name:
        return 0.5  # Acceptable: name but no personality
    return 0.0

def grade_conditional_updates(response: str, turn_idx: int) -> float:
    """Grade conditional memory updates"""
    if turn_idx == 3:  # First tier query (should be Gold)
        has_gold = contains_all_keywords(response, ["gold"])
        is_brief = is_concise(response, max_words=3)
        if has_gold and is_brief:
            return 1.0
        elif has_gold:
            return 0.5
        return 0.0
    elif turn_idx == 6:  # Second tier query (should be Bronze)
        has_bronze = contains_all_keywords(response, ["bronze"])
        is_brief = is_concise(response, max_words=3)
        if has_bronze and is_brief:
            return 1.0
        elif has_bronze:
            return 0.5
        return 0.0
    return None

def grade_memory_scalability(response: str, turn_idx: int) -> float:
    """Grade memory scalability"""
    # Turn indices: 100 = John's email, 101 = favorite city, 102 = Monday meeting
    if turn_idx == 100:
        has_email = "john@example.com" in response.lower()
        is_brief = is_concise(response, max_words=5)
        if has_email and is_brief:
            return 1.0
        elif has_email:
            return 0.5
        return 0.0
    elif turn_idx == 101:
        has_city = contains_all_keywords(response, ["phoenix"])
        is_brief = is_concise(response, max_words=3)
        if has_city and is_brief:
            return 1.0
        elif has_city:
            return 0.5
        return 0.0
    elif turn_idx == 102:
        has_meeting = contains_all_keywords(response, ["monday", "2pm"]) or contains_all_keywords(response, ["monday", "2:00"])
        is_brief = is_concise(response, max_words=6)
        if has_meeting and is_brief:
            return 1.0
        elif has_meeting:
            return 0.5
        return 0.0
    return None
```

## Step 3: Update Scenario List

Update the `SCENARIOS` list (around line 111):

```python
SCENARIOS: List[Scenario] = [
    # Existing scenarios
    CrossSessionRecall(),
    DegradationResilience(),
    MemoryEditingGovernance(),
    TaskContinuity(),
    # New scenarios
    TemporalConflictResolution(),
    MultiFactInference(),
    PersonalityConsistency(),
    ConditionalMemoryUpdates(),
    MemoryScalability(),
]
```

## Step 4: Update Grading Logic in `run_scenario`

Add new grading cases in the `run_scenario` function (around line 253):

```python
sim = try_embedding_similarity(sr.text, scenario.expected_answer()) if sr.text.strip() else None
acc = None

# Existing scenarios...
if scenario.name == "cross_session_recall" and idx == 2:
    acc = grade_cross_session(sr.text)
    if acc == 0.0 and sim is not None and sim >= 0.75:
        acc = 0.5
# ... other existing scenarios ...

# New scenarios
elif scenario.name == "temporal_conflict_resolution" and idx == 3:
    acc = grade_temporal_conflict(sr.text)
    if acc == 0.0 and sim is not None and sim >= 0.75:
        acc = 0.5

elif scenario.name == "multi_fact_inference" and idx == 3:
    acc = grade_multi_fact_inference(sr.text)
    if acc == 0.0 and sim is not None and sim >= 0.75:
        acc = 0.5

elif scenario.name == "personality_consistency" and idx in (1, 3):
    acc = grade_personality_consistency(sr.text, idx)
    # No similarity fallback for personality test

elif scenario.name == "conditional_memory_updates" and idx in (3, 6):
    acc = grade_conditional_updates(sr.text, idx)
    if acc == 0.0 and sim is not None and sim >= 0.75:
        acc = 0.5

elif scenario.name == "memory_scalability" and idx >= 100:
    acc = grade_memory_scalability(sr.text, idx)
    if acc == 0.0 and sim is not None and sim >= 0.75:
        acc = 0.5
```

## Step 5: Add Command-Line Filtering (Optional)

To run only specific scenarios, update `main()` in `src/benchmark.py`:

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["letta", "mem0", "both"], default="both")
    parser.add_argument("--sessions", type=int, default=1)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--out", type=str, default="out/results.json")
    parser.add_argument("--csv", type=str, default="out/results.csv")
    parser.add_argument("--scenarios", type=str, default="all",
                       help="Comma-separated list of scenarios to run, or 'all' (default)")
    args = parser.parse_args()

    # ... existing backend setup ...

    # Filter scenarios if specified
    scenarios_to_run = SCENARIOS
    if args.scenarios != "all":
        scenario_names = [s.strip() for s in args.scenarios.split(",")]
        scenarios_to_run = [s for s in SCENARIOS if s.name in scenario_names]
        if not scenarios_to_run:
            print(f"Error: No matching scenarios found for: {args.scenarios}")
            return

    all_records: List[Record] = []
    for be in backends:
        for sc in scenarios_to_run:
            all_records.extend(run_scenario(be, sc, sessions=args.sessions))

    # ... rest of main ...
```

## Step 6: Update System Prompts for Personality Test

For the personality consistency test, update Letta backend initialization in `src/backends/letta.py`:

```python
# Add a method to customize system prompt
def create_agent(self, *args, **kwargs) -> CreateResult:
    # ... existing code ...

    # Check if custom system prompt provided
    system = kwargs.get("system", self._default_system)

    # ... rest of create_agent ...
```

Then in the benchmark runner, pass custom system prompt for personality test:

```python
# In run_scenario function, before creating agent:
system_prompt = self._default_system
if scenario.name == "personality_consistency":
    system_prompt = "You are a formal, professional assistant who always addresses users as 'Sir' or 'Ma'am' and uses formal language."

cr: CreateResult = backend.create_agent(system=system_prompt)
```

## Step 7: Test the Implementation

```bash
# Test all scenarios
python -m src.benchmark --backend both --sessions 1 --noise 0.0

# Test only new scenarios
python -m src.benchmark --backend both --scenarios temporal_conflict_resolution,multi_fact_inference,personality_consistency,conditional_memory_updates,memory_scalability

# Test a specific challenging scenario
python -m src.benchmark --backend both --scenarios memory_scalability --sessions 1
```

## Expected Results

After implementation, you should see results like:

```
backend                          scenario  mean_latency_ms  accuracy  mean_similarity
  letta      temporal_conflict_resolution      3500.000000      0.90         0.820000
  letta          multi_fact_inference          4200.000000      0.85         0.750000
  letta      personality_consistency           3000.000000      0.95         0.900000
  letta      conditional_memory_updates        5000.000000      0.90         0.800000
  letta          memory_scalability            8000.000000      0.60         0.700000
   mem0      temporal_conflict_resolution      2500.000000      0.40         0.600000
   mem0          multi_fact_inference          3000.000000      0.50         0.650000
   mem0      personality_consistency           2200.000000      0.30         0.500000
   mem0      conditional_memory_updates        3500.000000      0.10         0.400000
   mem0          memory_scalability            2000.000000      0.95         0.900000
```

## Troubleshooting

**Memory Scalability test hangs:**
- Reduce to 50 facts instead of 100
- Increase timeout in `run_scenario` function

**Personality test shows 0.0 for both:**
- Verify system prompt is being passed correctly
- Check if persona block is being used in Letta

**Conditional updates fail for both:**
- This is expected for Mem0 (no reasoning)
- For Letta, verify the agent has access to memory tools

## Notes

- The **Memory Scalability** test may take significantly longer (30s-2min)
- **Conditional Updates** will likely fail for Mem0 (by design)
- **Personality Consistency** requires careful prompt engineering
- Consider running new tests separately first to verify they work
