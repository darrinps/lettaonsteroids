# Additional Test Scenarios for Letta vs Mem0

Five additional test scenarios designed to reveal where each system excels and where it struggles. These tests aren't about equality—they're about finding differentiators.

---

## 1. Temporal Conflict Resolution

**What It Tests**: Handling conflicting information over time without explicit "replace" instructions.

**Scenario**:
- Turn 1: "I work at Microsoft."
- Turn 2: "I work at Google." (no explicit "actually" or "replace")
- Turn 3: "I work at Amazon."
- Turn 4: "Where do I work?"

**Expected Behavior**:
- **Best Answer**: "Amazon" (most recent)
- **Acceptable**: Lists all three with recency indicators
- **Poor**: Returns "Microsoft" or random selection

**Why This Differentiates**:
- **Letta Advantage**: Agent can reason about temporal ordering, use timestamps, or explicitly update memories
- **Mem0 Challenge**: Semantic search returns all matches; needs external logic to determine recency
- **Grading**: 1.0 for "Amazon", 0.5 for mentioning recency, 0.0 for wrong answer

**Implementation**:
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
```

---

## 2. Multi-Fact Inference

**What It Tests**: Deriving new information by connecting multiple stored facts.

**Scenario**:
- Turn 1: "My budget for hotels is $200 per night."
- Turn 2: "Marriott hotels average $180 per night."
- Turn 3: "Hilton hotels average $220 per night."
- Turn 4: "Which hotel brand fits my budget?"

**Expected Behavior**:
- **Best Answer**: "Marriott" (requires reasoning: $180 < $200, but $220 > $200)
- **Poor**: Returns raw facts without conclusion

**Why This Differentiates**:
- **Letta Advantage**: LLM agent can perform numerical reasoning and draw conclusions
- **Mem0 Challenge**: Returns stored facts but can't infer; needs LLM post-processing with all context
- **Grading**: 1.0 for "Marriott", 0.5 for listing both with prices, 0.0 for no answer

**Implementation**:
```python
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
```

---

## 3. Memory Scalability Stress Test

**What It Tests**: Performance and accuracy with large memory sets (100+ facts).

**Scenario**:
- Turns 1-100: Store 100 different facts (cities, preferences, contacts, meetings, etc.)
  - "My favorite city is Phoenix."
  - "John's email is john@example.com."
  - "Meeting on Monday at 2pm."
  - ... (97 more varied facts)
- Turn 101: "What is John's email?"
- Turn 102: "What is my favorite city?"
- Turn 103: "When is my Monday meeting?"

**Expected Behavior**:
- **Best**: Correctly retrieves all 3 specific facts quickly (< 5s)
- **Good**: Retrieves correctly but slower (5-15s)
- **Poor**: Wrong answers or extreme latency (> 15s)

**Why This Differentiates**:
- **Mem0 Advantage**: Vector search scales well; semantic retrieval efficient
- **Letta Challenge**: Large structured blocks may slow down; context window limits
- **Grading**: 1.0 for all correct, 0.66 for 2/3, 0.33 for 1/3, 0.0 for none. Latency scored separately.

**Implementation**:
```python
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
            # ... add 95 more diverse facts
        ]

        turns = [Turn("user", fact) for fact in facts]
        turns.extend([
            Turn("user", "What is John's email?"),
            Turn("user", "What is my favorite city?"),
            Turn("user", "When is my Monday meeting?"),
        ])
        return turns

    def expected_answers(self) -> List[str]:
        return ["john@example.com", "Phoenix", "Monday 2pm"]
```

---

## 4. Personality Consistency

**What It Tests**: Maintaining a consistent persona/tone across interactions.

**Scenario**:
- System Prompt: "You are a formal, professional assistant who always addresses users as 'Sir' or 'Ma'am'."
- Turn 1: "My name is Alex."
- Turn 2: "What is my name?"
- Turn 3: "I prefer casual conversation."
- Turn 4: "What is my name?" (should still be formal despite preference)

**Expected Behavior**:
- **Best**: "Your name is Alex, Sir/Ma'am." (maintains formality)
- **Acceptable**: "Alex" (correct but no personality)
- **Poor**: Changes tone to casual after turn 3

**Why This Differentiates**:
- **Letta Advantage**: Persona block + system prompt maintains consistent personality/tone
- **Mem0 Challenge**: No concept of persona; returns raw facts; LLM post-processing may drift
- **Grading**: Check for personality consistency tokens ("Sir", "Ma'am", formality markers)

**Implementation**:
```python
class PersonalityConsistency(Scenario):
    name = "personality_consistency"
    system_prompt = "You are a formal, professional assistant who always addresses users as 'Sir' or 'Ma'am' and uses formal language."

    def build_session(self, noise: float = 0.0) -> List[Turn]:
        return [
            Turn("user", "My name is Alex."),
            Turn("user", "What is my name?"),
            Turn("user", "I prefer casual conversation style."),
            Turn("user", "What is my name?"),
        ]

    def grade_response(self, response: str, turn_idx: int) -> float:
        has_name = "alex" in response.lower()
        formal_markers = ["sir", "ma'am", "mister", "ms.", "your name is"]
        has_formality = any(marker in response.lower() for marker in formal_markers)

        if turn_idx in [1, 3]:  # Name queries
            if has_name and has_formality:
                return 1.0  # Perfect
            elif has_name:
                return 0.5  # Correct but no personality
            return 0.0
        return None
```

---

## 5. Conditional Memory Updates

**What It Tests**: Applying business logic/rules before updating memories.

**Scenario**:
- Turn 1: "My loyalty tier is Silver."
- Turn 2: "I have earned 1000 points."
- Turn 3: "Upgrade my tier to Gold only if I have over 500 points."
- Turn 4: "What is my loyalty tier?"
- Turn 5: "I spent 800 points."
- Turn 6: "Downgrade my tier to Bronze if I have under 500 points."
- Turn 7: "What is my loyalty tier?"

**Expected Behavior**:
- Turn 4: "Gold" (1000 > 500, so upgrade applied)
- Turn 7: "Bronze" (1000 - 800 = 200 < 500, so downgrade applied)

**Why This Differentiates**:
- **Letta Advantage**: Agent can evaluate conditions, perform arithmetic, and make decisions
- **Mem0 Challenge**: No reasoning capability; would need complex external orchestration
- **Grading**: 1.0 for both correct, 0.5 for one correct, 0.0 for both wrong

**Implementation**:
```python
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

    def expected_answers(self) -> Dict[int, str]:
        return {
            3: "Gold",   # After conditional upgrade
            6: "Bronze"  # After conditional downgrade
        }
```

---

## Summary: Expected Outcomes

| Test | Letta Expected | Mem0 Expected | Winner |
|------|---------------|---------------|---------|
| **Temporal Conflict** | 0.8-1.0<br>(Agent reasons about recency) | 0.3-0.5<br>(Returns all, needs external logic) | **Letta** |
| **Multi-Fact Inference** | 0.8-1.0<br>(LLM performs reasoning) | 0.5-0.7<br>(LLM post-processor helps) | **Letta** |
| **Scalability Stress** | 0.5-0.7<br>(May slow with large blocks) | 0.9-1.0<br>(Vector search scales well) | **Mem0** |
| **Personality Consistency** | 0.9-1.0<br>(Persona block maintains tone) | 0.2-0.4<br>(No personality concept) | **Letta** |
| **Conditional Updates** | 0.8-1.0<br>(Agent evaluates conditions) | 0.0-0.2<br>(No reasoning capability) | **Letta** |

## Key Insights

### Letta Strengths (Tests 1, 2, 4, 5):
- ✅ **Reasoning & Inference**: Can connect dots across memories
- ✅ **Temporal Awareness**: Understands recency and context
- ✅ **Personality/Persona**: Maintains consistent tone and behavior
- ✅ **Conditional Logic**: Can evaluate rules before acting
- ✅ **Agent-like Behavior**: Makes decisions autonomously

### Mem0 Strengths (Test 3):
- ✅ **Scalability**: Handles large memory sets efficiently
- ✅ **Search Performance**: Fast semantic retrieval
- ✅ **Simplicity**: Lighter weight for pure storage/retrieval
- ✅ **Flexibility**: Unstructured storage adapts easily

### Letta Challenges:
- ⚠️ **Scale**: Large structured blocks may impact performance
- ⚠️ **Complexity**: Requires more setup and configuration
- ⚠️ **Context Window**: Limited by LLM token limits

### Mem0 Challenges:
- ⚠️ **No Reasoning**: Can't infer or connect facts
- ⚠️ **No Personality**: Just returns raw data
- ⚠️ **No Logic**: Can't evaluate conditions or rules
- ⚠️ **Requires LLM**: Needs external LLM for formatting/reasoning

## Implementation Priority

**High Priority** (Most differentiating):
1. **Conditional Memory Updates** - Starkest difference in capabilities
2. **Multi-Fact Inference** - Shows reasoning gap clearly
3. **Temporal Conflict Resolution** - Real-world scenario

**Medium Priority**:
4. **Personality Consistency** - Good for agent vs service distinction
5. **Memory Scalability** - Important for production use cases but requires more setup

## Usage

Add these scenarios to `src/scenarios.py` and update `src/benchmark.py` to include the new grading functions. Each test is designed to run in the same framework as existing tests.

To run only new tests:
```bash
# After implementation
python -m src.benchmark --backend both --scenarios temporal_conflict,multi_fact_inference --out out/new_tests.json
```
