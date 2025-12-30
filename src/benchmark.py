"""
src/benchmark.py

Main benchmark runner and test orchestration for comparing memory backends.
This is the entry point for running Letta vs Mem0 performance tests.

Core functionality:
1. Test Scenario Definitions (9 total):
   - CrossSessionRecall: Basic memory persistence
   - DegradationResilience: Noisy input handling
   - MemoryEditingGovernance: Memory update logic
   - TaskContinuity: Multi-step process tracking
   - TemporalConflictResolution: Recency handling
   - MultiFactInference: Cross-fact reasoning
   - PersonalityConsistency: Persona maintenance
   - ConditionalMemoryUpdates: Business logic
   - MemoryScalability: Large-scale memory (50+ facts)

2. Grading System:
   - Accuracy: Keywords + conciseness (1.0 = perfect, 0.5 = partial, 0.0 = failed)
   - Similarity: OpenAI embeddings-based semantic similarity (0.0-1.0)
   - Word limits enforced per test (3-10 words depending on scenario)
   - Similarity fallback: ≥0.75 similarity awards 0.5 partial credit

3. Benchmark Execution:
   - Supports multiple backends (letta, mem0, both)
   - Configurable sessions and noise ratio
   - Records detailed per-turn metrics (latency, accuracy, similarity)
   - Outputs JSON (detailed) and CSV (summary) results

4. Metrics & Analysis:
   - Per-scenario mean latency, accuracy, mean similarity
   - Summary rows showing backend averages across all tests
   - Performance comparison tables

Usage:
    python -m src.benchmark --backend both --sessions 1 --noise 0.0
    python -m src.benchmark --backend letta --sessions 3 --noise 0.25

Output:
    - out/results.json: Full detailed records
    - out/results.csv: Summary table with averages
"""

import argparse
import json
import os
import re
import string
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import pandas as pd

# ====== Grading knobs (tunable via env) ======
SIM_THRESHOLD = float(os.environ.get("SIM_THRESHOLD", "0.30"))
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")  # optional
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def now_ms() -> float:
    return time.perf_counter() * 1000.0

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s.translate(_PUNCT_TABLE)

def contains_all_keywords(text: str, required: List[str]) -> bool:
    t = normalize_text(text)
    return all(k.lower() in t for k in required if k.strip())

def try_embedding_similarity(a: str, b: str) -> Optional[float]:
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI
        import math

        client = OpenAI(api_key=OPENAI_API_KEY)
        va = client.embeddings.create(model=EMBED_MODEL, input=a).data[0].embedding
        vb = client.embeddings.create(model=EMBED_MODEL, input=b).data[0].embedding

        dot = sum(x * y for x, y in zip(va, vb))
        na = math.sqrt(sum(x * x for x in va))
        nb = math.sqrt(sum(x * x for x in vb))
        if na == 0 or nb == 0:
            return None
        cos = dot / (na * nb)
        return max(0.0, min(1.0, (cos + 1.0) / 2.0))
    except Exception:
        return None

# ====== Scenarios ======
class Turn:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

class Scenario:
    name: str
    def build_session(self, noise: float = 0.0) -> List[Turn]:
        raise NotImplementedError
    def expected_answer(self) -> str:
        raise NotImplementedError

class CrossSessionRecall(Scenario):
    name = "cross_session_recall"
    def build_session(self, noise: float = 0.0) -> List[Turn]:
        return [
            Turn("user", "My name is Darrin Smith and I live in Phoenix."),
            Turn("user", "I prefer Marriott over Hilton."),
            Turn("user", "What is my name and city?"),
        ]
    def expected_answer(self) -> str:
        return "Darrin Phoenix"

class MemoryEditingGovernance(Scenario):
    name = "memory_editing_governance"
    def build_session(self, noise: float = 0.0) -> List[Turn]:
        return [
            Turn("user", "My favorite hotel brand is Hilton."),
            Turn("user", "Actually, replace that: my favorite hotel brand is Choice Hotels."),
            Turn("user", "What brand do I prefer now?"),
        ]
    def expected_answer(self) -> str:
        return "Choice"

class TaskContinuity(Scenario):
    name = "task_continuity"
    def build_session(self, noise: float = 0.0) -> List[Turn]:
        return [
            Turn("user", "My rollout has 3 steps: Select tech, Pilot property, Full deploy."),
            Turn("user", "What’s the next step after Select tech? Reply with just the keyword."),
            Turn("user", "List the 3 steps back as keywords in order."),
        ]
    def expected_answer(self) -> str:
        return "Select Pilot Full"

class DegradationResilience(Scenario):
    name = "degradation_resilience"
    def build_session(self, noise: float = 0.0) -> List[Turn]:
        return [
            Turn("user", "Remember: Phoenix and Choice."),
            Turn("user", "What city and brand do I prefer? Reply with just the keywords."),
        ]
    def expected_answer(self) -> str:
        return "Phoenix Choice"

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
        return "Gold Bronze"

class MemoryScalability(Scenario):
    name = "memory_scalability"
    def build_session(self, noise: float = 0.0) -> List[Turn]:
        # Generate 50 diverse facts (reduced from 100 for performance)
        facts = [
            "My favorite city is Phoenix.",
            "John's email is john@example.com.",
            "Meeting on Monday at 2pm.",
            "Project alpha deadline is June 15.",
            "Sarah's phone is 555-0123.",
            "My car is a Tesla Model 3.",
            "Favorite food is pizza.",
            "Wife's birthday is March 10.",
            "Office at 123 Main St.",
            "Pet dog named Max.",
            "Preferred airline is Delta.",
            "Gym membership expires Dec 2024.",
            "Doctor appointment Thursday 10am.",
            "Password hint is first pet name.",
            "Favorite color is blue.",
            "Laptop is MacBook Pro.",
            "Coffee order is large latte.",
            "Netflix password in phone notes.",
            "Dentist is Dr. Smith.",
            "Favorite band is Coldplay.",
            "Brother's name is Mike.",
            "Sister's name is Lisa.",
            "Mom's birthday is May 5.",
            "Dad's birthday is Aug 20.",
            "Anniversary is Sept 15.",
            "House number is 456.",
            "ZIP code is 85001.",
            "Bank is Chase.",
            "Credit card ends in 1234.",
            "Phone carrier is Verizon.",
            "Internet provider is Cox.",
            "Favorite restaurant is Olive Garden.",
            "Gym is LA Fitness.",
            "Yoga class on Wednesdays.",
            "Tennis on Saturdays.",
            "Golf handicap is 18.",
            "Shirt size is large.",
            "Shoe size is 10.",
            "Allergic to peanuts.",
            "Blood type is O positive.",
            "Emergency contact is Jane.",
            "Jane's phone is 555-9999.",
            "Work starts at 9am.",
            "Lunch break at noon.",
            "Work ends at 5pm.",
            "Commute is 30 minutes.",
            "Parking spot is B-23.",
            "Badge number is 7890.",
            "Manager is Bob Johnson.",
            "Team size is 8 people.",
        ]

        turns = [Turn("user", fact) for fact in facts]
        turns.extend([
            Turn("user", "What is John's email?"),
            Turn("user", "What is my favorite city?"),
            Turn("user", "When is my Monday meeting?"),
        ])
        return turns

    def expected_answer(self) -> str:
        return "john@example.com Phoenix Monday"

SCENARIOS: List[Scenario] = [
    CrossSessionRecall(),
    DegradationResilience(),
    MemoryEditingGovernance(),
    TaskContinuity(),
    TemporalConflictResolution(),
    MultiFactInference(),
    PersonalityConsistency(),
    ConditionalMemoryUpdates(),
    MemoryScalability(),
]

# ====== Record/Result types ======
@dataclass
class Record:
    scenario: str
    session: int
    step: str
    role: str
    ok: bool
    latency_ms: float
    error: Optional[str]
    backend: str
    response: str
    expected: str
    similarity: Optional[float]
    accuracy: Optional[float]

@dataclass
class CreateResult:
    ok: bool
    latency_ms: float
    error: Optional[str]
    agent_id: Optional[str]
    raw: Optional[Dict[str, Any]] = None

@dataclass
class SendResult:
    ok: bool
    latency_ms: float
    error: Optional[str]
    text: str
    raw: Optional[Dict[str, Any]] = None

# ====== Graders ======
def is_concise(text: str, max_words: int = 10) -> bool:
    """Check if response is concise (not a full sentence/paragraph)"""
    normalized = normalize_text(text)
    word_count = len([w for w in normalized.split() if w])
    return word_count <= max_words

def grade_cross_session(response: str) -> float:
    # Must contain both "darrin" and "phoenix", and be concise (not a full sentence)
    has_keywords = contains_all_keywords(response, ["darrin", "phoenix"])
    is_brief = is_concise(response, max_words=6)
    # Full credit only if both conditions met, partial if just keywords
    if has_keywords and is_brief:
        return 1.0
    elif has_keywords:
        return 0.5
    return 0.0

def grade_memory_edit(response: str) -> float:
    # Must contain "choice" and be concise
    has_keywords = contains_all_keywords(response, ["choice"])
    is_brief = is_concise(response, max_words=5)
    if has_keywords and is_brief:
        return 1.0
    elif has_keywords:
        return 0.5
    return 0.0

def grade_task_continuity(turn_idx: int, response: str) -> float:
    if turn_idx == 1:
        # Should respond with just "Pilot" or "Pilot property"
        has_pilot = contains_all_keywords(response, ["pilot"])
        is_brief = is_concise(response, max_words=4)
        if has_pilot and is_brief:
            return 1.0
        elif has_pilot:
            return 0.5
        return 0.0
    if turn_idx == 2:
        # Should list all 3 steps concisely
        has_all = contains_all_keywords(response, ["select", "pilot", "full"])
        is_brief = is_concise(response, max_words=10)
        if has_all and is_brief:
            return 1.0
        elif has_all:
            return 0.5
        return 0.0
    return 0.0

def grade_degradation(response: str) -> float:
    # Must contain both keywords and be concise
    has_keywords = contains_all_keywords(response, ["phoenix", "choice"])
    is_brief = is_concise(response, max_words=5)
    if has_keywords and is_brief:
        return 1.0
    elif has_keywords:
        return 0.5
    return 0.0

def grade_temporal_conflict(response: str) -> float:
    # Should return most recent employer (Amazon)
    has_amazon = contains_all_keywords(response, ["amazon"])
    is_brief = is_concise(response, max_words=3)
    if has_amazon and is_brief:
        return 1.0
    elif has_amazon:
        return 0.5
    return 0.0

def grade_multi_fact_inference(response: str) -> float:
    # Should infer Marriott fits budget ($180 < $200)
    has_marriott = contains_all_keywords(response, ["marriott"])
    is_brief = is_concise(response, max_words=3)
    if has_marriott and is_brief:
        return 1.0
    elif has_marriott:
        return 0.5
    return 0.0

def grade_personality_consistency(response: str, turn_idx: int) -> float:
    # Only grade name query turns (1 and 3)
    if turn_idx not in [1, 3]:
        return None

    has_name = contains_all_keywords(response, ["alex"])
    formal_markers = ["sir", "ma'am", "mister", "ms", "your name is", "your name:", "name:"]
    has_formality = any(marker in response.lower() for marker in formal_markers)

    if has_name and has_formality:
        return 1.0  # Perfect: name + personality maintained
    elif has_name:
        return 0.5  # Acceptable: name but no personality
    return 0.0

def grade_conditional_updates(response: str, turn_idx: int) -> float:
    # Grade tier queries at turn 3 (Gold) and turn 6 (Bronze)
    if turn_idx == 3:  # Should be Gold (1000 > 500)
        has_gold = contains_all_keywords(response, ["gold"])
        is_brief = is_concise(response, max_words=3)
        if has_gold and is_brief:
            return 1.0
        elif has_gold:
            return 0.5
        return 0.0
    elif turn_idx == 6:  # Should be Bronze (200 < 500)
        has_bronze = contains_all_keywords(response, ["bronze"])
        is_brief = is_concise(response, max_words=3)
        if has_bronze and is_brief:
            return 1.0
        elif has_bronze:
            return 0.5
        return 0.0
    return None

def grade_memory_scalability(response: str, turn_idx: int) -> float:
    # Grade queries after 50 facts stored
    # turn 50 = John's email, turn 51 = favorite city, turn 52 = Monday meeting
    if turn_idx == 50:
        has_email = "john@example.com" in response.lower()
        is_brief = is_concise(response, max_words=5)
        if has_email and is_brief:
            return 1.0
        elif has_email:
            return 0.5
        return 0.0
    elif turn_idx == 51:
        has_city = contains_all_keywords(response, ["phoenix"])
        is_brief = is_concise(response, max_words=3)
        if has_city and is_brief:
            return 1.0
        elif has_city:
            return 0.5
        return 0.0
    elif turn_idx == 52:
        has_meeting = contains_all_keywords(response, ["monday", "2pm"]) or contains_all_keywords(response, ["monday", "2:00"])
        is_brief = is_concise(response, max_words=6)
        if has_meeting and is_brief:
            return 1.0
        elif has_meeting:
            return 0.5
        return 0.0
    return None

# ====== Runner ======
def run_scenario(backend, scenario: Scenario, sessions: int = 1) -> List[Record]:
    records: List[Record] = []

    for s in range(sessions):
        # create
        t0 = now_ms()
        try:
            cr: CreateResult = backend.create_agent()
        except Exception as e:
            lat = now_ms() - t0
            records.append(Record(
                scenario=scenario.name, session=s, step="create", role="system",
                ok=False, latency_ms=lat, error=str(e), backend=backend.backend_name(),
                response="", expected="", similarity=None, accuracy=None
            ))
            continue

        records.append(Record(
            scenario=scenario.name, session=s, step="create", role="system",
            ok=cr.ok, latency_ms=cr.latency_ms, error=cr.error, backend=backend.backend_name(),
            response="", expected="", similarity=None, accuracy=None
        ))
        if not cr.ok or not cr.agent_id:
            continue

        agent_id = cr.agent_id

        # turns
        turns = scenario.build_session(noise=0.0)
        for idx, turn in enumerate(turns):
            tag = f"turn_{idx}"
            t1 = now_ms()
            try:
                sr: SendResult = backend.send(agent_id, turn.role, turn.content)
            except Exception as e:
                lat = now_ms() - t1
                records.append(Record(
                    scenario=scenario.name, session=s, step=tag, role=turn.role,
                    ok=False, latency_ms=lat, error=str(e), backend=backend.backend_name(),
                    response="", expected=scenario.expected_answer(), similarity=None, accuracy=None
                ))
                continue

            sim = try_embedding_similarity(sr.text, scenario.expected_answer()) if sr.text.strip() else None
            acc = None
            if scenario.name == "cross_session_recall" and idx == 2:
                acc = grade_cross_session(sr.text)
                # Only use similarity as tiebreaker for partial credit (0.5), not to bypass failed checks (0.0)
                if acc == 0.0 and sim is not None and sim >= 0.75:
                    acc = 0.5
            elif scenario.name == "memory_editing_governance" and idx == 2:
                acc = grade_memory_edit(sr.text)
                if acc == 0.0 and sim is not None and sim >= 0.75:
                    acc = 0.5
            elif scenario.name == "task_continuity" and idx in (1, 2):
                acc = grade_task_continuity(idx, sr.text)
                if acc == 0.0 and sim is not None and sim >= 0.75:
                    acc = 0.5
            elif scenario.name == "degradation_resilience" and idx == 1:
                acc = grade_degradation(sr.text)
                if acc == 0.0 and sim is not None and sim >= 0.75:
                    acc = 0.5
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
                # No similarity fallback for personality test (we want to see actual personality)
            elif scenario.name == "conditional_memory_updates" and idx in (3, 6):
                acc = grade_conditional_updates(sr.text, idx)
                if acc == 0.0 and sim is not None and sim >= 0.75:
                    acc = 0.5
            elif scenario.name == "memory_scalability" and idx in (50, 51, 52):
                acc = grade_memory_scalability(sr.text, idx)
                if acc == 0.0 and sim is not None and sim >= 0.75:
                    acc = 0.5

            records.append(Record(
                scenario=scenario.name, session=s, step=tag, role=turn.role,
                ok=sr.ok, latency_ms=sr.latency_ms, error=sr.error, backend=backend.backend_name(),
                response=sr.text, expected=scenario.expected_answer(), similarity=sim, accuracy=acc
            ))

        try:
            backend.teardown_agent(agent_id)
        except Exception:
            pass

    return records

def summarize(records: List[Record]) -> pd.DataFrame:
    rows = []
    from collections import defaultdict
    groups = defaultdict(list)
    for r in records:
        groups[(r.backend, r.scenario)].append(r)

    for (backend, scenario), recs in groups.items():
        lats = [r.latency_ms for r in recs if r.step.startswith("turn_") and r.ok]
        mean_latency = sum(lats) / len(lats) if lats else 0.0
        accs = [r.accuracy for r in recs if r.accuracy is not None]
        acc = sum(accs) / len(accs) if accs else 0.0
        sims = [r.similarity for r in recs if r.similarity is not None]
        mean_sim = sum(sims) / len(sims) if sims else 0.0

        rows.append({
            "backend": backend,
            "scenario": scenario,
            "mean_latency_ms": round(mean_latency, 6),
            "accuracy": round(acc, 6),
            "mean_similarity": round(mean_sim, 6),
        })

    df = pd.DataFrame(rows).sort_values(["backend", "scenario"]).reset_index(drop=True)

    # Add summary rows for each backend
    summary_rows = []
    for backend in df['backend'].unique():
        backend_data = df[df['backend'] == backend]
        summary_rows.append({
            "backend": backend,
            "scenario": "AVERAGE",
            "mean_latency_ms": round(backend_data['mean_latency_ms'].mean(), 6),
            "accuracy": round(backend_data['accuracy'].mean(), 6),
            "mean_similarity": round(backend_data['mean_similarity'].mean(), 6),
        })

    # Append summary rows
    summary_df = pd.DataFrame(summary_rows)
    df = pd.concat([df, summary_df], ignore_index=True)

    return df

def save_outputs(records: List[Record], out_json: str, out_csv: str, summary_df: pd.DataFrame):
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    payload = {"records": [asdict(r) for r in records]}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    summary_df.to_csv(out_csv, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["letta", "mem0", "both"], default="both")
    parser.add_argument("--sessions", type=int, default=1)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--out", type=str, default="out/results.json")
    parser.add_argument("--csv", type=str, default="out/results.csv")
    args = parser.parse_args()

    from src.backends.letta import LettaBackend
    try:
        from src.backends.mem0 import Mem0Backend
        HAVE_MEM0 = True
    except Exception:
        HAVE_MEM0 = False

    backends = []
    if args.backend in ("letta", "both"):
        backends.append(LettaBackend())
    if args.backend in ("mem0", "both") and HAVE_MEM0:
        backends.append(Mem0Backend())

    all_records: List[Record] = []
    for be in backends:
        for sc in SCENARIOS:
            all_records.extend(run_scenario(be, sc, sessions=args.sessions))

    summary = summarize(all_records)
    save_outputs(all_records, args.out, args.csv, summary)
    if not summary.empty:
        with pd.option_context("display.max_colwidth", 40):
            print(summary.to_string(index=False))
    else:
        print("No results. Check connectivity/config.")

if __name__ == "__main__":
    main()
