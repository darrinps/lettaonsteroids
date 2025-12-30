"""
src/scenarios.py

Defines test scenario data structures and the scenario builder function.
Contains:
- ScenarioTurn: Represents a single conversational turn (user/system message)
- Scenario: Groups multiple turns into a complete test scenario
- noisy(): Adds noise/scrambling to input text for degradation testing
- build_scenarios(): Factory function that creates all test scenarios (deprecated)

Note: As of the current version, test scenarios are defined directly in src/benchmark.py
using dedicated Scenario classes. This file may be used for legacy compatibility.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List
import random

@dataclass
class ScenarioTurn:
    role: str
    content: str
    expected_contains: List[str] | None = None

@dataclass
class Scenario:
    name: str
    description: str
    turns: List[ScenarioTurn]

def noisy(text: str, ratio: float) -> str:
    words = text.split()
    n = int(len(words) * ratio)
    for _ in range(max(1, n // 3)):
        i = random.randrange(0, len(words))
        words.insert(i, "…")
    random.shuffle(words[:n] if n>0 else words)
    return " ".join(words)

def build_scenarios(noise_ratio: float = 0.25) -> List[Scenario]:
    scenarios: List[Scenario] = []

    turns1 = [
        ScenarioTurn("system", "You are an EA helper. Remember user profile facts."),
        ScenarioTurn("user", "My name is Darrin Smith and I live in Phoenix."),
        ScenarioTurn("user", "I prefer Marriott over Hilton."),
        ScenarioTurn("user", "What's my name and city?", expected_contains=["Darrin", "Phoenix"]),
    ]
    scenarios.append(Scenario("cross_session_recall", "Persistent facts across dialogue.", turns1))

    turns2 = [
        ScenarioTurn("system", "Maintain a profile of the user. Facts may change."),
        ScenarioTurn("user", "My favorite hotel brand is Hilton."),
        ScenarioTurn("user", "Correction: Actually I prefer Choice Hotels now."),
        ScenarioTurn("user", "Which brand do I prefer now?", expected_contains=["Choice"]),
    ]
    scenarios.append(Scenario("memory_editing_governance", "Correct outdated fact with new truth.", turns2))

    turns3 = [
        ScenarioTurn("user", "I want to plan a 3-step rollout: Select tech, Pilot property, Full deploy."),
        ScenarioTurn("user", "What's the next step after selecting tech?", expected_contains=["Pilot"]),
        ScenarioTurn("user", "After the restart, what are my 3 steps?", expected_contains=["Select", "Pilot", "Full"]),
    ]
    scenarios.append(Scenario("task_continuity", "Maintain multi-step plan over interruptions.", turns3))

    base_q = "Remind me of my city and brand preference."
    noisy_q = noisy(base_q, noise_ratio)
    turns4 = [
        ScenarioTurn("user", "Store: city = Phoenix; brand = Choice Hotels."),
        ScenarioTurn("user", noisy_q, expected_contains=["Phoenix", "Choice"]),
    ]
    scenarios.append(Scenario("degradation_resilience", "Handle noisy/partial inputs.", turns4))

    return scenarios
