"""
Hand-written A/B pairs for offline testing — no Module 1 model, no API key required.

Each pair comes with a pre-computed emotion sequence (3–4 turns), a user
last-turn string, a deliberately-weak baseline response, and a
trajectory-aware conditioned response. Useful for exercising the evaluator
and the human-rater export in CI or in demos without GPU / OpenAI access.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from module2 import EmotionalTrajectoryTracker, build_conditioning_prompt


SIMULATED_PAIRS: List[Dict[str, Any]] = [
    {
        "conversation": [
            {"Anxiety": 0.72, "Stress": 0.81},
            {"Anxiety": 0.85, "Self-doubt": 0.76},
            {"Pressure": 0.89, "Frustration": 0.67},
        ],
        "user_last_turn": "My parents are expecting too much from me right now.",
        "baseline_response": (
            "That sounds like a lot. Have you tried talking to your parents about your expectations?"
        ),
        "conditioned_response": (
            "It sounds like you're carrying a lot right now — the anxiety about your exam "
            "and feeling weighed down by your parents' expectations on top of that. "
            "That's genuinely exhausting. You don't have to have everything figured out. "
            "What feels like the heaviest part right now?"
        ),
    },
    {
        "conversation": [
            {"Anger": 0.90, "Resentment": 0.70},
            {"Anger": 0.75, "Frustration": 0.60},
            {"Sadness": 0.65, "Disappointment": 0.55},
        ],
        "user_last_turn": "I just feel like nobody listens to me at work.",
        "baseline_response": (
            "That's frustrating. Maybe try scheduling a one-on-one with your manager."
        ),
        "conditioned_response": (
            "I can hear how drained you are — moving from anger to just feeling unheard "
            "and sad is its own kind of exhaustion. It's not just about work logistics; "
            "it feels like a deeper sense that your perspective doesn't matter to the people around you. "
            "That deserves to be acknowledged. What do you most wish they understood about you?"
        ),
    },
    {
        "conversation": [
            {"Joy": 0.85, "Excitement": 0.70},
            {"Joy": 0.60, "Nervousness": 0.45},
            {"Anxiety": 0.72, "Disappointment": 0.55},
            {"Sadness": 0.80, "Grief": 0.40},
        ],
        "user_last_turn": "I didn't get the job offer I was hoping for.",
        "baseline_response": (
            "Sorry to hear that. There will be other opportunities!"
        ),
        "conditioned_response": (
            "Oh, that's really hard — especially after going in hopeful and excited. "
            "It makes sense that the drop feels so sharp right now. "
            "Grieving a missed opportunity is real. Give yourself a moment before jumping to next steps. "
            "How are you feeling about everything else right now?"
        ),
    },
]


def build_records_from_simulated(
    pairs: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Turn each SIMULATED_PAIRS entry into an evaluation record."""
    pairs = pairs if pairs is not None else SIMULATED_PAIRS
    records = []
    for i, item in enumerate(pairs):
        tracker = EmotionalTrajectoryTracker()
        for turn in item["conversation"]:
            tracker.add_turn(turn)
        signals = tracker.compute()
        records.append({
            "conversation_id": f"sim_{i + 1:03d}",
            "conversation_turns": None,
            "user_last_turn": item["user_last_turn"],
            "predictions": item["conversation"],
            "signals": signals,
            "conditioning_prompt": build_conditioning_prompt(signals),
            "baseline_response": item["baseline_response"],
            "conditioned_response": item["conditioned_response"],
        })
    return records
