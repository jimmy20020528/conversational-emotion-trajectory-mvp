"""Heuristic emotional trajectory from a sequence of per-turn score vectors."""

from __future__ import annotations

from typing import Any

import numpy as np

from module1.labels import EMOTION_LABELS

# Rough "negative affect" proxy for escalation (project-specific heuristic).
_NEGATIVE = frozenset({"sadness", "anger", "fear", "disgust", "contempt"})


def _vec(scores: dict[str, float]) -> np.ndarray:
    return np.array([scores[k] for k in EMOTION_LABELS], dtype=np.float64)


def _negative_mass(scores: dict[str, float]) -> float:
    return float(sum(scores[k] for k in _NEGATIVE))


def compute_trajectory(turns: list[dict[str, Any]]) -> dict[str, Any]:
    """
    turns: each item must include key 'scores' -> dict[str,float] over EMOTION_LABELS.

    Returns metrics + a short English summary string for conditioning an LLM.
    """
    if not turns:
        return {
            "volatility": 0.0,
            "escalation_score": 0.0,
            "dominant_labels": [],
            "negative_trend": 0.0,
            "summary": "No user turns yet.",
        }

    vectors = [_vec(t["scores"]) for t in turns]
    neg_series = [_negative_mass(t["scores"]) for t in turns]
    dominants: list[str] = []
    for t in turns:
        s = t["scores"]
        top = max(EMOTION_LABELS, key=lambda k: s[k])
        dominants.append(top)

    # Volatility: mean L2 step size across consecutive turns.
    if len(vectors) >= 2:
        deltas = [np.linalg.norm(vectors[i] - vectors[i - 1]) for i in range(1, len(vectors))]
        volatility = float(np.mean(deltas))
    else:
        volatility = 0.0

    # Escalation: increase in negative mass from first third to last third (if enough turns).
    T = len(neg_series)
    if T >= 3:
        k = max(1, T // 3)
        early = float(np.mean(neg_series[:k]))
        late = float(np.mean(neg_series[-k:]))
        escalation_score = late - early
    else:
        escalation_score = float(neg_series[-1] - neg_series[0]) if T >= 2 else 0.0

    negative_trend = float(neg_series[-1] - neg_series[0]) if T >= 2 else 0.0

    # Dominant arc text
    arc = " → ".join(dominants)

    vol_label = "high" if volatility > 0.35 else ("moderate" if volatility > 0.18 else "low")
    esc_label = (
        "rising negative affect"
        if escalation_score > 0.08
        else ("stable" if abs(escalation_score) <= 0.08 else "easing")
    )

    summary = (
        f"Dominant emotion path: {arc}. "
        f"Volatility ({vol_label}, {volatility:.2f}). "
        f"Negative-affect trend: {esc_label} (Δ≈{escalation_score:+.2f}). "
        f"Latest dominant: {dominants[-1]}."
    )

    return {
        "volatility": volatility,
        "escalation_score": float(escalation_score),
        "negative_trend": negative_trend,
        "dominant_labels": dominants,
        "dominant_arc_text": arc,
        "summary": summary,
        "volatility_bucket": vol_label,
    }
