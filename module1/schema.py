"""Output contract for Module 2 (trajectory) consumers.

Each JSON line (JSONL) is one utterance with model scores. Intensity is aligned
per emotion: we use the same calibrated probability as the activation score
(sigmoid logits). Module 2 can treat `scores` as both presence strength and
per-emotion intensity, or threshold `scores` for active labels.
"""

from __future__ import annotations

from typing import Any, TypedDict

MODULE1_VERSION = "1.0.0"


class EmotionScores(TypedDict):
    joy: float
    sadness: float
    anger: float
    fear: float
    disgust: float
    surprise: float
    contempt: float


class TurnPrediction(TypedDict, total=False):
    """One row of module1 output (JSON object)."""

    dialog_id: str
    turn_id: int
    speaker: str
    text: str
    scores: EmotionScores
    intensity: EmotionScores  # same keys as scores; kept explicit for Module 2
    active_labels: list[str]  # labels with score >= threshold at export time
    threshold: float
    model_name: str
    module1_version: str
    logits: list[float]  # optional, length 7, same order as EMOTION_LABELS


def prediction_to_jsonable(
    *,
    dialog_id: str,
    turn_id: int,
    text: str,
    scores: dict[str, float],
    model_name: str,
    threshold: float = 0.5,
    speaker: str | None = None,
    logits: list[float] | None = None,
) -> dict[str, Any]:
    active = [k for k, v in scores.items() if v >= threshold]
    row: dict[str, Any] = {
        "dialog_id": dialog_id,
        "turn_id": turn_id,
        "text": text,
        "scores": dict(scores),
        "intensity": dict(scores),
        "active_labels": sorted(active),
        "threshold": threshold,
        "model_name": model_name,
        "module1_version": MODULE1_VERSION,
    }
    if speaker is not None:
        row["speaker"] = speaker
    if logits is not None:
        row["logits"] = logits
    return row
