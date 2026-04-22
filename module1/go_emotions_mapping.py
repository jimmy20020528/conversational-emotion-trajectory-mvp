"""Map GoEmotions fine-grained labels to our 7-way Ekman-style space.

GoEmotions `simplified` uses 28 classes (0–27) per `ClassLabel.names`. Keep
`GO_EMOTION_NAMES` identical to that order.
"""

from __future__ import annotations

from .labels import EMOTION_LABEL_TO_IDX

# Order must match Hugging Face `go_emotions` / `simplified` ClassLabel (0 .. 27).
GO_EMOTION_NAMES: tuple[str, ...] = (
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
)

# Each GoEmotion name maps to one or more canonical labels (multi-hot target).
_GO_TO_CANONICAL: dict[str, tuple[str, ...]] = {
    "admiration": ("joy",),
    "amusement": ("joy",),
    "anger": ("anger",),
    "annoyance": ("anger", "contempt"),
    "approval": ("joy",),
    "caring": ("joy",),
    "confusion": ("surprise",),
    "curiosity": ("surprise",),
    "desire": ("joy",),
    "disappointment": ("sadness",),
    "disapproval": ("contempt",),
    "disgust": ("disgust",),
    "embarrassment": ("fear",),
    "excitement": ("joy",),
    "fear": ("fear",),
    "gratitude": ("joy",),
    "grief": ("sadness",),
    "joy": ("joy",),
    "love": ("joy",),
    "nervousness": ("fear",),
    "optimism": ("joy",),
    "pride": ("joy",),
    "realization": ("surprise",),
    "relief": ("joy",),
    "remorse": ("sadness",),
    "sadness": ("sadness",),
    "surprise": ("surprise",),
    "neutral": (),
}


def go_label_ids_to_multihot(label_ids: list[int]) -> list[float]:
    """Convert a list of GoEmotions class ids to a 7-d multi-hot float vector."""
    vec = [0.0] * len(EMOTION_LABEL_TO_IDX)
    for lid in label_ids:
        if lid < 0 or lid >= len(GO_EMOTION_NAMES):
            continue
        name = GO_EMOTION_NAMES[lid]
        for emo in _GO_TO_CANONICAL.get(name, ()):
            idx = EMOTION_LABEL_TO_IDX[emo]
            vec[idx] = 1.0
    return vec
