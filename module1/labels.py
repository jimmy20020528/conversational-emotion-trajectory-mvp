"""Canonical emotion label order for Module 1 and downstream Module 2."""

# Fixed order — do not reorder without retraining and re-exporting artifacts.
EMOTION_LABELS: tuple[str, ...] = (
    "joy",
    "sadness",
    "anger",
    "fear",
    "disgust",
    "surprise",
    "contempt",
)

EMOTION_LABEL_TO_IDX: dict[str, int] = {e: i for i, e in enumerate(EMOTION_LABELS)}
