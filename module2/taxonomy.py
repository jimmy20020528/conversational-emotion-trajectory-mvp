"""Emotion taxonomy: cluster mappings, valence/arousal scores, and colour palette."""

from __future__ import annotations

from typing import Dict

# ---------------------------------------------------------------------------
# Cluster map: fine-grained label → 7 core Ekman clusters (+ Contempt)
# ---------------------------------------------------------------------------
EMOTION_CLUSTERS: Dict[str, str] = {
    # Joy
    "joy": "Joy", "happiness": "Joy", "excitement": "Joy", "amusement": "Joy",
    "gratitude": "Joy", "love": "Joy", "pride": "Joy", "relief": "Joy",
    "optimism": "Joy",
    # Sadness
    "sadness": "Sadness", "grief": "Sadness", "disappointment": "Sadness",
    "loneliness": "Sadness", "melancholy": "Sadness",
    # Anger
    "anger": "Anger", "frustration": "Anger", "annoyance": "Anger",
    "rage": "Anger", "resentment": "Anger",
    # Fear
    "fear": "Fear", "anxiety": "Fear", "nervousness": "Fear", "stress": "Fear",
    "worry": "Fear", "panic": "Fear", "dread": "Fear", "apprehension": "Fear",
    "self-doubt": "Fear", "pressure": "Fear", "overwhelm": "Fear",
    # Disgust
    "disgust": "Disgust", "revulsion": "Disgust",
    # Surprise
    "surprise": "Surprise", "shock": "Surprise", "amazement": "Surprise",
    # Contempt (kept separate per project spec)
    "contempt": "Contempt", "disdain": "Contempt", "scorn": "Contempt",
}

# Dimensional affect scores per cluster
VALENCE: Dict[str, float] = {
    "Joy":      +1.0,
    "Surprise": +0.2,
    "Sadness":  -0.7,
    "Fear":     -0.8,
    "Anger":    -0.9,
    "Disgust":  -0.85,
    "Contempt": -0.6,
}

AROUSAL: Dict[str, float] = {
    "Joy":      0.7,
    "Surprise": 0.9,
    "Sadness":  0.3,
    "Fear":     0.85,
    "Anger":    0.95,
    "Disgust":  0.6,
    "Contempt": 0.5,
}

CLUSTER_COLORS: Dict[str, str] = {
    "Joy":      "#4CAF50",
    "Fear":     "#FF9800",
    "Anger":    "#F44336",
    "Sadness":  "#2196F3",
    "Disgust":  "#795548",
    "Surprise": "#E91E63",
    "Contempt": "#9C27B0",
}
