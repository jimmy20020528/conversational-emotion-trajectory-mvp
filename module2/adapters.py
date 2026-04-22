"""
Module 2 — Dataset Adapters & Module 3 Conditioning Prompt Builder
===================================================================
DatasetAdapters
    Converts raw rows from supported datasets into the Dict[str, float]
    format expected by EmotionalTrajectoryTracker.add_turn().

    Supported datasets:
      - GoEmotions (logit/probability vector or binary label indices)
      - DailyDialog (7-class integer label)
      - MELD (emotion string + sentiment string)
      - EmpatheticDialogues (situation-level emotion string)

build_conditioning_prompt
    Generates the LLM system-prompt prefix for Module 3 (Adaptive Dialogue
    Generator) from a TrajectorySignals object.
"""

from __future__ import annotations

from typing import Dict, List

from .tracker import TrajectorySignals


# ---------------------------------------------------------------------------
# Dataset Adapters
# ---------------------------------------------------------------------------

class DatasetAdapters:
    """
    Converts raw dataset rows into Dict[str, float] for the tracker.

    All public methods are static and return a single-turn emotion dict.
    To build a full conversation, iterate over utterances and call
    tracker.add_turn(adapter_method(...)) for each one.
    """

    # GoEmotions 28-class label index order
    GOEMOTION_LABELS: List[str] = [
        "admiration", "amusement", "anger", "annoyance", "approval", "caring",
        "confusion", "curiosity", "desire", "disappointment", "disapproval",
        "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
        "joy", "love", "nervousness", "optimism", "pride", "realization",
        "relief", "remorse", "sadness", "surprise", "neutral",
    ]

    @staticmethod
    def from_goemotion_logits(
        logits: List[float],
        threshold: float = 0.3,
    ) -> Dict[str, float]:
        """
        GoEmotions — use raw sigmoid probabilities (28-dim vector).
        Labels with probability >= threshold are included.

        Example
        -------
        >>> logits = [0.05] * 28
        >>> logits[14] = 0.82  # fear
        >>> logits[19] = 0.61  # nervousness
        >>> DatasetAdapters.from_goemotion_logits(logits)
        {'fear': 0.82, 'nervousness': 0.61}
        """
        labels = DatasetAdapters.GOEMOTION_LABELS
        return {
            labels[i]: float(p)
            for i, p in enumerate(logits)
            if p >= threshold and i < len(labels)
        }

    @staticmethod
    def from_goemotion_indices(
        label_indices: List[int],
        score: float = 0.8,
    ) -> Dict[str, float]:
        """
        GoEmotions — binary multi-label format (list of active label indices).
        Assigns a uniform `score` to each present label.

        Example
        -------
        >>> DatasetAdapters.from_goemotion_indices([14, 19], score=0.85)
        {'fear': 0.85, 'nervousness': 0.85}
        """
        labels = DatasetAdapters.GOEMOTION_LABELS
        return {labels[i]: score for i in label_indices if i < len(labels)}

    @staticmethod
    def from_dailydialog(emotion_id: int, intensity: float = 0.75) -> Dict[str, float]:
        """
        DailyDialog — 7-class integer label.
        0=no_emotion, 1=anger, 2=disgust, 3=fear, 4=happiness, 5=sadness, 6=surprise.

        Returns an empty dict for no_emotion (neutral turns).

        Example
        -------
        >>> DatasetAdapters.from_dailydialog(3)
        {'Fear': 0.75}
        """
        mapping = {
            0: {},
            1: {"Anger":    intensity},
            2: {"Disgust":  intensity},
            3: {"Fear":     intensity},
            4: {"Joy":      intensity},
            5: {"Sadness":  intensity},
            6: {"Surprise": intensity},
        }
        return mapping.get(emotion_id, {})

    @staticmethod
    def from_meld(emotion: str, sentiment: str) -> Dict[str, float]:
        """
        MELD — emotion string + sentiment string.
        Combines both into a single emotion dict with a small sentiment boost.

        Example
        -------
        >>> DatasetAdapters.from_meld("fear", "negative")
        {'fear': 0.75, 'Sadness': 0.2}
        """
        base: Dict[str, float] = {emotion.lower(): 0.75}
        sentiment_boost: Dict[str, Dict[str, float]] = {
            "positive": {"Joy":     0.3},
            "negative": {"Sadness": 0.2},
            "neutral":  {},
        }
        base.update(sentiment_boost.get(sentiment.lower(), {}))
        return base

    @staticmethod
    def from_empathetic(emotion_label: str, score: float = 0.8) -> Dict[str, float]:
        """
        EmpatheticDialogues — situation-level emotion string.
        In practice, pair with RoBERTa scores from the utterance text.

        Example
        -------
        >>> DatasetAdapters.from_empathetic("afraid")
        {'afraid': 0.8}
        """
        return {emotion_label.lower(): score}


# ---------------------------------------------------------------------------
# Module 3 conditioning prompt builder
# ---------------------------------------------------------------------------

def build_conditioning_prompt(signals: TrajectorySignals) -> str:
    """
    Build the emotion-conditioning instruction string for Module 3.

    This string is prepended to the LLM system prompt so the Adaptive
    Dialogue Generator can tailor its tone to the user's emotional
    trajectory.

    Returns
    -------
    str
        e.g. "[EMOTIONAL CONTEXT — Module 2 output]\\nUser emotional
        trajectory: escalating anxiety → pressure. ..."

    Example
    -------
    >>> prompt = build_conditioning_prompt(signals)
    >>> print(prompt)
    """
    esc = signals.escalation_score
    vol = signals.volatility_index

    trajectory_str = " → ".join(signals.emotion_sequence)
    if esc > 0.15:
        traj_prefix = "escalating"
    elif esc < -0.15:
        traj_prefix = "de-escalating"
    else:
        traj_prefix = "stable"

    if vol > 0.6:
        vol_label = "very high"
    elif vol > 0.35:
        vol_label = "high"
    elif vol > 0.15:
        vol_label = "moderate"
    else:
        vol_label = "low"

    if esc > 0.2 and vol > 0.3:
        tone = "validation + grounding + emotional de-escalation"
    elif esc > 0.1:
        tone = "empathetic acknowledgement + gentle reframing"
    elif esc < -0.1:
        tone = "reinforcing + warm encouragement"
    elif vol > 0.4:
        tone = "stabilising + calm, consistent reassurance"
    else:
        tone = "neutral + supportive"

    return (
        f"[EMOTIONAL CONTEXT — Module 2 output]\n"
        f"User emotional trajectory: {traj_prefix} {trajectory_str.lower()}.\n"
        f"Dominant state: {signals.dominant_state}. "
        f"Momentum: {signals.emotional_momentum:+.3f}. "
        f"Volatility: {vol_label} ({signals.volatility_index:.3f}). "
        f"Escalation: {signals.escalation_score:+.3f}.\n"
        f"Recommended tone: {tone}.\n"
        f"[END EMOTIONAL CONTEXT]"
    )
