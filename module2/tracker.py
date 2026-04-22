"""
Module 2 — Emotional Trajectory Tracker
========================================
Accepts per-turn multi-label emotion predictions from the upstream RoBERTa/
DeBERTa classifier (Module 1) and computes higher-order emotional signals
across the full conversation.

Outputs
-------
- Emotional Momentum    : EW-directional drift in valence
- Volatility Index      : Normalised std-dev of per-turn intensity
- Escalation Score      : Regression slope of negative-affect activation
- Dominant Emotional State : Recency-weighted dominant cluster
- Transition Matrix     : Row-normalised bigram probabilities
- Emotion State Graph   : Weighted networkx.DiGraph (see visualise.py)
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .taxonomy import EMOTION_CLUSTERS, VALENCE, AROUSAL


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class TurnSnapshot:
    """
    Per-utterance emotional state derived from the upstream classifier.

    Parameters
    ----------
    turn_idx : int
        0-based index of this turn in the conversation.
    raw_emotions : dict
        Direct classifier output, e.g. {"Anxiety": 0.72, "Stress": 0.81}.
    """

    turn_idx: int
    raw_emotions: Dict[str, float]
    clustered: Dict[str, float] = field(default_factory=dict)
    dominant: str = ""
    intensity: float = 0.0   # mean of raw scores
    valence: float = 0.0     # weighted valence in [-1, 1]
    arousal: float = 0.0     # weighted arousal in [0, 1]

    def __post_init__(self) -> None:
        self._cluster_emotions()

    def _cluster_emotions(self) -> None:
        """Map fine-grained labels to core clusters and compute affect dimensions."""
        acc: Dict[str, float] = defaultdict(float)
        for label, score in self.raw_emotions.items():
            cluster = EMOTION_CLUSTERS.get(label.lower(), label.title())
            acc[cluster] += score

        total = sum(acc.values()) or 1.0
        self.clustered = {k: v / total for k, v in acc.items()}
        self.dominant = max(self.clustered, key=self.clustered.get)
        self.intensity = float(np.mean(list(self.raw_emotions.values())))

        v, a, w = 0.0, 0.0, 0.0
        for cluster, weight in self.clustered.items():
            v += VALENCE.get(cluster, 0.0) * weight
            a += AROUSAL.get(cluster, 0.0) * weight
            w += weight
        if w:
            self.valence = round(v / w, 4)
            self.arousal = round(a / w, 4)

    def __repr__(self) -> str:
        return (
            f"TurnSnapshot(turn={self.turn_idx}, dominant={self.dominant!r}, "
            f"intensity={self.intensity:.3f}, valence={self.valence:+.3f}, "
            f"arousal={self.arousal:.3f})"
        )


@dataclass
class TrajectorySignals:
    """All higher-order signals computed over a complete conversation."""

    turns: int = 0
    emotional_momentum: float = 0.0
    volatility_index: float = 0.0
    escalation_score: float = 0.0
    dominant_state: str = ""
    transition_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    emotion_sequence: List[str] = field(default_factory=list)
    intensity_series: List[float] = field(default_factory=list)
    valence_series: List[float] = field(default_factory=list)
    arousal_series: List[float] = field(default_factory=list)
    snapshots: List[TurnSnapshot] = field(default_factory=list)

    def summary(self) -> str:
        """Return a formatted multi-line summary string."""
        lines = [
            "═" * 56,
            "  EMOTIONAL TRAJECTORY SUMMARY",
            "═" * 56,
            f"  Turns analysed     : {self.turns}",
            f"  Dominant state     : {self.dominant_state}",
            f"  Emotional momentum : {self.emotional_momentum:+.4f}",
            f"  Volatility index   : {self.volatility_index:.4f}",
            f"  Escalation score   : {self.escalation_score:+.4f}",
            "",
            "  Emotion sequence   : " + " → ".join(self.emotion_sequence),
            "",
            "  Turn-by-turn detail:",
        ]
        for snap in self.snapshots:
            raw_str = ", ".join(
                f"{k}({v:.2f})" for k, v in snap.raw_emotions.items()
            )
            lines.append(
                f"    T{snap.turn_idx + 1}: [{raw_str}]  "
                f"→ {snap.dominant}  "
                f"intensity={snap.intensity:.3f}  "
                f"valence={snap.valence:+.3f}  "
                f"arousal={snap.arousal:.3f}"
            )
        lines += ["", "  Transition probabilities:"]
        for src, targets in self.transition_matrix.items():
            for tgt, prob in sorted(targets.items(), key=lambda x: -x[1]):
                if prob > 0:
                    lines.append(f"    {src:<12} → {tgt:<12}  p={prob:.2f}")
        lines.append("═" * 56)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class EmotionalTrajectoryTracker:
    """
    Stateful tracker for higher-order emotional signals across a conversation.

    Usage
    -----
    >>> tracker = EmotionalTrajectoryTracker()
    >>> tracker.add_turn({"Anxiety": 0.72, "Stress": 0.81})
    >>> tracker.add_turn({"Anxiety": 0.85, "Self-doubt": 0.76})
    >>> tracker.add_turn({"Pressure": 0.89, "Frustration": 0.67})
    >>> signals = tracker.compute()
    >>> print(signals.summary())

    Parameters
    ----------
    window : int
        Rolling window size for momentum / escalation (default 3).
    """

    def __init__(self, window: int = 3) -> None:
        self.window = window
        self._snapshots: List[TurnSnapshot] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def add_turn(self, emotions: Dict[str, float]) -> TurnSnapshot:
        """Ingest a single turn's emotion predictions and return its snapshot."""
        snap = TurnSnapshot(
            turn_idx=len(self._snapshots),
            raw_emotions={k: float(v) for k, v in emotions.items()},
        )
        self._snapshots.append(snap)
        return snap

    def reset(self) -> None:
        """Clear all stored turns."""
        self._snapshots.clear()

    def compute(self) -> TrajectorySignals:
        """Compute all higher-order signals over the accumulated turns."""
        n = len(self._snapshots)
        if n == 0:
            raise ValueError("No turns added yet — call add_turn() first.")

        signals = TrajectorySignals(
            turns=n,
            snapshots=list(self._snapshots),
        )

        signals.emotion_sequence = [s.dominant for s in self._snapshots]
        signals.intensity_series = [s.intensity for s in self._snapshots]
        signals.valence_series   = [s.valence   for s in self._snapshots]
        signals.arousal_series   = [s.arousal   for s in self._snapshots]

        # Recency-weighted dominant state
        cluster_scores: Dict[str, float] = defaultdict(float)
        for i, snap in enumerate(self._snapshots):
            weight = (i + 1) / n
            for cluster, score in snap.clustered.items():
                cluster_scores[cluster] += score * weight
        signals.dominant_state = max(cluster_scores, key=cluster_scores.get)

        signals.emotional_momentum = self._compute_momentum(signals.valence_series)
        signals.volatility_index   = self._compute_volatility(signals.intensity_series)
        signals.escalation_score   = self._compute_escalation(
            signals.valence_series, signals.arousal_series
        )
        signals.transition_matrix  = self._build_transition_matrix(
            signals.emotion_sequence
        )
        return signals

    # ── Computation helpers ───────────────────────────────────────────────────

    @staticmethod
    def _compute_momentum(valence_series: List[float]) -> float:
        """
        Emotional Momentum = exponentially-weighted mean of first differences
        in valence. Sign = direction; magnitude = speed. Range: [-1, +1].
        """
        if len(valence_series) < 2:
            return 0.0
        diffs = [
            valence_series[i] - valence_series[i - 1]
            for i in range(1, len(valence_series))
        ]
        weights = [math.exp(i) for i in range(len(diffs))]
        w_total = sum(weights)
        momentum = sum(d * w for d, w in zip(diffs, weights)) / w_total
        return float(np.clip(momentum, -1.0, 1.0))

    @staticmethod
    def _compute_volatility(intensity_series: List[float]) -> float:
        """
        Volatility Index = std-dev of per-turn intensities, normalised to
        [0, 1] (max possible std for Uniform[0,1] ≈ 0.5).
        """
        if len(intensity_series) < 2:
            return 0.0
        std = float(np.std(intensity_series))
        return float(np.clip(std / 0.5, 0.0, 1.0))

    @staticmethod
    def _compute_escalation(
        valence_series: List[float],
        arousal_series: List[float],
    ) -> float:
        """
        Escalation Score = linear regression slope of negative-affect
        activation, defined as: arousal × (1 − (valence + 1) / 2).

        +1.0 → strong escalation toward high-arousal negative states
        -1.0 → de-escalation / calming
         0.0 → stable
        """
        n = len(valence_series)
        if n < 2:
            return 0.0
        neg_activation = [
            ar * (1.0 - (vl + 1.0) / 2.0)
            for vl, ar in zip(valence_series, arousal_series)
        ]
        x = np.arange(n, dtype=float)
        slope, _ = np.polyfit(x, neg_activation, 1)
        return float(np.clip(slope * n, -1.0, 1.0))

    @staticmethod
    def _build_transition_matrix(
        emotion_sequence: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """
        Build a row-normalised transition probability matrix from bigrams of
        the dominant-emotion sequence.
        """
        states = sorted(set(emotion_sequence))
        counts: Dict[str, Dict[str, float]] = {
            s: {t: 0.0 for t in states} for s in states
        }
        for a, b in zip(emotion_sequence, emotion_sequence[1:]):
            counts[a][b] += 1.0
        for src in states:
            total = sum(counts[src].values())
            if total > 0:
                for tgt in states:
                    counts[src][tgt] /= total
        return counts
