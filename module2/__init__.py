"""
Module 2 — Conversational Emotion Trajectory
=============================================
Public API
----------
    # Legacy simple API (kept for backward compat with the first MVP — used by app.py)
    from module2 import compute_trajectory

    # Rich trajectory API (momentum, transition matrix, state graph, etc.)
    from module2 import EmotionalTrajectoryTracker, TrajectorySignals, TurnSnapshot
    from module2 import build_emotion_state_graph, plot_trajectory
    from module2 import DatasetAdapters, build_conditioning_prompt
    from module2.infer import load_model, predict_emotions, run_full_pipeline
"""

from .trajectory import compute_trajectory
from .tracker import EmotionalTrajectoryTracker, TrajectorySignals, TurnSnapshot
from .visualise import build_emotion_state_graph, plot_trajectory
from .adapters import DatasetAdapters, build_conditioning_prompt

__all__ = [
    # Legacy API
    "compute_trajectory",
    # Core tracker
    "EmotionalTrajectoryTracker",
    "TrajectorySignals",
    "TurnSnapshot",
    # Visualisation
    "build_emotion_state_graph",
    "plot_trajectory",
    # Adapters & Module 3 bridge
    "DatasetAdapters",
    "build_conditioning_prompt",
]
