"""
Module 2 — Conversational Emotion Trajectory
=============================================
Public API
----------
    from module2 import EmotionalTrajectoryTracker, TrajectorySignals, TurnSnapshot
    from module2 import build_emotion_state_graph, plot_trajectory
    from module2 import DatasetAdapters, build_conditioning_prompt
    from module2.infer import load_model, predict_emotions, run_full_pipeline
"""

from .tracker import EmotionalTrajectoryTracker, TrajectorySignals, TurnSnapshot
from .visualise import build_emotion_state_graph, plot_trajectory
from .adapters import DatasetAdapters, build_conditioning_prompt

__all__ = [
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
    # infer sub-module (load_model, predict_emotions, run_full_pipeline)
    # imported explicitly: from module2.infer import load_model, ...
]
