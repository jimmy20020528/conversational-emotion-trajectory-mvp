"""Adaptive reply generation + evaluation (Module 3)."""

from .generate import generate_reply, generate_baseline_reply, generate_ab_pair
from .prompts import BASELINE_SYSTEM_PROMPT, CONDITIONED_SYSTEM_PREFIX
from .simulated import SIMULATED_PAIRS, build_records_from_simulated
from .records import build_records_from_turns, build_records_from_text
from .evaluate import (
    AutomatedEvaluator,
    ResponseScores,
    print_evaluation_report,
    export_results_csv,
)
from .human_rater import export_human_rater_csv, load_human_ratings

__all__ = [
    # Generation
    "generate_reply",
    "generate_baseline_reply",
    "generate_ab_pair",
    # Prompts
    "BASELINE_SYSTEM_PROMPT",
    "CONDITIONED_SYSTEM_PREFIX",
    # Records
    "SIMULATED_PAIRS",
    "build_records_from_simulated",
    "build_records_from_turns",
    "build_records_from_text",
    # Evaluation
    "AutomatedEvaluator",
    "ResponseScores",
    "print_evaluation_report",
    "export_results_csv",
    # Human raters
    "export_human_rater_csv",
    "load_human_ratings",
]
