"""
Build evaluation records from raw conversations.

A "record" is the unit the evaluator + human rater export consume:
    {
        conversation_id, conversation_turns, user_last_turn,
        predictions, signals, conditioning_prompt,
        baseline_response, conditioned_response
    }

Two builders:
  - `build_records_from_turns`  — when Module 1 scores already exist (Streamlit path)
  - `build_records_from_text`   — when only raw utterance strings exist (notebook / CLI)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from module2 import EmotionalTrajectoryTracker, build_conditioning_prompt

from .generate import generate_ab_pair


def build_records_from_turns(
    conversations: List[List[Dict[str, float]]],
    user_last_turns: List[str],
    *,
    model: str = "gpt-4o-mini",
    generate_pairs: bool = True,
    conversation_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Each `conversations[i]` is a list of per-turn score dicts (already produced
    by Module 1). `user_last_turns[i]` is the text of the final user message
    (what the assistant is replying to).

    Parameters
    ----------
    generate_pairs : bool
        If True, call the LLM (or template fallback) to produce both replies.
        If False, leave response fields as placeholders so a human can fill in.
    """
    if len(conversations) != len(user_last_turns):
        raise ValueError("conversations and user_last_turns must be the same length")

    records: List[Dict[str, Any]] = []
    for i, (scores_list, last_text) in enumerate(zip(conversations, user_last_turns)):
        tracker = EmotionalTrajectoryTracker()
        for scores in scores_list:
            tracker.add_turn(scores)
        signals = tracker.compute()
        cond = build_conditioning_prompt(signals)

        if generate_pairs:
            baseline, conditioned = generate_ab_pair(
                last_text,
                {"summary": ""},  # the conditioning prompt carries the emotion context
                conditioning_prompt=cond,
                model=model,
            )
        else:
            baseline = "[FILL IN BASELINE RESPONSE]"
            conditioned = "[FILL IN CONDITIONED RESPONSE]"

        cid = conversation_ids[i] if conversation_ids else f"conv_{i + 1:03d}"
        records.append({
            "conversation_id": cid,
            "conversation_turns": None,
            "user_last_turn": last_text,
            "predictions": scores_list,
            "signals": signals,
            "conditioning_prompt": cond,
            "baseline_response": baseline,
            "conditioned_response": conditioned,
        })
    return records


def build_records_from_text(
    conversations: List[List[str]],
    *,
    model_dir: Optional[str] = None,
    generate_pairs: bool = True,
    threshold: float = 0.30,
    model: str = "gpt-4o-mini",
) -> List[Dict[str, Any]]:
    """
    Raw-text path: takes a list of conversations (each a list of utterance
    strings), runs Module 1 on them, then builds A/B records.

    `model_dir` is the path to the fine-tuned Module 1 folder. Defaults to
    `outputs/module1_goemotions/module1_model`. Pass `generate_pairs=False`
    to get records with placeholder responses (useful for offline setup).
    """
    from module2.infer import load_model, predict_emotions

    model_dir = model_dir or "outputs/module1_goemotions/module1_model"
    load_model(model_dir, inference_threshold=threshold, verbose=False)

    records: List[Dict[str, Any]] = []
    for i, convo in enumerate(conversations):
        predictions = predict_emotions(convo, threshold=threshold)

        tracker = EmotionalTrajectoryTracker()
        for turn_emotions in predictions:
            tracker.add_turn(turn_emotions)
        signals = tracker.compute()
        cond = build_conditioning_prompt(signals)

        last_text = convo[-1]
        if generate_pairs:
            baseline, conditioned = generate_ab_pair(
                last_text,
                {"summary": ""},
                conditioning_prompt=cond,
                model=model,
            )
        else:
            baseline = "[FILL IN BASELINE RESPONSE]"
            conditioned = "[FILL IN CONDITIONED RESPONSE]"

        records.append({
            "conversation_id": f"conv_{i + 1:03d}",
            "conversation_turns": convo,
            "user_last_turn": last_text,
            "predictions": predictions,
            "signals": signals,
            "conditioning_prompt": cond,
            "baseline_response": baseline,
            "conditioned_response": conditioned,
        })
    return records
