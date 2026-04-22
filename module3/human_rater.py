"""
Human-rater A/B export.

export_human_rater_csv writes an anonymised A/B sheet: each conversation
produces two rows (label A vs B, order randomised per conversation), with
empty empathy / appropriateness / helpfulness columns for the rater to fill.

load_human_ratings reads the completed CSV back and computes per-response
composite scores grouped by response_label.
"""

from __future__ import annotations

import csv
import os
import random
from typing import Any, Dict, List


def export_human_rater_csv(
    records: List[Dict[str, Any]],
    path: str = "human_rater_sheet.csv",
    *,
    append: bool = False,
    seed: int = 42,
) -> None:
    """
    Export A/B pairs to CSV.

    append=True lets the Streamlit app accumulate conversations into a single
    file across user sessions.
    """
    rng = random.Random(seed)
    rows = []
    for rec in records:
        pairs = [
            ("A", rec["baseline_response"]),
            ("B", rec["conditioned_response"]),
        ]
        rng.shuffle(pairs)
        for label, response in pairs:
            rows.append({
                "conversation_id": rec["conversation_id"],
                "response_label": label,
                "user_last_turn": rec["user_last_turn"],
                "response_text": response,
                "empathy_score": "",
                "appropriateness_score": "",
                "helpfulness_score": "",
                "notes": "",
            })

    mode = "a" if (append and os.path.exists(path)) else "w"
    write_header = (mode == "w") or os.path.getsize(path) == 0

    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def load_human_ratings(path: str):
    """Read a completed rater sheet and return a DataFrame with a composite
    column (mean of empathy, appropriateness, helpfulness)."""
    import pandas as pd

    df = pd.read_csv(path)
    for col in ["empathy_score", "appropriateness_score", "helpfulness_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["composite"] = df[[
        "empathy_score", "appropriateness_score", "helpfulness_score",
    ]].mean(axis=1)
    return df
