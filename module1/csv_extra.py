"""Optional Kaggle / local CSV merged into GoEmotions training (same 7-way multihot)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, concatenate_datasets
from transformers import PreTrainedTokenizerBase

from .labels import EMOTION_LABEL_TO_IDX


def _vec_from_canonicals(names: list[str]) -> list[float]:
    vec = [0.0] * len(EMOTION_LABEL_TO_IDX)
    for c in names:
        if c not in EMOTION_LABEL_TO_IDX:
            raise ValueError(
                f"Unknown emotion {c!r}. Allowed: {list(EMOTION_LABEL_TO_IDX)}"
            )
        vec[EMOTION_LABEL_TO_IDX[c]] = 1.0
    return vec


def load_label_map(path: Path) -> dict[str, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    # keys normalized to str; values are lists of canonical names
    return {str(k): list(v) for k, v in data.items()}


def tokenized_from_labeled_csv(
    tokenizer: PreTrainedTokenizerBase,
    csv_path: Path,
    *,
    text_col: str,
    label_col: str,
    label_map: dict[str, list[str]],
    max_length: int,
    skip_unknown_keys: bool = True,
    keep_zero_vector_rows: bool = True,
) -> Dataset:
    """
    CSV rows → same columns as GoEmotions train: input_ids, attention_mask, labels.

    label_map: raw string in CSV -> list of canonical labels, e.g.
      {"happy": ["joy"], "sad": ["sadness"], "neutral": []}
    """
    df = pd.read_csv(csv_path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"CSV must contain columns {text_col!r} and {label_col!r}; got {list(df.columns)}"
        )

    map_lower = {k.lower(): v for k, v in label_map.items()}

    texts: list[str] = []
    vecs: list[list[float]] = []

    for _, row in df.iterrows():
        t = row[text_col]
        if pd.isna(t) or not str(t).strip():
            continue
        raw = row[label_col]
        if pd.isna(raw):
            continue
        key = str(raw).strip()
        canon = label_map.get(key)
        if canon is None:
            canon = map_lower.get(key.lower())
        if canon is None and skip_unknown_keys:
            continue
        if canon is None:
            canon = []
        vec = _vec_from_canonicals(canon)
        if sum(vec) == 0 and not keep_zero_vector_rows:
            continue
        texts.append(str(t))
        vecs.append(vec)

    if not texts:
        raise ValueError("No rows left after filtering; check label_map vs CSV labels.")

    ds = Dataset.from_dict({"text": texts, "label_vec": vecs})

    def _tok(batch: dict[str, Any]) -> dict[str, Any]:
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        enc["label_vec"] = batch["label_vec"]
        return enc

    tok = ds.map(_tok, batched=True, remove_columns=["text"])
    tok = tok.rename_column("label_vec", "labels")
    return tok


def merge_train_with_csv(train_split: Dataset, extra: Dataset) -> Dataset:
    """Drop HF-only columns (e.g. id) and concatenate train + extra."""
    t = train_split
    if "id" in t.column_names:
        t = t.remove_columns(["id"])
    return concatenate_datasets([t, extra])
