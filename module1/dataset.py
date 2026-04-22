"""Load GoEmotions and prepare multi-label tensors for training."""

from __future__ import annotations

from typing import Any

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .go_emotions_mapping import GO_EMOTION_NAMES, go_label_ids_to_multihot
from .labels import EMOTION_LABELS


def load_go_emotions_simplified() -> DatasetDict:
    """HF `simplified` config: multi-label lists + train/validation/test splits."""
    return load_dataset("go_emotions", "simplified")


def prepare_tokenized(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 128,
) -> DatasetDict:
    ds = load_go_emotions_simplified()
    hf_names = tuple(ds["train"].features["labels"].feature.names)
    if hf_names != GO_EMOTION_NAMES:
        raise ValueError(
            "Update GO_EMOTION_NAMES in go_emotions_mapping.py to match the dataset."
        )

    def _add_vec(example: dict[str, Any]) -> dict[str, Any]:
        return {"label_vec": go_label_ids_to_multihot(example["labels"])}

    ds = ds.map(_add_vec)
    ds = ds.remove_columns(["labels"])

    def _tok(batch: dict[str, Any]) -> dict[str, Any]:
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        enc["label_vec"] = batch["label_vec"]
        return enc

    tokenized = ds.map(_tok, batched=True)
    tokenized = tokenized.rename_column("label_vec", "labels")
    return tokenized


def get_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def assert_label_width() -> None:
    assert len(EMOTION_LABELS) == 7
