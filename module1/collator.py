"""Batching: pad inputs and stack multi-label targets."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class MultiLabelCollator:
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        labels = [f.pop("labels") for f in features]
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["labels"] = torch.tensor(labels, dtype=torch.float32)
        return batch
