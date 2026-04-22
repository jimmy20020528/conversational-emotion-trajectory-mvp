"""Load saved Module 1 classifier and score a single utterance (for Streamlit / APIs)."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .labels import EMOTION_LABELS


def load_classifier(model_dir: Path | str):
    """Returns (model, tokenizer, device)."""
    path = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device


@torch.inference_mode()
def predict_scores(
    model,
    tokenizer,
    device: torch.device,
    text: str,
    *,
    max_length: int = 128,
) -> dict[str, float]:
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    logits = model(**enc).logits.float()
    probs = torch.sigmoid(logits).cpu().numpy()[0]
    return {lab: float(probs[i]) for i, lab in enumerate(EMOTION_LABELS)}
