"""
Module 2 — Module 1 Model Inference Bridge
===========================================
Loads the fine-tuned RoBERTa model saved by Module 1 and runs it on
utterance strings to produce the Dict[str, float] inputs that
EmotionalTrajectoryTracker.add_turn() expects.

Typical usage
-------------
    from module2.infer import load_model, predict_emotions, run_full_pipeline

    load_model("outputs/module1_goemotions/module1_model")

    conversation = [
        "I'm really stressed about my exam tomorrow.",
        "I studied but I feel like I'm going to mess everything up.",
        "My parents are expecting too much.",
    ]
    predictions, signals = run_full_pipeline(conversation)
    print(signals.summary())

Notes
-----
- All files are loaded from disk — no internet connection required.
- Call load_model() once at startup; predict_emotions() reuses the
  cached model and tokenizer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from .tracker import EmotionalTrajectoryTracker, TrajectorySignals

# ---------------------------------------------------------------------------
# Module-level state (populated by load_model)
# ---------------------------------------------------------------------------
_model = None
_tokenizer = None
_device = None
_labels: List[str] = []
_threshold: float = 0.30

# Default GoEmotions 28-label order (fallback if config.json lacks id2label)
_DEFAULT_GOEMOTION_LABELS: List[str] = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_dir: str,
    inference_threshold: float = 0.30,
    verbose: bool = True,
) -> None:
    """
    Load the Module 1 RoBERTa model and tokenizer from *model_dir*.

    Must be called before predict_emotions() or run_full_pipeline().

    Parameters
    ----------
    model_dir : str
        Path to the saved model directory (contains model.safetensors,
        config.json, tokenizer.json, etc.).
    inference_threshold : float
        Sigmoid probability cutoff. Labels below this are excluded per turn.
    verbose : bool
        Print loading progress to stdout.

    Raises
    ------
    FileNotFoundError
        If any required file is missing from model_dir.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    global _model, _tokenizer, _device, _labels, _threshold

    model_path = Path(model_dir)
    required = [
        "model.safetensors", "config.json", "tokenizer.json",
        "tokenizer_config.json", "vocab.json", "merges.txt",
        "special_tokens_map.json",
    ]
    if verbose:
        print("Checking model directory:")
    missing = []
    for fname in required:
        fpath = model_path / fname
        if verbose:
            size = f"({fpath.stat().st_size / 1024 / 1024:.1f} MB)" if fpath.exists() else ""
            status = "✓" if fpath.exists() else "✗ MISSING"
            print(f"  {status}  {fname} {size}")
        if not fpath.exists():
            missing.append(fname)
    if missing:
        raise FileNotFoundError(
            f"Missing files in {model_path}:\n  " + "\n  ".join(missing)
        )

    # Label mapping
    with open(model_path / "config.json") as f:
        cfg = json.load(f)
    if "id2label" in cfg:
        _labels = [cfg["id2label"][str(i)] for i in range(len(cfg["id2label"]))]
        if verbose:
            print(f"\nLabel mapping loaded from config.json — {len(_labels)} labels.")
    else:
        _labels = list(_DEFAULT_GOEMOTION_LABELS)
        if verbose:
            print(f"\nNo id2label in config — using default GoEmotions order.")
    if verbose:
        print("  Labels:", _labels)

    # Tokenizer
    if verbose:
        print("\nLoading tokenizer...")
    _tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if verbose:
        print(f"  ✓ Tokenizer loaded  (vocab size: {_tokenizer.vocab_size:,})")

    # Model weights
    if verbose:
        print("\nLoading model weights from model.safetensors...")
    _model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path),
        local_files_only=True,
    )

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = _model.to(_device)
    _model.eval()
    _threshold = inference_threshold

    if verbose:
        print(f"  ✓ Weights loaded")
        print(f"\n{'─' * 40}")
        print(f"  Model     : {_model.config.model_type} for sequence classification")
        print(f"  Labels    : {_model.config.num_labels}")
        print(f"  Device    : {_device}")
        print(f"  Parameters: {sum(p.numel() for p in _model.parameters()):,}")
        print(f"{'─' * 40}")

    # Optional: display training args
    try:
        import torch as _torch
        training_args = _torch.load(
            model_path / "training_args.bin", weights_only=False
        )
        if verbose:
            print(f"\nTraining args recovered:")
            print(f"  num_train_epochs    : {training_args.num_train_epochs}")
            print(f"  per_device_train_bs : {training_args.per_device_train_batch_size}")
            print(f"  learning_rate       : {training_args.learning_rate}")
    except Exception as e:
        if verbose:
            print(f"\n  (training_args.bin could not be parsed: {e})")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_emotions(
    utterances: List[str],
    threshold: Optional[float] = None,
    batch_size: int = 8,
) -> List[Dict[str, float]]:
    """
    Run the Module 1 model on a list of utterance strings.

    load_model() must be called before this function.

    Parameters
    ----------
    utterances : list[str]
        One string per conversational turn.
    threshold : float or None
        Override the sigmoid probability cutoff set in load_model().
    batch_size : int
        Number of utterances to process per forward pass.

    Returns
    -------
    list[dict[str, float]]
        One dict per utterance, e.g. [{"fear": 0.82, "nervousness": 0.61}, ...]
        Ready to pass directly into EmotionalTrajectoryTracker.add_turn().

    Raises
    ------
    RuntimeError
        If load_model() has not been called yet.
    """
    import torch

    if _model is None or _tokenizer is None:
        raise RuntimeError(
            "Model not loaded. Call module2.infer.load_model(model_dir) first."
        )

    cutoff = threshold if threshold is not None else _threshold
    all_results: List[Dict[str, float]] = []

    for i in range(0, len(utterances), batch_size):
        batch = utterances[i: i + batch_size]
        encoded = _tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(_device)

        with torch.no_grad():
            logits = _model(**encoded).logits        # (batch, num_labels)
            probs  = torch.sigmoid(logits).cpu()     # multi-label sigmoid

        for row in probs:
            emotions = {
                _labels[j]: float(row[j])
                for j in range(len(_labels))
                if float(row[j]) >= cutoff
            }
            # Guarantee at least the top-1 label even if nothing clears threshold
            if not emotions:
                top_idx = int(row.argmax())
                emotions = {_labels[top_idx]: float(row[top_idx])}
            all_results.append(emotions)

    return all_results


# ---------------------------------------------------------------------------
# End-to-end pipeline helper
# ---------------------------------------------------------------------------

def run_full_pipeline(
    conversation: List[str],
    threshold: Optional[float] = None,
) -> Tuple[List[Dict[str, float]], TrajectorySignals]:
    """
    End-to-end pipeline: raw text → Module 1 predictions → Module 2 signals.

    load_model() must be called before this function.

    Parameters
    ----------
    conversation : list[str]
        Ordered list of utterance strings (one per turn).
    threshold : float or None
        Override sigmoid cutoff (uses load_model default if None).

    Returns
    -------
    (predictions, TrajectorySignals)
        predictions : list of per-turn emotion dicts from Module 1
        signals     : TrajectorySignals with all Module 2 outputs
    """
    predictions = predict_emotions(conversation, threshold=threshold)

    tracker = EmotionalTrajectoryTracker()
    for turn_emotions in predictions:
        tracker.add_turn(turn_emotions)
    signals = tracker.compute()

    return predictions, signals
