"""Fine-tune a encoder model for 7-way multi-label emotion classification."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from .collator import MultiLabelCollator
from .csv_extra import load_label_map, merge_train_with_csv, tokenized_from_labeled_csv
from .dataset import get_tokenizer, prepare_tokenized
from .labels import EMOTION_LABELS


class MultiLabelTrainer(Trainer):
    """HF 4.57+ may not return loss for some heads; we apply BCE-with-logits explicitly."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        return (loss, outputs) if return_outputs else loss


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    y_pred = (probs >= 0.5).astype(np.int32)
    y_true = labels.astype(np.int32)
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="roberta-base",
        help="Hugging Face model id (e.g. roberta-base, microsoft/deberta-v3-small).",
    )
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--output_dir", default="outputs/module1_goemotions")
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="If >0, stop training after this many steps (smoke tests).",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bf16 if supported (CUDA Ampere+).",
    )
    parser.add_argument(
        "--kaggle_csv",
        default=None,
        help="Path to a Kaggle/local CSV to merge into training (same tokenizer).",
    )
    parser.add_argument(
        "--csv_text_col",
        default="text",
        help="Column name for utterance text in --kaggle_csv.",
    )
    parser.add_argument(
        "--csv_label_col",
        default="label",
        help="Column name for emotion label(s) in --kaggle_csv.",
    )
    parser.add_argument(
        "--csv_label_map",
        default=None,
        help="JSON file: raw label string -> list of canonical emotions, e.g. configs/kaggle_label_map.example.json",
    )
    parser.add_argument(
        "--data_mix",
        default=None,
        help=(
            "JSON file describing multiple CSVs to merge into GoEmotions training. "
            "Shape: {\"extras\": [{\"csv\": ..., \"text_col\": ..., \"label_col\": ..., \"label_map\": ...}, ...]}. "
            "When provided, --kaggle_csv is ignored."
        ),
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tokenizer = get_tokenizer(args.model_name)
    ds = prepare_tokenized(tokenizer, max_length=args.max_length)

    extras_spec: list[dict] = []
    if args.data_mix:
        mix = json.loads(Path(args.data_mix).read_text(encoding="utf-8"))
        extras_spec = list(mix.get("extras", []))
    elif args.kaggle_csv:
        if not args.csv_label_map:
            raise ValueError("--csv_label_map is required when using --kaggle_csv")
        extras_spec = [{
            "csv": args.kaggle_csv,
            "text_col": args.csv_text_col,
            "label_col": args.csv_label_col,
            "label_map": args.csv_label_map,
        }]

    for spec in extras_spec:
        csv_path = Path(spec["csv"])
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        lmap = load_label_map(Path(spec["label_map"]))
        extra = tokenized_from_labeled_csv(
            tokenizer,
            csv_path,
            text_col=spec.get("text_col", "text"),
            label_col=spec.get("label_col", "label"),
            label_map=lmap,
            max_length=args.max_length,
        )
        print(f"  merged {csv_path.name}: +{len(extra)} rows")
        ds["train"] = merge_train_with_csv(ds["train"], extra)

    collator = MultiLabelCollator(tokenizer)

    id2label = {str(i): lab for i, lab in enumerate(EMOTION_LABELS)}
    label2id = {lab: i for i, lab in enumerate(EMOTION_LABELS)}

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(EMOTION_LABELS),
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dataset_desc = "go_emotions/simplified"
    if extras_spec:
        extras_names = ", ".join(Path(s["csv"]).name for s in extras_spec)
        dataset_desc = f"go_emotions/simplified + {extras_names}"
    meta = {
        "model_name": args.model_name,
        "emotion_labels": list(EMOTION_LABELS),
        "dataset": dataset_desc,
        "max_length": args.max_length,
    }
    (out / "module1_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    ta: dict = {
        "output_dir": str(out),
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1_macro",
        "greater_is_better": True,
        "logging_steps": 50,
        "report_to": "none",
        "bf16": args.bf16,
        "seed": args.seed,
    }
    if args.max_steps > 0:
        ta["max_steps"] = args.max_steps
        step = max(1, args.max_steps // 2)
        ta["eval_strategy"] = "steps"
        ta["eval_steps"] = step
        ta["save_strategy"] = "steps"
        ta["save_steps"] = step
    else:
        ta["num_train_epochs"] = args.num_train_epochs
        ta["eval_strategy"] = "epoch"
        ta["save_strategy"] = "epoch"

    training_args = TrainingArguments(**ta)

    trainer = MultiLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=_compute_metrics,
    )

    trainer.train()
    export_dir = out / "module1_model"
    trainer.save_model(str(export_dir))
    tokenizer.save_pretrained(str(export_dir))

    metrics = trainer.evaluate()
    (out / "eval_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
