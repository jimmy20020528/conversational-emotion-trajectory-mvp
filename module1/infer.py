"""Batch inference: JSONL in → JSONL out for Module 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .labels import EMOTION_LABELS
from .schema import prediction_to_jsonable


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Folder with saved model + tokenizer (e.g. outputs/module1_goemotions/module1_model).",
    )
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    rows = _read_jsonl(Path(args.input_jsonl))
    texts = [r.get("text", "") for r in rows]
    out_rows: list[dict] = []

    for i in tqdm(range(0, len(texts), args.batch_size), desc="infer"):
        batch_texts = texts[i : i + args.batch_size]
        batch_meta = rows[i : i + args.batch_size]
        enc = tokenizer(
            batch_texts,
            truncation=True,
            max_length=args.max_length,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits.float()
        logits_cpu = logits.cpu().numpy()
        probs = torch.sigmoid(logits).cpu().numpy()

        for j, prob in enumerate(probs):
            meta = batch_meta[j]
            scores = {lab: float(prob[k]) for k, lab in enumerate(EMOTION_LABELS)}
            logits_row = [float(x) for x in logits_cpu[j]]
            out_rows.append(
                prediction_to_jsonable(
                    dialog_id=str(meta.get("dialog_id", "")),
                    turn_id=int(meta.get("turn_id", 0)),
                    text=str(meta.get("text", "")),
                    scores=scores,
                    model_name=str(model_dir),
                    threshold=args.threshold,
                    speaker=meta.get("speaker"),
                    logits=logits_row,
                )
            )

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(Path(args.output_jsonl), out_rows)


if __name__ == "__main__":
    main()
