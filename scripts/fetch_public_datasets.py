"""
Download and flatten public emotion datasets from HuggingFace Hub into the
two-column CSV format expected by module1/csv_extra.py.

Each CSV has columns: text, label   (string label, one row per utterance).

Usage
-----
    python scripts/fetch_public_datasets.py                 # all three
    python scripts/fetch_public_datasets.py dair daily      # subset

Datasets covered
----------------
  dair        dair-ai/emotion           (Twitter, 6 classes, ~20k rows)
  daily       daily_dialog              (dialogue, 7 classes, ~100k utterances)
  tweet       tweet_eval/emotion        (Twitter, 4 classes, ~5k rows)

Notes
-----
- All are permissively licensed for research use. Check each dataset card.
- DailyDialog utterances are FLATTENED: each dialog's turns become separate rows.
- `no_emotion` / `neutral` are KEPT in the CSV with literal label strings;
  the label_map JSONs map them to empty lists (= zero label vector), and
  csv_extra.py skips zero-vector rows by default.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "external"


DAIR_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
DAILYDIALOG_LABELS = ["no_emotion", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]
TWEET_EVAL_LABELS = ["anger", "joy", "optimism", "sadness"]


def fetch_dair(out_dir: Path) -> Path:
    from datasets import load_dataset

    print("[dair-ai/emotion] loading...")
    ds = load_dataset("dair-ai/emotion", "split")
    frames = []
    for split in ("train", "validation", "test"):
        if split not in ds:
            continue
        df = ds[split].to_pandas()[["text", "label"]]
        df["label"] = df["label"].apply(lambda i: DAIR_LABELS[int(i)])
        df["split"] = split
        frames.append(df)
    out = out_dir / "dair_emotion.csv"
    pd.concat(frames).to_csv(out, index=False)
    print(f"  wrote {out}  ({sum(len(f) for f in frames)} rows)")
    return out


def fetch_daily_dialog(out_dir: Path) -> Path:
    """
    DailyDialog — the canonical `daily_dialog` repo on HF is a loading script,
    which modern `datasets` refuses. We route around it via HF's automatic
    Parquet conversion (`refs/convert/parquet`), which exists for every
    dataset whose script is runnable by the HF converter.
    Fields: dialog (list[str]), emotion (list[int]), act (list[int]).
    """
    from datasets import load_dataset

    print("[daily_dialog] loading via refs/convert/parquet (auto-converted)...")
    ds = load_dataset("daily_dialog", revision="refs/convert/parquet")

    rows: list[dict] = []
    for split in ("train", "validation", "test"):
        if split not in ds:
            continue
        for ex in ds[split]:
            dialog = ex["dialog"]
            emotions = ex["emotion"]
            for utt, e in zip(dialog, emotions):
                text = (utt or "").strip()
                if not text:
                    continue
                rows.append({
                    "text": text,
                    "label": DAILYDIALOG_LABELS[int(e)],
                    "split": split,
                })
    out = out_dir / "daily_dialog.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  wrote {out}  ({len(rows)} utterances)")
    return out


def fetch_tweet_eval(out_dir: Path) -> Path:
    from datasets import load_dataset

    print("[tweet_eval/emotion] loading...")
    ds = load_dataset("tweet_eval", "emotion")
    frames = []
    for split in ("train", "validation", "test"):
        if split not in ds:
            continue
        df = ds[split].to_pandas()[["text", "label"]]
        df["label"] = df["label"].apply(lambda i: TWEET_EVAL_LABELS[int(i)])
        df["split"] = split
        frames.append(df)
    out = out_dir / "tweet_eval_emotion.csv"
    pd.concat(frames).to_csv(out, index=False)
    print(f"  wrote {out}  ({sum(len(f) for f in frames)} rows)")
    return out


_FETCHERS = {
    "dair":  fetch_dair,
    "daily": fetch_daily_dialog,
    "tweet": fetch_tweet_eval,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch public emotion datasets as CSV.")
    parser.add_argument(
        "which",
        nargs="*",
        default=list(_FETCHERS.keys()),
        help=f"Datasets to fetch. Default: all ({', '.join(_FETCHERS)}).",
    )
    parser.add_argument("--out_dir", default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    unknown = [w for w in args.which if w not in _FETCHERS]
    if unknown:
        print(f"Unknown dataset(s): {unknown}. Allowed: {list(_FETCHERS)}", file=sys.stderr)
        sys.exit(2)

    for key in args.which:
        _FETCHERS[key](out_dir)


if __name__ == "__main__":
    main()
