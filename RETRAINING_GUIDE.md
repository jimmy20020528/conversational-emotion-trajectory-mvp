# Retraining Module 1 for broader generalization

Current model is RoBERTa-base fine-tuned on **GoEmotions only** (Reddit comments). Macro F1 = 0.602. The live demo fails on customer-complaint / imperative sentences (e.g. "I want my money back" → misclassified as joy) because the training distribution never saw that register.

This guide retrains Module 1 on a **multi-domain mix** — Reddit + Twitter + dialogue — to generalize to everyday conversational use. You'll do the training; this doc just gives you the recipe.

---

## 0. Compute — where to run it

| Setup | Time estimate (3 epochs, ~200k rows, roberta-base) | Notes |
|---|---|---|
| Apple M1/M2 MPS (`device=mps`) | ~4–6 h | Works but transformers' `Trainer` flakiness on MPS is real. Best to run a smoke test first. |
| Colab free T4 | ~50–90 min | Easiest. Mount Drive to persist the `outputs/` folder. |
| Colab Pro A100 / local RTX 3090+ | ~10–20 min | Also enables `--bf16`. |
| CPU only | don't — ~12 h | Use Colab. |

Training is not supported on the current CPU venv in reasonable time.

---

## 1. Fetch the three extra public datasets (~1 min)

```bash
source .venv/bin/activate
pip install -r requirements.txt   # datasets is already listed

python scripts/fetch_public_datasets.py
# -> data/external/dair_emotion.csv         (~20k rows, Twitter, 6 labels)
# -> data/external/daily_dialog.csv         (~100k utterances, dialogue, 7 labels)
# -> data/external/tweet_eval_emotion.csv   (~5k rows, Twitter, 4 labels)
```

You can also fetch a subset: `python scripts/fetch_public_datasets.py dair daily`.

### What each dataset adds
- **dair-ai/emotion** — Twitter microblog, short texts, 6 classes. Fixes short-sentence weakness.
- **DailyDialog** — everyday conversation utterances (hospital, shopping, job etc.), 7 classes. Fixes the customer-service / dialogue register gap that the current model fails on.
- **tweet_eval/emotion** — small Twitter set, 4 classes. Additional short-text coverage.

### Label-map JSONs (already written for you)
- [configs/dair_emotion_map.json](configs/dair_emotion_map.json)
- [configs/daily_dialog_map.json](configs/daily_dialog_map.json)
- [configs/tweet_eval_emotion_map.json](configs/tweet_eval_emotion_map.json)

All three collapse onto the **existing 7 canonical labels** (`joy, sadness, anger, fear, disgust, surprise, contempt`). Note:
- DailyDialog `no_emotion` → empty vector (rows get dropped unless you change `keep_zero_vector_rows=True` in `csv_extra.py`).
- `love` (dair) and `optimism` (tweet_eval) are mapped to `joy` to preserve the 7-label contract.
- **None of these datasets labels contempt**. Contempt will remain GoEmotions-only. If you care about contempt, plan to add a Kaggle dataset that does label it (see §5 below).

---

## 2. Run training — one command

```bash
python -m module1.train \
  --model_name microsoft/deberta-v3-base \
  --data_mix configs/example_data_mix.json \
  --output_dir outputs/module1_multidomain \
  --num_train_epochs 3 \
  --per_device_train_batch_size 16 \
  --learning_rate 2e-5 \
  --bf16
```

### Why these choices

| Flag | Why |
|---|---|
| `--model_name microsoft/deberta-v3-base` | DeBERTa-v3-base is a strict drop-in upgrade over RoBERTa-base at the same size — typically +2–4 F1 on emotion tasks. Keep RoBERTa if you want an apples-to-apples comparison; then re-run with `roberta-base` for a baseline. |
| `--data_mix configs/example_data_mix.json` | Merges all 3 CSVs into the GoEmotions train split in one call (see the new multi-CSV support in `module1/train.py`). |
| `--output_dir outputs/module1_multidomain` | **Different folder** from the current model so you can A/B the old vs new in the Streamlit sidebar by just changing the path. |
| `--num_train_epochs 3` | Standard for BCE multi-label. Go 4–5 on DeBERTa if val loss still dropping. |
| `--per_device_train_batch_size 16` | Safe for a 16 GB GPU. Raise to 32 on A100 to cut wall-clock. |
| `--learning_rate 2e-5` | Default for both RoBERTa and DeBERTa fine-tuning. Don't change unless you see loss explosion. |
| `--bf16` | Required on Ampere+ GPUs for reasonable speed with DeBERTa-v3. Drop if on T4 (use default fp32 or `--fp16` via a code tweak). |

---

## 3. Smoke test first (30 seconds)

Before the full run:

```bash
python -m module1.train \
  --model_name microsoft/deberta-v3-base \
  --data_mix configs/example_data_mix.json \
  --output_dir outputs/module1_smoke \
  --max_steps 20 \
  --per_device_train_batch_size 4
```

This does 20 optimizer steps and saves a junk model; verifies the data mix loads, label maps apply, and the CSVs you downloaded have the expected columns. Delete `outputs/module1_smoke/` afterward.

---

## 4. Point the Streamlit app at the new model

In the app sidebar, change the "Module 1 model folder" text input from:
```
outputs/module1_goemotions/module1_model
```
to:
```
outputs/module1_multidomain/module1_model
```

Streamlit caches the model — click "Clear conversation" (or restart) to force a reload.

Re-run the pizza case:
```
"i don't like this pizza, it tastes so bad"
"it's such a waste of money"
"I want my money back"
```

**What to expect** if the retrain generalizes well:
- Turn 1 dominant: `disgust` (was `contempt`) or `anger` — both defensible for food tasting bad
- Turn 3 dominant: `anger` (was misfiring to `joy`) — DailyDialog has lots of angry complaint turns
- Emotion arc: `disgust → anger → anger` (or similar), escalating; momentum negative
- This matches the human intuition for the conversation

If the retrain didn't help (new model still says `joy` for "I want my money back"), dig into §5.

---

## 5. If results still don't generalize — diagnosis checklist

Before retraining again, inspect what went in:

```bash
# Confirm row counts per dataset
wc -l data/external/*.csv

# Confirm label distribution in each (check contempt is still rare)
python -c "
import pandas as pd
for f in ['data/external/dair_emotion.csv',
          'data/external/daily_dialog.csv',
          'data/external/tweet_eval_emotion.csv']:
    df = pd.read_csv(f)
    print(f, '\n', df['label'].value_counts(), '\n')
"
```

Expected: DailyDialog will be ~85% `no_emotion`. Those get dropped by `csv_extra.py`, leaving ~10–15k usable emotional rows. If DailyDialog contributes < 5k rows after dropping, increase the no-emotion keep rate (see §6 option C).

### Per-class F1 after training
Check `outputs/module1_multidomain/eval_metrics.json` — the default only reports macro/micro. To see which class is still weak, add this one-shot snippet:

```python
from sklearn.metrics import classification_report
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from module1.labels import EMOTION_LABELS
from module1.go_emotions_mapping import go_label_ids_to_multihot

tok = AutoTokenizer.from_pretrained("outputs/module1_multidomain/module1_model")
m   = AutoModelForSequenceClassification.from_pretrained("outputs/module1_multidomain/module1_model")
m.eval()

val = load_dataset("go_emotions", "simplified", split="validation")
y_true, y_pred = [], []
for ex in val:
    y_true.append(go_label_ids_to_multihot(ex["labels"]))
    enc = tok(ex["text"], truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        p = torch.sigmoid(m(**enc).logits)[0].numpy()
    y_pred.append((p >= 0.5).astype(int))
print(classification_report(np.array(y_true), np.array(y_pred), target_names=EMOTION_LABELS))
```

---

## 6. Further levers if you need more lift

### A. **Drop `contempt` if you don't have a training source for it**
Contempt has basically no signal in any of these public datasets. Realistic path: either (a) accept low contempt F1 as a known limitation, or (b) find a Kaggle dataset that explicitly labels contempt and add it as a 4th extra.

### B. **Class-weighted BCE for rare labels**
`compute_loss` in `module1/train.py` currently uses plain `binary_cross_entropy_with_logits`. For rare labels (contempt, disgust), add `pos_weight`:

```python
pos_weight = torch.tensor([1.0, 1.0, 1.0, 1.0, 2.5, 1.0, 3.0]).to(labels.device)
# order matches EMOTION_LABELS: joy, sadness, anger, fear, disgust, surprise, contempt
loss = F.binary_cross_entropy_with_logits(logits, labels.float(), pos_weight=pos_weight)
```

Start with modest weights (2×–3×) for the minority classes. Higher weights often hurt precision badly.

### C. **Keep some neutral utterances for hard-negative mining**
Set `keep_zero_vector_rows=True` in the `tokenized_from_labeled_csv` call (edit `module1/csv_extra.py` line in the function body). DailyDialog's neutral turns then become explicit "all-zeros" targets — the model learns that not every sentence is emotional, which reduces false positives on bland text.

### D. **Longer context, bigger batches, longer training**
Bump `--max_length` 128 → 192 (costs ~40% more VRAM), `--num_train_epochs` 3 → 5. Diminishing returns past 5 on this size.

---

## 7. What files you'll need to keep in sync across teammates

After your retrain completes, share with the team:
- `outputs/module1_multidomain/module1_model/` — the **whole folder**, ~480 MB (same constraint as before: `model.safetensors` alone won't load)
- `outputs/module1_multidomain/eval_metrics.json`
- `outputs/module1_multidomain/module1_meta.json` (auto-written by `train.py`, records which datasets were used)
- The [configs/example_data_mix.json](configs/example_data_mix.json) you used, so the run is reproducible

Don't commit `data/external/*.csv` — they're large-ish and trivially re-fetchable with `scripts/fetch_public_datasets.py`.

---

## 8. TL;DR command sequence

```bash
# Once:
python scripts/fetch_public_datasets.py

# Smoke test (30s):
python -m module1.train \
  --model_name microsoft/deberta-v3-base \
  --data_mix configs/example_data_mix.json \
  --output_dir outputs/module1_smoke --max_steps 20 --per_device_train_batch_size 4

# Real training (T4 ~1h, A100 ~15min):
python -m module1.train \
  --model_name microsoft/deberta-v3-base \
  --data_mix configs/example_data_mix.json \
  --output_dir outputs/module1_multidomain \
  --num_train_epochs 3 --per_device_train_batch_size 16 --bf16

# Then in the Streamlit app, switch the sidebar model path to:
#   outputs/module1_multidomain/module1_model
```
