# Presentation handoff — Module 1 + Web App

Hand this to your teammate who is presenting. It covers **what I built**, **why**, **how it works**, **what numbers to quote**, and a **live demo script**.

---

## 1. One-slide overview (say this first)

> "I built the **utterance-level emotion classifier (Module 1)** and the **Streamlit web app** that ties all three modules together into a live demo.
>
> The classifier is a **fine-tuned RoBERTa-base** trained on GoEmotions, mapped from 28 labels down to our 7 canonical emotions (joy, sadness, anger, fear, disgust, surprise, contempt). It outputs multi-label sigmoid scores, so each turn gets a full 7-dimensional emotion vector — not just a single top label.
>
> The Streamlit app loads the model once, lets the user send multi-turn messages, scores every turn in real time, feeds those scores into Module 2 for the trajectory, renders a live 7-emotion line chart, and on demand generates a trajectory-conditioned reply via Module 3."

---

## 2. Module 1 — Emotion classifier

### What it is
- Base model: **`roberta-base`** (125M params), loaded via HuggingFace `transformers`.
- Head: linear classifier, **7 output logits**, sigmoid activation (not softmax).
- Task formulation: **multi-label classification** (`problem_type="multi_label_classification"` in `config.json`). This is the correct formulation because one utterance can legitimately express multiple emotions (e.g. sadness + fear).
- Training data: HuggingFace `go_emotions/simplified` (~58k Reddit comments, 28 original labels).
- Label mapping: 28 GoEmotions labels → 7 canonical labels via a hand-written mapping in `module1/go_emotions_mapping.py` (needed because our downstream modules assume a fixed 7-label space).
- Training: 3 epochs, max_length=128, on GPU.

### Numbers to quote on the slide
| Metric | Value |
|---|---|
| Macro F1 | **0.602** |
| Micro F1 | **0.708** |
| Eval loss | 0.174 |
| Epochs | 3 |
| Model size | ~480 MB (`model.safetensors`) |

Source: [outputs/module1_goemotions/eval_metrics.json](outputs/module1_goemotions/eval_metrics.json)

### Why these decisions (anticipate these questions)
- **"Why RoBERTa instead of DeBERTa / BERT?"** RoBERTa-base is a well-tuned transformer that trains fast and is a standard baseline for emotion tasks. The repo is designed so DeBERTa-v3-small can be swapped in with `--model_name microsoft/deberta-v3-small`; we chose RoBERTa for speed to get a working end-to-end pipeline first.
- **"Why sigmoid / multi-label instead of softmax?"** Emotions co-occur. A user can be both sad AND anxious in the same sentence. Softmax forces a single label and throws away that information; sigmoid lets us keep all seven scores, which is exactly what Module 2 needs to compute an emotion trajectory.
- **"Why 7 labels, not 28?"** The downstream modules (trajectory tracker + dialogue generator) are built around a small, stable emotion basis. 28 labels would make the trajectory vector too sparse and visualizations unreadable. 7 matches Ekman's basic emotions + contempt, which is the most common practical grouping.

### Output schema (each utterance produces)
```json
{
  "dialog_id": "...",
  "turn_id": 2,
  "text": "I studied but I feel like I'm going to mess everything up.",
  "scores":     {"joy": 0.02, "sadness": 0.68, "anger": 0.11, "fear": 0.74, ...},
  "intensity":  {...same as scores...},
  "active_labels": ["sadness", "fear"],
  "logits": [...7 raw logits...]
}
```

---

## 3. Web app — Streamlit demo

### What it does (the live story)
1. User types a message in the chat input.
2. App tokenizes the text and runs Module 1 → gets 7 emotion scores.
3. App appends `{text, scores}` to `st.session_state.turns`.
4. App calls Module 2 `compute_trajectory()` over **all** turns so far → gets volatility, escalation, dominant-label arc, summary.
5. App re-renders:
   - The chat history.
   - Volatility + escalation metrics.
   - The trajectory summary sentence.
   - A live line chart showing how all 7 emotions have moved across turns.
6. User clicks "Generate trajectory-conditioned reply" → Module 3 produces a reply (OpenAI if key is set, otherwise a template that still quotes the trajectory summary).

### Architecture highlights to mention
- **`st.cache_resource`** on model loading — the 480 MB model loads **once**, not per message. Critical for usability.
- **Stateful session** via `st.session_state` — conversation persists across reruns without a database.
- **Sidebar controls** — model path override (for teammates with different folder layouts) + max-token slider + Clear-conversation button.
- **Zero-config fallback for Module 3** — if no `OPENAI_API_KEY`, the app still works end-to-end, just with a deterministic template reply.

### Why Streamlit (anticipate this)
- Fastest path from "Python function that works" to "clickable demo".
- Built-in support for line charts, metric widgets, chat input, session state, caching — no frontend code needed.
- Single file ([app.py](app.py), ~100 lines).

---

## 4. Live demo script (2–3 minutes)

**Before class**: activate venv, `streamlit run app.py`, confirm model loaded in sidebar, **clear the conversation**.

**Demo flow — narrate as you click:**

1. **Turn 1** — type: *"I'm really stressed about my exam tomorrow."*
   → "Notice the app immediately scored this — fear and sadness are the dominant emotions. Only one turn, so no trajectory yet."

2. **Turn 2** — type: *"I studied but I feel like I'm going to mess everything up."*
   → "Now we have two turns. Volatility is the L2 distance between these two 7-d vectors. The line chart starts to show movement."

3. **Turn 3** — type: *"My parents are expecting too much from me."*
   → "Escalation score just went positive — that means negative-affect mass is rising over time. The summary now reads something like: `Dominant emotion path: fear → sadness → anger. Volatility moderate. Negative-affect trend: rising.`"

4. Click **Generate reply**.
   → "Module 3 takes the **trajectory summary** — not just the last message — and conditions the LLM prompt on it. So the reply validates the escalation pattern, not just the last sentence."

5. Click **Clear conversation** to reset.

**Key line to land**: *"The difference from a normal emotion classifier is that we're not predicting a single label per sentence — we're tracking how emotions **move** across a conversation, and the app shows that movement live."*

---

## 5. Limitations to mention honestly (shows maturity)

- **Macro F1 = 0.602**, below our 0.80 target — GoEmotions is a hard, imbalanced dataset and 3 epochs of RoBERTa-base isn't state of the art. DeBERTa-v3 + more data would help.
- Trained on **Reddit comments only**, so performance on a conversational/therapeutic register is likely lower than eval F1 suggests.
- "Contempt" has low support in the mapped GoEmotions labels — its scores are the noisiest of the seven.
- `intensity` currently equals the sigmoid score; a separate intensity regression head was out of MVP scope.

---

## 6. What's handed off to downstream teammates

- **Module 1 runtime API** ([module1/runtime.py](module1/runtime.py)):
  ```python
  load_classifier(model_dir) -> (model, tokenizer, device)
  predict_scores(model, tokenizer, device, text) -> dict[str, float]
  ```
- **Module 1 batch CLI** ([module1/infer.py](module1/infer.py)): JSONL in → JSONL out with scores/active_labels/logits.
- **Model artifact**: the entire folder [outputs/module1_goemotions/module1_model/](outputs/module1_goemotions/module1_model/) (~480 MB). Whole folder is required — not just `model.safetensors`.
- **Stable label order** in [module1/labels.py](module1/labels.py). Do not reorder without retraining.

---

## 7. Files I own (for the "contributions" slide)

```
module1/
  train.py, infer.py, runtime.py          ← training, batch inference, live API
  labels.py, go_emotions_mapping.py       ← 28→7 label schema
  dataset.py, collator.py, csv_extra.py   ← data loading
  schema.py                               ← output JSON schema
outputs/module1_goemotions/module1_model/ ← trained model artifact
app.py                                    ← Streamlit demo
requirements.txt                          ← pinned deps
QUICKRUN.txt, README.md                   ← run instructions
```

---

## 8. Quick-reference cheatsheet

- **Framework stack**: PyTorch + HuggingFace `transformers` + Streamlit + pandas.
- **Model**: `roberta-base`, fine-tuned, multi-label, 7 classes, sigmoid.
- **Training data**: GoEmotions (simplified) from HuggingFace Hub.
- **Eval**: macro F1 0.602, micro F1 0.708.
- **Inference latency**: ~single-digit ms per utterance on CPU after model load (real number depends on machine — measure before presenting if asked).
- **Model size**: 480 MB on disk.
- **Demo**: `streamlit run app.py` — browser opens, chat, watch the line chart move, generate reply.
