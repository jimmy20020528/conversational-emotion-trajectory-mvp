# Conversational Emotional Trajectory & Adaptive Dialogue

End-to-end MVP for the AAI 6620 final project: multi-turn user text → per-turn multi-label emotion scores → conversation-level trajectory features → trajectory-conditioned LLM reply, exposed through a Streamlit UI.

See [PROPOSAL](PROPOSAL.md) for full research framing. This README tracks **what is actually built today** and **what is still missing** relative to the proposal.

---

## What works right now

### Module 1 — Utterance-level emotion classifier
- Fine-tuned `roberta-base` on HuggingFace `go_emotions/simplified`, with 28 original labels mapped down to **7 canonical emotions**: `joy, sadness, anger, fear, disgust, surprise, contempt` (see [module1/labels.py](module1/labels.py), [module1/go_emotions_mapping.py](module1/go_emotions_mapping.py)).
- Multi-label, sigmoid output (`problem_type=multi_label_classification`).
- Saved model at [outputs/module1_goemotions/module1_model/](outputs/module1_goemotions/module1_model/) (≈ 480 MB — **the whole folder is required**, not just `model.safetensors`).
- Eval: **macro F1 = 0.602, micro F1 = 0.708** ([eval_metrics.json](outputs/module1_goemotions/eval_metrics.json)).
- Runtime helper: [module1/runtime.py](module1/runtime.py) → `load_classifier()` + `predict_scores()`.
- Batch inference CLI: [module1/infer.py](module1/infer.py) (produces `artifacts/turn_emotions.jsonl`).

### Module 2 — Emotional trajectory tracker
- [module2/trajectory.py](module2/trajectory.py) `compute_trajectory(turns)` returns:
  - `volatility` — mean L2 step between consecutive 7-d score vectors.
  - `escalation_score` — negative-affect mass, first third vs last third of conversation.
  - `dominant_labels` per turn + `dominant_arc_text` (e.g. `sadness → fear → anger`).
  - `summary` — 2–4 sentence English blurb used to condition the LLM.
  - `volatility_bucket` (low / moderate / high).

### Module 3 — Adaptive dialogue generator
- [module3/generate.py](module3/generate.py) `generate_reply(last_user_text, traj)`:
  - If `OPENAI_API_KEY` is set → calls OpenAI Chat Completions (stdlib `urllib`, no extra dep) with an empathy-oriented system prompt and the trajectory summary.
  - Otherwise → deterministic template reply that still includes the trajectory summary.

### Streamlit demo
- [app.py](app.py): multi-turn chat input (`st.session_state`), model cached via `st.cache_resource`, per-turn metrics, 7-emotion line chart across turns, "Generate trajectory-conditioned reply" button, sidebar with model-path override and Clear-conversation button.

---

## How to run it

```bash
# 1. from the project root
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. launch the UI
streamlit run app.py

# 3. optional: real LLM replies instead of template fallback
export OPENAI_API_KEY="sk-..."       # Windows cmd:  set OPENAI_API_KEY=sk-...
```

The Streamlit sidebar defaults to `outputs/module1_goemotions/module1_model` — no change needed if you kept the folder layout.

### Using the model from Python directly
```python
from module1.runtime import load_classifier, predict_scores
from module2.trajectory import compute_trajectory
from module3.generate import generate_reply

model, tok, dev = load_classifier("outputs/module1_goemotions/module1_model")

turns = []
for text in ["I'm really stressed about my exam tomorrow.",
             "I studied but I feel like I'm going to mess everything up.",
             "My parents are expecting too much."]:
    scores = predict_scores(model, tok, dev, text)
    turns.append({"text": text, "scores": scores})

traj = compute_trajectory(turns)
print(traj["summary"])
print(generate_reply(turns[-1]["text"], traj))
```

### Batch inference (JSONL → JSONL for downstream analysis)
```bash
.venv/bin/python -m module1.infer \
  --model_dir outputs/module1_goemotions/module1_model \
  --input_jsonl data/sample_conversations.jsonl \
  --output_jsonl artifacts/turn_emotions.jsonl
```

---

## Repo layout
```
module1/      utterance classifier: training, inference, runtime, label mapping
module2/      trajectory features (volatility, escalation, summary)
module3/      LLM / template reply generation
outputs/      trained model + eval metrics        (module1_model/ is the model)
artifacts/    example JSONL outputs from Module 1
data/         sample multi-turn conversations
configs/      kaggle label-map example
app.py        Streamlit UI
```

---

## What is still missing vs. the proposal

Scope honestly: the current build is an MVP. Several items from the proposal are **not implemented yet**.

### Classification quality
- [ ] Macro F1 target **0.80** — currently **0.602**. Options: DeBERTa-v3 base, longer training, class-weighting, data augmentation with Kaggle CSVs (pipeline already wired, see `configs/kaggle_label_map.example.json`).
- [ ] Training data is **GoEmotions only**. Proposal also lists DailyDialog, EmpatheticDialogues, MELD — none of those are merged in yet.
- [ ] Separate `intensity` head — today `intensity == sigmoid score`; a distinct regression head was never trained.

### Trajectory modeling
- [ ] **Emotion transition matrix** (per-user P(eᵢ → eⱼ)) — not implemented.
- [ ] **Emotion State Graph** visualization — current UI only shows a line chart, no graph view.
- [ ] **Emotional momentum** as a first-class metric — volatility and escalation exist, momentum as defined in the proposal does not.
- [ ] Trajectory-level evaluation: no labeled test set for state-transition accuracy, drift detection, or early-escalation detection yet.

### Dialogue generation
- [ ] **Baseline vs. emotion-conditioned A/B** — only the conditioned path is wired; a no-trajectory baseline call for comparison is not.
- [ ] **Human rater study** (20–30 evaluators, empathy/appropriateness/helpfulness) — out of scope for MVP; proposal flagged it as stretch.
- [ ] Non-OpenAI LLM fallback (e.g. local model) — currently OpenAI or template only.

### App / UX
- [ ] Per-turn emotion chips in the chat bubbles (right now scores only appear in the line chart).
- [ ] Side-by-side baseline vs. conditioned reply panel.
- [ ] Export / save a session transcript + trajectory as JSON.

### Engineering
- [ ] Unit tests for `compute_trajectory` edge cases (0 turns, 1 turn, all-zero scores).
- [ ] CI + lint.
- [ ] Model card / dataset card.

---

## Known limitations to mention in the writeup
- Classifier was trained only on GoEmotions (Reddit comments), so performance on conversational / therapeutic-register text is likely lower than eval F1 suggests.
- "Contempt" is a **mapped** label from GoEmotions; its support in training is small, so its scores are noisier than the other six.
- `escalation_score` uses a hand-picked negative-affect set (`sadness, anger, fear, disgust, contempt`). That heuristic is defensible but not learned.
- `generate_reply` does **not** retain dialogue history beyond the last user turn + trajectory summary; it is not a full dialogue agent.

---

## Handoff notes for teammates
- Ship the **entire** `outputs/module1_goemotions/module1_model/` folder, not just `model.safetensors`. `config.json`, both tokenizer files, `merges.txt`, `vocab.json`, and `special_tokens_map.json` are all required by `AutoModel.from_pretrained` / `AutoTokenizer.from_pretrained`.
- The `checkpoint-2714/5428/8142/` folders under `outputs/module1_goemotions/` are intermediate training snapshots — safe to delete, do not load from them.
- `training_args.bin` inside the model folder is archival only; deletable if size matters.
- Do not reorder `EMOTION_LABELS` in [module1/labels.py](module1/labels.py) without retraining — Module 2 assumes the order is stable.
