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
Two complementary APIs live side-by-side:

**Legacy / simple** — [module2/trajectory.py](module2/trajectory.py) `compute_trajectory(turns)`:
  - `volatility` — mean L2 step between consecutive 7-d score vectors.
  - `escalation_score` — negative-affect mass, first third vs last third of conversation.
  - `dominant_labels` per turn + `dominant_arc_text` (e.g. `sadness → fear → anger`).
  - `summary` — 2–4 sentence English blurb used to condition the LLM.

**Rich / proposal-aligned** — [module2/tracker.py](module2/tracker.py) `EmotionalTrajectoryTracker` → `TrajectorySignals`:
  - `emotional_momentum` — exponentially-weighted drift of per-turn valence (signed, bounded).
  - `volatility_index` — std-dev of per-turn intensity, normalised to `[0, 1]`.
  - `escalation_score` — linear-regression slope of negative-affect activation `arousal × (1 − (valence+1)/2)`.
  - `dominant_state` — recency-weighted dominant cluster across the conversation.
  - `transition_matrix` — row-normalised bigram probabilities over the dominant-emotion sequence.
  - `valence_series`, `arousal_series`, `intensity_series` — full dimensional-affect timelines.
  - `snapshots` — per-turn `TurnSnapshot` objects (valence / arousal / intensity / dominant).

Supporting pieces:
  - [module2/taxonomy.py](module2/taxonomy.py) — 7-cluster map + per-cluster valence / arousal / colour.
  - [module2/visualise.py](module2/visualise.py) — `build_emotion_state_graph()` (networkx `DiGraph`) + `plot_trajectory()` (4-panel matplotlib figure).
  - [module2/adapters.py](module2/adapters.py) — `DatasetAdapters` for GoEmotions / DailyDialog / MELD / EmpatheticDialogues rows, plus `build_conditioning_prompt(signals)` — emits the emotion-aware instruction block that Module 3 prepends to its system prompt.
  - [module2/infer.py](module2/infer.py) — convenience `load_model()` / `predict_emotions()` / `run_full_pipeline()` bridge when you want to skip Module 1's helper.

### Module 3 — Adaptive dialogue generator **+ evaluation pipeline**
Generation ([module3/generate.py](module3/generate.py) + [module3/prompts.py](module3/prompts.py)):
- `generate_baseline_reply(text)` — generic helpful-assistant prompt, **no** emotion context. The A/B baseline.
- `generate_reply(text, traj, conditioning_prompt=...)` — trajectory-conditioned. When `conditioning_prompt` (from `module2.build_conditioning_prompt(signals)`) is supplied, it is **prepended** to the system prompt so the LLM sees dominant state, momentum, volatility, escalation, and a recommended tone.
- `generate_ab_pair(text, traj, conditioning_prompt=...)` — returns `(baseline, conditioned)` in one call.
- All three use OpenAI Chat Completions via stdlib `urllib` when `OPENAI_API_KEY` is set; otherwise fall back to deterministic templates (so A/B still works offline).

Records ([module3/records.py](module3/records.py), [module3/simulated.py](module3/simulated.py)):
- `build_records_from_turns(conversations, user_last_turns)` — pre-scored input.
- `build_records_from_text(conversations)` — raw text; runs Module 1 on the fly.
- `build_records_from_simulated()` — 3 hand-written A/B pairs for offline testing.

Automated evaluation ([module3/evaluate.py](module3/evaluate.py)):
- `AutomatedEvaluator(use_heavy_models=False|True)` — 3 metrics (BERTScore + Empathy + Specificity), weighted composite `0.5·empathy + 0.3·bertscore + 0.2·specificity`.
- Heavy mode loads `bert-score` + `facebook/bart-large-mnli` (~1.5 GB); lightweight mode uses Jaccard token overlap + empathy-keyword matching + rule-based specificity — fast, no download, used in the Streamlit app.
- `print_evaluation_report(scores)` — per-conversation table, aggregate means, proposal's **≥ 20% improvement target** check.
- `export_results_csv(scores, path)` — raw score dump.

Human rater study ([module3/human_rater.py](module3/human_rater.py)):
- `export_human_rater_csv(records, path, append=True)` — anonymised A/B CSV, label order randomised per conversation, blank empathy/appropriateness/helpfulness columns for 1–5 ratings.
- `load_human_ratings(path)` — reads the completed sheet back and adds a composite column.

### Streamlit demo
- [app.py](app.py) exposes:
  - Chat-style multi-turn input (`st.session_state`), model cached via `st.cache_resource`.
  - Trajectory panel: volatility, momentum, escalation, dominant state, dominant-emotion arc, live summary.
  - Latest-turn top-3 bar chart + 7-emotion line chart across turns.
  - Valence / arousal timeline (once ≥ 2 turns).
  - Transition matrix table.
  - Emotion state graph (networkx + matplotlib), toggleable in the sidebar.
  - **Baseline vs conditioned A/B** (sidebar toggle, on by default): side-by-side responses + auto-scored BERTScore / Empathy / Specificity / Overall Δ% deltas (lightweight fallback metrics — heavy BERTScore/NLI models available via the `AutomatedEvaluator(use_heavy_models=True)` Python API).
  - "Append this conversation to the human-rater CSV" button — writes to `artifacts/human_rater_sheet.csv` (append mode), so a session accumulates A/B pairs for later distribution to 20–30 evaluators.
  - Expandable panel showing the exact emotional-conditioning context sent to the LLM.
  - Sidebar: model-path override, max-tokens slider, graph toggle, A/B toggle, Clear-conversation.

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
from module2 import EmotionalTrajectoryTracker, build_conditioning_prompt, compute_trajectory
from module3.generate import generate_reply

model, tok, dev = load_classifier("outputs/module1_goemotions/module1_model")

conversation = [
    "I'm really stressed about my exam tomorrow.",
    "I studied but I feel like I'm going to mess everything up.",
    "My parents are expecting too much.",
]

tracker = EmotionalTrajectoryTracker()
turns = []
for text in conversation:
    scores = predict_scores(model, tok, dev, text)
    turns.append({"text": text, "scores": scores})
    tracker.add_turn(scores)

signals = tracker.compute()
print(signals.summary())                        # full multi-line analysis

traj = compute_trajectory(turns)                # legacy summary dict for Module 3
cond = build_conditioning_prompt(signals)       # prepended to LLM system prompt
print(generate_reply(turns[-1]["text"], traj, conditioning_prompt=cond))
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
module2/
  trajectory.py   legacy simple API (compute_trajectory)
  tracker.py      rich API (momentum, transition matrix, snapshots)
  taxonomy.py     emotion clusters + valence / arousal / colour table
  adapters.py     dataset adapters + build_conditioning_prompt for Module 3
  visualise.py    networkx emotion state graph + 4-panel matplotlib plot
  infer.py        standalone load_model / predict_emotions bridge
module3/
  generate.py     baseline + conditioned + ab-pair reply generators
  prompts.py      BASELINE_SYSTEM_PROMPT + CONDITIONED_SYSTEM_PREFIX
  simulated.py    SIMULATED_PAIRS (offline A/B fixtures)
  records.py      build_records_from_turns / build_records_from_text
  evaluate.py     AutomatedEvaluator (3 metrics) + report + CSV export
  human_rater.py  export / load human-rater A/B sheets
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
- [x] **Emotion transition matrix** — ✅ built in `tracker.py::_build_transition_matrix`, rendered in the UI as a table.
- [x] **Emotion State Graph** — ✅ `build_emotion_state_graph()` (networkx DiGraph) + rendered inline in Streamlit.
- [x] **Emotional momentum** as a first-class metric — ✅ exponentially-weighted valence drift in `tracker.py::_compute_momentum`.
- [x] **Dimensional affect (valence / arousal)** — ✅ per-turn + plotted across the conversation.
- [ ] Trajectory-level evaluation: no labeled test set for state-transition accuracy, drift detection, or early-escalation detection yet.

### Dialogue generation
- [x] **Trajectory-aware conditioning prompt** — ✅ `build_conditioning_prompt(signals)` prepended to the LLM system prompt.
- [x] **Baseline vs. emotion-conditioned A/B** — ✅ `generate_ab_pair()` returns both; Streamlit shows them side-by-side with automated scoring.
- [x] **Automated evaluation** — ✅ `AutomatedEvaluator` with BERTScore + empathy + specificity, weighted composite, ≥ 20% target check.
- [x] **Human rater export** — ✅ `export_human_rater_csv()` writes randomised A/B sheets for 20–30 evaluators. Rater CSV grows as the Streamlit user clicks "Append to CSV".
- [ ] **Human rater study actually run** — code is ready; the 20–30 rater data collection itself is out of scope for MVP.
- [ ] Non-OpenAI LLM fallback (e.g. local model) — currently OpenAI or template only.

### App / UX
- [x] **Side-by-side baseline vs. conditioned panel** — ✅ in Streamlit.
- [ ] Per-turn emotion chips in the chat bubbles (right now scores only appear in the line chart).
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
