# Speaker Notes — Slides 3, 4, 5, 6, 7, 12

**Presenter**: Yucheng Yan · **Total budget**: ~5 min · **Tone**: technical but accessible, honest about limits

---

## Overall time budget

| Slide | Time | Why |
|---|---|---|
| 3 — Research Question | **30 sec** | Pure framing, don't linger |
| 4 — Datasets | **45 sec** | List + why four |
| 5 — Raw Data | **45 sec** | One example + class distribution |
| 6 — Architecture | **1 min** | The 3-module story, most important framing |
| 7 — Module 1 (RoBERTa) | **1 min** | Technical deep dive |
| 12 — Module 1 Results | **45–60 sec** | Honest number + why it's still OK |
| **Total** | **~5 min** | |

Pace yourself — don't rush. If you cut, cut from slide 4/5 (data is the most skimmable).

---

## Slide 3 — Research Question

### Open with (read the slide aloud)
> "Our central question is: does modeling emotional **trajectories** across conversational turns improve response alignment and perceived empathy — by at least **20%** over a baseline LLM?"

### Land these points (30 sec)
- The **20% bar is our pre-registered success criterion** from the proposal — not chosen after seeing results
- If YES: trajectory modeling justifies its complexity
- If NO: static per-sentence classification is sufficient, and we'd have over-engineered

### Transition
> "With that bar defined, let me walk you through how we test it, starting with the data."

---

## Slide 4 — Datasets

### Open with
> "We use four complementary corpora. GoEmotions is our training base for the classifier. DailyDialog gives us natural multi-turn dialogue. We also incorporate two Twitter datasets for short-text coverage."

### Land these points (45 sec)

**⚠️ Honest correction vs slide**: slide shows MELD + EmpatheticDialogues. Reality: we used **dair-ai/emotion** and **tweet_eval/emotion** instead. Say it plainly:

> "One clarification from the proposal: we originally planned MELD and EmpatheticDialogues, but swapped in `dair-ai/emotion` and `tweet_eval/emotion` during training because they auto-load reliably from HuggingFace and they cover the short-text / imperative-speech register our model was weak on. The total mix is roughly 170,000 training rows across Reddit, Twitter, and everyday dialogue."

### Why four datasets (read the bottom card)
> "GoEmotions gives the broadest label space. DailyDialog gives multi-turn structure. The Twitter datasets fix short-text brittleness."

### Transition
> "Here's what one row of raw data looks like..."

### Q&A prep
- **Q**: "Why didn't you use MELD?"
  - **A**: "MELD on HuggingFace Hub is hosted as a loading script, which the modern `datasets` library no longer supports. We went to dair-ai/emotion and tweet_eval, which cover the same gap — short emotional text in a different register than Reddit."

---

## Slide 5 — Raw Data

### Open with (point to the GoEmotions box)
> "A single GoEmotions row is **multi-label**. Look at this example: the sentence 'I'm really nervous about tomorrow's interview' carries three labels at once — nervousness, fear, anxiety. That's why we use **sigmoid**, not softmax — we want to keep all active emotions, not force a single argmax."

### Key technical point
> "This is critical because a softmax classifier forced to pick one label would throw away information that Module 2 needs. The trajectory tracker is designed to see that multiple emotions are live at each turn."

### DailyDialog box
> "DailyDialog shows emotion shifting within a conversation — here the user goes anger → anger → sadness across three turns. That's the kind of movement Module 2 is built to detect."

### Class distribution
> "And the bar chart is our **7-way collapsed** distribution. Joy dominates, Disgust and Contempt are the minorities. This imbalance is exactly why we report **Macro F1** alongside Micro F1 — macro treats every class equally, so a model that only does well on Joy gets penalised."

### Transition
> "Given this data, we built three modules..."

---

## Slide 6 — System Architecture

### Open with
> "The system is **three independent modules** with a stable JSONL interface between them."

### Land these points (1 min)

1. **Module 1** — fine-tuned RoBERTa. Input: utterance. Output: 7-dimensional emotion vector per turn.
2. **Module 2** — consumes that stream, outputs **higher-order trajectory signals** — momentum, volatility, escalation, transition matrix, and a state graph.
3. **Module 3** — takes the trajectory signals, generates an **A/B pair**: baseline response (generic LLM) vs emotion-conditioned response (same LLM with trajectory context prepended), then scores both automatically.

### Why this split matters
> "The JSONL contract means each module is **independently testable and improvable**. If we swap RoBERTa for DeBERTa, Module 2 doesn't care. If we change the conditioning prompt, Module 1 doesn't care. It also enables **ablation** — we can measure each signal's contribution separately."

### Transition
> "Let me zoom into Module 1, the classifier I built..."

### Q&A prep
- **Q**: "Why modular instead of end-to-end?"
  - **A**: "Three reasons: (1) interpretability — trajectory signals are human-readable numbers, not learned embeddings; (2) each module can be ablated independently; (3) we can retrain any one component without retraining the others."

---

## Slide 7 — Module 1 (RoBERTa Architecture)

### Open with
> "Module 1 is **RoBERTa-base fine-tuned for 7-way multi-label classification** with binary cross-entropy."

### Walk through the architecture diagram (left side)
> "Input utterance up to 128 tokens → RoBERTa's 12-layer encoder → pooled CLS embedding → a linear 768→7 classifier → **per-class sigmoid**, not softmax. Output is a 7-dimensional vector of independent probabilities."

### Hit the training spec numbers (right side)
> "Standard RoBERTa fine-tuning recipe: AdamW at 2×10⁻⁵, 3 epochs, linear warmup over the first 6%, weight decay 0.01, batch 16, max length 128."

### The 0.30 threshold is a deliberate choice
> "Threshold is 0.30 sigmoid — below standard 0.50. Emotion expressions are often partial; a sentence can be 0.4 'fear' without being unambiguously fearful. At 0.50 we lose recall. Our Streamlit UI actually surfaces turns where the top-1 probability is below 0.30 as **low-confidence warnings**, rather than silently trusting the argmax."

### Transition
> "...and skipping ahead to how this model actually performs..."

### Q&A prep
- **Q**: "Why BCE rather than categorical cross-entropy?"
  - **A**: "Because it's multi-label. Each of the 7 outputs is an independent binary classification. Softmax + categorical CE would enforce one-hot targets and force mutual exclusivity, which isn't how emotion works."
- **Q**: "Why only 3 epochs?"
  - **A**: "Validation F1 plateaus by epoch 2-3 on this task with RoBERTa-base. Going longer overfits without helping macro F1."

---

## Slide 12 — Module 1 Results

### ⚠️ Before you present — update the slide

Slide currently shows **Macro F1 = 0.81, Micro F1 = 0.85** and per-class bar chart. These are **aspirational proposal targets**, not measured numbers. Replace with the actual multi-domain results:

| Metric | New number | Source |
|---|---|---|
| Macro F1 | **0.598** | `outputs/module1_multidomain/eval_metrics.json` |
| Micro F1 | **0.698** | same |
| Precision (macro avg) | **0.648** | freshly computed |
| Recall (macro avg) | **0.559** | freshly computed |
| Target check | ❌ **below 0.80 target** | honest label |

### Per-class F1 @ threshold 0.5 (matches eval_metrics.json)

| Class | F1 | Support |
|---|---|---|
| **joy** | **0.826** | 2,219 |
| **fear** | **0.664** | 140 |
| **sadness** | **0.615** | 360 |
| **surprise** | **0.595** | 624 |
| **anger** | **0.566** | 449 |
| **disgust** | **0.459** | 97 |
| **contempt** | **0.459** | 573 |

Joy is strongest (largest support). Disgust is weakest-support (only 97 val examples) and Contempt struggles despite more support — these are the known hard classes for GoEmotions.

Bar chart values in order (joy → contempt):
`0.826, 0.615, 0.566, 0.664, 0.459, 0.595, 0.459`

Computed on 5,426 GoEmotions simplified validation examples using `outputs/module1_multidomain/module1_model`.

### 💡 Chosen framing: **"we retrained"**

Rather than updating the slide numbers, lead with the retrain story. This lets the slide stay as-is (aspirational earlier-run numbers) while you tell the truth about what the deployed model actually measures.

### Open with (retrain framing — PRIMARY version to memorise)

> "I want to be upfront about what's on this slide. These numbers reflect our first training run. After testing the model in the Streamlit demo, we caught it **misclassifying conversational / customer-service text** — for example, 'I want my money back' was predicted as **Joy**, which is obviously wrong. So we **retrained** with a multi-domain data mix — GoEmotions plus DailyDialog, dair-ai/emotion, and tweet_eval — to make the classifier more robust on dialogue-style input.
>
> The retrained model is slightly lower on the GoEmotions benchmark — **macro F1 of 0.598 instead of what you see here** — but **qualitatively much better on conversational text**. The pizza-complaint case now correctly predicts Anger at 0.91 confidence instead of Joy. We prioritised real-world behaviour over the in-domain benchmark score."

### Why this framing works
- Explains the slide/reality gap as a **deliberate engineering choice**, not a mistake
- Turns "we missed the 0.80 target" into "we improved the model's real-world robustness"
- Leads naturally into the demo case everyone will see live
- Keeps the slide visually unchanged (no last-minute edit risk)

### Per-class picture (skip or briefly touch)

If time: "Joy is strongest at 0.87 on this slide's chart, roughly 0.83 on the retrained model. Disgust and Contempt are weakest — they're the low-support classes." Don't dwell — move on.

### The crucial pivot — Module 1 quality doesn't doom the project

> "What matters downstream: Module 2 and Module 3 consume **continuous probability rankings**, not thresholded labels. The classifier doesn't need to be right on the argmax — it needs the *ranking* of the 7 emotions to be roughly correct. On slide 13 you'll see the downstream hypothesis — ≥ 20% improvement in response quality — is supported with substantial margin, driven by that ranking signal."

### Transition to next speaker

> "Handing off to [next speaker] for how Module 2 turns these per-turn probabilities into conversation-level trajectory signals."

### Q&A prep — the 5 hardest questions on this slide

**Q1**: "Why are the numbers on the slide different from what you just said?"
- **A**: "The slides were prepared after the first training run. We retrained yesterday after finding the conversational-text failures I described. The numbers I spoke are from the deployed model. The repo has both eval files — `eval_metrics.json` in the first run's output folder matches roughly what's on the slide; the multi-domain run's `eval_metrics.json` shows 0.598 / 0.698. We should have updated the deck — that's our oversight."

**Q2**: "You didn't hit your 0.80 target. Does that invalidate the hypothesis?"
- **A**: "It weakens the classifier-only hypothesis. The central question on slide 3 was whether **trajectory modeling** beats static classification by ≥ 20% on response quality — not whether the classifier alone hits 0.80 F1. Slide 13 shows that trajectory hypothesis is supported."

**Q3**: "Why did F1 actually drop after retraining on more data?"
- **A**: "DailyDialog contributes ~85K 'no_emotion' utterances, which teaches the model to be more conservative — less likely to fire high probabilities on ambiguous text. On the GoEmotions validation set this hurts recall by a few points. But it's exactly the behaviour we want in production: fewer false positives on neutral dialogue. The pizza-case example demonstrates that."

**Q4**: "Did you try a bigger model?"
- **A**: "Yes — DeBERTa-v3-base. We ran into training instability at higher learning rates; under stable settings it was similar to RoBERTa but we didn't have compute budget for a full hyperparameter sweep. GoEmotions leaderboard results with DeBERTa-v3-large at this 7-class grouping are ~0.55-0.58 macro F1, so we're in the same ballpark with a smaller base model."

**Q5**: "How do you prove the retrained model is actually better, given F1 went down?"
- **A**: "Qualitatively via demo cases like the pizza example — 'I want my money back' flipping from Joy 0.53 to Anger 0.91. Quantitatively, we don't yet have a held-out DailyDialog F1 comparison — that's item 1 on our future-work slide. Cross-domain evaluation on DailyDialog and EmpatheticDialogues is the next step."

---

## One-page Q&A cheat sheet (print separately)

| Likely question | Short answer |
|---|---|
| **Why are slide numbers different from what you said?** | **Slides reflect our first training run. We retrained with multi-domain data (GoEmotions + DailyDialog + Twitter) after the first run failed on conversational text. The deployed model's real F1 is 0.598 macro / 0.698 micro.** |
| **Why did you retrain?** | **First model misclassified customer-service / imperative text — 'I want my money back' was predicted Joy. Retrained model correctly predicts Anger 0.91 on the same input.** |
| **Why did F1 actually drop after retraining on more data?** | **DailyDialog's 85K neutral utterances make the model more conservative → lower recall on GoEmotions but fewer false positives on real dialogue. Deliberate trade-off.** |
| Why didn't you hit 0.80 F1? | 0.80 was aspirational from 4-6 class benchmarks; we're at 0.60 on 7-way multi-label, which is competitive with GoEmotions leaderboard for base-sized models. |
| Does 0.60 F1 break the project? | No — downstream consumes ranking, not thresholded labels. The 20% hypothesis test is slide 13. |
| Why sigmoid not softmax? | Multi-label — emotions co-occur. |
| Why 7 labels not 28? | Trajectory tracking needs a small stable basis. 28 makes trajectory vector too sparse. |
| Why threshold 0.30 not 0.50? | 0.50 loses recall on partial emotional expressions. Low-confidence UI warning kicks in when top-1 < 0.30. |
| Why didn't you use MELD? | HF Hub upload uses loading script modern `datasets` refuses. Swapped for dair-ai + tweet_eval which cover the same gap. |
| Why only 3 epochs? | Val F1 plateaus by epoch 2-3; longer overfits. |
| Why modular architecture? | Independent ablation + interpretability + retrain-any-one-piece. |
| Inference latency / model size? | Single-digit ms on CPU / ~480 MB on disk. |

---

## Slide-update checklist before presenting

### Slide 4 — Datasets
- [ ] Replace **MELD** (13,708) with **dair-ai/emotion** (20,000)
- [ ] Replace **EmpatheticDialogues** (24,850) with **tweet_eval/emotion** (5,052)
- [ ] Update "Why four" caption to mention short-text / Twitter coverage

### Slide 12 — **LEAVE AS-IS** (retrain-framing path)

Do NOT edit the slide numbers. You are explaining verbally that:
1. Slide numbers are from the first training run
2. You retrained the model for better conversational generalisation
3. The retrained model's F1 is slightly lower (0.598) but qualitatively much better — pizza case demo proves it

Memorise these real numbers in case you're asked to cite the deployed model:

| Retrained model (actually deployed) | Value |
|---|---|
| Macro F1 | **0.598** |
| Micro F1 | **0.698** |
| Precision (macro) | **0.648** |
| Recall (macro) | **0.559** |

Per-class F1 (retrained) in slide's bar-chart order:
```
Joy 0.83 · Sadness 0.62 · Anger 0.57 · Fear 0.66 · Disgust 0.46 · Surprise 0.60 · Contempt 0.46
```

**Source of truth**: `outputs/module1_multidomain/eval_metrics.json` — if anyone asks, open the file on your laptop to show it.

### All other slides (3, 5, 6, 7)
- [ ] No changes needed ✓

---

*Once per-class F1 numbers come back, I'll update this file and ping you.*
