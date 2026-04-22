# Metrics & Mathematical Specification

Complete reference for every piece of data and every metric displayed in the web app, plus the exact formulas that produce them.

**Scope**: this is the paper-ready companion to [README.md](README.md). If you need to justify a number on a slide, cite an equation, or write the report's "Methods" section, this file is the source of truth.

---

## Table of contents
1. [Data & training sets](#1-data--training-sets)
2. [Module 1 — utterance-level classification](#2-module-1--utterance-level-classification)
3. [Module 2 — trajectory features](#3-module-2--trajectory-features)
4. [Module 3 — adaptive reply & evaluation](#4-module-3--adaptive-reply--evaluation)
5. [UI metric → formula map](#5-ui-metric--formula-map)
6. [Dimensional affect table (reference)](#6-dimensional-affect-table-reference)

---

## 1. Data & training sets

### 1.1 Multi-source training mix (current best model)

| Source | Domain | Rows used | How labels are mapped |
|---|---|---|---|
| [`go_emotions/simplified`](https://huggingface.co/datasets/go_emotions) | Reddit comments | 43,410 train / 5,426 val | 28 fine-grained → **7 canonical** via `module1/go_emotions_mapping.py` |
| [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) | Twitter microblog | 20,000 | 6 classes (sadness, joy, love, anger, fear, surprise) → 7 canonical; `love` folded into `joy` |
| `daily_dialog` (`refs/convert/parquet`) | Everyday dialogue | 102,979 utterances | 7 classes (no_emotion, anger, disgust, fear, happiness, sadness, surprise); `no_emotion` → all-zero vector (kept as hard negatives) |
| [`tweet_eval/emotion`](https://huggingface.co/datasets/tweet_eval) | Twitter | 5,052 | 4 classes (anger, joy, optimism, sadness); `optimism` folded into `joy` |
| **Total** | mixed | **~171,441** | → 7-way multi-hot |

The **7 canonical labels** are fixed as the ordered tuple:
```
(joy, sadness, anger, fear, disgust, surprise, contempt)
```
This order is hard-coded in [`module1/labels.py`](module1/labels.py) and MUST NOT be reordered without retraining.

### 1.2 Evaluation set

All reported F1 / loss numbers are computed on **GoEmotions `simplified` validation split** (5,426 examples). Cross-domain evaluation on the DailyDialog / TweetEval validation splits is not wired up yet.

### 1.3 Label mapping from GoEmotions 28-label set → 7 canonical

| GoEmotions label(s) | Canonical cluster |
|---|---|
| admiration, amusement, approval, caring, desire, excitement, gratitude, joy, love, optimism, pride, relief | **joy** |
| disappointment, embarrassment, grief, nervousness, remorse, sadness | **sadness** |
| anger, annoyance, disapproval | **anger** |
| fear | **fear** |
| disgust | **disgust** |
| confusion, curiosity, realization, surprise | **surprise** |
| contempt (virtual — not in GoEmotions, mapped from `disgust`-adjacent annotations where possible) | **contempt** |
| neutral | all-zero vector |

See [`module1/go_emotions_mapping.py`](module1/go_emotions_mapping.py) for the authoritative mapping.

---

## 2. Module 1 — utterance-level classification

### 2.1 Architecture

Fine-tuned encoder: `roberta-base` (125M parameters) or `microsoft/deberta-v3-base` (184M).

Per utterance $x$:

$$\mathbf{z} = f_{\theta}(\text{tokenize}(x)) \in \mathbb{R}^{7}$$

where $f_{\theta}$ is the transformer encoder followed by a linear classification head over the `[CLS]` (or equivalent) pooled embedding.

### 2.2 Multi-label probabilities

Independent sigmoid over each of the 7 logits:

$$p_c(x) = \sigma(z_c) = \frac{1}{1 + e^{-z_c}}, \quad c \in \{1, \ldots, 7\}$$

Critically this is **not** softmax — each label's probability is independent, so a single utterance can legitimately have `joy = 0.7` and `surprise = 0.6` simultaneously.

### 2.3 Loss (during training)

Binary cross-entropy with logits, averaged over batch and label axes:

$$\mathcal{L}(\mathbf{z}, \mathbf{y}) = -\frac{1}{7} \sum_{c=1}^{7}\bigl[y_c \log \sigma(z_c) + (1 - y_c)\log(1 - \sigma(z_c))\bigr]$$

Implementation: [`module1/train.py::MultiLabelTrainer.compute_loss`](module1/train.py) uses `F.binary_cross_entropy_with_logits`.

### 2.4 Training hyperparameters (current best checkpoint)

| Parameter | Value |
|---|---|
| Base model | `roberta-base` |
| Max sequence length | 128 |
| Epochs | 3 |
| Effective batch size | 64 (Blackwell 6000) |
| Optimizer | AdamW |
| Learning rate | $2 \times 10^{-5}$ |
| Warmup ratio | 0.06 |
| Weight decay | 0.01 |
| Precision | `bf16` |
| Scheduler | linear-decay after warmup |

### 2.5 Inference thresholding

- **Default threshold** (batch CLI in `module1/infer.py`): $\tau = 0.5$ — a label is reported as "active" if $p_c \geq \tau$.
- **Low-confidence flag** (Streamlit UI): a turn is flagged if $\max_c p_c < \tau_{\text{conf}}$ where $\tau_{\text{conf}} = 0.30$ ([`app.py`](app.py)). This surfaces cases where no single label is strongly predicted.

---

## 3. Module 2 — trajectory features

All Module 2 formulas live in [`module2/tracker.py`](module2/tracker.py).

### 3.1 Cluster mapping & dimensional affect

Module 2 operates on 7 **clusters** (same 7 as Module 1's labels, but capitalised: Joy / Sadness / Anger / Fear / Disgust / Surprise / Contempt). Each cluster carries two dimensional-affect scalars:

| Cluster | Valence $v_c$ | Arousal $a_c$ |
|---|---|---|
| Joy | $+1.00$ | $0.70$ |
| Surprise | $+0.20$ | $0.90$ |
| Sadness | $-0.70$ | $0.30$ |
| Fear | $-0.80$ | $0.85$ |
| Anger | $-0.90$ | $0.95$ |
| Disgust | $-0.85$ | $0.60$ |
| Contempt | $-0.60$ | $0.50$ |

Values fixed in [`module2/taxonomy.py`](module2/taxonomy.py). These are **hand-set** based on the circumplex model of affect (Russell 1980, Ekman) — not learned.

### 3.2 Per-turn snapshot

Given raw classifier output $\mathbf{p}^{(t)} = (p_1^{(t)}, \ldots, p_K^{(t)})$ for turn $t$ (where the labels could be the 7 canonical ones, or a richer subset from a different dataset):

**Cluster mass** (re-normalisation after string-based cluster lookup):
$$m_c^{(t)} = \frac{\sum_{l \in \text{cluster}(c)} p_l^{(t)}}{\sum_{c'}\sum_{l \in \text{cluster}(c')} p_l^{(t)}}$$

so $\sum_c m_c^{(t)} = 1$.

**Dominant cluster**:
$$d^{(t)} = \arg\max_c\, m_c^{(t)}$$

**Intensity** (mean of raw probabilities, not normalised):
$$I^{(t)} = \frac{1}{K}\sum_{l=1}^{K} p_l^{(t)}$$

**Weighted valence / arousal** (using the dimensional scalars above):
$$V^{(t)} = \frac{\sum_c v_c \cdot m_c^{(t)}}{\sum_c m_c^{(t)}} = \sum_c v_c \cdot m_c^{(t)}$$

$$A^{(t)} = \sum_c a_c \cdot m_c^{(t)}$$

(denominator is 1 by construction)

### 3.3 Higher-order trajectory signals

Given an ordered sequence of $n$ snapshots $\{(V^{(t)}, A^{(t)}, I^{(t)}, d^{(t)})\}_{t=1}^{n}$:

#### 3.3.1 Emotional Momentum

Exponentially-weighted mean of first-differences in valence:

$$M = \text{clip}\!\left( \frac{\sum_{t=1}^{n-1} e^{t-1}\,(V^{(t+1)} - V^{(t)})}{\sum_{t=1}^{n-1} e^{t-1}},\ -1,\ +1 \right)$$

- **Sign** = direction of valence drift (+ = moving positive, − = moving negative)
- **Magnitude** = speed of drift
- **Exponential weighting** means the most recent transition dominates — a late-conversation sentiment flip swings $M$ heavily.

Implementation: [`module2/tracker.py::_compute_momentum`](module2/tracker.py).

#### 3.3.2 Volatility Index

Standard deviation of per-turn intensity, normalised by the theoretical max std of a Uniform(0,1) random variable ($\approx 0.5$):

$$\mathrm{Vol} = \text{clip}\!\left( \frac{\sigma\!\bigl(\{I^{(1)}, \ldots, I^{(n)}\}\bigr)}{0.5},\ 0,\ 1 \right)$$

- $0$ = intensity is constant across turns (steady emotional load)
- $1$ = intensity fluctuates between the extremes

Implementation: [`module2/tracker.py::_compute_volatility`](module2/tracker.py).

#### 3.3.3 Escalation Score

Let the **negative-affect activation** series be:

$$N^{(t)} = A^{(t)} \cdot \left(1 - \frac{V^{(t)} + 1}{2}\right)$$

This is high when the user is in a **negative** ($V < 0$) **high-arousal** ($A$ near 1) state (e.g., anger), low for calm positives (e.g., contentment).

Fit a linear regression $N^{(t)} \approx \beta_0 + \beta_1 t$ and define:

$$\mathrm{Esc} = \text{clip}(\beta_1 \cdot n,\ -1,\ +1)$$

- $> 0$: negative-affect activation is trending **up** (user is escalating)
- $< 0$: trending **down** (de-escalating)
- $\approx 0$: stable

Multiplying by $n$ normalises the slope into the $[-1, 1]$ range roughly independent of conversation length.

Implementation: [`module2/tracker.py::_compute_escalation`](module2/tracker.py).

#### 3.3.4 Dominant state (recency-weighted)

For each cluster $c$, sum its per-turn mass with a linear recency weight $w^{(t)} = t/n$:

$$S_c = \sum_{t=1}^{n} \frac{t}{n} \cdot m_c^{(t)}$$

Then:

$$d^* = \arg\max_c\, S_c$$

Recency weighting means the most recent turn contributes $n/n = 1$, the first contributes $1/n$. So a brief change at the end shifts the dominant only if the shift is large; a persistent state wins.

Implementation: [`module2/tracker.py::compute`](module2/tracker.py) (first half).

#### 3.3.5 Transition matrix

Let $\mathcal{D} = (d^{(1)}, \ldots, d^{(n)})$ be the sequence of per-turn dominant clusters. Build bigram counts:

$$C_{ij} = \bigl|\{ t : d^{(t)} = i,\ d^{(t+1)} = j\}\bigr|$$

Row-normalise to probabilities:

$$T_{ij} = P(d^{(t+1)} = j \mid d^{(t)} = i) = \frac{C_{ij}}{\sum_{j'} C_{ij'}}\quad\text{if } \textstyle\sum_{j'} C_{ij'} > 0$$

Each row sums to 1 (or to 0 if state $i$ never appears as a non-terminal state).

Implementation: [`module2/tracker.py::_build_transition_matrix`](module2/tracker.py).

### 3.4 Emotion state graph

A weighted directed graph $G = (V, E)$ derived from $T$:
- $V$ = $\{c : c \in \mathcal{D}\}$ — clusters that actually appeared
- Node attributes: `mean_intensity = mean(I^{(t)} : d^{(t)} = c)`, `count`, `valence = v_c`
- $E = \{(i, j, w_{ij}) : T_{ij} > 0\}$, edge weight = transition probability
- Rendered with `networkx.spring_layout` + matplotlib (serialised to PNG and shown via `st.image`)

### 3.5 Conditioning prompt generation

A rule-based translation of $(d^*, M, \mathrm{Vol}, \mathrm{Esc})$ into an English system-prompt prefix for Module 3. See [`module2/adapters.py::build_conditioning_prompt`](module2/adapters.py). The decision table is:

| Condition | Implied tone |
|---|---|
| $\mathrm{Esc} > 0.2$ and $\mathrm{Vol} > 0.3$ | validation + grounding + emotional de-escalation |
| $\mathrm{Esc} > 0.1$ | empathetic acknowledgement + gentle reframing |
| $\mathrm{Esc} < -0.1$ | reinforcing + warm encouragement |
| $\mathrm{Vol} > 0.4$ | stabilising + calm, consistent reassurance |
| (otherwise) | neutral + supportive |

Volatility label buckets:
- $\mathrm{Vol} > 0.6$ → "very high"
- $0.35 < \mathrm{Vol} \leq 0.6$ → "high"
- $0.15 < \mathrm{Vol} \leq 0.35$ → "moderate"
- $\mathrm{Vol} \leq 0.15$ → "low"

Escalation label:
- $\mathrm{Esc} > 0.15$ → "escalating"
- $\mathrm{Esc} < -0.15$ → "de-escalating"
- else → "stable"

---

## 4. Module 3 — adaptive reply & evaluation

### 4.1 Reply generation

Two parallel system prompts (the A/B experimental design):

$$P_{\text{base}} = \text{"You are a helpful conversational assistant. Respond naturally..."}$$

$$P_{\text{cond}} = \text{"You are an emotionally intelligent..."} \oplus \text{build\_conditioning\_prompt}(\text{signals})$$

For the same user message $u$, both prompts are fed to the same LLM (OpenAI `gpt-4o-mini` when `OPENAI_API_KEY` is set, deterministic templates otherwise). The resulting $(r_{\text{base}}, r_{\text{cond}})$ pair is the unit of evaluation.

### 4.2 Automated evaluation metrics

Implementations in [`module3/evaluate.py::AutomatedEvaluator`](module3/evaluate.py).

#### 4.2.1 BERTScore F1

Full mode (`use_heavy_models=True`): contextual BERT embedding similarity. Let $\mathbf{B}(s)$ be the BERT token embedding matrix for string $s$.

$$\mathrm{BScore}(r, r_{\text{gold}}) = 2 \cdot \frac{\mathrm{Precision} \cdot \mathrm{Recall}}{\mathrm{Precision} + \mathrm{Recall}}$$

where Precision/Recall are computed via greedy alignment between $\mathbf{B}(r)$ and $\mathbf{B}(r_{\text{gold}})$. Uses rescaled variant (`rescale_with_baseline=True`).

Lightweight fallback (the Streamlit default): Jaccard token overlap:

$$\mathrm{BScore}_{\text{lite}}(r, r_{\text{gold}}) = \frac{|T(r) \cap T(r_{\text{gold}})|}{|T(r) \cup T(r_{\text{gold}})|}$$

where $T(\cdot)$ is the whitespace-tokenised lowercased token set.

**Gold reference** defaults to $r_{\text{cond}}$ itself (the conditioned response acts as the "target empathetic reply"). This is what the notebook uses and what the `.py` port preserves. This means $\mathrm{BScore}(r_{\text{cond}}, r_{\text{cond}}) = 1$ by construction — so interpret the baseline's score against it, not the conditioned's.

#### 4.2.2 Empathy score

Full mode: zero-shot natural language inference with `facebook/bart-large-mnli`. For a response $r$ and user dominant emotion $d$:

$$\mathrm{Emp}(r, d) = P_{\text{NLI}}\bigl(r \models \text{"this response to a user expressing } d \text{ is empathetic and validating"}\bigr)$$

Lightweight fallback: keyword count over the set
$$\mathcal{K} = \{\text{understand, hear, feel, difficult, hard, support, acknowledge, sense, exhausting, sounds like, that's, valid, matter, grieve, sorry}\}$$

$$\mathrm{Emp}_{\text{lite}}(r) = \min\!\left(1,\ \frac{|W(r) \cap \mathcal{K}|}{5}\right)$$

#### 4.2.3 Specificity

Rule-based scoring (always available, no heavy models):

$$\mathrm{Spec}(r, \text{signals}) = \text{clip}\bigl(s_1 + s_2 + s_3,\ 0,\ 1\bigr)$$

where:

$$s_1 = \begin{cases}0.4 & \text{if } d^* \in r \\ 0.2 & \text{if any of \{anxious, stress, fear, anger, sad, joy, disgust, contempt\}} \in r \\ 0 & \text{otherwise}\end{cases}$$

$$s_2 = \begin{cases}0.3 & \text{if } \mathrm{Esc} > 0.1 \text{ and any of \{more, building, growing, heavy, exhausting\}} \in r \\ 0.3 & \text{if } \mathrm{Esc} < -0.1 \text{ and any of \{better, improving, progress, calmer\}} \in r \\ 0.15 & \text{if } |\mathrm{Esc}| \leq 0.1 \\ 0 & \text{otherwise}\end{cases}$$

$$s_3 = \begin{cases}0.3 & \text{if } \mathrm{Vol} > 0.4 \text{ and any of \{lot, much, overwhelming, intense, hard\}} \in r \\ 0 & \text{otherwise}\end{cases}$$

#### 4.2.4 Composite score

Weighted average of the three metrics:

$$\mathrm{Overall}(r) = 0.5 \cdot \mathrm{Emp}(r) + 0.3 \cdot \mathrm{BScore}(r) + 0.2 \cdot \mathrm{Spec}(r)$$

Weights reflect the hypothesis priority: **empathy is the primary outcome** (proposal's central claim), BERTScore measures semantic alignment, specificity is a bonus.

#### 4.2.5 Improvement percentage

Per-conversation:

$$\Delta\% = \frac{\mathrm{Overall}(r_{\text{cond}}) - \mathrm{Overall}(r_{\text{base}})}{\mathrm{Overall}(r_{\text{base}})} \times 100$$

**Caveat**: when $\mathrm{Overall}(r_{\text{base}})$ is near zero, $\Delta\%$ is unstable — single-digit absolute changes can produce triple-digit percentages. Always report the absolute composites too.

Aggregate:

$$\overline{\Delta\%} = \frac{1}{|\text{conversations}|} \sum_i \Delta\%_i$$

The proposal's target is:
$$\overline{\Delta\%} \geq 20\%$$

### 4.3 Human-rater evaluation

After automated evaluation, per-conversation A/B pairs are exported to CSV with randomised labels. Each of 20–30 raters scores each response on 3 Likert-scale (1–5) dimensions:
- Empathy
- Appropriateness
- Helpfulness

Per-rater composite: $\frac{1}{3}(e + a + h)$. Per-response aggregate = mean across raters. Target is the same 20% improvement of the conditioned mean over the baseline mean.

**Status**: export pipeline implemented; actual rater data collection is out of MVP scope.

---

## 5. UI metric → formula map

Cheat sheet: each number you see in the Streamlit UI pointed back to its exact equation.

| UI element | Formula / source | File · symbol |
|---|---|---|
| **Top-3 scores** (per turn) | $\text{top-3 by } p_c$ | `module1/runtime.py::predict_scores` |
| **Score curves** (line chart) | $p_c$ vs turn index $t$ | same |
| **Low-confidence warning** | $\max_c p_c < 0.30$ | `app.py::_confidence_label` |
| **Volatility** metric card | $\mathrm{Vol}$ — §3.3.2 | `tracker.py::_compute_volatility` |
| **Momentum** metric card | $M$ — §3.3.1 | `tracker.py::_compute_momentum` |
| **Escalation** metric card | $\mathrm{Esc}$ — §3.3.3 | `tracker.py::_compute_escalation` |
| **Dominant state** | $d^*$ — §3.3.4 | `tracker.py::compute` |
| **Emotion arc** | $(d^{(1)}, d^{(2)}, \ldots, d^{(n)})$ | `tracker.py::compute` |
| **Valence / Arousal** line chart | $V^{(t)}$, $A^{(t)}$ — §3.2 | `tracker.py::TurnSnapshot.__post_init__` |
| **Transition matrix** | $T_{ij}$ — §3.3.5 | `tracker.py::_build_transition_matrix` |
| **Emotion state graph** | DiGraph from $T$ — §3.4 | `visualise.py::build_emotion_state_graph` |
| **Conditioning context** (expander) | Rule-based prompt — §3.5 | `adapters.py::build_conditioning_prompt` |
| **BERTScore (Jaccard)** | §4.2.1 lite | `evaluate.py::_bertscore` |
| **Empathy (keywords)** | §4.2.2 lite | `evaluate.py::_empathy_score` |
| **Specificity** | §4.2.3 | `evaluate.py::_specificity` |
| **Overall Δ%** | §4.2.5 | `evaluate.py::AutomatedEvaluator.evaluate` |

---

## 6. Dimensional affect table (reference)

Full taxonomy constants in one place for citation:

### Valence (−1 = most negative, +1 = most positive)

```
Joy      : +1.00
Surprise : +0.20
Sadness  : −0.70
Fear     : −0.80
Anger    : −0.90   (most negative)
Disgust  : −0.85
Contempt : −0.60
```

### Arousal (0 = calm, 1 = activated)

```
Joy      : 0.70
Surprise : 0.90
Sadness  : 0.30   (most calm)
Fear     : 0.85
Anger    : 0.95   (most activated)
Disgust  : 0.60
Contempt : 0.50
```

Grounded in Russell (1980)'s circumplex model and Scherer (2005)'s appraisal refinements; concrete values were hand-set by the project team. If you report them in a paper, cite them as "project-specific calibration based on prior literature, not learned" — do not claim they were optimised.

---

## Appendix A — Worked example

For the "pizza complaint + second chance" demo:

**Turns**:
1. "pizza tastes so bad" → dominant = Disgust, $V = -0.85$, $A = 0.60$, $I \approx 0.35$
2. "it's such a waste of money" → Contempt, $V = -0.60$, $A = 0.50$, $I \approx 0.45$
3. "I want my money back" → Anger, $V = -0.90$, $A = 0.95$, $I \approx 0.40$
4. "i hate this store" → Anger, $V = -0.90$, $A = 0.95$, $I \approx 0.40$
5. "actually i will give them a second chance" → Joy, $V = +1.00$, $A = 0.70$, $I \approx 0.15$

**Momentum** (exponential weights $\{e^0, e^1, e^2, e^3\} = \{1, 2.72, 7.39, 20.09\}$, sum 31.2):
- Diffs: $\{+0.25, -0.30, +0.00, +1.90\}$
- Weighted sum: $1 \cdot 0.25 + 2.72 \cdot (-0.30) + 7.39 \cdot 0 + 20.09 \cdot 1.90 \approx 37.6$
- Divide by 31.2: $\approx 1.21$
- Clipped to $+1.00$ ✓ (matches UI)

**Escalation** ($N$ series): $[0.555, 0.400, 0.903, 0.903, 0.000]$
- Linear slope $\beta_1 \approx -0.08$ over $t \in \{0,1,2,3,4\}$
- $\mathrm{Esc} = \beta_1 \cdot 5 \approx -0.40$ (matches UI's −0.388) ✓

**Transition matrix** from sequence `[Disgust, Contempt, Anger, Anger, Joy]`:

| from \ to | Anger | Contempt | Disgust | Joy |
|---|---|---|---|---|
| Anger | 0.5 | 0 | 0 | 0.5 |
| Contempt | 1.0 | 0 | 0 | 0 |
| Disgust | 0 | 1.0 | 0 | 0 |
| Joy | 0 | 0 | 0 | 0 |

Anger row sums to 1 with two observations (→Anger, →Joy). Joy row is all-zero because it was the final turn.

---

*This document is the formal specification. For a narrative / presentation-oriented version see [PRESENTATION_HANDOFF.md](PRESENTATION_HANDOFF.md).*
