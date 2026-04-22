"""
Automated evaluation — three complementary metrics with graceful fallbacks.

Composite score (per response):
    overall = 0.50 · empathy + 0.30 · bertscore + 0.20 · specificity

Improvement percentage per conversation:
    (overall_conditioned − overall_baseline) / overall_baseline × 100

The notebook uses `conditioned_response` as the implicit gold reference when
no hand-written references are provided — that is deliberately preserved here
so behavior matches the notebook report.

Heavy optional deps:
  - `bert-score` (real contextual BERTScore) — Jaccard token overlap fallback
  - `facebook/bart-large-mnli` via transformers pipeline — keyword fallback
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ResponseScores:
    """Per-conversation evaluation scores."""
    conversation_id: str
    bertscore_baseline: float = 0.0
    bertscore_conditioned: float = 0.0
    empathy_baseline: float = 0.0
    empathy_conditioned: float = 0.0
    specificity_baseline: float = 0.0
    specificity_conditioned: float = 0.0
    length_baseline: int = 0
    length_conditioned: int = 0
    overall_baseline: float = 0.0
    overall_conditioned: float = 0.0
    improvement_pct: float = 0.0


class AutomatedEvaluator:
    """Three-metric evaluator: BERTScore + Empathy + Specificity."""

    def __init__(self, *, use_heavy_models: bool = True):
        """
        Parameters
        ----------
        use_heavy_models : bool
            If True, try to load `bert-score` and `facebook/bart-large-mnli`
            (~1.5 GB download on first call). If False, go straight to the
            lightweight fallbacks — fine for quick scoring in a UI.
        """
        self._use_heavy = use_heavy_models
        self._bertscore_model = None
        self._nli_pipeline = None
        self._loaded = False

    def _load_models(self) -> None:
        if self._loaded:
            return
        if self._use_heavy:
            try:
                from bert_score import BERTScorer
                self._bertscore_model = BERTScorer(lang="en", rescale_with_baseline=True)
            except ImportError:
                pass
            try:
                from transformers import pipeline
                self._nli_pipeline = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli",
                    device=-1,
                )
            except ImportError:
                pass
        self._loaded = True

    def _bertscore(self, response: str, reference: str) -> float:
        if self._bertscore_model is not None:
            _, _, F = self._bertscore_model.score([response], [reference])
            return float(F[0])
        # Jaccard token overlap fallback
        ref_tokens = set(reference.lower().split())
        res_tokens = set(response.lower().split())
        if not ref_tokens:
            return 0.0
        return len(ref_tokens & res_tokens) / len(ref_tokens | res_tokens)

    def _empathy_score(self, response: str, dominant_emotion: str) -> float:
        if self._nli_pipeline is not None:
            result = self._nli_pipeline(
                response,
                candidate_labels=["empathetic and validating", "dismissive or cold"],
                hypothesis_template=(
                    f"This response to a user expressing {dominant_emotion.lower()} is {{}}."
                ),
            )
            scores = dict(zip(result["labels"], result["scores"]))
            return float(scores.get("empathetic and validating", 0.0))
        # Keyword fallback
        empathy_keywords = {
            "understand", "hear", "feel", "difficult", "hard", "support",
            "acknowledge", "sense", "exhausting", "sounds like", "that's",
            "valid", "matter", "grieve", "sorry",
        }
        words = set(response.lower().split())
        return min(1.0, len(words & empathy_keywords) / 5.0)

    @staticmethod
    def _specificity(response: str, signals: Any) -> float:
        resp_lower = response.lower()
        score = 0.0

        dominant = getattr(signals, "dominant_state", "") or ""
        if dominant.lower() in resp_lower:
            score += 0.4
        elif any(w in resp_lower for w in [
            "anxious", "stress", "fear", "anger",
            "sad", "joy", "disgust", "contempt",
        ]):
            score += 0.2

        esc = float(getattr(signals, "escalation_score", 0.0))
        vol = float(getattr(signals, "volatility_index", 0.0))

        if esc > 0.1:
            if any(w in resp_lower for w in ["more", "building", "growing", "heavy", "exhausting"]):
                score += 0.3
        elif esc < -0.1:
            if any(w in resp_lower for w in ["better", "improving", "progress", "calmer"]):
                score += 0.3
        else:
            score += 0.15

        if vol > 0.4:
            if any(w in resp_lower for w in ["lot", "much", "overwhelming", "intense", "hard"]):
                score += 0.3

        return min(1.0, score)

    def evaluate(
        self,
        records: List[Dict[str, Any]],
        gold_references: Optional[List[str]] = None,
    ) -> List[ResponseScores]:
        self._load_models()

        results: List[ResponseScores] = []
        for i, rec in enumerate(records):
            signals = rec["signals"]
            baseline = rec["baseline_response"]
            conditioned = rec["conditioned_response"]
            reference = gold_references[i] if gold_references else conditioned

            bs_b = self._bertscore(baseline, reference)
            bs_c = self._bertscore(conditioned, reference)
            emp_b = self._empathy_score(baseline, signals.dominant_state)
            emp_c = self._empathy_score(conditioned, signals.dominant_state)
            spec_b = self._specificity(baseline, signals)
            spec_c = self._specificity(conditioned, signals)

            overall_b = 0.5 * emp_b + 0.3 * bs_b + 0.2 * spec_b
            overall_c = 0.5 * emp_c + 0.3 * bs_c + 0.2 * spec_c
            improvement = ((overall_c - overall_b) / overall_b * 100) if overall_b > 0 else 0.0

            results.append(ResponseScores(
                conversation_id=rec["conversation_id"],
                bertscore_baseline=round(bs_b, 4),
                bertscore_conditioned=round(bs_c, 4),
                empathy_baseline=round(emp_b, 4),
                empathy_conditioned=round(emp_c, 4),
                specificity_baseline=round(spec_b, 4),
                specificity_conditioned=round(spec_c, 4),
                length_baseline=len(baseline.split()),
                length_conditioned=len(conditioned.split()),
                overall_baseline=round(overall_b, 4),
                overall_conditioned=round(overall_c, 4),
                improvement_pct=round(improvement, 2),
            ))
        return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_evaluation_report(scores: List[ResponseScores]) -> None:
    """Pretty-print baseline → conditioned deltas, aggregate means, and the
    proposal's ≥ 20% improvement target."""
    import pandas as pd

    print("\n" + "═" * 75)
    print("  MODULE 3 — RESPONSE EVALUATION REPORT")
    print("═" * 75)

    df = pd.DataFrame([vars(s) for s in scores])

    print("\nPer-Conversation Scores (Baseline → Conditioned)\n")
    print(f"{'Conv':<12}{'BERTScore':>18}{'Empathy':>18}{'Specificity':>18}{'Overall':>18}{'Δ%':>8}")
    print("-" * 92)
    for s in scores:
        print(
            f"{s.conversation_id:<12}"
            f"  {s.bertscore_baseline:.3f} → {s.bertscore_conditioned:.3f}"
            f"  {s.empathy_baseline:.3f} → {s.empathy_conditioned:.3f}"
            f"  {s.specificity_baseline:.3f} → {s.specificity_conditioned:.3f}"
            f"  {s.overall_baseline:.3f} → {s.overall_conditioned:.3f}"
            f"  {s.improvement_pct:>+.1f}%"
        )

    print("\n" + "─" * 75)
    print("AGGREGATE (mean across all conversations)\n")
    metrics = {
        "BERTScore":   ("bertscore_baseline",   "bertscore_conditioned"),
        "Empathy":     ("empathy_baseline",     "empathy_conditioned"),
        "Specificity": ("specificity_baseline", "specificity_conditioned"),
        "Overall":     ("overall_baseline",     "overall_conditioned"),
    }
    for label, (col_b, col_c) in metrics.items():
        mean_b = df[col_b].mean()
        mean_c = df[col_c].mean()
        delta = mean_c - mean_b
        pct = (delta / mean_b * 100) if mean_b > 0 else 0.0
        bar = "▓" * int(abs(pct) / 5) if abs(pct) > 0 else "·"
        print(
            f"  {label:<14}: {mean_b:.4f} → {mean_c:.4f}  "
            f"(Δ {delta:+.4f}  |  {pct:+.1f}%  {bar})"
        )

    avg_improvement = df["improvement_pct"].mean()
    print(f"\n  Mean overall improvement : {avg_improvement:+.2f}%")

    target = 20.0
    status = (
        "HYPOTHESIS SUPPORTED"
        if avg_improvement >= target
        else f"Below {target}% target"
    )
    print(f"  Target (>= {target}%)        : {status}")

    print("\n  Response lengths (words):")
    print(f"    Baseline avg    : {df['length_baseline'].mean():.1f}")
    print(f"    Conditioned avg : {df['length_conditioned'].mean():.1f}")
    print("═" * 75)


def export_results_csv(scores: List[ResponseScores], path: str = "evaluation_results.csv") -> None:
    import pandas as pd
    pd.DataFrame([vars(s) for s in scores]).to_csv(path, index=False)
