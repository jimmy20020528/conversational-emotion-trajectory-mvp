"""
Streamlit demo: multi-turn chat → Module 1 scores → Module 2 trajectory → Module 3 reply.

Run from project root:
  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from module1.labels import EMOTION_LABELS
from module1.runtime import load_classifier, predict_scores
from module2.trajectory import compute_trajectory
from module2 import EmotionalTrajectoryTracker, build_conditioning_prompt
from module3 import (
    generate_reply,
    generate_baseline_reply,
    AutomatedEvaluator,
    export_human_rater_csv,
)

ROOT = Path(__file__).resolve().parent
RATER_CSV = ROOT / "artifacts" / "human_rater_sheet.csv"


def _compute_rich_signals(turns: list[dict]):
    """Run the full EmotionalTrajectoryTracker over all turns so far."""
    if not turns:
        return None
    tracker = EmotionalTrajectoryTracker()
    for t in turns:
        tracker.add_turn(t["scores"])
    return tracker.compute()


@st.cache_resource
def _cached_evaluator() -> AutomatedEvaluator:
    # Lightweight fallbacks only — avoid 1.5 GB NLI/BERTScore download in the UI.
    return AutomatedEvaluator(use_heavy_models=False)


def main() -> None:
    st.set_page_config(page_title="Emotion trajectory demo", layout="wide")
    st.title("Conversational emotion trajectory (MVP)")

    with st.sidebar:
        model_dir = st.text_input(
            "Module 1 model folder",
            value=str(ROOT / "outputs/module1_multidomain/module1_model"),
        )
        max_length = st.slider("Max tokens", 32, 256, 128)
        show_graph = st.checkbox("Show emotion state graph", value=True)
        run_ab = st.checkbox("Run baseline vs conditioned A/B", value=True)
        if st.button("Clear conversation"):
            for k in ("turns", "last_baseline", "last_conditioned", "last_cond"):
                st.session_state.pop(k, None)
            st.rerun()

    if "turns" not in st.session_state:
        st.session_state.turns = []

    path = Path(model_dir)
    if not path.is_dir():
        st.warning(f"Model path not found: `{path}`. Train Module 1 first or fix the path.")
        return

    @st.cache_resource
    def _cached_load(p: str):
        return load_classifier(Path(p))

    model, tokenizer, device = _cached_load(str(path.resolve()))

    user_msg = st.chat_input("Type a message as the user…")
    if user_msg:
        scores = predict_scores(
            model, tokenizer, device, user_msg, max_length=max_length
        )
        st.session_state.turns.append({"text": user_msg, "scores": scores})
        # Invalidate stored A/B when conversation changes
        for k in ("last_baseline", "last_conditioned", "last_cond"):
            st.session_state.pop(k, None)

    turns = st.session_state.turns
    traj = compute_trajectory(turns) if turns else None
    signals = _compute_rich_signals(turns)

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.subheader("Chat")
        for t in turns:
            with st.chat_message("user"):
                st.write(t["text"])

    with col_b:
        st.subheader("Trajectory (Module 2)")
        if traj:
            if len(turns) < 2:
                st.info(
                    "Only **one** user turn so far — **volatility**, **momentum** and "
                    "**escalation** stay at 0 because they compare *between* turns. "
                    "Send another message to see them move."
                )
            m1, m2, m3 = st.columns(3)
            m1.metric("Volatility", f"{signals.volatility_index:.3f}" if signals else "0.000")
            m2.metric("Momentum", f"{signals.emotional_momentum:+.3f}" if signals else "+0.000")
            m3.metric("Escalation", f"{signals.escalation_score:+.3f}" if signals else "+0.000")
            if signals:
                st.markdown(f"**Dominant state:** `{signals.dominant_state}`")
                st.markdown("**Emotion arc:** " + " → ".join(signals.emotion_sequence))
                # Summary derived from the rich tracker — consistent with the metric cards above.
                # (The legacy compute_trajectory summary was kept earlier but used a different
                # negative-mass heuristic that can disagree with the valence/arousal-based
                # escalation_score shown in the metric; one source of truth is clearer.)
                esc = signals.escalation_score
                esc_desc = (
                    "rising negative affect"
                    if esc > 0.08
                    else "de-escalating / easing"
                    if esc < -0.08
                    else "stable"
                )
                mom = signals.emotional_momentum
                mom_desc = (
                    "improving"
                    if mom > 0.08
                    else "worsening"
                    if mom < -0.08
                    else "steady"
                )
                st.write(
                    f"Trajectory: **{esc_desc}**, valence **{mom_desc}**. "
                    f"Latest dominant emotion: **{signals.dominant_state}**."
                )
            st.caption(
                "Dominant label = argmax over 7 probabilities. The classifier can misfire on short or "
                "out-of-domain sentences; check raw scores below."
            )
        else:
            st.info("Send a message to compute trajectory.")

    if turns:
        latest = turns[-1]["scores"]
        ranked = sorted(latest.items(), key=lambda kv: -kv[1])
        top3 = ", ".join(f"**{k}** {v:.2f}" for k, v in ranked[:3])
        st.markdown(f"**Latest turn — top-3 scores:** {top3}")
        bar_df = pd.DataFrame([latest]).T.rename(columns={0: "p"})
        st.bar_chart(bar_df, height=220)

        # Use explicit "T1", "T2"... string labels so Streamlit treats the
        # x-axis as ordinal categories. Integer indices sometimes get auto-
        # detected as quantitative, producing weird tick ranges (e.g. 0-135
        # on 5 turns).
        turn_labels = [f"T{i + 1}" for i in range(len(turns))]
        rows = []
        for i, t in enumerate(turns):
            row = {"turn": turn_labels[i]}
            row.update({k: t["scores"][k] for k in EMOTION_LABELS})
            rows.append(row)
        df = pd.DataFrame(rows).set_index("turn")
        st.subheader("Score curves")
        st.line_chart(df, y_label="probability", x_label="turn")

        if signals and len(turns) >= 2:
            st.subheader("Valence & arousal")
            va_df = pd.DataFrame(
                {
                    "valence": signals.valence_series,
                    "arousal": signals.arousal_series,
                },
                index=turn_labels,
            )
            va_df.index.name = "turn"
            st.line_chart(va_df, y_label="score", x_label="turn")

    if signals and len(turns) >= 2 and signals.transition_matrix:
        st.subheader("Transition matrix")
        tm_rows = []
        for src, targets in signals.transition_matrix.items():
            row = {"from → to": src}
            row.update({tgt: round(prob, 2) for tgt, prob in targets.items()})
            tm_rows.append(row)
        if tm_rows:
            st.dataframe(pd.DataFrame(tm_rows).set_index("from → to"))

    if show_graph and signals and len(turns) >= 2:
        st.subheader("Emotion state graph")
        try:
            import io
            import matplotlib
            matplotlib.use("Agg")  # headless backend — no tkinter/figure-manager side effects
            import matplotlib.pyplot as plt
            import networkx as nx
            from module2.visualise import build_emotion_state_graph
            from module2.taxonomy import CLUSTER_COLORS

            G = build_emotion_state_graph(signals)
            if len(G.nodes) >= 1:
                fig, ax = plt.subplots(figsize=(6, 4))
                pos = nx.spring_layout(G, seed=42, k=2.0)
                node_sizes = [G.nodes[n].get("mean_intensity", 0.5) * 1800 + 400 for n in G.nodes]
                node_colors = [CLUSTER_COLORS.get(n, "#888") for n in G.nodes]
                nx.draw_networkx_nodes(
                    G, pos, ax=ax, node_size=node_sizes,
                    node_color=node_colors, alpha=0.85,
                )
                nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_weight="bold")
                weights = [d.get("weight", 0.1) for _, _, d in G.edges(data=True)]
                widths = [max(1.0, w * 3) for w in weights]
                nx.draw_networkx_edges(
                    G, pos, ax=ax, width=widths, arrows=True, arrowsize=14,
                    alpha=0.75, connectionstyle="arc3,rad=0.2",
                    min_source_margin=15, min_target_margin=15,
                )
                edge_labels = {
                    (u, v): f"{d['weight']:.2f}"
                    for u, v, d in G.edges(data=True) if d.get("weight", 0) > 0.1
                }
                nx.draw_networkx_edge_labels(
                    G, pos, edge_labels=edge_labels, ax=ax, font_size=7,
                )
                ax.set_axis_off()
                # Serialize to PNG bytes and render via st.image — skips
                # st.pyplot entirely, which was leaking a stray "0" under
                # the figure on Streamlit 1.50.x.
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
                plt.close(fig)
                st.image(buf.getvalue(), use_column_width=True)
        except ImportError:
            st.caption("Install `matplotlib` + `networkx` to see the emotion state graph.")

    st.divider()

    # ─── Module 3: generation (A/B or single) ────────────────────────────────
    if turns:
        button_label = (
            "Generate baseline + conditioned replies (Module 3 A/B)"
            if run_ab
            else "Generate trajectory-conditioned reply (Module 3)"
        )
        if st.button(button_label):
            last = turns[-1]["text"]
            cond_prompt = build_conditioning_prompt(signals) if signals else None
            if run_ab:
                baseline = generate_baseline_reply(last)
                conditioned = generate_reply(last, traj, conditioning_prompt=cond_prompt)
                st.session_state.last_baseline = baseline
                st.session_state.last_conditioned = conditioned
            else:
                conditioned = generate_reply(last, traj, conditioning_prompt=cond_prompt)
                st.session_state.last_conditioned = conditioned
                st.session_state.pop("last_baseline", None)
            if cond_prompt:
                st.session_state.last_cond = cond_prompt

    # ─── Display replies + A/B metrics ───────────────────────────────────────
    if st.session_state.get("last_conditioned"):
        if st.session_state.get("last_baseline"):
            st.subheader("Baseline vs conditioned (A/B)")
            b_col, c_col = st.columns(2)
            with b_col:
                st.markdown("**Baseline** _(no emotion context)_")
                st.write(st.session_state["last_baseline"])
            with c_col:
                st.markdown("**Conditioned** _(trajectory-aware)_")
                st.write(st.session_state["last_conditioned"])

            if signals:
                evaluator = _cached_evaluator()
                rec = {
                    "conversation_id": "current_session",
                    "signals": signals,
                    "baseline_response": st.session_state["last_baseline"],
                    "conditioned_response": st.session_state["last_conditioned"],
                }
                scores = evaluator.evaluate([rec])[0]
                st.markdown("**Automated scores** _(lightweight fallbacks — Jaccard + keyword; "
                            "heavy BERTScore/NLI metrics available via `module3.AutomatedEvaluator(use_heavy_models=True)`)_")
                s1, s2, s3, s4 = st.columns(4)
                s1.metric(
                    "BERTScore (Jaccard)",
                    f"{scores.bertscore_conditioned:.2f}",
                    delta=f"{scores.bertscore_conditioned - scores.bertscore_baseline:+.2f}",
                )
                s2.metric(
                    "Empathy (keywords)",
                    f"{scores.empathy_conditioned:.2f}",
                    delta=f"{scores.empathy_conditioned - scores.empathy_baseline:+.2f}",
                )
                s3.metric(
                    "Specificity",
                    f"{scores.specificity_conditioned:.2f}",
                    delta=f"{scores.specificity_conditioned - scores.specificity_baseline:+.2f}",
                )
                s4.metric(
                    "Overall Δ",
                    f"{scores.improvement_pct:+.1f}%",
                    delta=f"{scores.overall_conditioned - scores.overall_baseline:+.3f}",
                )
                st.caption(
                    f"Baseline composite: {scores.overall_baseline:.3f} | "
                    f"Conditioned composite: {scores.overall_conditioned:.3f} | "
                    f"Target (proposal): ≥ +20%"
                )

                if st.button("Append this conversation to the human-rater CSV"):
                    RATER_CSV.parent.mkdir(parents=True, exist_ok=True)
                    conv_id = f"ui_{len(turns):02d}turns_{hash(turns[-1]['text']) & 0xFFFF:04x}"
                    rec_for_rater = dict(rec, conversation_id=conv_id,
                                         user_last_turn=turns[-1]["text"])
                    export_human_rater_csv([rec_for_rater], str(RATER_CSV), append=True)
                    st.success(f"Appended to {RATER_CSV.relative_to(ROOT)}")
        else:
            st.subheader("Assistant reply")
            st.write(st.session_state["last_conditioned"])

        if st.session_state.get("last_cond"):
            with st.expander("Emotional conditioning context sent to the LLM"):
                st.code(st.session_state["last_cond"], language="text")


if __name__ == "__main__":
    main()
