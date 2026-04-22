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
from module3.generate import generate_reply

ROOT = Path(__file__).resolve().parent


def main() -> None:
    st.set_page_config(page_title="Emotion trajectory demo", layout="wide")
    st.title("Conversational emotion trajectory (MVP)")

    with st.sidebar:
        model_dir = st.text_input(
            "Module 1 model folder",
            value=str(ROOT / "outputs/module1_goemotions/module1_model"),
        )
        max_length = st.slider("Max tokens", 32, 256, 128)
        if st.button("Clear conversation"):
            st.session_state.turns = []
            st.session_state.pop("last_reply", None)
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

    turns = st.session_state.turns
    traj = compute_trajectory(turns) if turns else None

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
                    "Only **one** user turn so far — **volatility** and **escalation** stay at 0 "
                    "because they compare *between* turns. Send another message to see them move."
                )
            st.metric("Volatility", f"{traj['volatility']:.3f}")
            st.metric("Escalation (neg. affect)", f"{traj['escalation_score']:+.3f}")
            st.write(traj["summary"])
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

        rows = []
        for i, t in enumerate(turns):
            row = {"turn": i}
            row.update({k: t["scores"][k] for k in EMOTION_LABELS})
            rows.append(row)
        df = pd.DataFrame(rows).set_index("turn")
        st.subheader("Score curves")
        st.line_chart(df)

    st.divider()
    if turns and st.button("Generate trajectory-conditioned reply (Module 3)"):
        last = turns[-1]["text"]
        reply = generate_reply(last, traj)
        st.session_state.last_reply = reply

    if st.session_state.get("last_reply"):
        st.subheader("Assistant reply")
        st.write(st.session_state.last_reply)


if __name__ == "__main__":
    main()
