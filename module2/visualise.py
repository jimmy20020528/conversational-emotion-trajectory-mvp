"""
Module 2 — Visualisation
=========================
Provides:
  - build_emotion_state_graph  : weighted networkx.DiGraph of emotion transitions
  - plot_trajectory            : four-panel matplotlib figure
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from .taxonomy import VALENCE, CLUSTER_COLORS
from .tracker import TrajectorySignals


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_emotion_state_graph(signals: TrajectorySignals) -> nx.DiGraph:
    """
    Build a directed weighted graph of emotion transitions.

    Nodes
    -----
    Each unique dominant emotion cluster seen in the conversation.
    Node attributes: mean_intensity, count, valence.

    Edges
    -----
    A → B for each observed transition; weight = transition probability.

    Returns
    -------
    networkx.DiGraph
    """
    G = nx.DiGraph()

    cluster_intensity: dict = defaultdict(list)
    for snap in signals.snapshots:
        cluster_intensity[snap.dominant].append(snap.intensity)

    for cluster, intensities in cluster_intensity.items():
        G.add_node(
            cluster,
            mean_intensity=float(np.mean(intensities)),
            count=len(intensities),
            valence=VALENCE.get(cluster, 0.0),
        )

    for src, targets in signals.transition_matrix.items():
        for tgt, prob in targets.items():
            if prob > 0 and src in G.nodes and tgt in G.nodes:
                G.add_edge(src, tgt, weight=prob)

    return G


# ---------------------------------------------------------------------------
# Four-panel trajectory plot
# ---------------------------------------------------------------------------

def plot_trajectory(
    signals: TrajectorySignals,
    save_path: Optional[str] = None,
) -> None:
    """
    Four-panel figure:
      1. Intensity & valence across turns
      2. Arousal across turns
      3. Higher-order signal bar chart (momentum / volatility / escalation)
      4. Emotion state graph

    Parameters
    ----------
    signals : TrajectorySignals
        Output of EmotionalTrajectoryTracker.compute().
    save_path : str or None
        If provided, save the figure to this path (PNG/PDF/SVG).
    """
    fig = plt.figure(figsize=(16, 10), facecolor="#0d0d14")
    fig.suptitle(
        "Emotional Trajectory Analysis — Module 2",
        fontsize=17, fontweight="bold", color="#e8e0f0", y=0.97,
    )

    ACCENT = "#b57bee"
    WARN   = "#f0826a"
    CALM   = "#6abed0"
    GRID   = "#1e1e2e"
    TEXT   = "#ccc8e8"

    turns = list(range(1, signals.turns + 1))

    # ── Panel 1: Intensity & Valence ─────────────────────────────────────────
    ax1 = fig.add_subplot(2, 2, 1, facecolor=GRID)
    ax1.plot(
        turns, signals.intensity_series, "o-", color=ACCENT,
        linewidth=2, markersize=7, label="Intensity", zorder=3,
    )
    ax1v = ax1.twinx()
    ax1v.plot(
        turns, signals.valence_series, "s--", color=CALM,
        linewidth=1.5, markersize=5, label="Valence", zorder=2,
    )
    ax1v.axhline(0, color="#555", linewidth=0.8, linestyle=":")
    ax1v.set_ylabel("Valence (−1 → +1)", color=CALM, fontsize=9)
    ax1v.tick_params(colors=CALM)
    ax1.set_title("Intensity & Valence per Turn", color=TEXT, fontsize=11, pad=8)
    ax1.set_xlabel("Turn", color=TEXT)
    ax1.set_ylabel("Intensity", color=ACCENT)
    ax1.tick_params(colors=TEXT)
    ax1.set_xticks(turns)
    ax1.set_ylim(0, 1.05)
    for i, (t, snap) in enumerate(zip(turns, signals.snapshots)):
        ax1.annotate(
            snap.dominant, (t, signals.intensity_series[i]),
            textcoords="offset points", xytext=(0, 10),
            ha="center", fontsize=7.5, color=TEXT,
        )
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1v.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2, labels1 + labels2,
        fontsize=8, facecolor="#1a1a2e", labelcolor=TEXT, loc="lower left",
    )
    _style_axes(ax1, TEXT)

    # ── Panel 2: Arousal ─────────────────────────────────────────────────────
    ax2 = fig.add_subplot(2, 2, 2, facecolor=GRID)
    ax2.fill_between(turns, signals.arousal_series, alpha=0.3, color=WARN)
    ax2.plot(turns, signals.arousal_series, "o-", color=WARN, linewidth=2, markersize=7)
    esc = signals.escalation_score
    esc_label = (
        f"Escalation: {esc:+.3f} "
        f"({'↑ escalating' if esc > 0.1 else '↓ de-escalating' if esc < -0.1 else '→ stable'})"
    )
    ax2.text(
        0.97, 0.05, esc_label, transform=ax2.transAxes,
        ha="right", fontsize=8.5, color=WARN,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#2a1a1a", edgecolor=WARN, alpha=0.8),
    )
    ax2.set_title("Arousal Trajectory", color=TEXT, fontsize=11, pad=8)
    ax2.set_xlabel("Turn", color=TEXT)
    ax2.set_ylabel("Arousal", color=WARN)
    ax2.tick_params(colors=TEXT)
    ax2.set_xticks(turns)
    ax2.set_ylim(0, 1.05)
    _style_axes(ax2, TEXT)

    # ── Panel 3: Signal bar chart ─────────────────────────────────────────────
    ax3 = fig.add_subplot(2, 2, 3, facecolor=GRID)
    metrics = {
        "Momentum":   signals.emotional_momentum,
        "Volatility": signals.volatility_index,
        "Escalation": signals.escalation_score,
    }
    colors = [CALM if v >= 0 else WARN for v in metrics.values()]
    bars = ax3.barh(
        list(metrics.keys()), list(metrics.values()),
        color=colors, height=0.45, edgecolor="#333",
    )
    ax3.axvline(0, color="#888", linewidth=1)
    ax3.set_xlim(-1.1, 1.1)
    ax3.set_title("Higher-Order Signal Scores", color=TEXT, fontsize=11, pad=8)
    ax3.tick_params(colors=TEXT)
    _style_axes(ax3, TEXT)
    for bar, val in zip(bars, metrics.values()):
        ax3.text(
            val + (0.04 if val >= 0 else -0.04),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}", va="center",
            ha="left" if val >= 0 else "right",
            fontsize=9, color=TEXT,
        )

    # ── Panel 4: Emotion State Graph ──────────────────────────────────────────
    ax4 = fig.add_subplot(2, 2, 4, facecolor=GRID)
    G = build_emotion_state_graph(signals)
    if len(G.nodes) > 0:
        _draw_graph(G, ax4, ACCENT, WARN, CALM, TEXT)
    else:
        ax4.text(
            0.5, 0.5, "Not enough turns for graph",
            transform=ax4.transAxes, ha="center", va="center",
            color=TEXT, fontsize=11,
        )
    ax4.set_title("Emotion State Graph", color=TEXT, fontsize=11, pad=8)
    _style_axes(ax4, TEXT)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        plt.savefig(
            save_path, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
        print(f"Saved → {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _draw_graph(G, ax, ACCENT, WARN, CALM, TEXT) -> None:
    pos = nx.spring_layout(G, seed=42, k=2.5)
    node_sizes  = [G.nodes[n].get("mean_intensity", 0.5) * 2500 + 600 for n in G.nodes]
    node_colors = [CLUSTER_COLORS.get(n, WARN) for n in G.nodes]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                           node_color=node_colors, alpha=0.85)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8,
                            font_color=TEXT, font_weight="bold")

    edges   = list(G.edges(data=True))
    weights = [d.get("weight", 0.1) for _, _, d in edges]
    widths  = [max(1.0, w * 4) for w in weights]
    edge_cols = [ACCENT if w > 0.5 else "#888" for w in weights]

    nx.draw_networkx_edges(
        G, pos, ax=ax, width=widths, edge_color=edge_cols,
        arrows=True, arrowsize=18, alpha=0.85,
        connectionstyle="arc3,rad=0.2",
        min_source_margin=20, min_target_margin=20,
    )
    edge_labels = {
        (u, v): f"{d['weight']:.2f}"
        for u, v, d in G.edges(data=True) if d.get("weight", 0) > 0.1
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
                                 font_size=7, font_color="#ccc")


def _style_axes(ax, text_color: str) -> None:
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.tick_params(colors=text_color)
    ax.xaxis.label.set_color(text_color)
    ax.yaxis.label.set_color(text_color)
