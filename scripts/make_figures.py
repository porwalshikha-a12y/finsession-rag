"""Regenerate Figures 5.1-5.3 from the recorded experiment runs.

Reads results/results_main7.json (A0-A6) and results/results_all.json (A7-A10)
and writes PNG (400 dpi) + PDF (vector) into figures/.

Usage:
    python3 scripts/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

# --- design tokens -----------------------------------------------------------
BLUE, ORANGE, SLATE = "#2a78d6", "#eb6834", "#6f6e69"
INK, SUBTLE, GRID, RULE = "#1a1a19", "#52514e", "#eceae5", "#c9c7c0"
THOUSANDS = FuncFormatter(lambda v, _: f"{v:,.0f}")
WIDTH = 6.5  # inches: matches a 1-inch-margin A4 text column

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8.5,
    "axes.labelsize": 9,
    "axes.labelcolor": INK,
    "axes.labelpad": 7,
    "axes.edgecolor": RULE,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.color": INK,
    "xtick.color": SUBTLE,
    "ytick.color": INK,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8.5,
    "xtick.major.width": 0.8,
    "ytick.major.size": 0,
    "figure.dpi": 400,
    "savefig.dpi": 400,
    "pdf.fonttype": 42,   # embed TrueType so Word/print keep the text selectable
    "ps.fonttype": 42,
})

LABELS = {
    "A0_no_memory": "A0   no memory",
    "A1_unbounded": "A1   unbounded",
    "A2_lru": "A2   LRU (3)",
    "A3_lfu": "A3   LFU (3)",
    "A4_redundancy": "A4   redundancy (3)",
    "A5_cost_aware": "A5   cost-aware (3)",
    "A6_hybrid": "A6   replication of A5",
    "A7_no_reuse_gate": "A7   gate disabled",
    "A8_lru_b5": "A8   LRU (5)",
    "A9_cost_aware_b5": "A9   cost-aware (5)",
    "A10_hybrid": "A10  hybrid retrieval",
}
ORDER = list(LABELS)


def load() -> dict[str, dict]:
    rows = []
    for name in ("results_main7.json", "results_all.json"):
        rows += json.loads((ROOT / "results" / name).read_text())
    arms: dict[str, dict] = {}
    for s in rows:
        arms.setdefault(s["arm"], {})[s["session_id"]] = s["queries"]
    sessions = sorted(arms[ORDER[0]])
    stats = {}
    for a in ORDER:
        qq = [q for s in sessions for q in arms[a][s]]
        accepted = sum(q["memory_hits"] for q in qq)
        rejected = sum(q["reuse_rejected"] for q in qq)
        cand = accepted + rejected
        stats[a] = {
            "tokens_per_session": sum(q["tokens_spent"] for q in qq) / len(sessions),
            "accuracy": 100 * sum(1 for q in qq if q["judged_correct"] is True) / len(qq),
            "candidates": cand,
            "rejection": 100 * rejected / cand if cand else 0.0,
        }
    return stats


def tidy_barh(ax):
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.xaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", pad=2)
    ax.invert_yaxis()


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight",
                    pad_inches=0.04, facecolor="white")
    plt.close(fig)
    print("wrote", f"figures/{stem}.png", "+ .pdf")


def fig_5_1(st):
    vals = [st[a]["tokens_per_session"] for a in ORDER]
    colours = [SLATE if a == "A0_no_memory" else
               ORANGE if a == "A7_no_reuse_gate" else BLUE for a in ORDER]
    base = st["A0_no_memory"]["tokens_per_session"]

    fig, ax = plt.subplots(figsize=(WIDTH, 4.0))
    ax.barh(range(len(ORDER)), vals, color=colours, height=0.66, zorder=3)
    ax.set_yticks(range(len(ORDER)), [LABELS[a] for a in ORDER])

    for i, (a, v) in enumerate(zip(ORDER, vals)):
        delta = ("baseline" if a == "A0_no_memory"
                 else f"−{100 * (base - v) / base:.1f}%")
        ax.text(v + base * 0.014, i, f"{v:,.0f}", va="center",
                fontsize=8, color=INK)
        ax.text(base * 1.30, i, delta, va="center", ha="right",
                fontsize=8, color=SUBTLE)

    ax.set_xlabel("Mean tokens per session")
    ax.xaxis.set_major_formatter(THOUSANDS)
    ax.set_xlim(0, base * 1.32)
    ax.set_xticks([0, 5000, 10000, 15000, 20000])
    tidy_barh(ax)
    save(fig, "fig5_1_cost_by_arm")


def fig_5_2(st):
    pair = ["A5_cost_aware", "A7_no_reuse_gate"]
    names = ["A5\ngate enabled", "A7\ngate disabled"]
    panels = (("tokens_per_session", "Mean tokens per session", "{:,.0f}", True),
              ("accuracy", "Correctness (%)", "{:.1f}%", False))

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.3))
    for ax, (key, lab, fmt, thou) in zip(axes, panels):
        vals = [st[a][key] for a in pair]
        ax.bar(names, vals, color=[BLUE, ORANGE], width=0.46, zorder=3)
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.025, fmt.format(v),
                    ha="center", fontsize=8.5, color=INK)

        # change annotation, drawn between the two bar tops
        top = max(vals) * 1.17
        ax.plot([0, 0, 1, 1], [top * 0.94, top, top, top * 0.94],
                color=SUBTLE, lw=0.8, zorder=2)
        drop = (f"−{100 * (vals[0] - vals[1]) / vals[0]:.1f}%" if thou
                else f"−{vals[0] - vals[1]:.1f} pp")
        ax.text(0.5, top * 1.015, drop, ha="center", va="bottom",
                fontsize=8.5, color=SUBTLE)

        ax.set_ylabel(lab)
        ax.set_ylim(0, max(vals) * 1.34)
        if thou:
            ax.yaxis.set_major_formatter(THOUSANDS)
        ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0, pad=4)
        ax.spines["bottom"].set_color(RULE)

    fig.tight_layout(w_pad=3.0)
    save(fig, "fig5_2_gate_ablation")


def fig_5_3(st):
    arms = [a for a in ORDER if a not in ("A0_no_memory", "A7_no_reuse_gate")]
    vals = [st[a]["rejection"] for a in arms]
    lo, hi = min(vals), max(vals)

    fig, ax = plt.subplots(figsize=(WIDTH, 3.5))
    ax.axvspan(lo, hi, color=BLUE, alpha=0.07, zorder=1)
    ax.barh(range(len(arms)), vals, color=BLUE, height=0.62, zorder=3)
    ax.set_yticks(range(len(arms)), [LABELS[a] for a in arms])

    for i, (a, v) in enumerate(zip(arms, vals)):
        ax.text(v + 1.4, i, f"{v:.1f}%", va="center", fontsize=8,
                color=INK, fontweight="medium")
        ax.text(99, i, f"n = {st[a]['candidates']}", va="center", ha="right",
                fontsize=7.5, color=SUBTLE)

    ax.annotate(f"range {lo:.1f}–{hi:.1f}%", xy=((lo + hi) / 2, -0.85),
                ha="center", va="bottom", fontsize=7.5, color=SUBTLE, style="italic",
                annotation_clip=False)

    ax.set_xlabel("Candidate matches rejected by the reuse gate (%)")
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 20))
    tidy_barh(ax)
    save(fig, "fig5_3_gate_rejection")


if __name__ == "__main__":
    s = load()
    fig_5_1(s)
    fig_5_2(s)
    fig_5_3(s)
