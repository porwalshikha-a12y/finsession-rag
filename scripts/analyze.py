"""Produce the results-chapter tables and figures.

Usage:  python -m scripts.analyze
Reads   results/results_all.csv (+ results/traces_*.jsonl)
Writes  results/analysis/*.csv and results/analysis/*.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analysis import (accuracy_with_ci, consistency, groundedness,
                          load_results, pairwise_tests, per_run, summarize)

OUT = Path("results/analysis")
OUT.mkdir(parents=True, exist_ok=True)

# --- chart style: one accent hue, thin marks, recessive axes ---------------
ACCENT = "#4269D0"
MUTED = "#8A94A6"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": "#1a1d23",
    "xtick.color": MUTED, "ytick.color": MUTED,
    "grid.color": "#E5E7EB", "grid.linewidth": 0.6,
})


def barh(series, title, xlabel, fname, fmt="{:.0f}", lower_better=False):
    """Horizontal bar chart, values direct-labelled, no legend (one series)."""
    s = series.dropna().sort_values(ascending=lower_better)
    fig, ax = plt.subplots(figsize=(6.2, 0.42 * len(s) + 1.3))
    ax.barh(range(len(s)), s.values, height=0.62, color=ACCENT, zorder=3)
    ax.set_yticks(range(len(s)), [i.replace("_", " ") for i in s.index])
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", pad=10)
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    pad = max(s.values) * 0.015
    for i, v in enumerate(s.values):
        ax.text(v + pad, i, fmt.format(v), va="center", fontsize=8, color="#1a1d23")
    ax.set_xlim(0, max(s.values) * 1.18)
    fig.tight_layout()
    fig.savefig(OUT / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"   figure -> {OUT / fname}")


def main() -> None:
    df = load_results()
    print(f"loaded {len(df)} rows | arms: {df['arm'].nunique()} | "
          f"runs: {df['run'].nunique()}\n")

    # ---- tables ----
    summary = summarize(df)
    summary.to_csv(OUT / "summary.csv")
    print("PER-ARM SUMMARY")
    print(summary.to_string(), "\n")

    ci = accuracy_with_ci(df)
    ci.to_csv(OUT / "accuracy_ci.csv")
    print("ACCURACY WITH 95% BOOTSTRAP CI")
    print(ci.to_string(), "\n")

    if df["run"].nunique() > 1:
        runs = per_run(df)
        runs.to_csv(OUT / "per_run_variance.csv")
        print("ACROSS REPEATED RUNS (mean +/- std)")
        print(runs.to_string(), "\n")
    else:
        print("(single run per arm — use --repeats N to get variance)\n")

    tests = pairwise_tests(df)
    if not tests.empty:
        tests.to_csv(OUT / "pairwise_tests.csv", index=False)
        print("PAIRED SIGNIFICANCE TESTS (McNemar, exact binomial)")
        print(tests.to_string(index=False), "\n")

    cons = consistency(df)
    if not cons.empty:
        cons.to_csv(OUT / "consistency.csv")
        print("PARAPHRASE CONSISTENCY (RQ3)")
        print(cons.to_string(), "\n")

    gr = groundedness()
    if not gr.empty:
        gr.to_csv(OUT / "groundedness.csv")
        print("EVIDENCE GROUNDING & REUSE (RQ4 / H4)")
        print(gr.to_string(), "\n")

    # ---- figures ----
    barh(summary["mean_tokens_per_q"], "Mean tokens per question",
         "tokens (lower is better)", "tokens_per_question.png", lower_better=True)
    barh(summary["accuracy"], "Answer accuracy", "proportion correct",
         "accuracy.png", fmt="{:.2f}")
    if summary["tokens_per_correct"].notna().any():
        barh(summary["tokens_per_correct"], "Cost per correct answer",
             "tokens per correct answer (lower is better)",
             "cost_per_correct.png", lower_better=True)
    if summary["memory_hits"].sum() > 0:
        barh(summary["hit_rate"], "Memory hit rate",
             "hits / (hits + retrievals)", "hit_rate.png", fmt="{:.2f}")

    print(f"\nAll tables and figures written to {OUT}/")


if __name__ == "__main__":
    main()
