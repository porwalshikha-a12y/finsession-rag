"""Analysis and statistics over experiment results.

Turns results/results_all.csv into the numbers and tests that go in the
results chapter:

  summarize()       per-arm accuracy, tokens, cost-per-correct, memory stats
  mcnemar()         paired significance test between two arms (exact binomial)
  bootstrap_ci()    confidence interval for any mean
  consistency()     paraphrase-probe agreement rate (RQ3)
  groundedness()    are memory-reused answers as evidence-grounded as fresh
                    retrievals? (RQ4 / H4)

No scipy dependency: the exact binomial test is computed directly.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

# gpt-4o-mini list prices (USD per 1M tokens) — indicative only.
PRICE_IN, PRICE_OUT = 0.15, 0.60


# ------------------------------------------------------------------ loading
def load_results(path: str | Path = "results/results_all.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize the judge column to 0/1 floats (it may arrive as bool or text)
    df["correct"] = (
        df["correct"].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
        if df["correct"].dtype == object
        else df["correct"].astype(float)
    )
    if "run" not in df.columns:
        df["run"] = 0
    return df


# --------------------------------------------------------------- summarising
def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """One row per arm: the headline table of the results chapter."""
    rows = []
    for arm, g in df.groupby("arm", sort=False):
        n_correct = g["correct"].sum()
        acc = g["correct"].mean()
        tokens = g["tokens"].sum()
        rows.append({
            "arm": arm,
            "questions": len(g),
            "runs": g["run"].nunique(),
            "accuracy": round(acc, 3),
            "mean_tokens_per_q": round(g["tokens"].mean(), 0),
            "total_tokens": int(tokens),
            # headline metric: what does one correct answer cost?
            "tokens_per_correct": round(tokens / n_correct, 0) if n_correct else np.nan,
            "est_usd": round(tokens / 1e6 * (PRICE_IN + PRICE_OUT) / 2, 4),
            "retrievals": int(g["retrievals"].sum()),
            "memory_hits": int(g["memory_hits"].sum()),
            "hit_rate": round(g["memory_hits"].sum() / max(1, g["retrievals"].sum()
                                                           + g["memory_hits"].sum()), 3),
        })
    out = pd.DataFrame(rows).set_index("arm")

    # savings versus the memoryless baseline, if present
    base = "A0_no_memory"
    if base in out.index:
        b_tok = out.loc[base, "mean_tokens_per_q"]
        b_tpc = out.loc[base, "tokens_per_correct"]
        out["token_saving_%"] = ((b_tok - out["mean_tokens_per_q"]) / b_tok * 100).round(1)
        out["cost_per_correct_saving_%"] = (
            (b_tpc - out["tokens_per_correct"]) / b_tpc * 100).round(1)
    return out


def per_run(df: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- std across repeated runs — the answer to 'is this just noise?'"""
    g = df.groupby(["arm", "run"]).agg(accuracy=("correct", "mean"),
                                       mean_tokens=("tokens", "mean"))
    out = g.groupby("arm").agg(
        runs=("accuracy", "count"),
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        tokens_mean=("mean_tokens", "mean"), tokens_std=("mean_tokens", "std"),
    ).round(3)
    return out


# ---------------------------------------------------------------- statistics
def _binom_two_sided(b: int, c: int) -> float:
    """Exact two-sided binomial p-value for McNemar with n = b + c, p = 0.5."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def mcnemar(df: pd.DataFrame, arm_a: str, arm_b: str) -> dict:
    """Paired test on the SAME questions: does arm_a differ from arm_b?

    b = a correct & b wrong;  c = a wrong & b correct.
    Uses the exact binomial test (correct for the small samples here).
    """
    key = ["session_id", "qid", "run"]
    a = df[df.arm == arm_a].set_index(key)["correct"]
    b = df[df.arm == arm_b].set_index(key)["correct"]
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    n_b = int(((joined.a == 1) & (joined.b == 0)).sum())
    n_c = int(((joined.a == 0) & (joined.b == 1)).sum())
    return {
        "arm_a": arm_a, "arm_b": arm_b, "n_paired": len(joined),
        "a_only_correct": n_b, "b_only_correct": n_c,
        "p_value": round(_binom_two_sided(n_b, n_c), 4),
        "significant_at_05": _binom_two_sided(n_b, n_c) < 0.05,
    }


def pairwise_tests(df: pd.DataFrame, baseline: str = "A0_no_memory") -> pd.DataFrame:
    """Every arm against the baseline, plus cost-aware against each heuristic."""
    arms = list(df["arm"].unique())
    tests = [(baseline, a) for a in arms if a != baseline]
    ca = "A5_cost_aware"
    if ca in arms:
        tests += [(h, ca) for h in arms
                  if h.startswith(("A2", "A3", "A4")) and h != ca]
    return pd.DataFrame([mcnemar(df, a, b) for a, b in tests])


def bootstrap_ci(values, n_boot: int = 10_000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean (no distributional assumption)."""
    v = np.asarray([x for x in values if not pd.isna(x)], dtype=float)
    if len(v) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (round(float(lo), 3), round(float(hi), 3))


def accuracy_with_ci(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm, g in df.groupby("arm", sort=False):
        lo, hi = bootstrap_ci(g["correct"])
        rows.append({"arm": arm, "accuracy": round(g["correct"].mean(), 3),
                     "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows).set_index("arm")


# ------------------------------------------------------------------ RQ3 / H4
def consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Paraphrase-probe agreement: does memory make answers more consistent?"""
    probes = df[df["paraphrase_of"].notna()]
    if probes.empty:
        return pd.DataFrame(columns=["arm", "probes", "consistent_rate"])
    col = "consistent_with_original"
    probes = probes.copy()
    probes[col] = probes[col].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}) \
        if probes[col].dtype == object else probes[col].astype(float)
    return (probes.groupby("arm")
            .agg(probes=(col, "count"), consistent_rate=(col, "mean"))
            .round(3))


def groundedness(traces_dir: str | Path = "results") -> pd.DataFrame:
    """From trace files: how often were verified facts numerically grounded,
    and how often was a memory hit used instead of fresh retrieval?

    Answers the on-thesis question: does reuse degrade evidence grounding?
    """
    rows = []
    for path in sorted(Path(traces_dir).glob("traces_*.jsonl")):
        arm = path.stem.replace("traces_", "")
        steps = []
        for line in path.open():
            rec = json.loads(line)
            steps.extend(rec.get("sub_steps", []))
        if not steps:
            continue
        retrieved = [s for s in steps if s.get("source") == "retrieval"]
        from_mem = [s for s in steps if s.get("source") == "memory"]
        verified = [s for s in retrieved if s.get("verified")]
        grounded = [s for s in verified if s.get("grounded", True)]
        dropped = sum(len(s.get("dropped_chunks", []) or []) for s in retrieved)
        rows.append({
            "arm": arm,
            "sub_steps": len(steps),
            "from_memory": len(from_mem),
            "from_retrieval": len(retrieved),
            "verified_rate": round(len(verified) / len(retrieved), 3) if retrieved else np.nan,
            "grounded_rate": round(len(grounded) / len(verified), 3) if verified else np.nan,
            "chunks_dropped_by_security": dropped,
        })
    return pd.DataFrame(rows).set_index("arm") if rows else pd.DataFrame()
