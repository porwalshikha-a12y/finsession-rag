"""Run the experiment grid: arms x sessions -> results/*.csv + traces.

Usage:
    python -m src.run_experiment                # all arms from config.yaml
    python -m src.run_experiment --arm A0_no_memory
Requires: a built index (python -m scripts.build_index) and session files
in data/sessions/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from .agent import RAGAgent
from .eviction import make_policy
from .index import Embedder, make_store
from .judge import judge_answer, judge_consistency
from .llm import LLM
from .memory import SemanticMemory
from .sessions import load_sessions


def build_memory(arm: dict, cfg: dict, embedder: Embedder) -> SemanticMemory | None:
    if not arm.get("memory"):
        return None
    eviction = arm.get("eviction", "none")
    policy = None if eviction in (None, "none") else make_policy(eviction)
    budget = None
    if "budget" in arm:
        budget = cfg["memory"]["budgets"][arm["budget"]]
    return SemanticMemory(
        embedder=embedder,
        eviction_policy=policy,
        budget=budget,
        similarity_threshold=cfg["memory"]["similarity_threshold"],
    )


def run_arm(arm: dict, cfg: dict, index, embedder: Embedder,
            sessions: list[dict], out_dir: Path) -> pd.DataFrame:
    rows = []
    traces_path = out_dir / f"traces_{arm['name']}.jsonl"
    answers_by_qid: dict[tuple, str] = {}

    with traces_path.open("w") as tf:
        for session in sessions:
            llm = LLM(**cfg["llm"])  # fresh ledger per session
            memory = build_memory(arm, cfg, embedder)
            agent = RAGAgent(
                llm=llm, index=index, memory=memory,
                top_k=cfg["retrieval"]["top_k"],
                max_steps=cfg["agent"]["max_steps"],
                reuse_verification=cfg["memory"]["reuse_verification"],
                security=cfg.get("security", {}).get("enabled", True),
            )
            for q in session["queries"]:
                trace = agent.answer(q["question"])
                answers_by_qid[(session["session_id"], q["qid"])] = trace.answer
                rows.append({
                    "arm": arm["name"],
                    "session_id": session["session_id"],
                    "qid": q["qid"],
                    "type": q.get("type", ""),
                    "paraphrase_of": q.get("paraphrase_of"),
                    "question": q["question"],
                    "gold": q.get("gold_answer", ""),
                    "answer": trace.answer,
                    "tokens": trace.tokens_spent,
                    "retrievals": trace.retrievals,
                    "memory_hits": trace.memory_hits,
                })
                tf.write(json.dumps({
                    "arm": arm["name"], "session_id": session["session_id"],
                    "qid": q["qid"], "sub_steps": trace.sub_steps,
                }) + "\n")
            if memory is not None:
                rows[-1]["memory_stats"] = json.dumps(memory.stats.snapshot())

    df = pd.DataFrame(rows)

    # ---- scoring (judge + consistency probes) ----
    judge_llm = LLM(**cfg["llm"])
    correct, consistent = [], []
    for _, r in df.iterrows():
        ok = judge_answer(judge_llm, r["question"], r["gold"], r["answer"]) if r["gold"] else None
        correct.append(ok)
        if r["paraphrase_of"]:
            orig = answers_by_qid.get((r["session_id"], r["paraphrase_of"]))
            consistent.append(
                judge_consistency(judge_llm, orig, r["answer"]) if orig else None
            )
        else:
            consistent.append(None)
    df["correct"] = correct
    df["consistent_with_original"] = consistent
    df["judge_tokens"] = judge_llm.ledger.total_tokens
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--arm", default=None, help="run a single arm by name")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out_dir = Path(cfg["paths"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    embedder = Embedder(**cfg["embeddings"])
    index = make_store(cfg, embedder)
    sessions = load_sessions(cfg["paths"]["sessions_dir"])
    print(f"{len(sessions)} sessions, {sum(len(s['queries']) for s in sessions)} queries")

    arms = cfg["arms"] if args.arm is None else [a for a in cfg["arms"] if a["name"] == args.arm]
    all_dfs = []
    for arm in arms:
        print(f"\n=== arm: {arm['name']} ===")
        df = run_arm(arm, cfg, index, embedder, sessions, out_dir)
        df.to_csv(out_dir / f"results_{arm['name']}.csv", index=False)
        acc = df["correct"].dropna().mean() if df["correct"].notna().any() else float("nan")
        print(f"accuracy={acc:.3f}  mean_tokens/query={df['tokens'].mean():.0f}  "
              f"memory_hits={df['memory_hits'].sum()}")
        all_dfs.append(df)

    pd.concat(all_dfs).to_csv(out_dir / "results_all.csv", index=False)
    print(f"\nwrote {out_dir}/results_all.csv")


if __name__ == "__main__":
    main()
