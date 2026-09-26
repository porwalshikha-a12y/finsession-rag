#!/usr/bin/env python3
"""
Run FinSession-RAG experiments — DB version with better error handling.
"""

import json
import sys
import argparse
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import traceback

import yaml
from tqdm import tqdm
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.index import Embedder, FaissIndex
from src.llm import LLM, TokenLedger
from src.agent import RAGAgent
from src.memory import SemanticMemory
from src.eviction import make_eviction_policy
from src.judge import judge_answer, judge_consistency

# For pgvector support
try:
    import psycopg
    HAS_PSYCOPG = True
except ImportError:
    HAS_PSYCOPG = False


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load config.yaml."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_sessions_from_db(conn) -> Dict[str, Dict[str, Any]]:
    """Load sessions and queries from PostgreSQL."""
    sessions = {}
    with conn.cursor() as cur:
        cur.execute("SELECT session_id, company FROM sessions ORDER BY session_id")
        for session_id, company in cur.fetchall():
            cur.execute(
                "SELECT query_id, question, query_type, expected_answer, paraphrase_of "
                "FROM queries WHERE session_id = %s ORDER BY query_id",
                (session_id,)
            )
            queries = [
                {
                    "query_id": qid,
                    "question": q,
                    "type": qtype,
                    "expected_answer": ans,
                    "paraphrase_of": para,
                }
                for qid, q, qtype, ans, para in cur.fetchall()
            ]
            sessions[session_id] = {
                "session_id": session_id,
                "company": company,
                "queries": queries
            }
    return sessions


class PgvectorIndex:
    """Pgvector index wrapper for RAGAgent."""

    def __init__(self, embedder: Embedder, dsn: str, table: str):
        self.embedder = embedder
        self.dsn = dsn
        self.table = table
        self._conn = None
        self._chunk_count = None

    def connect(self):
        """Connect to PostgreSQL."""
        if not HAS_PSYCOPG:
            raise ImportError("psycopg is required for pgvector backend. Install: pip install psycopg[binary]")
        try:
            self._conn = psycopg.connect(self.dsn)
            with self._conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self.table}")
                self._chunk_count = cur.fetchone()[0]
        except Exception as e:
            raise RuntimeError(f"Failed to connect to pgvector at {self.dsn}: {e}")

    def count(self) -> int:
        """Return number of chunks in index."""
        if self._chunk_count is None:
            self.connect()
        return self._chunk_count

    def search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Search by query text (embed then search by vector)."""
        if not self._conn:
            self.connect()

        try:
            # Embed query
            query_vec = self.embedder.encode([query])[0]

            # Search in pgvector
            with self._conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id, text, doc_id, page, company, year, 1 - (embedding <=> %s::vector) as similarity
                    FROM {self.table}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_vec.tolist(), query_vec.tolist(), top_k)
                )
                rows = cur.fetchall()

            results = []
            for row in rows:
                try:
                    result = {
                        "id": row[0],
                        "text": row[1],
                        "metadata": {
                            "doc_id": row[2],
                            "page": row[3],
                            "company": row[4],
                            "year": row[5]
                        },
                        "score": float(row[6])
                    }
                    results.append(result)
                except (IndexError, TypeError) as e:
                    print(f"ERROR parsing row: {row}")
                    print(f"Error: {e}")
                    traceback.print_exc()
                    continue

            return results
        except Exception as e:
            print(f"ERROR in pgvector search: {e}")
            traceback.print_exc()
            return []

    def close(self):
        """Close connection."""
        if self._conn:
            self._conn.close()



# ---------------------------------------------------------------- hybrid
class _FlatDenseAdapter:
    """Presents the runner's PgvectorIndex output in the flat shape RRF needs.

    PgvectorIndex nests provenance under "metadata"; reciprocal_rank_fusion
    keys on top-level doc_id/page. Adapting here means the hybrid arm reuses
    the exact query path that has already served every other arm, instead of
    introducing a second, unproven database path.
    """

    def __init__(self, inner):
        self.inner = inner

    def search(self, query: str, top_k: int = 4, company=None, year=None) -> List[Dict]:
        # company/year are accepted for interface compatibility; the runner's
        # index does not support metadata filtering, so they are ignored.
        out = []
        for h in self.inner.search(query, top_k=top_k):
            m = h.get("metadata") or {}
            out.append({
                "doc_id": m.get("doc_id", h.get("doc_id")),
                "page": m.get("page", h.get("page")),
                "text": h.get("text", ""),
                "score": float(h.get("score", 0.0) or 0.0),
                "company": m.get("company"),
                "year": m.get("year"),
            })
        return out


def build_bm25_store(conn, table: str):
    """Build the BM25 index from the chunk corpus held in PostgreSQL.

    BM25 is computed in-process (rank_bm25's BM25Okapi) over text read from
    Postgres. Postgres' own full-text search ranks with ts_rank, which is NOT
    BM25 — using it would make the 'BM25 + dense via RRF' claim inaccurate.
    At this corpus size the build costs a few seconds.
    """
    from src.hybrid import BM25Store
    with conn.cursor() as cur:
        cur.execute(f"SELECT doc_id, page, text FROM {table}")
        chunks = [{"doc_id": r[0], "page": r[1], "text": r[2] or ""}
                  for r in cur.fetchall()]
    store = BM25Store()
    store.build(chunks)
    return store, len(chunks)


class HybridIndex:
    """Dense (pgvector) + sparse (BM25) retrieval merged by RRF."""

    def __init__(self, dense_index, bm25_store, rrf_k: int = 60):
        from src.hybrid import HybridRetriever
        self._dense = dense_index
        self.retriever = HybridRetriever(_FlatDenseAdapter(dense_index),
                                         bm25_store, rrf_k=rrf_k)

    def search(self, query: str, top_k: int = 4) -> List[Dict]:
        return self.retriever.search(query, top_k=top_k)

    def count(self) -> int:
        return self._dense.count()


def make_agent(config: Dict, arm_config: Dict, embedder: Embedder,
               index: Union[FaissIndex, PgvectorIndex],
               hybrid_index=None) -> RAGAgent:
    """Create RAGAgent for an arm."""
    llm = LLM(
        model=config["llm"]["model"],
        api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=config["llm"].get("temperature", 0.0),
        max_output_tokens=config["llm"].get("max_output_tokens", 600),
    )

    memory = None
    if arm_config.get("memory", False):
        budget_name = arm_config.get("budget", "tight")
        budget = None if budget_name is None else config["memory"]["budgets"][budget_name]
        eviction_policy_name = arm_config.get("eviction", "none")
        eviction_policy = make_eviction_policy(
            eviction_policy_name,
            embedder=embedder,                      # redundancy needs it
            half_life=config["memory"].get("cost_aware", {}).get("half_life", 10),
        )

        memory = SemanticMemory(
            embedder=embedder,
            budget=budget,
            similarity_threshold=config["memory"]["similarity_threshold"],
            eviction_policy=eviction_policy,
        )

    # Per-arm retrieval backend. This is the line whose absence meant
    # A6's "retrieval: hybrid" was silently ignored.
    arm_index = index
    if arm_config.get("retrieval") == "hybrid":
        if hybrid_index is None:
            raise RuntimeError(
                f"arm {arm_config.get('name')} requests hybrid retrieval but no "
                "hybrid index was built")
        arm_index = hybrid_index

    agent = RAGAgent(
        llm=llm,
        index=arm_index,
        memory=memory,
        top_k=config["retrieval"]["top_k"],
        max_steps=config["agent"]["max_steps"],
        # per-arm override so an ablation arm can disable the gate while every
        # other arm keeps it; falls back to the global default.
        reuse_verification=arm_config.get(
            "reuse_verification", config["memory"].get("reuse_verification", True)),
        security=config["security"].get("enabled", True),
    )

    return agent


def _judge(judge_llm, question: str, query: Dict[str, str],
           answer: str) -> Optional[bool]:
    """LLM-as-judge vs the gold answer; None when there is no gold to judge.

    The judge runs on its own LLM instance so grading tokens never land in
    the arm's ledger and inflate the cost metric.
    """
    gold = query.get("expected_answer") or query.get("gold_answer")
    if not gold or judge_llm is None or not answer:
        return None
    try:
        return judge_answer(judge_llm, question, gold, answer)
    except Exception as e:
        print(f"WARNING: judge failed for {query.get('query_id')}: {e}")
        return None


def evaluate_query(agent: RAGAgent, query: Dict[str, str],
                   judge_llm: Optional[LLM] = None) -> Dict[str, Any]:
    """Evaluate a single query and return result.

    `verdict` is a cheap shape heuristic on the answer text (did the model
    produce something and not declare failure) — it is NOT a correctness
    measure. `judged_correct` is the correctness measure: an LLM judge
    comparing the answer against the query's expected_answer. It is None
    when no gold answer exists for the query, so the two never get
    conflated in analysis."""
    query_id = query.get("query_id", "unknown")
    question = query.get("question", "")

    if not question:
        return {
            "query_id": query_id,
            "question": question,
            "answer": "",
            "evidence": "",
            "verdict": "UNKNOWN",
            "error": "Empty question"
        }

    try:
        trace = agent.answer(question)

        # Audit trail. Memory-served facts count as evidence too — they carry
        # the provenance of the retrieval that originally produced them — so a
        # cached answer is as auditable as a freshly retrieved one.
        evidence_parts = []
        for step in trace.sub_steps:
            if not step.get("verified"):
                continue
            prov = step.get("provenance") or []
            cites = ", ".join(f"{d} p.{pg}" for d, pg in
                              dict.fromkeys(tuple(x) for x in prov)) or "no provenance"
            if step.get("source") == "memory":
                evidence_parts.append(
                    f"[CACHED from \"{step.get('stored_q')}\" | {cites}]")
            else:
                evidence_parts.append(f"[{step.get('sub_q')} | {cites}]")

        evidence = " ".join(evidence_parts) if evidence_parts else ""

        if not trace.answer or len(trace.answer) < 10:
            verdict = "NOT_SUPPORTED"
        elif "could not" in trace.answer.lower() or "missing" in trace.answer.lower():
            verdict = "PARTIAL"
        else:
            verdict = "SUPPORTED"

        return {
            "query_id": query_id,
            "question": question,
            "answer": trace.answer,
            "evidence": evidence,
            "verdict": verdict,
            "judged_correct": _judge(judge_llm, question, query, trace.answer),
            # snapshot after this query: lookups/hits/writes/evictions so far
            # in this session. Without it the eviction column is always empty
            # and RQ1 has no evidence that any eviction happened at all.
            "memory_stats": (agent.memory.stats.snapshot() | {"entries": len(agent.memory.entries)}
                             if agent.memory is not None else {}),
            "tokens_spent": trace.tokens_spent,
            "memory_hits": trace.memory_hits,
            "reuse_rejected": trace.reuse_rejected,
            "planner_loops": trace.planner_loops,
            "retrievals": trace.retrievals,
            "sub_steps": trace.sub_steps,
        }

    except Exception as e:
        print(f"ERROR in evaluate_query for {query_id}: {e}")
        traceback.print_exc()
        return {
            "query_id": query_id,
            "question": question,
            "answer": "",
            "evidence": "",
            "verdict": "UNKNOWN",
            "error": str(e)
        }


def ensure_results_columns(conn) -> None:
    """Add metric columns the original results table never had.

    judged_correct (accuracy vs gold), consistent_with_original (RQ3
    paraphrase consistency) and reuse_rejected (candidates the reuse gate
    refused) were computed but never persisted, so any analysis run against
    Postgres rather than results_all.json silently lost all three.
    """
    with conn.cursor() as cur:
        cur.execute("ALTER TABLE results ADD COLUMN IF NOT EXISTS judged_correct BOOLEAN")
        cur.execute("ALTER TABLE results ADD COLUMN IF NOT EXISTS consistent_with_original BOOLEAN")
        cur.execute("ALTER TABLE results ADD COLUMN IF NOT EXISTS reuse_rejected INT")
        cur.execute("ALTER TABLE results ADD COLUMN IF NOT EXISTS planner_loops INT")
    conn.commit()


def save_result_to_db(conn, arm_name: str, session_id: str, result: Dict[str, Any]) -> None:
    """Save one query result to results table."""
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO results
               (arm, session_id, query_id, question, answer, evidence, verdict,
                tokens_spent, memory_hits, retrievals, memory_stats, error,
                judged_correct, consistent_with_original, reuse_rejected, planner_loops)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                arm_name,
                session_id,
                result.get("query_id"),
                result.get("question"),
                result.get("answer"),
                result.get("evidence"),
                result.get("verdict"),
                result.get("tokens_spent", 0),
                result.get("memory_hits", 0),
                result.get("retrievals", 0),
                json.dumps(result.get("memory_stats", {})),
                result.get("error", ""),
                result.get("judged_correct"),
                result.get("consistent_with_original"),
                result.get("reuse_rejected", 0),
                result.get("planner_loops", 0),
            )
        )


def run_arm_on_session(
    agent: RAGAgent,
    arm_name: str,
    session: Dict[str, Any],
    db_conn,
    disable_pbar: bool = False,
    judge_llm: Optional[LLM] = None,
) -> Dict[str, Any]:
    """Run an arm on a single session."""
    session_id = session.get("session_id", "unknown")
    queries = session.get("queries", [])

    results = {
        "arm": arm_name,
        "session_id": session_id,
        "queries": []
    }

    # The agent (and its memory) is built once per ARM and reused across every
    # session, so without this reset one session's facts stay live in the next
    # one — cross-company leakage, inflated hit rates, and sessions that are no
    # longer independent observations for the paired tests. Memory is
    # session-scoped in this study; enforce it here.
    if agent.memory is not None:
        agent.memory.reset()

    pbar_desc = f"{arm_name:20s} | {session_id:30s}"
    answers_so_far: Dict[str, str] = {}
    for query in tqdm(queries, desc=pbar_desc, disable=disable_pbar):
        result = evaluate_query(agent, query, judge_llm=judge_llm)

        # RQ3: a paraphrase should resolve to the same fact as the query it
        # rephrases. Both were asked in THIS session, so memory was live for
        # the second one — which is exactly what makes this a memory test.
        src = query.get("paraphrase_of")
        result["consistent_with_original"] = None
        if src and judge_llm is not None:
            original = answers_so_far.get(src)
            if original and result.get("answer"):
                try:
                    result["consistent_with_original"] = judge_consistency(
                        judge_llm, original, result["answer"])
                except Exception as e:
                    print(f"WARNING: consistency judge failed for {query.get('query_id')}: {e}")

        answers_so_far[query.get("query_id")] = result.get("answer", "")
        results["queries"].append(result)

        save_result_to_db(db_conn, arm_name, session_id, result)
        db_conn.commit()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run FinSession-RAG experiments (DB version)"
    )
    parser.add_argument(
        "--arm",
        type=str,
        default=None,
        help="Run only these arms; comma-separated (e.g. A7_no_reuse_gate,A8_lru_b5)"
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Run only this session (e.g., apple_fy2022_basics)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: results/)"
    )

    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    print(f"  Loading .env from: {PROJECT_ROOT / '.env'}")

    config_path = PROJECT_ROOT / "config.yaml"
    output_dir = args.output or (PROJECT_ROOT / "results")

    if not config_path.exists():
        print(f"ERROR: {config_path} not found")
        sys.exit(1)

    print("=" * 80)
    print("FINSESSION-RAG: Run Experiments (DB Version)")
    print("=" * 80)
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    print()

    try:
        db_conn = psycopg.connect("postgresql:///finsession")
        ensure_results_columns(db_conn)
    except Exception as e:
        print(f"ERROR: Failed to connect to PostgreSQL: {e}")
        sys.exit(1)

    config = load_config(config_path)
    sessions = load_sessions_from_db(db_conn)
    all_arms = config.get("arms", [])

    if args.arm:
        wanted = [x.strip() for x in args.arm.split(",") if x.strip()]
        all_arms = [a for a in all_arms if a["name"] in wanted]
        if not all_arms:
            print(f"ERROR: Arm '{args.arm}' not found in config")
            sys.exit(1)

    sessions_to_run = sessions
    if args.session:
        if args.session not in sessions:
            print(f"ERROR: Session '{args.session}' not found")
            sys.exit(1)
        sessions_to_run = {args.session: sessions[args.session]}

    print(f"Arms to run: {len(all_arms)}")
    for arm in all_arms:
        print(f"  - {arm['name']}")
    print(f"Sessions to run: {len(sessions_to_run)}")
    print()

    print("Loading embeddings and index...")
    embedder = Embedder(model=config["embeddings"]["model"])

    vector_store_config = config.get("vector_store", {})
    backend = vector_store_config.get("backend", "faiss").lower()

    if backend == "pgvector":
        dsn = vector_store_config.get("dsn") or os.environ.get("PGVECTOR_DSN", "postgresql:///finsession")
        table = vector_store_config.get("table", "chunks")
        print(f"  Using pgvector backend: {dsn}")
        index = PgvectorIndex(embedder, dsn, table)
        index.connect()
    else:
        print(f"  Using FAISS backend")
        index_dir = PROJECT_ROOT / "data" / "index"
        if not index_dir.exists():
            print(f"ERROR: {index_dir} not found. Run: python -m scripts.build_index")
            sys.exit(1)
        index = FaissIndex.load(index_dir, embedder)

    print(f"  ✓ Index loaded: {index.count()} chunks")

    # Build BM25 only when a selected arm actually uses hybrid retrieval.
    hybrid_index = None
    if any(a.get("retrieval") == "hybrid" for a in all_arms):
        vs = config.get("vector_store", {})
        rrf_k = vs.get("rrf_k", 60)
        print("  Building BM25 index from the pgvector corpus...")
        bm25_store, n_chunks = build_bm25_store(db_conn, vs.get("table", "chunks"))
        hybrid_index = HybridIndex(index, bm25_store, rrf_k=rrf_k)
        print(f"  ✓ BM25 index built: {n_chunks} chunks (RRF k={rrf_k})")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        print("WARNING: OPENAI_API_KEY not set. Queries will fail.")
        print()

    # One judge for the whole grid, on its own ledger so grading tokens are
    # never counted as an arm's retrieval/answer cost.
    judge_llm = LLM(
        model=config["llm"]["model"],
        api_key=api_key,
        temperature=0.0,
        max_output_tokens=config["llm"].get("max_output_tokens", 600),
    )

    for arm_config in all_arms:
        arm_name = arm_config["name"]
        print(f"\n{'='*80}")
        print(f"ARM: {arm_name}")
        print(f"{'='*80}")
        print(f"  Memory: {arm_config.get('memory', False)}")
        if arm_config.get("memory"):
            print(f"  Budget: {arm_config.get('budget', 'tight')}")
            print(f"  Eviction: {arm_config.get('eviction', 'none')}")

        agent = make_agent(config, arm_config, embedder, index,
                           hybrid_index=hybrid_index)

        arm_results = []
        for session_id, session in sessions_to_run.items():
            result = run_arm_on_session(agent, arm_name, session, db_conn,
                                        disable_pbar=False, judge_llm=judge_llm)
            arm_results.append(result)
            all_results.append(result)

        total_queries = sum(len(r["queries"]) for r in arm_results)
        total_supported = sum(
            1 for r in arm_results for q in r["queries"]
            if q.get("verdict") == "SUPPORTED"
        )
        total_tokens = sum(
            q.get("tokens_spent", 0) for r in arm_results for q in r["queries"]
        )
        total_hits = sum(
            q.get("memory_hits", 0) for r in arm_results for q in r["queries"]
        )

        print(f"\n  Summary:")
        print(f"    Queries: {total_queries}")
        print(f"    Supported: {total_supported}/{total_queries} ({100*total_supported/total_queries:.1f}%)")
        print(f"    Tokens: {total_tokens:,}")
        print(f"    Memory hits: {total_hits}")

    results_file = output_dir / "results_all.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    if isinstance(index, PgvectorIndex):
        index.close()
    db_conn.close()

    print(f"\n✅ Results saved to PostgreSQL and {results_file}")


if __name__ == "__main__":
    main()
