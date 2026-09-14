#!/usr/bin/env python3
"""
Run FinSession-RAG experiments — DB version (sessions and results in PostgreSQL).

Usage:
    python -m src.run_experiment                    # Run all arms, all sessions
    python -m src.run_experiment --arm A0_no_memory # Single arm
"""

import json
import sys
import argparse
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

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
        # Get all sessions
        cur.execute("SELECT session_id, company FROM sessions ORDER BY session_id")
        for session_id, company in cur.fetchall():
            # Get queries for this session
            cur.execute(
                "SELECT query_id, question, query_type, expected_answer FROM queries WHERE session_id = %s",
                (session_id,)
            )
            queries = [
                {
                    "query_id": qid,
                    "question": q,
                    "type": qtype,
                    "expected_answer": ans
                }
                for qid, q, qtype, ans in cur.fetchall()
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
            # Count chunks
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

        # Embed query
        query_vec = self.embedder.encode([query])[0]

        # Search in pgvector
        try:
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
                results.append({
                    "id": row[0],
                    "text": row[1],
                    "metadata": {
                        "doc_id": row[2],
                        "page": row[3],
                        "company": row[4],
                        "year": row[5]
                    },
                    "score": float(row[6])
                })
            return results
        except Exception as e:
            print(f"WARNING: pgvector search failed: {e}")
            return []

    def close(self):
        """Close connection."""
        if self._conn:
            self._conn.close()


def make_agent(config: Dict, arm_config: Dict, embedder: Embedder, index: Union[FaissIndex, PgvectorIndex]) -> RAGAgent:
    """Create RAGAgent for an arm."""
    # LLM setup
    llm = LLM(
        model=config["llm"]["model"],
        api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=config["llm"].get("temperature", 0.0),
        max_output_tokens=config["llm"].get("max_output_tokens", 600),
    )

    # Memory setup
    memory = None
    if arm_config.get("memory", False):
        budget = config["memory"]["budgets"][arm_config.get("budget", "tight")]
        eviction_policy_name = arm_config.get("eviction", "none")
        eviction_policy = make_eviction_policy(eviction_policy_name)

        memory = SemanticMemory(
            embedder=embedder,
            budget=budget,  # FIXED: was max_entries
            similarity_threshold=config["memory"]["similarity_threshold"],
            eviction_policy=eviction_policy,
        )

    # Agent
    agent = RAGAgent(
        llm=llm,
        index=index,
        memory=memory,
        top_k=config["retrieval"]["top_k"],
        max_steps=config["agent"]["max_steps"],
        reuse_verification=config["memory"].get("reuse_verification", True),
        security=config["security"].get("enabled", True),
    )

    return agent


def evaluate_query(agent: RAGAgent, query: Dict[str, str]) -> Dict[str, Any]:
    """Evaluate a single query and return result."""
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
        # Run agent
        trace = agent.answer(question)

        # Extract evidence from sub-steps
        evidence_parts = []
        for step in trace.sub_steps:
            if step.get("verified"):
                evidence_parts.append(f"[Evidence for: {step.get('sub_q')}]")

        evidence = " ".join(evidence_parts) if evidence_parts else ""

        # Determine verdict based on answer quality
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
            "tokens_spent": trace.tokens_spent,
            "memory_hits": trace.memory_hits,
            "retrievals": trace.retrievals,
            "sub_steps": trace.sub_steps,
        }

    except Exception as e:
        return {
            "query_id": query_id,
            "question": question,
            "answer": "",
            "evidence": "",
            "verdict": "UNKNOWN",
            "error": str(e)
        }


def save_result_to_db(conn, arm_name: str, session_id: str, result: Dict[str, Any]) -> None:
    """Save one query result to results table."""
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO results
               (arm, session_id, query_id, question, answer, evidence, verdict,
                tokens_spent, memory_hits, retrievals, memory_stats, error)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
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
                result.get("error", "")
            )
        )


def run_arm_on_session(
    agent: RAGAgent,
    arm_name: str,
    session: Dict[str, Any],
    db_conn,
    disable_pbar: bool = False,
) -> Dict[str, Any]:
    """Run an arm on a single session."""
    session_id = session.get("session_id", "unknown")
    queries = session.get("queries", [])

    results = {
        "arm": arm_name,
        "session_id": session_id,
        "queries": []
    }

    pbar_desc = f"{arm_name:20s} | {session_id:30s}"
    for query in tqdm(queries, desc=pbar_desc, disable=disable_pbar):
        result = evaluate_query(agent, query)
        results["queries"].append(result)

        # Save to DB immediately
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
        help="Run only this arm (e.g., A0_no_memory)"
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

    # Load .env
    load_dotenv(PROJECT_ROOT / ".env")
    print(f"  Loading .env from: {PROJECT_ROOT / '.env'}")

    # Setup
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

    # Connect to database
    try:
        db_conn = psycopg.connect("postgresql:///finsession")
    except Exception as e:
        print(f"ERROR: Failed to connect to PostgreSQL: {e}")
        sys.exit(1)

    # Load config and sessions from DB
    config = load_config(config_path)
    sessions = load_sessions_from_db(db_conn)
    all_arms = config.get("arms", [])

    # Filter by args
    if args.arm:
        all_arms = [a for a in all_arms if a["name"] == args.arm]
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
    print(f"Total evaluations: {len(all_arms) * len(sessions_to_run) * 4}")
    print()

    # Load embeddings and pgvector index
    print("Loading embeddings and index...")
    embedder = Embedder(model=config["embeddings"]["model"])

    # Determine backend
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

    print(f"  ✓ Index loaded: {index.count()} chunks\n")

    # Run experiments
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        print("WARNING: OPENAI_API_KEY not set. Queries will fail.")
        print("Set it with: export OPENAI_API_KEY=sk-...")
        print()

    for arm_config in all_arms:
        arm_name = arm_config["name"]
        print(f"\n{'='*80}")
        print(f"ARM: {arm_name}")
        print(f"{'='*80}")
        print(f"  Memory: {arm_config.get('memory', False)}")
        if arm_config.get("memory"):
            print(f"  Budget: {arm_config.get('budget', 'tight')}")
            print(f"  Eviction: {arm_config.get('eviction', 'none')}")

        # Create agent for this arm
        agent = make_agent(config, arm_config, embedder, index)

        # Run on each session
        arm_results = []
        for session_id, session in sessions_to_run.items():
            result = run_arm_on_session(agent, arm_name, session, db_conn, disable_pbar=False)
            arm_results.append(result)
            all_results.append(result)

        # Aggregate stats for arm
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

    # Save results to JSON as well (for compatibility)
    results_file = output_dir / "results_all.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Clean up
    if isinstance(index, PgvectorIndex):
        index.close()
    db_conn.close()

    print(f"\n✅ Results saved to PostgreSQL and {results_file}")
    print(f"\nQuery results: SELECT * FROM results WHERE arm = 'A0_no_memory' LIMIT 10;")


if __name__ == "__main__":
    main()
