#!/usr/bin/env python3
import json
import sys
from pathlib import Path

try:
    import psycopg
except ImportError:
    print("ERROR: psycopg required. Install: pip install psycopg[binary]")
    sys.exit(1)

PROJECT_ROOT = Path.cwd()
SESSIONS_DIR = PROJECT_ROOT / "data" / "sessions"
GOLD_FILE = PROJECT_ROOT / "data" / "gold_answers.json"


def load_gold() -> dict:
    """Curated gold answers, keyed by query_id.

    Kept in their own file rather than inside the session JSON so the
    reviewed gold set stays a separate, auditable artifact: each entry
    carries the filing and page it came from. Missing file is not fatal —
    the migration just loads NULL gold, as it did before.
    """
    if not GOLD_FILE.exists():
        print(f"WARNING: {GOLD_FILE.name} not found - expected_answer will be NULL")
        return {}
    gold = json.loads(GOLD_FILE.read_text())
    print(f"Loaded {len(gold)} gold answers from {GOLD_FILE.name}")
    return gold


def migrate():
    if not SESSIONS_DIR.exists():
        print(f"ERROR: {SESSIONS_DIR} not found")
        sys.exit(1)

    try:
        conn = psycopg.connect("postgresql:///finsession")
    except Exception as e:
        print(f"ERROR: Failed to connect to PostgreSQL: {e}")
        sys.exit(1)

    gold = load_gold()

    with conn.cursor() as cur:
        # paraphrase_of links a probe to the query it rephrases, so RQ3's
        # consistency check can compare the two answers. Added here rather
        # than in create_tables.sql so an existing database upgrades itself.
        cur.execute("ALTER TABLE queries ADD COLUMN IF NOT EXISTS paraphrase_of TEXT")
        session_files = sorted(SESSIONS_DIR.glob("*.json"))
        print(f"Found {len(session_files)} session files\n")

        for session_file in session_files:
            with open(session_file) as f:
                session_data = json.load(f)

            session_id = session_data.get("session_id", session_file.stem)
            company = session_data.get("company", "Unknown")
            queries = session_data.get("queries", [])

            try:
                cur.execute(
                    "INSERT INTO sessions (session_id, company, query_count) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                    (session_id, company, len(queries))
                )
                print(
                    f"✓ Session: {session_id} ({company}, {len(queries)} queries)")
            except Exception as e:
                print(f"✗ Session {session_id}: {e}")
                continue

            for query in queries:
                try:
                    cur.execute(
                        """INSERT INTO queries (query_id, session_id, question, query_type,
                                                expected_answer, expected_answer_marker, paraphrase_of)
                           VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING""",
                        # NB: the session JSON uses "query_type", not "type".
                        # Reading the wrong key silently wrote NULL for every row.
                        (query.get("query_id"), session_id, query.get("question"),
                         query.get("query_type") or query.get("type"),
                         # gold_answers.json wins; fall back to anything
                         # carried in the session file itself.
                         (gold.get(query.get("query_id"), {}).get("expected_answer")
                          or query.get("expected_answer")
                          or query.get("gold_answer")),
                         query.get("expected_answer_marker"),
                         query.get("paraphrase_of"))
                    )
                except Exception as e:
                    print(f"  ✗ Query {query.get('query_id')}: {e}")

        conn.commit()

        cur.execute("SELECT count(*), count(expected_answer), count(query_type), count(paraphrase_of) FROM queries")
        total, with_gold, with_type, with_para = cur.fetchone()
        print(f"\nqueries: {total} | with gold answer: {with_gold} | "
              f"with query_type: {with_type} | paraphrase probes: {with_para}")
        if with_gold < total:
            missing = total - with_gold
            print(f"WARNING: {missing} queries have no gold answer")

    conn.close()
    print(f"\n✅ Migration complete!")


if __name__ == "__main__":
    migrate()
