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


def migrate():
    if not SESSIONS_DIR.exists():
        print(f"ERROR: {SESSIONS_DIR} not found")
        sys.exit(1)

    try:
        conn = psycopg.connect("postgresql:///finsession")
    except Exception as e:
        print(f"ERROR: Failed to connect to PostgreSQL: {e}")
        sys.exit(1)

    with conn.cursor() as cur:
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
                        """INSERT INTO queries (query_id, session_id, question, query_type, expected_answer, expected_answer_marker)
                           VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING""",
                        (query.get("query_id"), session_id, query.get("question"),
                         query.get("type"), query.get("expected_answer"), query.get("expected_answer_marker"))
                    )
                except Exception as e:
                    print(f"  ✗ Query {query.get('query_id')}: {e}")

        conn.commit()
    conn.close()
    print(f"\n✅ Migration complete!")


if __name__ == "__main__":
    migrate()
