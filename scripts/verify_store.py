"""Prove that retrieval hits PostgreSQL/pgvector, not local files.

Evidence collected:
 1. which store class the factory returns (and its DSN)
 2. Postgres server-side scan counters BEFORE and AFTER a search — these
    live inside the database, so they can only move if the query really
    reached Postgres
 3. the SQL query plan, showing the pgvector index/table being read
 4. a sample retrieved chunk with its provenance

Usage:  python -m scripts.verify_store
"""
from __future__ import annotations

import yaml

from src.index import Embedder, FaissIndex, PgVectorIndex, make_store

cfg = yaml.safe_load(open("config.yaml"))
print("=" * 68)
print("1. CONFIGURED BACKEND")
print("=" * 68)
print("   config.vector_store :", cfg.get("vector_store"))

embedder = Embedder(**cfg["embeddings"])
store = make_store(cfg, embedder)
print("   store class         :", type(store).__name__)
print("   module              :", type(store).__module__)

if isinstance(store, FaissIndex):
    print("\n   >>> This is the FAISS in-process store (reads data/index/ files).")
    raise SystemExit(1)

assert isinstance(store, PgVectorIndex)
print("   DSN                 :", store.dsn)
print("   table               :", store.table)
print("   rows in table       :", store.count())

# --- 2. server-side counters -------------------------------------------
def scan_stats() -> tuple[int, int]:
    with store.conn.cursor() as cur:
        cur.execute(
            "SELECT coalesce(seq_scan,0), coalesce(seq_tup_read,0) "
            "FROM pg_stat_user_tables WHERE relname = %s", (store.table,)
        )
        row = cur.fetchone()
    return (row[0], row[1]) if row else (0, 0)


print()
print("=" * 68)
print("2. POSTGRES SERVER-SIDE COUNTERS (these live inside the database)")
print("=" * 68)
before = scan_stats()
print(f"   before search: seq_scan={before[0]}  tuples_read={before[1]}")

QUERY = "What was Apple's total net sales in fiscal 2022?"
hits = store.search(QUERY, top_k=3)

# stats collector updates asynchronously; ask Postgres to flush
with store.conn.cursor() as cur:
    cur.execute("SELECT pg_stat_force_next_flush()")
after = scan_stats()
print(f"   after  search: seq_scan={after[0]}  tuples_read={after[1]}")
moved = after[1] > before[1] or after[0] > before[0]
print(f"   counters moved: {moved}   <-- proof the query executed in Postgres"
      if moved else
      "   counters unchanged (index-only scan may not bump seq_scan; see plan below)")

# --- 3. query plan ------------------------------------------------------
print()
print("=" * 68)
print("3. QUERY PLAN (what Postgres actually did)")
print("=" * 68)
qv = embedder.encode([QUERY])[0]
with store.conn.cursor() as cur:
    cur.execute(
        f"EXPLAIN ANALYZE SELECT doc_id, page, 1 - (embedding <=> %s) AS score "
        f"FROM {store.table} ORDER BY embedding <=> %s LIMIT 3", (qv, qv)
    )
    for line in cur.fetchall():
        print("   ", line[0])

# --- 4. what came back --------------------------------------------------
print()
print("=" * 68)
print("4. RETRIEVED CHUNKS")
print("=" * 68)
print(f"   query: {QUERY!r}")
for h in hits:
    print(f"   [{h['score']:.3f}] {h['doc_id']} p.{h['page']}: {h['text'][:90]}...")

print()
print("=" * 68)
print("CONCLUSION: retrieval is served by PostgreSQL + pgvector.")
print("No FAISS index file is read; data/index/ need not exist.")
print("=" * 68)
