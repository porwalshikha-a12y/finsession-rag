"""Tests for the pgvector store.

These are SKIPPED automatically when no Postgres is reachable, so the
offline suite still runs anywhere. To run them, start Postgres and set
PGVECTOR_DSN (default: postgresql:///finsession).
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from src.index import PgVectorIndex, _parse_doc_id

DSN = os.environ.get("PGVECTOR_DSN", "postgresql:///finsession")
TABLE = "chunks_test"


class FakeEmbedder:
    """Deterministic 8-dim embeddings — no model download, no network."""

    model_name = "fake"

    @property
    def dim(self) -> int:
        return 8

    def encode(self, texts):
        out = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.standard_normal(8).astype("float32")
            out.append(v / np.linalg.norm(v))
        return np.stack(out)


def _pg_available() -> bool:
    try:
        import psycopg

        with psycopg.connect(DSN, connect_timeout=3):
            return True
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(
    not _pg_available(), reason=f"no Postgres at {DSN}"
)

CHUNKS = [
    {"doc_id": "AAPL_2022_10K", "page": 1, "text": "Apple net sales were 394.3 billion in fiscal 2022."},
    {"doc_id": "AAPL_2021_10K", "page": 7, "text": "Apple net sales were 365.8 billion in fiscal 2021."},
    {"doc_id": "MSFT_2022_10K", "page": 3, "text": "Microsoft revenue was 198.3 billion in fiscal 2022."},
    {"doc_id": "PEP_2022_10K", "page": 9, "text": "PepsiCo net revenue was 86.4 billion in 2022."},
]


@pytest.fixture(scope="module")
def store():
    idx = PgVectorIndex(FakeEmbedder(), dsn=DSN, table=TABLE)
    idx.build(CHUNKS)
    yield idx
    idx.conn.execute(f"DROP TABLE IF EXISTS {TABLE}")
    idx.close()


def test_build_stores_all_chunks(store):
    assert store.count() == len(CHUNKS)


def test_search_returns_exact_match_first(store):
    hits = store.search(CHUNKS[0]["text"], top_k=2)
    assert hits[0]["doc_id"] == "AAPL_2022_10K"
    assert hits[0]["score"] > 0.99          # cosine similarity of identical text
    assert set(hits[0]) >= {"doc_id", "page", "text", "score"}


def test_filter_by_company(store):
    hits = store.search("net sales", top_k=5, company="MSFT")
    assert hits and all(h["doc_id"].startswith("MSFT") for h in hits)


def test_filter_by_year(store):
    hits = store.search("net sales", top_k=5, year=2021)
    assert hits and all("2021" in h["doc_id"] for h in hits)


def test_incremental_insert(store):
    before = store.count()
    store.build([{"doc_id": "JNJ_2022_10K", "page": 2,
                  "text": "Johnson & Johnson sales were 94.9 billion."}],
                recreate=False)
    assert store.count() == before + 1


def test_reload_sees_persisted_rows(store):
    reopened = PgVectorIndex.load(FakeEmbedder(), dsn=DSN, table=TABLE)
    assert reopened.count() == store.count()
    reopened.close()


@pytest.mark.parametrize("doc_id,expected", [
    ("AAPL_2022_10K", ("AAPL", 2022)),
    ("PEP_2015_10K", ("PEP", 2015)),
    ("weird-name", ("weird", None)),
])
def test_parse_doc_id(doc_id, expected):
    assert _parse_doc_id(doc_id) == expected
