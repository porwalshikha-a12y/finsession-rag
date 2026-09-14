"""Embeddings + vector store.

Embeddings are computed locally with sentence-transformers (free, offline).
Three interchangeable stores are provided:

* PgVectorIndex  - PostgreSQL + pgvector (default). A real database: the
  corpus persists in Postgres, supports metadata filtering (company/year)
  and incremental inserts.
* HybridPgVectorIndex - pgvector + BM25 via Reciprocal Rank Fusion. Combines
  dense semantic retrieval with sparse keyword matching for financial text.
* FaissIndex    - the original in-process FAISS index, kept as a fallback
  and for offline runs where no database is available.

All expose the same interface used by the agent:
    build(chunks)                       -> index a corpus
    search(query, top_k, where=None)    -> [{doc_id, page, text, score}]
so nothing else in the project changes when you swap stores.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


# --------------------------------------------------------------- embeddings
class Embedder:
    # Loaded models are cached so the (slow) load happens once per process.
    _cache: dict = {}

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", batch_size: int = 32,
                 model_name: str | None = None):
        # `model` matches the key in config.yaml (embeddings.model);
        # `model_name` is accepted as an alias for older call sites.
        name = model_name or model
        if name not in self._cache:
            from sentence_transformers import SentenceTransformer  # lazy import

            self._cache[name] = SentenceTransformer(name)
        self.model = self._cache[name]
        self.model_name = name
        self.batch_size = batch_size

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,      # so cosine == inner product
            show_progress_bar=len(texts) > 256,
        )
        return np.asarray(vecs, dtype="float32")


# ------------------------------------------------------------ pgvector store
class PgVectorIndex:
    """PostgreSQL + pgvector store.

    Vectors are normalized, so cosine distance (<=>) is the right operator
    and `score = 1 - distance` is cosine similarity, matching FAISS's inner
    product exactly. Same numbers, different storage engine.
    """

    def __init__(self, embedder: Embedder, dsn: str | None = None,
                 table: str = "chunks"):
        self.embedder = embedder
        self.table = table
        self.dsn = dsn or os.environ.get(
            "PGVECTOR_DSN", "postgresql:///finsession"
        )
        self._conn = None

    # -- connection ------------------------------------------------------
    @property
    def conn(self):
        if self._conn is None or self._conn.closed:
            import psycopg
            from pgvector.psycopg import register_vector

            self._conn = psycopg.connect(self.dsn, autocommit=True)
            self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            register_vector(self._conn)
        return self._conn

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()

    # -- build -----------------------------------------------------------
    def build(self, chunks: list[dict], recreate: bool = True) -> None:
        """Embed and store chunks. recreate=False appends (incremental ingest)."""
        dim = self.embedder.dim
        if recreate:
            self.conn.execute(f"DROP TABLE IF EXISTS {self.table}")
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
                id      bigserial PRIMARY KEY,
                doc_id  text NOT NULL,
                page    int  NOT NULL,
                company text,
                year    int,
                text    text NOT NULL,
                embedding vector({dim}) NOT NULL
            )
        """)

        vecs = self.embedder.encode([c["text"] for c in chunks])
        with self.conn.cursor() as cur:
            with cur.copy(
                f"COPY {self.table} (doc_id, page, company, year, text, embedding) "
                f"FROM STDIN WITH (FORMAT BINARY)"
            ) as copy:
                copy.set_types(["text", "int4", "text", "int4", "text", "vector"])
                for c, v in zip(chunks, vecs):
                    company, year = _parse_doc_id(c["doc_id"])
                    copy.write_row([c["doc_id"], int(c["page"]), company, year,
                                    c["text"], v])

        # Approximate-NN index. At this corpus size exact scan is already
        # fast; HNSW keeps it fast if the corpus grows.
        self.conn.execute(
            f"CREATE INDEX IF NOT EXISTS {self.table}_hnsw ON {self.table} "
            f"USING hnsw (embedding vector_cosine_ops)"
        )
        self.conn.execute(f"ANALYZE {self.table}")

    # -- search ----------------------------------------------------------
    def search(self, query: str, top_k: int = 4,
               company: str | None = None, year: int | None = None) -> list[dict]:
        """Top-k most similar chunks, optionally filtered by company/year.

        Metadata filtering is the capability FAISS could not offer.
        """
        qv = self.embedder.encode([query])[0]
        where, params = [], [qv]
        if company:
            where.append("company = %s")
            params.append(company)
        if year:
            where.append("year = %s")
            params.append(year)
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        params.extend([qv, top_k])

        sql = (f"SELECT doc_id, page, text, 1 - (embedding <=> %s) AS score "
               f"FROM {self.table} {clause} "
               f"ORDER BY embedding <=> %s LIMIT %s")
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [{"doc_id": r[0], "page": r[1], "text": r[2], "score": float(r[3])}
                for r in rows]

    def count(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {self.table}")
            return int(cur.fetchone()[0])

    # -- persistence is the database itself ------------------------------
    def save(self, *_args, **_kwargs) -> None:
        """No-op: rows are already durable in Postgres."""

    @classmethod
    def load(cls, embedder: Embedder, dsn: str | None = None,
             table: str = "chunks") -> "PgVectorIndex":
        idx = cls(embedder, dsn=dsn, table=table)
        idx.count()  # fail fast if the table is missing
        return idx


# -------------------------------------------------------- hybrid store (NEW)
class HybridPgVectorIndex:
    """Hybrid retrieval: pgvector (dense) + BM25 (sparse) via Reciprocal Rank Fusion.

    This wrapper combines both retrieval signals:
    - Dense: semantic understanding via pgvector embeddings
    - Sparse (BM25): exact matching on company names, ticker symbols, fiscal years

    RRF (Reciprocal Rank Fusion) merges rankings without parameter tuning.
    """

    def __init__(
        self,
        embedder: Embedder,
        dsn: str | None = None,
        table: str = "chunks",
        rrf_k: int = 60,
    ):
        self.embedder = embedder
        self.table = table
        self.dsn = dsn or os.environ.get(
            "PGVECTOR_DSN", "postgresql:///finsession"
        )
        self.rrf_k = rrf_k

        # Initialize both retrievers
        self.dense_index = PgVectorIndex(embedder, dsn=dsn, table=table)
        from .hybrid import BM25Store, HybridRetriever
        self.bm25_store = BM25Store()
        self.hybrid = HybridRetriever(self.dense_index, self.bm25_store, rrf_k=rrf_k)

    def build(self, chunks: list[dict], recreate: bool = True) -> None:
        """Build both dense index (pgvector) and sparse index (BM25)."""
        self.dense_index.build(chunks, recreate=recreate)
        self.bm25_store.build(chunks)

    def search(
        self,
        query: str,
        top_k: int = 4,
        company: str | None = None,
        year: int | None = None,
    ) -> list[dict]:
        """Hybrid search via RRF.

        Returns top-k results merged from dense and BM25 retrievals.
        Each result includes:
        - doc_id, page, text, combined_score (RRF)
        - dense_rank, bm25_rank, merged_rank (for analysis)
        """
        return self.hybrid.search(query, top_k=top_k, company=company, year=year)

    def count(self) -> int:
        """Count indexed chunks."""
        return self.dense_index.count()

    def close(self) -> None:
        """Close database connection."""
        self.dense_index.close()

    def save(self, *_args, **_kwargs) -> None:
        """No-op: state persisted in Postgres and memory."""

    @classmethod
    def load(
        cls,
        embedder: Embedder,
        dsn: str | None = None,
        table: str = "chunks",
        rrf_k: int = 60,
    ) -> "HybridPgVectorIndex":
        """Load both dense and sparse indices."""
        idx = cls(embedder, dsn=dsn, table=table, rrf_k=rrf_k)
        idx.dense_index.count()  # Verify Postgres is accessible

        # Load chunks from pgvector into BM25
        with idx.dense_index.conn.cursor() as cur:
            cur.execute(f"SELECT doc_id, page, text FROM {table}")
            chunks = [
                {"doc_id": r[0], "page": r[1], "text": r[2]}
                for r in cur.fetchall()
            ]
        idx.bm25_store.build(chunks)

        return idx


def _parse_doc_id(doc_id: str) -> tuple[str | None, int | None]:
    """'AAPL_2022_10K' -> ('AAPL', 2022). Best-effort; None if unparseable."""
    parts = doc_id.replace("-", "_").split("_")
    company = parts[0] if parts else None
    year = next((int(p) for p in parts if p.isdigit() and len(p) == 4), None)
    return company, year


# --------------------------------------------------------------- FAISS store
class FaissIndex:
    """Original in-process FAISS store (kept as a fallback)."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.index = None
        self.chunks: list[dict] = []

    def build(self, chunks: list[dict]) -> None:
        import faiss

        self.chunks = chunks
        vecs = self.embedder.encode([c["text"] for c in chunks])
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)

    def search(self, query: str, top_k: int = 4, **_ignored) -> list[dict]:
        qv = self.embedder.encode([query])
        scores, ids = self.index.search(qv, top_k)
        out = []
        for score, cid in zip(scores[0], ids[0]):
            if cid == -1:
                continue
            hit = dict(self.chunks[cid])
            hit["score"] = float(score)
            out.append(hit)
        return out

    def count(self) -> int:
        return len(self.chunks)

    def save(self, dir_path: str | Path) -> None:
        import faiss

        d = Path(dir_path)
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(d / "chunks.faiss"))
        with (d / "chunks.jsonl").open("w") as f:
            for c in self.chunks:
                f.write(json.dumps(c) + "\n")

    @classmethod
    def load(cls, dir_path: str | Path, embedder: Embedder) -> "FaissIndex":
        import faiss

        d = Path(dir_path)
        vi = cls(embedder)
        vi.index = faiss.read_index(str(d / "chunks.faiss"))
        vi.chunks = [json.loads(line) for line in (d / "chunks.jsonl").open()]
        return vi


# Backwards-compatible alias (older code/scripts referred to VectorIndex).
VectorIndex = FaissIndex


# ------------------------------------------------------------------ factory
def make_store(cfg: dict, embedder: Embedder, load: bool = True):
    """Build or load the store named in config['vector_store']['backend']."""
    vs = cfg.get("vector_store", {"backend": "pgvector"})
    backend = vs.get("backend", "pgvector")
    if backend == "pgvector":
        dsn = vs.get("dsn") or os.environ.get("PGVECTOR_DSN")
        table = vs.get("table", "chunks")
        return (PgVectorIndex.load(embedder, dsn=dsn, table=table) if load
                else PgVectorIndex(embedder, dsn=dsn, table=table))
    if backend == "hybrid":
        dsn = vs.get("dsn") or os.environ.get("PGVECTOR_DSN")
        table = vs.get("table", "chunks")
        rrf_k = vs.get("rrf_k", 60)
        return (HybridPgVectorIndex.load(embedder, dsn=dsn, table=table, rrf_k=rrf_k) if load
                else HybridPgVectorIndex(embedder, dsn=dsn, table=table, rrf_k=rrf_k))
    if backend == "faiss":
        d = cfg["paths"]["index_dir"]
        return FaissIndex.load(d, embedder) if load else FaissIndex(embedder)
    raise ValueError(f"unknown vector_store.backend: {backend}")
