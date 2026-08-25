"""Parse all PDFs in data/filings/, chunk, embed, and store them.

Storage backend comes from config.yaml -> vector_store.backend
("pgvector" by default, "faiss" as a fallback).

Usage:  python -m scripts.build_index
"""
from __future__ import annotations

from pathlib import Path

import yaml

from src.chunking import chunk_pages
from src.index import Embedder, FaissIndex, make_store
from src.ingest import parse_pdf

cfg = yaml.safe_load(open("config.yaml"))

pdfs = sorted(Path(cfg["paths"]["filings_dir"]).glob("*.pdf"))
if not pdfs:
    raise SystemExit(
        "No PDFs in data/filings/. Download FinanceBench 10-Ks first, e.g.\n"
        "  https://github.com/patronus-ai/financebench  (pdfs/ folder)\n"
        "Pick 8-12 filings: 4 companies x 2-3 fiscal years."
    )

all_pages = []
for pdf in pdfs:
    all_pages.extend(parse_pdf(pdf))
print(f"parsed {len(all_pages)} pages from {len(pdfs)} PDFs")

chunks = chunk_pages(
    all_pages,
    max_tokens=cfg["chunking"]["max_tokens"],
    overlap_tokens=cfg["chunking"]["overlap_tokens"],
    tokenizer=cfg["chunking"]["tokenizer"],
)
print(f"built {len(chunks)} chunks")

embedder = Embedder(**cfg["embeddings"])
store = make_store(cfg, embedder, load=False)   # fresh store, not loaded
store.build(chunks)

backend = cfg.get("vector_store", {}).get("backend", "pgvector")
if isinstance(store, FaissIndex):
    store.save(cfg["paths"]["index_dir"])
    print(f"saved FAISS index -> {cfg['paths']['index_dir']}")
else:
    print(f"stored {store.count()} chunks in Postgres "
          f"(table '{cfg['vector_store']['table']}')")
