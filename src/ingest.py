"""Parse 10-K PDFs into page-level text with pdfplumber.

Parser choice (pdfplumber) follows arXiv:2604.12047, where it ranked first
across FinanceBench and TableQuest — cite that instead of re-benchmarking.

Usage:
    python -m src.ingest data/filings/AAPL_2023_10K.pdf
Produces data/filings/AAPL_2023_10K.pages.jsonl (one JSON object per page).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pdfplumber
from tqdm import tqdm


def parse_pdf(pdf_path: str | Path) -> list[dict]:
    """Return [{doc_id, page, text}] for every non-empty page."""
    pdf_path = Path(pdf_path)
    doc_id = pdf_path.stem
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(tqdm(pdf.pages, desc=f"parsing {doc_id}")):
            text = page.extract_text() or ""
            # Append extracted tables as tab-separated rows so table content
            # is retrievable even when extract_text mangles the layout.
            for table in page.extract_tables() or []:
                rows = ["\t".join((cell or "") for cell in row) for row in table]
                text += "\n" + "\n".join(rows)
            text = text.strip()
            if text:
                pages.append({"doc_id": doc_id, "page": i + 1, "text": text})
    return pages


def parse_to_jsonl(pdf_path: str | Path) -> Path:
    pdf_path = Path(pdf_path)
    out_path = pdf_path.with_suffix("").with_suffix(".pages.jsonl")
    pages = parse_pdf(pdf_path)
    with out_path.open("w") as f:
        for p in pages:
            f.write(json.dumps(p) + "\n")
    print(f"wrote {len(pages)} pages -> {out_path}")
    return out_path


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        parse_to_jsonl(arg)
