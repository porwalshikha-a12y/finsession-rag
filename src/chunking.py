"""Sentence chunking: 512-token cap, 128-token (25%) overlap.

Configuration follows the recommendation of arXiv:2604.12047 (sentence
chunking = near-optimal, low-cost; 25% overlap = the single biggest win).
"""
from __future__ import annotations

import re

import tiktoken

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


class _WordEncoding:
    """Offline fallback if the tiktoken encoding file can't be downloaded:
    counts whitespace words as tokens. Slightly coarser boundaries, same
    interface — experiments on your machine will use real tiktoken."""

    def encode(self, text: str) -> list[str]:
        return text.split()

    def decode(self, tokens: list[str]) -> str:
        return " ".join(tokens)


def get_encoding(name: str):
    try:
        return tiktoken.get_encoding(name)
    except Exception:
        print("WARNING: tiktoken encoding unavailable, using word-count fallback")
        return _WordEncoding()


def split_sentences(text: str) -> list[str]:
    """Cheap rule-based sentence splitter (good enough for filings; swap in
    nltk/pysbd later if you want — keep the interface identical)."""
    parts = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        parts.extend(s.strip() for s in _SENT_SPLIT.split(line) if s.strip())
    return parts


def chunk_pages(
    pages: list[dict],
    max_tokens: int = 512,
    overlap_tokens: int = 128,
    tokenizer: str = "cl100k_base",
) -> list[dict]:
    """Greedy sentence packing per page.

    Returns [{chunk_id, doc_id, page, text, n_tokens}].
    Overlap: each new chunk starts with the trailing sentences (~overlap_tokens
    worth) of the previous chunk, so facts straddling a boundary stay intact.
    """
    enc = get_encoding(tokenizer)
    chunks: list[dict] = []

    for p in pages:
        sents = split_sentences(p["text"])
        if not sents:
            continue
        lens = [len(enc.encode(s)) for s in sents]
        # Oversized "sentences" (mangled tables) get hard-split by tokens.
        norm_sents, norm_lens = [], []
        for s, ln in zip(sents, lens):
            if ln <= max_tokens:
                norm_sents.append(s)
                norm_lens.append(ln)
            else:
                toks = enc.encode(s)
                for i in range(0, len(toks), max_tokens):
                    piece = enc.decode(toks[i : i + max_tokens])
                    norm_sents.append(piece)
                    norm_lens.append(min(max_tokens, len(toks) - i))

        cur: list[int] = []  # indices into norm_sents
        cur_tok = 0
        i = 0
        while i < len(norm_sents):
            if cur_tok + norm_lens[i] <= max_tokens or not cur:
                cur.append(i)
                cur_tok += norm_lens[i]
                i += 1
            else:
                _emit(chunks, norm_sents, cur, p)
                # Build overlap tail from the end of the finished chunk.
                tail, tail_tok = [], 0
                for j in reversed(cur):
                    if tail_tok + norm_lens[j] > overlap_tokens:
                        break
                    tail.insert(0, j)
                    tail_tok += norm_lens[j]
                cur = list(tail)
                cur_tok = tail_tok
        if cur:
            _emit(chunks, norm_sents, cur, p)

    for idx, c in enumerate(chunks):
        c["chunk_id"] = idx
        c["n_tokens"] = len(enc.encode(c["text"]))
    return chunks


def _emit(chunks: list[dict], sents: list[str], idxs: list[int], page: dict) -> None:
    chunks.append(
        {
            "doc_id": page["doc_id"],
            "page": page["page"],
            "text": " ".join(sents[i] for i in idxs),
        }
    )
