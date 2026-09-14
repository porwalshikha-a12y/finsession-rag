"""Hybrid retrieval: BM25 (keyword) + dense embeddings with Reciprocal Rank Fusion.

BM25 excels at exact matching (company names, ticker symbols, fiscal years).
Dense embeddings excel at semantic understanding (profitability ≈ earnings power).
RRF combines both signals without parameter tuning.

Financial Query Expansion: Expands financial terminology to improve keyword matching.
E.g., "net sales" → "net sales total revenue revenues net revenues turnover"
"""
from __future__ import annotations

from typing import Optional
import math


# Financial domain lexicon for query expansion
FINANCIAL_LEXICON = {
    # Revenue-related terms
    "net sales": ["total revenue", "revenues", "net revenues", "turnover", "top line"],
    "total revenue": ["net sales", "revenues", "net revenues", "turnover"],
    "revenues": ["net sales", "total revenue", "net revenues"],
    "revenue": ["net sales", "total revenue", "revenues", "turnover"],

    # Profitability terms
    "net income": ["earnings", "profit", "net earnings", "bottom line", "net profit"],
    "earnings": ["net income", "profit", "net earnings", "income"],
    "profit": ["net income", "earnings", "net earnings", "bottom line"],
    "ebitda": ["operating income", "income from operations"],
    "operating income": ["ebitda", "income from operations", "operating profit"],

    # Margin terms
    "gross profit": ["gross margin", "cost of revenue"],
    "gross margin": ["gross profit", "cost of revenue"],
    "operating margin": ["operating income", "operating profit"],

    # Assets & Liabilities
    "total assets": ["assets", "balance sheet assets"],
    "current assets": ["liquid assets", "short term assets"],
    "current liabilities": ["short term debt", "accounts payable"],
    "total liabilities": ["debt", "obligations"],
    "debt": ["liabilities", "borrowings", "loans", "long term debt"],
    "equity": ["shareholders equity", "stockholders equity", "net worth"],

    # Cash Flow
    "cash flow": ["operating cash flow", "free cash flow", "cash from operations"],
    "operating cash flow": ["cash flow from operations", "cash from operations"],
    "free cash flow": ["fcf", "cash flow"],

    # Financial Ratios
    "eps": ["earnings per share", "diluted eps"],
    "earnings per share": ["eps", "diluted eps"],
    "roe": ["return on equity"],
    "roa": ["return on assets"],
    "current ratio": ["liquidity ratio"],

    # Growth & Performance
    "growth": ["increase", "decline", "year over year", "yoy", "change"],
    "year over year": ["yoy", "growth", "change"],
    "yoy": ["year over year", "growth", "change"],

    # Tax & Legal
    "tax rate": ["effective tax rate", "income tax"],
    "provision": ["reserve", "expense"],

    # Industry-specific (10-K terms)
    "segment": ["business segment", "reporting segment"],
    "10-k": ["annual report", "fiscal year"],
    "fiscal year": ["fy", "year ended", "fiscal period"],
    "fiscal": ["annual", "year ended"],
    "form 10-k": ["annual report", "10-k"],
}


def expand_financial_query(query: str) -> str:
    """Expand financial queries with domain-specific synonyms.

    For example:
        "net sales 2022" → "net sales total revenue revenues net revenues turnover top line 2022"

    This improves keyword matching for financial documents where terminology
    variations are common (e.g., "net sales" vs "total revenue" vs "revenues").

    Args:
        query: Original query string

    Returns:
        Expanded query with additional synonyms
    """
    query_lower = query.lower()
    expanded_terms = [query]  # Keep original query

    # Check for matches in lexicon (case-insensitive)
    for key, synonyms in FINANCIAL_LEXICON.items():
        if key in query_lower:
            # Add all synonyms for this key
            expanded_terms.extend(synonyms)

    # Join all terms (original + expanded synonyms)
    # Deduplicate while preserving order
    seen = set()
    unique_terms = []
    for term in expanded_terms:
        term_lower = term.lower()
        if term_lower not in seen:
            unique_terms.append(term)
            seen.add(term_lower)

    return " ".join(unique_terms)


def reciprocal_rank_fusion(
    dense_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
    top_k: int = 4,
) -> list[dict]:
    """Merge dense and BM25 results using Reciprocal Rank Fusion.

    RRF formula: score = Σ (1 / (k + rank_i))
    where rank_i is 0-indexed position in result list i.

    Args:
        dense_results: Results from pgvector (cosine similarity), ordered by similarity
        bm25_results: Results from BM25 (keyword matching), ordered by BM25 score
        k: Constant in RRF formula (default 60, standard in literature)
        top_k: Number of final results to return

    Returns:
        Merged and ranked results, each with combined RRF score
    """
    # Build score dict keyed by (doc_id, page) tuple for uniqueness
    scores: dict[tuple[str, int], dict] = {}

    # Score from dense retrieval
    for rank, result in enumerate(dense_results):
        key = (result["doc_id"], result["page"])
        rrf_score = 1.0 / (k + rank)
        scores[key] = {
            **result,
            "dense_rank": rank,
            "dense_rrf": rrf_score,
            "bm25_rank": None,
            "bm25_rrf": 0.0,
            "combined_score": rrf_score,  # Will update below
        }

    # Add scores from BM25 retrieval
    for rank, result in enumerate(bm25_results):
        key = (result["doc_id"], result["page"])
        rrf_score = 1.0 / (k + rank)

        if key in scores:
            # Already seen in dense results: add BM25 contribution
            scores[key]["bm25_rank"] = rank
            scores[key]["bm25_rrf"] = rrf_score
            scores[key]["combined_score"] = (
                scores[key]["dense_rrf"] + rrf_score
            )
        else:
            # New from BM25: create entry
            scores[key] = {
                **result,
                "dense_rank": None,
                "dense_rrf": 0.0,
                "bm25_rank": rank,
                "bm25_rrf": rrf_score,
                "combined_score": rrf_score,
            }

    # Sort by combined RRF score (descending) and return top_k
    merged = sorted(
        scores.values(),
        key=lambda x: x["combined_score"],
        reverse=True,
    )

    # Attach merged rank and keep both dense/BM25 signals for analysis
    for merged_rank, result in enumerate(merged[:top_k]):
        result["merged_rank"] = merged_rank

    return merged[:top_k]


def bm25_batch_score(
    texts: list[str],
    query: str,
) -> list[float]:
    """Compute BM25 scores for a batch of texts against a query.

    Uses rank_bm25 library. Returns scores (higher = more relevant).
    """
    from rank_bm25 import BM25Okapi

    # Tokenize (simple: split on whitespace and punctuation)
    def tokenize(text: str) -> list[str]:
        import re
        # Split on whitespace and common punctuation, lowercase
        tokens = re.findall(r"\b\w+\b", text.lower())
        return tokens

    corpus_tokens = [tokenize(text) for text in texts]
    query_tokens = tokenize(query)

    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(query_tokens)

    return scores.tolist()


class BM25Store:
    """In-memory BM25 index for chunks.

    Stores chunk texts and metadata; computes BM25 scores on-demand.
    Used alongside pgvector for hybrid retrieval.
    """

    def __init__(self):
        self.chunks: list[dict] = []  # {doc_id, page, text, ...}
        self.tokenized_corpus: list[list[str]] = []
        self.bm25_index = None

    def build(self, chunks: list[dict]) -> None:
        """Index chunks for BM25 retrieval."""
        from rank_bm25 import BM25Okapi
        import re

        self.chunks = chunks

        # Tokenize all chunks
        def tokenize(text: str) -> list[str]:
            tokens = re.findall(r"\b\w+\b", text.lower())
            return tokens

        self.tokenized_corpus = [tokenize(c["text"]) for c in chunks]
        self.bm25_index = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        """Retrieve top-k chunks by BM25 score with financial query expansion.

        Uses financial lexicon to expand queries with domain synonyms.
        E.g., "net sales" expands to include "total revenue", "revenues", etc.
        """
        import re

        if self.bm25_index is None:
            return []

        def tokenize(text: str) -> list[str]:
            tokens = re.findall(r"\b\w+\b", text.lower())
            return tokens

        # Expand query with financial synonyms
        expanded_query = expand_financial_query(query)
        query_tokens = tokenize(expanded_query)
        scores = self.bm25_index.get_scores(query_tokens)

        # Get top-k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results = []
        for idx in top_indices:
            results.append({
                **self.chunks[idx],
                "score": float(scores[idx]),  # BM25 score (not normalized)
                "expanded_query": expanded_query,  # Track expansion for debugging
            })

        return results


# ============================================================================
# Integration: HybridRetriever (combines pgvector + BM25 via RRF)
# ============================================================================

class HybridRetriever:
    """Combines dense (pgvector) and sparse (BM25) retrieval via RRF.

    Constructor takes both index objects (PgVectorIndex + BM25Store).
    search() calls both, then merges via RRF.
    """

    def __init__(self, dense_index, bm25_store, rrf_k: int = 60):
        """
        Args:
            dense_index: PgVectorIndex instance
            bm25_store: BM25Store instance
            rrf_k: Constant in RRF formula (1/(k+rank))
        """
        self.dense_index = dense_index
        self.bm25_store = bm25_store
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 4,
        company: Optional[str] = None,
        year: Optional[int] = None,
    ) -> list[dict]:
        """Hybrid search: dense + BM25 (with financial query expansion) via RRF.

        Dense retrieval (pgvector) captures semantic similarity.
        Sparse retrieval (BM25) with financial query expansion captures exact matches
        and domain-specific terminology variations.

        Args:
            query: The question/search text
            top_k: Number of final results
            company: Optional metadata filter (pgvector only)
            year: Optional metadata filter (pgvector only)

        Returns:
            Top-k merged results with RRF scoring. Each result includes:
            - dense_rank: Position in dense results (None if only in BM25)
            - bm25_rank: Position in BM25 results (None if only in dense)
            - combined_score: RRF-merged score from both signals
            - expanded_query: Expanded query used for BM25 (for debugging)
        """
        # Get dense results from pgvector (semantic matching)
        dense_results = self.dense_index.search(
            query, top_k=top_k * 2, company=company, year=year
        )

        # Get BM25 results with financial query expansion (keyword matching)
        # Note: Query expansion happens inside bm25_store.search()
        bm25_results = self.bm25_store.search(query, top_k=top_k * 2)

        # Merge via Reciprocal Rank Fusion
        merged = reciprocal_rank_fusion(
            dense_results, bm25_results, k=self.rrf_k, top_k=top_k
        )

        return merged
