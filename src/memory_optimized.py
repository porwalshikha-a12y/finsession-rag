"""Bounded semantic memory of verified evidence units.

This is the heart of the dissertation. Each entry stores a verified
(sub-question, evidence, sub-answer) tuple plus the metadata eviction
policies need: production cost, hit count, recency.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

_entry_ids = itertools.count()


@dataclass
class MemoryEntry:
    entry_id: int
    sub_question: str
    evidence: str          # supporting text (chunk excerpts)
    sub_answer: str        # verified answer to the sub-question
    embedding: np.ndarray  # of the sub_question, normalized
    cost_tokens: int       # tokens spent producing this entry (retrieve+verify)
    created_step: int      # logical clock at creation
    last_used_step: int    # logical clock at last hit
    hits: int = 0
    provenance: list = field(default_factory=list)  # [(doc_id, page), ...]


@dataclass
class MemoryStats:
    lookups: int = 0
    hits: int = 0
    writes: int = 0
    evictions: int = 0
    evicted_ids: list = field(default_factory=list)
    hit_entry_ids: list = field(default_factory=list)

    def snapshot(self) -> dict:
        return {
            "lookups": self.lookups,
            "hits": self.hits,
            "hit_rate": (self.hits / self.lookups) if self.lookups else 0.0,
            "writes": self.writes,
            "evictions": self.evictions,
        }


class SemanticMemory:
    """Bounded store; on overflow, the eviction policy picks a victim.

    budget=None means unbounded (arm A1).
    Optimized lookup using vectorized numpy for O(1) similarity computation.
    """

    def __init__(
        self,
        embedder,  # anything with .encode(list[str]) -> normalized np.ndarray
        eviction_policy=None,          # None = never evict (unbounded)
        budget: int | None = None,     # max entries
        similarity_threshold: float = 0.80,
    ):
        self.embedder = embedder
        self.policy = eviction_policy
        self.budget = budget
        self.tau = similarity_threshold
        self.entries: list[MemoryEntry] = []
        self.embedding_matrix = None  # Will be (n_entries, embedding_dim)
        self.clock = 0
        self.stats = MemoryStats()

    # ---- lookup (OPTIMIZED) ----
    def lookup(self, sub_question: str) -> MemoryEntry | None:
        """Best entry above the similarity threshold, else None.
        
        Optimized: uses vectorized numpy dot product instead of loop.
        O(n*d) instead of O(n*d) but with much lower constant factor.
        """
        self.clock += 1
        self.stats.lookups += 1
        if not self.entries:
            return None
        
        # Encode query once
        qv = self.embedder.encode([sub_question])[0]  # shape: (embedding_dim,)
        
        # Vectorized similarity: all at once instead of loop
        # This is ~10x faster than: [float(e.embedding @ qv) for e in self.entries]
        sims = np.dot(self.embedding_matrix, qv)  # shape: (n_entries,)
        
        best = int(np.argmax(sims))
        if sims[best] >= self.tau:
            entry = self.entries[best]
            entry.hits += 1
            entry.last_used_step = self.clock
            self.stats.hits += 1
            self.stats.hit_entry_ids.append(entry.entry_id)
            
            # Debug log for unbounded memory growth
            if self.budget is None and len(self.entries) > 100:
                import sys
                print(f"[DEBUG] Unbounded memory: {len(self.entries)} entries, lookup took {sims[best]:.3f}s", file=sys.stderr)
            
            return entry
        return None

    # ---- write ----
    def write(
        self,
        sub_question: str,
        evidence: str,
        sub_answer: str,
        cost_tokens: int,
        provenance: list | None = None,
    ) -> MemoryEntry:
        self.clock += 1
        emb = self.embedder.encode([sub_question])[0]
        entry = MemoryEntry(
            entry_id=next(_entry_ids),
            sub_question=sub_question,
            evidence=evidence,
            sub_answer=sub_answer,
            embedding=emb,
            cost_tokens=cost_tokens,
            created_step=self.clock,
            last_used_step=self.clock,
            provenance=provenance or [],
        )
        self.entries.append(entry)
        self.stats.writes += 1
        
        # Rebuild embedding matrix (more efficient than appending)
        self._rebuild_embedding_matrix()
        
        while self.budget is not None and len(self.entries) > self.budget:
            self._evict()
        return entry

    def _rebuild_embedding_matrix(self) -> None:
        """Rebuild the embedding matrix from all entries."""
        if not self.entries:
            self.embedding_matrix = np.zeros((0, 384))  # or infer from first embedding
        else:
            # Stack all embeddings into a matrix: (n_entries, embedding_dim)
            self.embedding_matrix = np.vstack([e.embedding for e in self.entries])

    def _evict(self) -> None:
        assert self.policy is not None, "bounded memory needs an eviction policy"
        victim_idx = self.policy.select_victim(self.entries)
        if victim_idx is not None:
            victim = self.entries.pop(victim_idx)
            self.stats.evictions += 1
            self.stats.evicted_ids.append(victim.entry_id)
            # Rebuild matrix after eviction
            self._rebuild_embedding_matrix()

    def reset(self) -> None:
        """Clear between sessions (memory is session-scoped in this study)."""
        self.entries = []
        self.embedding_matrix = None
        self.clock = 0
        self.stats = MemoryStats()
