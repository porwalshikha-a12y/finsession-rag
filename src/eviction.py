"""Memory eviction policies for bounded semantic memory.

Every policy answers one question: when the store is over budget, which
entry do we give up? They share one interface so an experiment arm differs
ONLY in the policy object it is handed.

  none        no eviction (unbounded; arm A1)
  lru         least recently used
  lfu         least frequently used
  redundancy  the entry whose information is most nearly covered by another
  cost_aware  the entry with the lowest retention value (the contribution)

Entries are `src.memory.MemoryEntry`: they carry `cost_tokens`, `hits`,
`created_step` and `last_used_step` on the memory's logical clock, plus the
normalised embedding of the sub-question.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from .memory import MemoryEntry


class EvictionPolicy(ABC):
    """Base class. `select_victim` returns an index into `entries`."""

    @abstractmethod
    def select_victim(self, entries: list[MemoryEntry]) -> Optional[int]:
        """Index of the entry to evict (for list.pop), or None to evict nothing."""


class NoEviction(EvictionPolicy):
    """Never evict — unbounded memory (arm A1)."""

    name = "none"

    def select_victim(self, entries: list[MemoryEntry]) -> Optional[int]:
        return None


class LRUEviction(EvictionPolicy):
    """Evict the entry untouched for the longest."""

    name = "lru"

    def select_victim(self, entries: list[MemoryEntry]) -> Optional[int]:
        if not entries:
            return None
        return min(range(len(entries)), key=lambda i: entries[i].last_used_step)


class LFUEviction(EvictionPolicy):
    """Evict the least-reused entry; ties broken by least-recently-used.

    The tie-break matters here: at realistic hit rates most entries sit at
    hits == 0, so without it `min` just returns the first index and the
    policy silently degenerates into FIFO.
    """

    name = "lfu"

    def select_victim(self, entries: list[MemoryEntry]) -> Optional[int]:
        if not entries:
            return None
        return min(
            range(len(entries)),
            key=lambda i: (entries[i].hits, entries[i].last_used_step),
        )


class RedundancyEviction(EvictionPolicy):
    """Evict the entry most similar to some OTHER surviving entry.

    The intuition: if two entries say nearly the same thing, dropping one
    loses least, because what remains still covers the ground.

    Similarity is over the stored sub-question embeddings, which
    `SemanticMemory` already computed and normalised — so this costs one
    matrix multiply and no extra embedding calls. `embedder` is only a
    fallback for entries that predate stored embeddings.
    """

    name = "redundancy"

    def __init__(self, embedder=None):
        self.embedder = embedder

    def _matrix(self, entries: list[MemoryEntry]) -> Optional[np.ndarray]:
        embs = [getattr(e, "embedding", None) for e in entries]
        if all(e is not None for e in embs):
            return np.vstack(embs)
        if self.embedder is not None:
            return np.asarray(self.embedder.encode([e.sub_question for e in entries]))
        return None

    def select_victim(self, entries: list[MemoryEntry]) -> Optional[int]:
        if not entries:
            return None
        if len(entries) < 2:
            return 0
        mat = self._matrix(entries)
        if mat is None:                      # cannot measure similarity
            return LRUEviction().select_victim(entries)
        sims = mat @ mat.T
        np.fill_diagonal(sims, -1.0)         # ignore self-similarity
        # max (not mean) similarity to any other entry: redundancy is a
        # property of the closest neighbour, not of the average neighbour.
        return int(np.argmax(sims.max(axis=1)))


class CostAwareEviction(EvictionPolicy):
    """Evict the entry with the lowest retention value:

        value = cost_tokens * (1 + hits) * 0.5 ** (age / half_life)

    Keep what was expensive to produce AND shows reuse; let unused entries
    decay so the store cannot ossify. `age` is measured on the memory's
    logical clock, taken as the most recent `last_used_step` across the
    store — the store's own notion of "now".

    Note the direction: cost MULTIPLIES value, so an expensive entry is
    retained longer. Dividing by cost would invert the whole thesis.
    """

    name = "cost_aware"

    def __init__(self, half_life: float = 10.0):
        self.half_life = float(half_life)

    def select_victim(self, entries: list[MemoryEntry]) -> Optional[int]:
        if not entries:
            return None
        now = max(e.last_used_step for e in entries)

        def value(e: MemoryEntry) -> float:
            age = max(0, now - e.last_used_step)
            decay = 0.5 ** (age / self.half_life) if self.half_life > 0 else 1.0
            return max(e.cost_tokens, 1) * (1.0 + e.hits) * decay

        # tie-break by least-recently-used so the ordering is total
        return min(range(len(entries)),
                   key=lambda i: (value(entries[i]), entries[i].last_used_step))


POLICIES = {
    "none": NoEviction,
    "unbounded": NoEviction,
    "lru": LRUEviction,
    "least_recently_used": LRUEviction,
    "lfu": LFUEviction,
    "least_frequently_used": LFUEviction,
    "redundancy": RedundancyEviction,
    "semantic_redundancy": RedundancyEviction,
    "cost_aware": CostAwareEviction,
    "cost-aware": CostAwareEviction,
}


def make_eviction_policy(name: str, embedder=None,
                         half_life: float = 10.0) -> EvictionPolicy:
    """Build a policy by config name."""
    key = (name or "none").lower().strip()
    if key not in POLICIES:
        raise ValueError(f"Unknown eviction policy: {name}")
    cls = POLICIES[key]
    if cls is RedundancyEviction:
        return cls(embedder=embedder)
    if cls is CostAwareEviction:
        return cls(half_life=half_life)
    return cls()


# -- backwards-compatible aliases -------------------------------------------
# app.py / api.py import `make_policy`; tests/test_offline.py imports the
# short class names. Keeping both spellings avoids editing those call sites.
make_policy = make_eviction_policy
LRU = LRUEviction
LFU = LFUEviction
CostAware = CostAwareEviction
SemanticRedundancy = RedundancyEviction
