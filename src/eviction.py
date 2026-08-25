"""Eviction policies: given the entry list, return the index of the victim.

A2 LRU, A3 LFU, A4 semantic-redundancy, A5 cost-aware (the SmartEvict-style
contribution). All share one interface so arms differ ONLY in policy.
"""
from __future__ import annotations

import numpy as np

from .memory import MemoryEntry


class EvictionPolicy:
    name = "base"

    def select_victim(self, entries: list[MemoryEntry], clock: int) -> int:
        raise NotImplementedError


class LRU(EvictionPolicy):
    """Evict the least recently used entry."""

    name = "lru"

    def select_victim(self, entries, clock):
        return min(range(len(entries)), key=lambda i: entries[i].last_used_step)


class LFU(EvictionPolicy):
    """Evict the least frequently used entry (ties -> older)."""

    name = "lfu"

    def select_victim(self, entries, clock):
        return min(
            range(len(entries)),
            key=lambda i: (entries[i].hits, entries[i].last_used_step),
        )


class SemanticRedundancy(EvictionPolicy):
    """Evict the entry most similar to another remaining entry — its
    information is most nearly covered by what stays."""

    name = "redundancy"

    def select_victim(self, entries, clock):
        if len(entries) == 1:
            return 0
        embs = np.stack([e.embedding for e in entries])
        sims = embs @ embs.T
        np.fill_diagonal(sims, -1.0)
        max_sim_to_other = sims.max(axis=1)
        return int(np.argmax(max_sim_to_other))


class CostAware(EvictionPolicy):
    """SmartEvict-adapted: evict the entry with the lowest retention value

        value = cost_tokens * (1 + hits) * recency_decay

    i.e. keep entries that were expensive to produce AND show reuse.
    `half_life` controls how fast unused entries decay. This is the
    hand-designed variant; a learned reuse predictor (ported from
    SmartEvict) can replace `1 + hits` later without touching the interface.
    """

    name = "cost_aware"

    def __init__(self, half_life: float = 20.0):
        self.half_life = half_life

    def select_victim(self, entries, clock):
        def value(e: MemoryEntry) -> float:
            age = max(0, clock - e.last_used_step)
            decay = 0.5 ** (age / self.half_life)
            return e.cost_tokens * (1.0 + e.hits) * decay

        return min(range(len(entries)), key=lambda i: value(entries[i]))


POLICIES = {
    "lru": LRU,
    "lfu": LFU,
    "redundancy": SemanticRedundancy,
    "cost_aware": CostAware,
}


def make_policy(name: str) -> EvictionPolicy:
    return POLICIES[name]()
