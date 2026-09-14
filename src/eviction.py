"""Memory eviction policies for bounded semantic memory.

Policies:
  - none: No eviction (unbounded)
  - lru: Least Recently Used
  - lfu: Least Frequently Used
  - redundancy: Remove semantically redundant entries
  - cost_aware: Remove entries with lowest cost-effectiveness (cost/reuse_probability)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import heapq


@dataclass
class MemoryEntry:
    """A single memory entry (verified fact)."""
    entry_id: str
    sub_question: str
    sub_answer: str
    evidence: str
    cost_tokens: int = 0
    provenance: list = field(default_factory=list)
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    access_count: int = 0
    last_access: float = field(default_factory=lambda: datetime.now().timestamp())


class EvictionPolicy(ABC):
    """Base class for eviction policies."""

    @abstractmethod
    def select_victim(self, entries: list[MemoryEntry]) -> Optional[int]:
        """Select an entry to evict.

        Args:
            entries: List of MemoryEntry objects

        Returns:
            Index of the entry to evict (for use with list.pop), or None if no eviction needed
        """
        pass


class NoEviction(EvictionPolicy):
    """No eviction—memory is unbounded."""

    def select_victim(self, entries: list[MemoryEntry]) -> Optional[int]:
        return None


class LRUEviction(EvictionPolicy):
    """Least Recently Used: evict the entry with the oldest last_used_step."""

    def select_victim(self, entries: list[MemoryEntry]) -> Optional[int]:
        if not entries:
            return None
        return min(range(len(entries)), key=lambda i: entries[i].last_used_step)


class LFUEviction(EvictionPolicy):
    """Least Frequently Used: evict the entry with the lowest hit count."""

    def select_victim(self, entries: list[MemoryEntry]) -> Optional[int]:
        if not entries:
            return None
        return min(range(len(entries)), key=lambda i: entries[i].hits)


class RedundancyEviction(EvictionPolicy):
    """Remove semantically redundant entries.

    Evict the entry most similar to others in the memory.
    Requires embedder for similarity computation.
    """

    def __init__(self, embedder=None, similarity_threshold: float = 0.95):
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold

    def select_victim(self, entries: list[MemoryEntry]) -> Optional[int]:
        """Find and evict the most redundant entry."""
        if not entries or len(entries) < 2 or not self.embedder:
            # Fall back to LRU if not enough entries
            return LRUEviction().select_victim(entries)

        # Embed all answers
        answers = [entry.sub_answer for entry in entries]

        try:
            embeddings = self.embedder.encode(answers)

            # Find entry with highest average similarity to others
            max_redundancy = -1
            victim_idx = 0

            for i in range(len(entries)):
                similarities = []
                for j in range(len(entries)):
                    if i != j:
                        sim = float(embeddings[i] @ embeddings[j])
                        similarities.append(sim)

                avg_sim = sum(similarities) / len(similarities) if similarities else 0
                if avg_sim > max_redundancy:
                    max_redundancy = avg_sim
                    victim_idx = i

            return victim_idx if max_redundancy > 0 else 0

        except Exception:
            # Fall back to LRU on error
            return LRUEviction().select_victim(entries)


class CostAwareEviction(EvictionPolicy):
    """Cost-aware eviction: remove entries with lowest cost-effectiveness.

    Cost-effectiveness = hits / cost_tokens
    Prioritizes keeping cheap entries and frequently reused entries.
    """

    def select_victim(self, entries: list[MemoryEntry]) -> Optional[int]:
        if not entries:
            return None

        # Compute cost-effectiveness for each entry
        candidates = []
        for i, entry in enumerate(entries):
            # Avoid division by zero
            cost = max(entry.cost_tokens, 1)
            # Reuse is a proxy: entries used once are "low value"
            # entries used many times are "high value"
            effectiveness = entry.hits / cost
            candidates.append((effectiveness, i, entry.last_used_step))

        # Evict the entry with lowest effectiveness
        # If tied on effectiveness, break tie by LRU (oldest last_used_step)
        victim = min(candidates, key=lambda x: (x[0], x[2]))
        return victim[1]


def make_eviction_policy(name: str, embedder=None) -> EvictionPolicy:
    """Factory function to create eviction policy by name."""
    name = name.lower().strip()

    if name in ["none", "unbounded"]:
        return NoEviction()
    elif name in ["lru", "least_recently_used"]:
        return LRUEviction()
    elif name in ["lfu", "least_frequently_used"]:
        return LFUEviction()
    elif name in ["redundancy", "semantic_redundancy"]:
        return RedundancyEviction(embedder=embedder)
    elif name in ["cost_aware", "cost-aware"]:
        return CostAwareEviction()
    else:
        raise ValueError(f"Unknown eviction policy: {name}")
