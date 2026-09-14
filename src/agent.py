"""The agentic RAG loop: plan -> (memory lookup | retrieve) -> verify -> synthesize.

Deliberately minimal (plain Python, no framework) so every token is
accounted for and the ONLY difference between experiment arms is the
memory configuration.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .index import VectorIndex
from .llm import LLM
from .memory import SemanticMemory
from .security import HARDENING_CLAUSE, filter_hits, is_grounded, spotlight

PLANNER_SYS = """You decompose questions about SEC filings into retrieval sub-questions.
Given the main question and facts already gathered, either:
- output NEXT: <one specific sub-question whose answer is still needed>, or
- output DONE if the gathered facts are sufficient to answer.
Output exactly one line."""

VERIFIER_SYS = """You are a strict verifier. Given a sub-question and retrieved passages,
answer ONLY from the passages.
Line 1: SUPPORTED or NOT_SUPPORTED
Line 2 (if SUPPORTED): the concise answer to the sub-question, with figures and units."""

REUSE_CHECK_SYS = """Does the stored fact answer the new sub-question? Stored facts can be
about a different metric, period, or company than asked — be strict.
Answer YES or NO only."""

SYNTH_SYS = """Answer the user's question about SEC filings using ONLY the verified facts
provided. Be concise and include figures and units. If the facts are insufficient,
say what is missing."""


@dataclass
class QueryTrace:
    question: str
    sub_steps: list = field(default_factory=list)
    answer: str = ""
    memory_hits: int = 0
    retrievals: int = 0
    tokens_before: int = 0
    tokens_after: int = 0

    @property
    def tokens_spent(self) -> int:
        return self.tokens_after - self.tokens_before


class RAGAgent:
    def __init__(
        self,
        llm: LLM,
        index: VectorIndex,
        memory: SemanticMemory | None = None,
        top_k: int = 4,
        max_steps: int = 6,
        reuse_verification: bool = True,
        security: bool = True,
    ):
        self.llm = llm
        self.index = index
        self.memory = memory
        self.top_k = top_k
        self.max_steps = max_steps
        self.reuse_verification = reuse_verification
        self.security = security

    def answer(self, question: str) -> QueryTrace:
        trace = QueryTrace(question=question,
                           tokens_before=self.llm.ledger.total_tokens)
        # "Q: ... A: ..." strings for planner + synthesizer
        facts: list[str] = []

        for _ in range(self.max_steps):
            plan = self.llm.chat(
                PLANNER_SYS,
                f"Main question: {question}\n\nGathered facts:\n"
                + ("\n".join(facts) if facts else "(none)"),
                role="planner",
            )
            if plan.upper().startswith("DONE"):
                break
            sub_q = plan.split(
                ":", 1)[-1].strip() if ":" in plan else plan.strip()

            fact = self._resolve_subquestion(sub_q, trace)
            if fact:
                facts.append(fact)
            else:
                facts.append(
                    f"Q: {sub_q} A: (could not be verified from the corpus)")

        trace.answer = self.llm.chat(
            SYNTH_SYS,
            f"Question: {question}\n\nVerified facts:\n"
            + ("\n".join(facts) if facts else "(none)"),
            role="synthesizer",
        )
        trace.tokens_after = self.llm.ledger.total_tokens
        return trace

    # ------------------------------------------------------------------
    def _resolve_subquestion(self, sub_q: str, trace: QueryTrace) -> str | None:
        # 1) memory first
        if self.memory is not None:
            entry = self.memory.lookup(sub_q)
            if entry is not None and self._reuse_ok(sub_q, entry):
                trace.memory_hits += 1
                trace.sub_steps.append({"sub_q": sub_q, "source": "memory",
                                        "entry_id": entry.entry_id})
                return f"Q: {sub_q} A: {entry.sub_answer}"

        # 2) retrieve + verify
        tokens_before = self.llm.ledger.total_tokens
        hits = self.index.search(sub_q, top_k=self.top_k)
        trace.retrievals += 1

        dropped = []
        if self.security:
            # Drop chunks containing instruction-like text before any LLM
            # sees them; wrap survivors in explicit data delimiters.
            hits, flagged = filter_hits(hits)
            dropped = [{"doc_id": f["doc_id"], "page": f["page"], "flags": f["flags"]}
                       for f in flagged]
            passages = spotlight(hits)
            verifier_sys = VERIFIER_SYS + HARDENING_CLAUSE
        else:
            passages = "\n\n".join(
                f"[{h['doc_id']} p.{h['page']}]\n{h['text']}" for h in hits
            )
            verifier_sys = VERIFIER_SYS

        if not hits:  # everything was flagged — treat as unverifiable
            trace.sub_steps.append({"sub_q": sub_q, "source": "retrieval",
                                    "verified": False, "dropped_chunks": dropped})
            return None

        verdict = self.llm.chat(
            verifier_sys, f"Sub-question: {sub_q}\n\nPassages:\n{passages}", role="verifier"
        )
        lines = verdict.splitlines()
        supported = bool(lines) and lines[0].strip(
        ).upper().startswith("SUPPORTED")
        cost = self.llm.ledger.total_tokens - tokens_before

        if not supported:
            trace.sub_steps.append({"sub_q": sub_q, "source": "retrieval",
                                    "verified": False, "dropped_chunks": dropped})
            return None

        sub_answer = " ".join(l.strip() for l in lines[1:]).strip() or lines[0]

        # Grounding gate: memory is the amplifier, so writes are gated harder
        # than answers. A fact whose numbers don't appear in the evidence is
        # still returned for THIS answer (the verifier passed it) but is never
        # cached and re-served across the session.
        grounded = is_grounded(sub_answer, passages) if self.security else True
        if self.memory is not None and grounded:
            self.memory.write(
                sub_question=sub_q,
                evidence=passages[:2000],
                sub_answer=sub_answer,
                cost_tokens=cost,
                # provenance=[(h["doc_id"], h["page"]) for h in hits],
                provenance=[(h["metadata"]["doc_id"], h["metadata"]["page"])
                            for h in hits],
            )
        trace.sub_steps.append({"sub_q": sub_q, "source": "retrieval",
                                "verified": True, "cost_tokens": cost,
                                "grounded": grounded, "dropped_chunks": dropped})
        return f"Q: {sub_q} A: {sub_answer}"

    def _reuse_ok(self, sub_q: str, entry) -> bool:
        """Cheap gate against near-miss reuse (H4). Skippable via config."""
        if not self.reuse_verification:
            return True
        ans = self.llm.chat(
            REUSE_CHECK_SYS,
            f"New sub-question: {sub_q}\nStored fact: Q: {entry.sub_question} "
            f"A: {entry.sub_answer}",
            role="reuse_check",
        )
        return ans.strip().upper().startswith("YES")
