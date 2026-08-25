"""LLM-as-judge answer scoring (protocol follows arXiv:2604.12047).

At this project's scale (~150 answers per arm) you should ALSO manually
verify every judgment — the judge gives you a first pass and consistency.
"""
from __future__ import annotations

from .llm import LLM

JUDGE_SYS = """You grade answers to questions about SEC filings.
Given the question, the gold answer, and a candidate answer, reply with
exactly one word:
CORRECT   - candidate matches the gold answer (allow rounding/formatting differences)
INCORRECT - candidate contradicts or misses the gold answer
Numbers must match to within rounding. Extra correct context is fine."""


def judge_answer(llm: LLM, question: str, gold: str, candidate: str) -> bool:
    verdict = llm.chat(
        JUDGE_SYS,
        f"Question: {question}\nGold answer: {gold}\nCandidate answer: {candidate}",
        role="judge",
    )
    return verdict.strip().upper().startswith("CORRECT")


CONSISTENCY_SYS = """You compare two answers to the same underlying question.
Reply with exactly one word:
CONSISTENT   - they convey the same key facts/figures
INCONSISTENT - they disagree on a key fact or figure"""


def judge_consistency(llm: LLM, answer_a: str, answer_b: str) -> bool:
    verdict = llm.chat(
        CONSISTENCY_SYS, f"Answer A: {answer_a}\nAnswer B: {answer_b}", role="judge"
    )
    return verdict.strip().upper().startswith("CONSISTENT")
