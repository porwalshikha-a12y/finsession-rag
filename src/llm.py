"""Thin OpenAI-compatible LLM client with token accounting.

Every call's token usage is accumulated in a CostLedger so each experiment
arm can report exactly what it spent (the headline metric of the project).
Works with OpenAI, Groq, Ollama, or any OpenAI-compatible endpoint via
OPENAI_BASE_URL.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from openai import OpenAI


@dataclass
class CostLedger:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    by_role: dict = field(default_factory=dict)  # e.g. {"planner": {...}, "verifier": {...}}

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, role: str, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.calls += 1
        slot = self.by_role.setdefault(role, {"prompt": 0, "completion": 0, "calls": 0})
        slot["prompt"] += prompt
        slot["completion"] += completion
        slot["calls"] += 1

    def snapshot(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "calls": self.calls,
            "by_role": {k: dict(v) for k, v in self.by_role.items()},
        }


class LLM:
    def __init__(self, model: str, temperature: float = 0.0, max_output_tokens: int = 600):
        base_url = os.environ.get("OPENAI_BASE_URL") or None
        self.client = OpenAI(base_url=base_url)  # reads OPENAI_API_KEY from env
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.ledger = CostLedger()

    def chat(self, system: str, user: str, role: str = "generic") -> str:
        """One chat completion. `role` labels the call for cost breakdowns."""
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        usage = resp.usage
        self.ledger.add(role, usage.prompt_tokens, usage.completion_tokens)
        return resp.choices[0].message.content.strip()
