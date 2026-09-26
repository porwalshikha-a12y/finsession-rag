"""OpenAI-compatible LLM client with per-role token accounting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import os

from openai import OpenAI


@dataclass
class TokenLedger:
    """Track tokens per role.

    `per_role` (role -> total tokens) is the original interface and is
    unchanged. The prompt/completion split, the call count and `by_role`
    were added for the Streamlit interface and the API, which price the
    two token types differently; nothing in the experiment path uses them.
    """
    per_role: dict = field(default_factory=dict)
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    per_role_detail: dict = field(default_factory=dict)

    def add(self, role: str, tokens: int,
            prompt: int | None = None, completion: int | None = None):
        """Record tokens for a role.

        `prompt`/`completion` are optional so older call sites that pass
        only a total keep working; the total is then attributed to prompt.
        """
        if prompt is None and completion is None:
            prompt, completion = tokens, 0
        prompt = prompt or 0
        completion = completion or 0

        self.per_role[role] = self.per_role.get(role, 0) + tokens
        self.total_tokens += tokens
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.calls += 1

        d = self.per_role_detail.setdefault(
            role, {"prompt": 0, "completion": 0, "total": 0, "calls": 0})
        d["prompt"] += prompt
        d["completion"] += completion
        d["total"] += tokens
        d["calls"] += 1

    @property
    def by_role(self) -> dict:
        """Per-role breakdown with the prompt/completion split."""
        return self.per_role_detail

    def snapshot(self) -> dict:
        """Plain-dict view, for JSON responses."""
        return {
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "calls": self.calls,
            "per_role": dict(self.per_role),
            "by_role": {r: dict(v) for r, v in self.per_role_detail.items()},
        }

    def reset(self) -> None:
        self.per_role.clear()
        self.per_role_detail.clear()
        self.total_tokens = self.prompt_tokens = self.completion_tokens = 0
        self.calls = 0

    def __str__(self) -> str:
        parts = [f"total={self.total_tokens}"]
        for role in sorted(self.per_role.keys()):
            parts.append(f"{role}={self.per_role[role]}")
        return ", ".join(parts)


class LLM:
    """OpenAI-compatible client with token accounting."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_output_tokens: int = 600,
    ):
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.ledger = TokenLedger()

        # Initialize OpenAI client
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=api_key)

    def chat(
        self,
        system: str,
        user_message: str,
        role: str = "default",
        temperature: Optional[float] = None,
    ) -> str:
        """Send message to LLM and return response.

        Args:
            system: System prompt
            user_message: User message
            role: Role name for token accounting (e.g., "planner", "verifier")
            temperature: Override default temperature

        Returns:
            LLM response text
        """
        temperature = temperature if temperature is not None else self.temperature

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                max_tokens=self.max_output_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_message},
                ],
            )

            # Extract response and token count
            answer = response.choices[0].message.content.strip()
            usage = response.usage
            tokens_used = usage.completion_tokens + usage.prompt_tokens
            self.ledger.add(role, tokens_used,
                            prompt=usage.prompt_tokens,
                            completion=usage.completion_tokens)

            return answer

        except Exception as e:
            print(f"ERROR in LLM.chat (role={role}): {e}")
            raise
