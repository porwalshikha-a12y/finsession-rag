"""OpenAI-compatible LLM client with per-role token accounting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import os

from openai import OpenAI


@dataclass
class TokenLedger:
    """Track tokens per role."""
    per_role: dict = field(default_factory=dict)
    total_tokens: int = 0

    def add(self, role: str, tokens: int):
        """Record tokens for a role."""
        self.per_role[role] = self.per_role.get(role, 0) + tokens
        self.total_tokens += tokens

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
            tokens_used = response.usage.completion_tokens + response.usage.prompt_tokens
            self.ledger.add(role, tokens_used)

            return answer

        except Exception as e:
            print(f"ERROR in LLM.chat (role={role}): {e}")
            raise
