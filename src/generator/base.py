"""LLM client abstraction.

A thin, text-in / text-out chat interface over an LLM provider. In ELODIE
these clients play two roles:

    * **Test Generator (Phase 1):** produce adversarial prompts for a rule.
    * **System Under Test (Phase 3):** the model being evaluated — given a
      prompt Q, return the raw response R.

A new provider only needs to subclass :class:`LLMClient` and implement
:meth:`generate`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMClient(ABC):
    """A chat LLM that maps a prompt (+ optional system message) to text."""

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
        name: str | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.name = name or f"{type(self).__name__}:{model}"

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Return the model's text response to ``prompt``."""
        raise NotImplementedError

    def __call__(self, prompt: str, **kwargs) -> str:
        """Alias for :meth:`generate` so a client is usable as a function."""
        return self.generate(prompt, **kwargs)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r})"
