"""Google Gemini client (via its OpenAI-compatible endpoint)."""

from __future__ import annotations

import os

from .openai_compatible import OpenAICompatibleClient

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class GeminiClient(OpenAICompatibleClient):
    """Gemini chat client. Reads GOOGLE_API_KEY (or GEMINI_API_KEY).

    Gemini 2.5 models *think* before answering, and on the OpenAI-compatible
    endpoint ``max_tokens`` caps thinking **and** visible output together — so a
    modest ``max_tokens`` gets eaten by reasoning and the answer is truncated
    mid-sentence. ``thinking`` defaults to ``False`` (``reasoning_effort=none``)
    so the whole budget goes to the response, which is what we want when the SUT
    is scored on its actual output. Set ``thinking=True`` to keep it.
    """

    def __init__(
        self,
        model: str = "gemini-3.5-flash",
        *,
        api_key: str | None = None,
        base_url: str = GEMINI_BASE_URL,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        name: str | None = None,
        thinking: bool = False,
    ) -> None:
        key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        super().__init__(
            model,
            base_url=base_url,
            api_key=key,
            api_key_env="GOOGLE_API_KEY",
            temperature=temperature,
            max_tokens=max_tokens,
            name=name,
            extra_body=None if thinking else {"reasoning_effort": "none"},
        )
