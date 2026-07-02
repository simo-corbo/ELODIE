"""Google Gemini client (via its OpenAI-compatible endpoint)."""

from __future__ import annotations

import os

from .openai_compatible import OpenAICompatibleClient

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class GeminiClient(OpenAICompatibleClient):
    """Gemini chat client. Reads GOOGLE_API_KEY (or GEMINI_API_KEY)."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        *,
        api_key: str | None = None,
        base_url: str = GEMINI_BASE_URL,
        temperature: float = 0.7,
        max_tokens: int = 512,
        name: str | None = None,
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
        )
