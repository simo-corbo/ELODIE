"""Client for any provider exposing the OpenAI chat-completions API.

Gemini (via its OpenAI-compatible endpoint) and DeepSeek both speak this
protocol, so they share this implementation and only differ in base URL,
default model, and which environment variable holds the API key.
"""

from __future__ import annotations

import os

from .base import LLMClient


class OpenAICompatibleClient(LLMClient):
    """Chat client backed by the ``openai`` SDK pointed at any base URL."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        api_key: str | None = None,
        api_key_env: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        name: str | None = None,
        extra_body: dict | None = None,
    ) -> None:
        super().__init__(model, temperature=temperature, max_tokens=max_tokens, name=name)
        key = api_key or (os.getenv(api_key_env) if api_key_env else None)
        if not key:
            raise ValueError(
                f"No API key found; pass api_key or set the {api_key_env} environment variable."
            )
        # Provider-specific request fields merged into the JSON body (e.g. Gemini's
        # thinking config). Left as-is for providers that don't need it.
        self.extra_body = extra_body
        from openai import OpenAI

        self._client = OpenAI(api_key=key, base_url=base_url)

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens,
            extra_body=self.extra_body,
        )
        return (response.choices[0].message.content or "").strip()
