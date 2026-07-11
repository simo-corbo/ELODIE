"""Client for a locally deployed model behind an OpenAI-compatible HTTP server.

Works with vLLM, llama.cpp, or text-generation-inference — anything exposing
``/v1/chat/completions``. This is the "LLaMA3 (local)" SUT from the schema.
"""

from __future__ import annotations

import os

import requests

from .base import LLMClient


class LocalLLMClient(LLMClient):
    """Chat client that POSTs to a local ``/v1/chat/completions`` endpoint."""

    def __init__(
        self,
        model: str = "local",
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: int = 120,
        temperature: float = 0.7,
        max_tokens: int = 512,
        name: str | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        super().__init__(model, temperature=temperature, max_tokens=max_tokens, name=name)
        # Reasoning models (e.g. gpt-oss) always emit an analysis channel; this
        # only sets how much they think. gpt-oss accepts "low" | "medium" |
        # "high" — "low" minimises reasoning (it can't be turned off entirely)
        # so it doesn't eat the max_tokens budget. Left unset for plain models.
        self.reasoning_effort = reasoning_effort
        base_url = base_url or os.getenv("LOCAL_LLM_BASE_URL")
        if not base_url:
            raise ValueError(
                "No base_url found; pass base_url or set the LOCAL_LLM_BASE_URL environment variable."
            )
        if not base_url.endswith("/v1/chat/completions"):
            base_url = base_url.rstrip("/") + "/v1/chat/completions"
        self.base_url = base_url
        self.api_key = api_key or os.getenv("LOCAL_LLM_API_KEY", "")
        self.timeout = timeout

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
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        response = requests.post(
            self.base_url, headers=headers, json=payload, timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
