"""DeepSeek client (via its OpenAI-compatible API)."""

from __future__ import annotations

from .openai_compatible import OpenAICompatibleClient

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekClient(OpenAICompatibleClient):
    """DeepSeek chat client. Reads DEEPSEEK_API_KEY."""

    def __init__(
        self,
        model: str = "deepseek-chat",
        *,
        api_key: str | None = None,
        base_url: str = DEEPSEEK_BASE_URL,
        temperature: float = 0.7,
        max_tokens: int = 512,
        name: str | None = None,
    ) -> None:
        super().__init__(
            model,
            base_url=base_url,
            api_key=api_key,
            api_key_env="DEEPSEEK_API_KEY",
            temperature=temperature,
            max_tokens=max_tokens,
            name=name,
        )
