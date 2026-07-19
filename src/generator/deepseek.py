"""DeepSeek client (via its OpenAI-compatible API)."""

from __future__ import annotations

from .openai_compatible import OpenAICompatibleClient

DEEPSEEK_BASE_URL = "https://router.huggingface.co/v1"


class DeepSeekClient(OpenAICompatibleClient):
    """DeepSeek chat client. Reads DEEPSEEK_API_KEY."""

    def __init__(
        self,
        model="deepseek-ai/DeepSeek-V4-Flash:fastest",
        api_key: str | None = None,
        base_url: str = DEEPSEEK_BASE_URL,
        temperature: float = 1,
        max_tokens: int = 2048,
        name: str | None = None,
        extra_body={
                "thinking": {
                    "type": "disabled"
                }
            },
    ) -> None:
        super().__init__(
            model,
            base_url=base_url,
            api_key=api_key,
            api_key_env="DEEPSEEK_API_KEY",
            temperature=temperature,
            max_tokens=max_tokens,
            name=name,
            extra_body=extra_body
        )
