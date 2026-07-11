"""DeepSeek client (via its OpenAI-compatible API)."""

from __future__ import annotations

from .openai_compatible import OpenAICompatibleClient

QWEN_BASE_URL = "https://router.huggingface.co/v1"


class QwenClient(OpenAICompatibleClient):
    """Qwen chat client. Reads QWEN_API_KEY."""

    def __init__(
        self,
        model="Qwen/Qwen3-32B:groq",
        api_key: str | None = None,
        base_url: str = QWEN_BASE_URL,
        temperature: float = 1,
        max_tokens: int = 2048,
        name: str | None = None,
        extra_body={
            "enable_thinking": False,
        },
    ) -> None:
        super().__init__(
            model,
            base_url=base_url,
            api_key=api_key,
            api_key_env="HF_TOKEN",
            temperature=temperature,
            max_tokens=max_tokens,
            name=name,
            extra_body=extra_body
        )
