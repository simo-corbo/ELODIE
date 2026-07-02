"""LLM clients for ELODIE — usable as the Test Generator or the System Under Test."""

from __future__ import annotations

from .base import LLMClient
from .deepseek import DeepSeekClient
from .gemini import GeminiClient
from .local_llm import LocalLLMClient
from .openai_compatible import OpenAICompatibleClient

_REGISTRY = {
    "gemini": GeminiClient,
    "deepseek": DeepSeekClient,
    "local": LocalLLMClient,
    "local_llm": LocalLLMClient,
    "vllm": LocalLLMClient,
}


def build_llm(config: dict) -> LLMClient:
    """Build an :class:`LLMClient` from a config dict.

    The dict must have a ``type`` key (``gemini`` | ``deepseek`` | ``local``);
    remaining keys (``model``, ``temperature``, ``max_tokens``, ``base_url``…)
    are passed to the client constructor. Extra keys not accepted by the client
    (e.g. ``num_prompts``) are ignored.
    """
    config = dict(config)
    kind = str(config.pop("type", "gemini")).lower()
    config.pop("num_prompts", None)  # generation-loop setting, not a client arg
    if kind not in _REGISTRY:
        raise ValueError(
            f"Unknown LLM type {kind!r}; choose from {sorted(set(_REGISTRY))}."
        )
    return _REGISTRY[kind](**config)


__all__ = [
    "LLMClient",
    "OpenAICompatibleClient",
    "GeminiClient",
    "DeepSeekClient",
    "LocalLLMClient",
    "build_llm",
]
