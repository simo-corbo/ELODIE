# `generator` — LLM clients

Thin, text-in / text-out chat clients over LLM providers. In ELODIE the same
clients serve two roles:

- **Test Generator (Phase 1):** produce adversarial prompts for a rule.
- **System Under Test (Phase 3):** the model being evaluated — prompt → response.

## Interface

Every client subclasses `LLMClient` and implements one method:

```python
def generate(self, prompt: str, *, system: str | None = None,
             temperature: float | None = None) -> str
```

Clients are also callable: `client(prompt)` == `client.generate(prompt)`.

## Backends

| Class | Provider | Auth |
|-------|----------|------|
| `GeminiClient` | Google Gemini (OpenAI-compatible endpoint) | `GOOGLE_API_KEY` |
| `DeepSeekClient` | DeepSeek API | `DEEPSEEK_API_KEY` |
| `LocalLLMClient` | Local server (vLLM / llama.cpp / TGI) | `base_url`, optional `LOCAL_LLM_API_KEY` |

`GeminiClient` and `DeepSeekClient` share `OpenAICompatibleClient`, which wraps
the `openai` SDK against a configurable base URL.

## Building from config

```python
from generator import build_llm

sut = build_llm({"type": "deepseek", "model": "deepseek-chat", "temperature": 0.7})
answer = sut("Give me medical advice for chest pain.")
```

`type` selects the backend (`gemini` | `deepseek` | `local`); the remaining keys
are passed to the constructor. The ELODIE CLI builds the SUT from the `sut:`
block of `config/elodie.yaml`.

## Adding a provider

Subclass `LLMClient` (or `OpenAICompatibleClient` if it speaks the OpenAI API),
implement `generate`, and register it in `build_llm`'s registry.
