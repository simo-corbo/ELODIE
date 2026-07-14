"""Llama client running the HuggingFace weights locally on the GPU.

Unlike the other clients in this package, this one holds the model in-process
(``transformers`` + ``torch``) instead of talking to an HTTP API. The weights
are pulled from the Hub on first use (gated repo — ``huggingface-cli login`` or
an ``HF_TOKEN`` in the environment) and cached under ``~/.cache/huggingface``.
"""

from __future__ import annotations

import os

from .base import LLMClient

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


class LlamaClient(LLMClient):
    """Llama chat client backed by a locally loaded HuggingFace checkpoint."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        device: str = "cuda",
        dtype: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        name: str | None = None,
        hf_token: str | None = None,
    ) -> None:
        super().__init__(model, temperature=temperature, max_tokens=max_tokens, name=name)
        self.device = device
        self.dtype = dtype
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self._tokenizer = None
        self._model = None

    def _load(self):
        """Load tokenizer + weights onto the GPU (once, on first generate)."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available; install a GPU-enabled torch build or "
                "pass device='cpu' (8B on CPU is impractically slow)."
            )

        if self.dtype is not None:
            torch_dtype = getattr(torch, self.dtype)
        elif self.device.startswith("cuda") and torch.cuda.get_device_capability()[0] >= 8:
            torch_dtype = torch.bfloat16
        else:
            # Turing and older (e.g. TITAN RTX, sm_75) only emulate bf16 — fp16
            # is the native path there. is_bf16_supported() counts emulation, so
            # check the compute capability directly.
            torch_dtype = torch.float16

        self._tokenizer = AutoTokenizer.from_pretrained(self.model, token=self.hf_token)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model, dtype=torch_dtype, token=self.hf_token
        ).to(self.device)
        self._model.eval()

        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def _build_inputs(self, prompt: str, system: str | None):
        """Apply the chat template, or fall back to plain text for base models."""
        if self._tokenizer.chat_template:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            return self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        return f"{system}\n\n{prompt}" if system else prompt

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
    ) -> str:
        import torch

        self._load()
        text = self._build_inputs(prompt, system)
        inputs = self._tokenizer(text, return_tensors="pt", add_special_tokens=not self._tokenizer.chat_template).to(
            self._model.device
        )

        temp = self.temperature if temperature is None else temperature
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=temp > 0,
                temperature=temp if temp > 0 else None,
                top_p=0.9 if temp > 0 else None,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        # Keep only the continuation, dropping the echoed prompt.
        completion = output[0][inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(completion, skip_special_tokens=True).strip()
