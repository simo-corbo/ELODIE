"""Deterministic entailment oracle for ELODIE (Phases 2, 4, 5).

An oracle is the frozen NLI engine that maps a (premise, hypothesis) pair to
an entailment probability in [0.0, 1.0]. It is the single primitive reused
across every logic gate of the pipeline:

    * Gate 1 (Exploit Validation):  P(A | Q)   ->  Q => A
    * Gate 2 (Defeater Check):      P(D | Q)   ->  Q => D
    * Gate 3 (Compliance Check):    P(B | R)   ->  R => B

A new backend only needs to subclass :class:`Oracle` and implement ``score``.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod

from transformers import pipeline

DEFAULT_DEBERTA_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"


class Oracle(ABC):
    """Base class for deterministic entailment oracles.

    An oracle only *scores*; the decision thresholds (one per gate) live in the
    pipeline config, not here — see :class:`elodie.evaluation.Thresholds`.
    """

    @abstractmethod
    def score(self, premise: str, hypothesis: str) -> float:
        """Return P(hypothesis | premise), a probability in [0.0, 1.0].

        Must be deterministic: identical inputs always yield the same score.
        """
        raise NotImplementedError

    def score_timed(self, premise: str, hypothesis: str) -> tuple[float, float]:
        """Return ``(score, elapsed_ms)`` — the score plus the wall-clock cost of it.

        This is the single measurement point for every gate's latency: each gate
        is one :meth:`score` call, so timing the call *is* timing the gate. The
        score is identical to :meth:`score`'s, so timing is never paid for with
        accuracy. Backends need not override this — the timer wraps whatever
        :meth:`score` they implement.
        """
        start = time.perf_counter()
        score = self.score(premise, hypothesis)
        return score, (time.perf_counter() - start) * 1000.0


class DebertaOracle(Oracle):
    """Entailment oracle backed by a DeBERTa-v3-large NLI cross-encoder.

    Scoring drives the model and tokenizer directly rather than going through the
    ``zero-shot-classification`` pipeline. The result is bit-identical — the
    pipeline's ``multi_label=True`` score for a single candidate label *is*
    ``sigmoid(logit_entail - logit_contradict)``, which is what we compute — but
    it avoids the pipeline's per-call Python overhead, which mattered once gates
    became timed: that overhead is not part of a gate's cost, yet a timer wrapped
    around a pipeline call charges it to the gate anyway. (It also silences
    transformers' "using the pipelines sequentially on GPU" warning, which advises
    batching. We deliberately do not batch: a batched call can only report
    throughput, ``batch_time / batch_size``, whereas a gate's latency is the cost
    of the one call it actually makes.)

    ``self.pipeline`` is still built and kept, since ``classify_wildjailbreak.py``
    reaches into it for the model and tokenizer to run its own batched pass.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_DEBERTA_MODEL,
        token: str | None = None,
    ) -> None:
        # `or None` coerces an empty HF_TOKEN (e.g. `HF_TOKEN=` in .env) to None:
        # the model is public, and an empty-string token makes the Hub send an
        # `Authorization: Bearer ` header with no value, which httpx rejects.
        self.pipeline = pipeline(
            "zero-shot-classification",
            model=model_name,
            token=token or os.getenv("HF_TOKEN") or None,
        )
        self.model = self.pipeline.model.eval()
        self.tokenizer = self.pipeline.tokenizer
        self.device = next(self.model.parameters()).device
        label2id = {k.lower(): v for k, v in self.model.config.label2id.items()}
        self._entail_id = label2id["entailment"]
        self._contradict_id = label2id["contradiction"]

    def _entail_logits(self, premise: str, hypothesis: str, max_length: int | None):
        """Forward one (premise, hypothesis) pair; return its (entail, contradict) logits.

        ``max_length`` caps the tokenised pair. It is not a mere performance knob:
        the premise is truncated to fit, so a shorter cap feeds the model *less of
        the prompt* and can legitimately change the score. See :meth:`score`.
        """
        import torch

        encoded = self.tokenizer(premise, hypothesis, truncation=True,
                                 max_length=max_length, return_tensors="pt")
        encoded = encoded.to(self.device)
        with torch.inference_mode():
            logits = self.model(**encoded).logits[0].float()
        return logits[self._entail_id], logits[self._contradict_id]

    def score(self, premise: str, hypothesis: str, *, max_length: int | None = None) -> float:
        """Absolute P(entailment) for (premise => hypothesis).

        ``max_length`` defaults to the model's own limit (512). Pass an explicit
        cap to reproduce a score computed under a tighter one — notably the Gate-1
        cache from ``classify_wildjailbreak.py``, which truncates pairs at 256
        tokens. That is not a rounding-level difference: a prompt longer than the
        cap is cut short, so scoring the same pair at 512 shows the model text the
        cached score never saw, and the two can disagree wildly.
        """
        import torch

        entail, contradict = self._entail_logits(premise, hypothesis, max_length)
        return float(torch.sigmoid(entail - contradict))

    def score_timed(self, premise: str, hypothesis: str, *,
                    max_length: int | None = None) -> tuple[float, float]:
        """Return ``(score, elapsed_ms)``, with the GPU actually waited on.

        CUDA kernels launch asynchronously, so a timer that only brackets the
        Python call can return before the GPU has finished and report a latency
        far below the true one. Synchronising on both sides makes the measurement
        the real cost of the forward pass rather than the cost of queueing it.
        """
        import torch

        cuda = self.device.type == "cuda"
        if cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        entail, contradict = self._entail_logits(premise, hypothesis, max_length)
        if cuda:
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return float(torch.sigmoid(entail - contradict)), elapsed_ms
