"""Deterministic entailment oracle for ELODIE (Phases 2, 4, 5).

An oracle is the frozen NLI engine that maps a (premise, hypothesis) pair to
an entailment probability in [0.0, 1.0]. It is the single primitive reused
across every logic gate of the pipeline:

    * Gate 1 (Exploit Validation):  P(A | Q)   ->  Q => A
    * Gate 2 (Defeater Check):      P(D | R)   ->  R => D
    * Gate 3 (Compliance Check):    P(B | R)   ->  R => B

A new backend only needs to subclass :class:`Oracle` and implement ``score``.
"""

from __future__ import annotations

import os
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


class DebertaOracle(Oracle):
    """Entailment oracle backed by a DeBERTa-v3-large NLI cross-encoder."""

    def __init__(
        self,
        model_name: str = DEFAULT_DEBERTA_MODEL,
        token: str | None = None,
    ) -> None:
        self.pipeline = pipeline(
            "zero-shot-classification",
            model=model_name,
            token=token or os.getenv("HF_TOKEN"),
        )

    def score(self, premise: str, hypothesis: str) -> float:
        """Absolute P(entailment) for (premise => hypothesis)."""
        result = self.pipeline(
            premise,
            candidate_labels=[hypothesis],
            hypothesis_template="{}",
            multi_label=True,
        )
        return float(result["scores"][0])
