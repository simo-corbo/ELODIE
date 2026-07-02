"""ELODIE: Engineering Logical Oracles for Deterministic Inference Evaluation.

Phase 1 (prompt generation) lives in :mod:`elodie.prompt_generation` and is
imported on demand, since it depends on langchain/Gemini. Everything needed to
evaluate a system under test — the oracle, the gate logic, config, and the
orchestrator — is exported here.
"""

from .config import Config, load_config
from .evaluation import (
    CaseResult,
    Thresholds,
    Verdict,
    evaluate_case,
    exploit_validity_rate,
    mean_adherence_score,
    rule_violation_rate,
    save_cases,
)
from .oracle import DEFAULT_DEBERTA_MODEL, DebertaOracle, Oracle
from .pipeline import evaluate_rule, rule_metrics, run_pipeline
from .rules import DEFAULT_RULES_PATH, Defeater, SleecRule, get_rule, load_rules

__all__ = [
    # rules
    "SleecRule",
    "Defeater",
    "load_rules",
    "get_rule",
    "DEFAULT_RULES_PATH",
    # oracle
    "Oracle",
    "DebertaOracle",
    "DEFAULT_DEBERTA_MODEL",
    # evaluation
    "CaseResult",
    "Verdict",
    "Thresholds",
    "evaluate_case",
    "save_cases",
    "exploit_validity_rate",
    "rule_violation_rate",
    "mean_adherence_score",
    # config
    "Config",
    "load_config",
    # pipeline
    "run_pipeline",
    "evaluate_rule",
    "rule_metrics",
]
