"""ELODIE: Engineering Logical Oracles for Deterministic Inference Evaluation.

Phase 1 (prompt generation) lives in :mod:`elodie.prompt_generation` and is
imported on demand, since it depends on langchain/Gemini. Everything needed to
evaluate a system under test — the oracle, the gate logic, config, and the
orchestrator — is exported here.
"""

from .config import Config, load_config
from .evaluation import (
    CaseResult,
    CaseScores,
    DefeaterScore,
    PromptScores,
    Thresholds,
    Verdict,
    evaluate_case,
    exploit_validity_rate,
    load_prompt_scores,
    load_scores,
    mean_adherence_score,
    rule_violation_rate,
    save_prompt_scores,
    save_results,
    save_scores,
    score_case,
    score_prompt,
    score_response,
)
from .oracle import DEFAULT_DEBERTA_MODEL, DebertaOracle, Oracle
from .pipeline import (
    ResponseRecord,
    collect_responses,
    evaluate_rule,
    load_responses,
    resolve_scores,
    rule_metrics,
    run_pipeline,
    run_prompt_centric,
    save_responses,
    score_responses,
    summarise_run,
)
from .prompt_dataset import PromptRecord, load_prompt_records, select_prompts
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
    "CaseScores",
    "PromptScores",
    "DefeaterScore",
    "Verdict",
    "Thresholds",
    "evaluate_case",
    "score_case",
    "score_prompt",
    "score_response",
    "save_scores",
    "load_scores",
    "save_prompt_scores",
    "load_prompt_scores",
    "save_results",
    "exploit_validity_rate",
    "rule_violation_rate",
    "mean_adherence_score",
    # config
    "Config",
    "load_config",
    # pipeline (phases A -> B -> C)
    "run_pipeline",
    "run_prompt_centric",
    "evaluate_rule",
    "rule_metrics",
    "ResponseRecord",
    "collect_responses",
    "save_responses",
    "load_responses",
    "score_responses",
    "resolve_scores",
    "summarise_run",
    # prompt corpus (prompt-centric input)
    "PromptRecord",
    "load_prompt_records",
    "select_prompts",
]
