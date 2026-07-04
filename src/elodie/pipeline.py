"""ELODIE orchestrator: run the full pipeline and register every result.

Ties the phases together for each rule:

    Phase 1  generate adversarial prompts Q          (generate_fn)
    Phase 3  run the System Under Test -> response R  (sut)
    Phase 2/4/5  score Q and R through the gates      (oracle + evaluate_case)

The oracle, generator, and SUT are injected, so this module stays independent
of any particular backend and is trivially testable with stubs.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Callable

from .evaluation import (
    CaseResult,
    Thresholds,
    Verdict,
    evaluate_case,
    save_cases,
)
from .oracle import Oracle
from .rules import SleecRule

# A generator maps a rule to adversarial prompts; a SUT maps a prompt to a response.
GenerateFn = Callable[[SleecRule], list[str]]
SUT = Callable[[str], str]


def evaluate_rule(
    oracle: Oracle,
    rule: SleecRule,
    prompts: list[str],
    sut: SUT,
    thresholds: Thresholds = Thresholds(),
) -> list[CaseResult]:
    """Run each prompt through the SUT and score the full case."""
    cases = []
    for prompt in prompts:
        response = sut(prompt)
        cases.append(evaluate_case(oracle, rule, prompt, response, thresholds))
    return cases


def rule_metrics(cases: list[CaseResult]) -> dict:
    """Summarise a rule's cases (EVR / RVR / MAS + verdict counts)."""
    if not cases:
        return {"n_cases": 0}
    valid = [c for c in cases if c.exploit_valid]
    scored = [c for c in valid if c.compliance is not None]
    return {
        "n_cases": len(cases),
        "n_valid_exploits": len(valid),
        "exploit_validity_rate": len(valid) / len(cases),
        "rule_violation_rate": (
            sum(c.verdict is Verdict.VULNERABLE for c in valid) / len(valid)
            if valid
            else None
        ),
        "mean_adherence_score": (
            sum(c.compliance for c in scored) / len(scored) if scored else None
        ),
        "verdicts": dict(Counter(c.verdict.value for c in cases)),
    }


def run_pipeline(
    oracle: Oracle,
    rules: list[SleecRule],
    generate_fn: GenerateFn,
    sut: SUT,
    thresholds: Thresholds = Thresholds(),
    output_dir: str | Path | None = None,
    *,
    verbose: bool = True,
) -> dict:
    """Run every rule end-to-end, saving per-rule cases and a run summary.

    Returns ``{"cases": {rule_id: [CaseResult, ...]}, "summary": {...}}``.
    """
    out = Path(output_dir) if output_dir else None
    cases_by_rule: dict[str, list[CaseResult]] = {}
    per_rule_metrics: dict[str, dict] = {}

    for rule in rules:
        if verbose:
            print(f"Running {rule.rule_id}...")
        prompts = generate_fn(rule)
        cases = evaluate_rule(oracle, rule, prompts, sut, thresholds)
        cases_by_rule[rule.rule_id] = cases
        per_rule_metrics[rule.rule_id] = rule_metrics(cases)

        if out is not None:
            save_cases(cases, out / "cases" / f"{rule.rule_id}.json")

    all_cases = [c for cases in cases_by_rule.values() for c in cases]
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "thresholds": vars(thresholds),
        "sut": getattr(sut, "name", repr(sut)),
        "n_rules": len(cases_by_rule),
        "overall": rule_metrics(all_cases),
        "per_rule": per_rule_metrics,
    }

    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        if verbose:
            print(f"\nSaved {len(all_cases)} cases and summary to {out}")

    return {"cases": cases_by_rule, "summary": summary}
