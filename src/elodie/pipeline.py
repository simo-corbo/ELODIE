"""ELODIE orchestrator: run the pipeline as decoupled, resumable phases.

    Phase A  System Under Test          collect_responses -> responses.jsonl
    Phase B  Oracle scoring (all gates,  score_responses  -> scores.jsonl
             all branches, threshold-free)
    Phase C  Post-hoc thresholds         resolve_scores   -> results.csv + summary.json

Each phase writes its artifact, so the expensive stages run once: the SUT
interaction (Phase A) and the oracle scoring (Phase B) are saved, and thresholds
are a *post-hoc* decision — re-resolve ``scores.jsonl`` (Phase C) at any
tau_1..tau_4 without touching the SUT or the oracle again.

The oracle, generator, and SUT are injected, so this module stays independent of
any particular backend and is trivially testable with stubs.
"""

from __future__ import annotations

import json
import time
from collections import Counter

from tqdm import tqdm
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .evaluation import (
    CaseResult,
    CaseScores,
    PromptScores,
    Thresholds,
    Verdict,
    evaluate_case,
    load_prompt_scores,
    save_prompt_scores,
    save_results,
    save_scores,
    score_prompt,
    score_response,
)
from .oracle import Oracle
from .prompt_dataset import PromptRecord
from .rules import SleecRule

# A generator maps a rule to adversarial prompts; a SUT maps a prompt to a response.
GenerateFn = Callable[[SleecRule], list[str]]
SUT = Callable[[str], str]

# One unit of work handed to Phase A: a prompt and the rules to score it against,
# each paired with a precomputed Gate-1 score (or None to score it in Phase B).
ScoringTask = tuple[str, list[tuple[str, float | None]]]


# --------------------------------------------------------------------------- #
# Phase A — System Under Test interaction (prompt -> response), saved to file.
# --------------------------------------------------------------------------- #
@dataclass
class ResponseRecord:
    """One SUT interaction: a prompt, its response, and the rules to score it on.

    Keyed by prompt (the SUT response depends only on Q), with the rules the
    prompt should be scored against and any precomputed Gate-1 scores carried
    alongside so Phase B need not re-derive them.
    """

    prompt: str
    response: str
    rules: list[str]
    exploit: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"prompt": self.prompt, "response": self.response,
                "rules": self.rules, "exploit": self.exploit}

    @classmethod
    def from_dict(cls, d: dict) -> "ResponseRecord":
        return cls(prompt=d["prompt"], response=d["response"],
                   rules=list(d.get("rules", [])), exploit=dict(d.get("exploit", {})))


def collect_responses(
    tasks: list[ScoringTask],
    sut: SUT,
    *,
    verbose: bool = True,
    resume_path: str | Path | None = None,
) -> list[ResponseRecord]:
    """Phase A: run the SUT once per unique prompt, collecting responses.

    Tasks that share a prompt are merged (the SUT is deterministic in Q), so a
    prompt tested against several rules still costs a single inference.

    When ``resume_path`` is given, each response is appended to that JSONL file
    the moment it is produced, and any prompts already recorded there (from an
    earlier, possibly interrupted run over the same tasks) are reloaded and not
    re-sent to the SUT. This makes a long SUT phase crash- and Ctrl-C-safe: a
    rerun picks up where it left off instead of re-querying every prompt.
    """
    merged: dict[str, dict[str, float | None]] = {}
    order: list[str] = []
    for prompt, rule_exploits in tasks:
        if prompt not in merged:
            merged[prompt] = {}
            order.append(prompt)
        for rid, exploit in rule_exploits:
            merged[prompt].setdefault(rid, exploit)

    # Resume: reload prompts already answered in a prior run so we neither
    # re-query the SUT nor lose that work to an earlier crash.
    done: dict[str, ResponseRecord] = {}
    if resume_path is not None and Path(resume_path).exists():
        for rec in load_responses(resume_path, tolerant=True):
            done[rec.prompt] = rec
        if verbose and done:
            print(f"  resuming: {len(done)} prompts already answered in {resume_path}")

    fh = None
    if resume_path is not None:
        path = Path(resume_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # If a prior crash left a truncated final line (no trailing newline),
        # start on a fresh line so the next record can't be glued onto it —
        # that way a partial line only ever damages itself.
        needs_newline = False
        if path.exists() and path.stat().st_size > 0:
            with path.open("rb") as rf:
                rf.seek(-1, 2)
                needs_newline = rf.read(1) != b"\n"
        fh = path.open("a", encoding="utf-8")
        if needs_newline:
            fh.write("\n")

    records: list[ResponseRecord] = []
    try:
        for prompt in tqdm(order, desc="SUT", unit="prompt", disable=not verbose):
            cached = done.get(prompt)
            if cached is not None:
                records.append(cached)
                continue
            response = sut(prompt)
            rmap = merged[prompt]
            record = ResponseRecord(
                prompt=prompt,
                response=response,
                rules=list(rmap),
                exploit={rid: e for rid, e in rmap.items() if e is not None},
            )
            records.append(record)
            if fh is not None:
                fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                fh.flush()
    finally:
        if fh is not None:
            fh.close()
    return records


def save_responses(responses: list[ResponseRecord], path: str | Path) -> None:
    """Write SUT interactions as JSON Lines (one prompt/response per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in responses:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")


def load_responses(path: str | Path, *, tolerant: bool = False) -> list[ResponseRecord]:
    """Read back a responses JSONL file (to resume from Phase B without the SUT).

    With ``tolerant=True``, lines that fail to parse are skipped rather than
    raising — a crash mid-append can leave a truncated final line, and we still
    want every complete record before it.
    """
    records = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(ResponseRecord.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError):
                if tolerant:
                    continue
                raise
    return records


# --------------------------------------------------------------------------- #
# Phase B — Oracle scoring: every gate, every branch, threshold-free.
# --------------------------------------------------------------------------- #
def score_responses(
    oracle: Oracle,
    responses: list[ResponseRecord],
    rules_by_id: dict[str, SleecRule],
    *,
    prompt_scores_path: str | Path | None = None,
    verbose: bool = True,
) -> list[CaseScores]:
    """Phase B: score each (rule, prompt, response) through all gates and branches.

    Splits the work into the SUT-independent prompt half (Gate 1 P(A|Q), Gate 2
    P(D|Q)) and the response half (Gate 3 P(B|R), P(B_D|R)). The prompt half is a
    cache keyed by ``(rule_id, prompt)``: when ``prompt_scores_path`` is given it
    is loaded from and written back to that shared file, so a second SUT over the
    same prompts recomputes **nothing** prompt-side — only Gate 3 against its own
    responses. Unknown rule ids (not in ``rules_by_id``) are skipped so a scores
    file survives a changed rule set.

    The shared cache is keyed only by ``(rule_id, prompt)``, so it is tied to the
    current rule set — delete ``prompt_scores.jsonl`` if you edit the antecedents
    or defeaters in ``sleec_rules.json``.
    """
    index: dict[tuple[str, str], PromptScores] = {}
    if prompt_scores_path is not None and Path(prompt_scores_path).exists():
        index = {(ps.rule_id, ps.prompt): ps for ps in load_prompt_scores(prompt_scores_path)}
    reused_at_start = len(index)

    total = sum(sum(1 for rid in r.rules if rid in rules_by_id) for r in responses)
    scores: list[CaseScores] = []
    computed_prompt_side = 0
    bar = tqdm(total=total, desc="scoring", unit="case", disable=not verbose)
    for rec in responses:
        for rid in rec.rules:
            rule = rules_by_id.get(rid)
            if rule is None:
                continue
            key = (rid, rec.prompt)
            ps = index.get(key)
            if ps is None:  # prompt half not cached yet -> score it once and keep it
                ps = score_prompt(oracle, rule, rec.prompt, exploit=rec.exploit.get(rid))
                index[key] = ps
                computed_prompt_side += 1
            scores.append(score_response(oracle, rule, ps, rec.response))
            bar.update(1)
    bar.close()

    if prompt_scores_path is not None and len(index) > reused_at_start:
        save_prompt_scores(list(index.values()), prompt_scores_path)
    if verbose and reused_at_start:
        print(f"  prompt-side: reused {reused_at_start} cached, computed {computed_prompt_side} new.")
    return scores


# --------------------------------------------------------------------------- #
# Phase C — Post-hoc: apply thresholds -> verdicts, metrics, results.csv.
# --------------------------------------------------------------------------- #
def resolve_scores(
    scores: list[CaseScores],
    thresholds: Thresholds = Thresholds(),
    *,
    rule_ids: list[str] | None = None,
) -> dict[str, list[CaseResult]]:
    """Phase C: resolve raw scores into verdicts at ``thresholds``, grouped by rule.

    ``rule_ids`` seeds the grouping so rules with no cases still appear (with
    ``n_cases: 0``) in the summary; any rule present in the scores is added too.
    """
    cases_by_rule: dict[str, list[CaseResult]] = {rid: [] for rid in (rule_ids or [])}
    for s in scores:
        cases_by_rule.setdefault(s.rule_id, []).append(s.resolve(thresholds))
    return cases_by_rule


def rule_metrics(cases: list[CaseResult]) -> dict:
    """Summarise a rule's cases (EVR / RVR / MAS + verdict counts)."""
    if not cases:
        return {"n_cases": 0}
    valid = [c for c in cases if c.exploit_valid]
    scored = [c.adherence for c in valid if c.adherence is not None]
    return {
        "n_cases": len(cases),
        "n_valid_exploits": len(valid),
        "exploit_validity_rate": len(valid) / len(cases),
        "rule_violation_rate": (
            sum(c.verdict is Verdict.VULNERABLE for c in valid) / len(valid)
            if valid
            else None
        ),
        "mean_adherence_score": (sum(scored) / len(scored) if scored else None),
        "verdicts": dict(Counter(c.verdict.value for c in cases)),
    }


def summarise_run(
    cases_by_rule: dict[str, list[CaseResult]],
    thresholds: Thresholds,
    sut_name: str,
    output_dir: str | Path | None = None,
    *,
    verbose: bool = True,
    **summary_extra,
) -> dict:
    """Write the resolved ``results.csv`` + ``summary.json`` and return the result dict.

    Shared by every entry point so they emit the same layout and metric shape.
    Only the post-hoc artifacts are written here; Phase A/B save their own files.
    ``summary_extra`` merges in mode-specific fields (corpus size, thresholds run).
    """
    out = Path(output_dir) if output_dir else None
    per_rule_metrics = {rid: rule_metrics(cases) for rid, cases in cases_by_rule.items()}
    all_cases = [c for cases in cases_by_rule.values() for c in cases]
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "thresholds": vars(thresholds),
        "sut": sut_name,
        "n_rules": len(cases_by_rule),
        **summary_extra,
        "overall": rule_metrics(all_cases),
        "per_rule": per_rule_metrics,
    }

    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        save_results(all_cases, out / "results.csv")
        (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        if verbose:
            print(f"\nResolved {len(all_cases)} cases -> {out / 'results.csv'} (+ summary.json)")

    return {"cases": cases_by_rule, "summary": summary}


# --------------------------------------------------------------------------- #
# End-to-end orchestrators (chain A -> B -> C, saving every artifact).
# --------------------------------------------------------------------------- #
def evaluate_rule(
    oracle: Oracle,
    rule: SleecRule,
    prompts: list[str],
    sut: SUT,
    thresholds: Thresholds = Thresholds(),
) -> list[CaseResult]:
    """Convenience: run each prompt through the SUT and resolve the case."""
    return [evaluate_case(oracle, rule, prompt, sut(prompt), thresholds) for prompt in prompts]


def _run_phases(
    oracle: Oracle,
    tasks: list[ScoringTask],
    rules_by_id: dict[str, SleecRule],
    sut: SUT,
    thresholds: Thresholds,
    out: Path | None,
    verbose: bool,
    *,
    prompt_scores_path: str | Path | None = None,
    **summary_extra,
) -> dict:
    """Drive Phase A -> B -> C for a prepared task list, saving each artifact.

    ``prompt_scores_path`` points at the SUT-shared prompt-side cache; when
    ``None`` and an output dir exists, it defaults to ``prompt_scores.jsonl`` in
    the output's parent (the base ``outputs/`` above the per-SUT dir).
    """
    if prompt_scores_path is None and out is not None:
        prompt_scores_path = out.parent / "prompt_scores.jsonl"
    responses = collect_responses(
        tasks, sut, verbose=verbose,
        resume_path=(out / "responses.jsonl") if out is not None else None,
    )
    if out is not None:
        # collect_responses streamed each record as it was produced (crash-safe);
        # this final pass rewrites the file cleanly (ordered, deduped, no partial
        # trailing line from an interrupted run).
        save_responses(responses, out / "responses.jsonl")
    scores = score_responses(oracle, responses, rules_by_id,
                             prompt_scores_path=prompt_scores_path, verbose=verbose)
    if out is not None:
        save_scores(scores, out / "scores.jsonl")
    cases_by_rule = resolve_scores(scores, thresholds, rule_ids=list(rules_by_id))
    return summarise_run(cases_by_rule, thresholds, getattr(sut, "name", repr(sut)),
                         out, verbose=verbose, **summary_extra)


def run_pipeline(
    oracle: Oracle,
    rules: list[SleecRule],
    generate_fn: GenerateFn,
    sut: SUT,
    thresholds: Thresholds = Thresholds(),
    output_dir: str | Path | None = None,
    *,
    prompt_scores_path: str | Path | None = None,
    verbose: bool = True,
) -> dict:
    """Rule-centric run: generate prompts per rule, then Phase A -> B -> C.

    Returns ``{"cases": {rule_id: [CaseResult, ...]}, "summary": {...}}``.
    """
    out = Path(output_dir) if output_dir else None
    rules_by_id = {r.rule_id: r for r in rules}

    tasks: list[ScoringTask] = []
    for rule in rules:
        if verbose:
            print(f"Generating prompts for {rule.rule_id}...")
        for prompt in generate_fn(rule):
            tasks.append((prompt, [(rule.rule_id, None)]))

    return _run_phases(oracle, tasks, rules_by_id, sut, thresholds, out, verbose,
                       prompt_scores_path=prompt_scores_path, mode="rule_centric")


def run_prompt_centric(
    oracle: Oracle,
    rules: list[SleecRule],
    records: list[PromptRecord],
    sut: SUT,
    thresholds: Thresholds = Thresholds(),
    output_dir: str | Path | None = None,
    *,
    complete: bool = False,
    prompt_scores_path: str | Path | None = None,
    verbose: bool = True,
) -> dict:
    """Prompt-centric run: test corpus prompts against the rules they trigger.

    Each prompt carries its per-rule Gate-1 scores (see :mod:`elodie.prompt_dataset`),
    fed straight into scoring so Phase B spends the oracle only on Gates 2/3. By
    default a prompt is scored against the rules whose cached P(A|Q) clears tau_1;
    ``complete`` scores it against **all** rules under test instead (so Gate-1
    non-triggers become recorded ``vacuous_pass`` cases once resolved). Because
    ``scores.jsonl`` is threshold-free, tau_1 itself can still be raised post-hoc.
    """
    out = Path(output_dir) if output_dir else None
    rules_by_id = {r.rule_id: r for r in rules}
    all_rule_ids = list(rules_by_id)

    tasks: list[ScoringTask] = []
    for rec in records:
        if complete:
            rids = all_rule_ids
        else:
            rids = [rid for rid in rec.triggered_rules(thresholds.exploit) if rid in rules_by_id]
        if rids:
            tasks.append((rec.prompt, [(rid, rec.rule_scores.get(rid)) for rid in rids]))

    return _run_phases(oracle, tasks, rules_by_id, sut, thresholds, out, verbose,
                       prompt_scores_path=prompt_scores_path,
                       mode="prompt_centric", complete=complete,
                       n_prompts=len(records), n_prompts_used=len(tasks))
