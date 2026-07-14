"""ELODIE negative control: run the pipeline on prompts that *fail* Gate 1.

The ordinary prompt-centric run (``run_pipeline.py``) tests each corpus prompt
against the rules it **triggers** — those whose cached Gate-1 score P(A|Q) clears
tau_1. This is its mirror image: same corpus, same oracle, same gates, same
timings, same SUTs, but it selects the prompts that clear tau_1 for **no** rule,
and tests each against the rule it comes *closest* to triggering (its argmax,
still below tau_1 — the near miss).

Every case therefore fails Gate 1 by construction and resolves to
``vacuous_pass``. That is the point: it is the negative control. What the run
measures is Gate 3 — does the SUT still answer safely when the prompt was never
adversarial? — and the per-gate latencies, which are directly comparable to the
positive run's because they come from the very same code.

Nothing here re-implements the pipeline. Selection is the only new logic; Phase A
(``collect_responses``), Phase B (``score_responses``) and Phase C
(``resolve_scores``) are the pipeline's own, so a case is scored, timed and
resolved by exactly the code that produced ``results.csv``.

Output — one CSV, the same columns as ``results.csv``, in the SUT's own folder:

    outputs/<sut-slug>/results_below_threshold.csv

Alongside it, two caches make a rerun (and the next SUT) cheap:

    outputs/prompt_scores_below_threshold.jsonl        Gate 1/2 scores + latencies,
                                                       SUT-independent, shared by
                                                       every SUT
    outputs/<sut-slug>/.below_threshold_responses.jsonl  crash-safe SUT resume

Usage:
    python run_below_threshold.py                                  # config's SUT
    python run_below_threshold.py --sut qwen --sut-model Qwen/Qwen3-32B:groq
    python run_below_threshold.py --max-prompts 10000 --stratify   # balanced sample
    python run_below_threshold.py --rules-per-prompt 3             # 3 nearest misses
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from elodie.config import DEFAULT_CONFIG_PATH, load_config
from elodie.evaluation import save_results
from elodie.oracle import DebertaOracle
from elodie.pipeline import (
    ScoringTask,
    collect_responses,
    gate_timings,
    resolve_scores,
    score_responses,
)
from elodie.prompt_dataset import PromptRecord, _stratified_sample, load_prompt_records
from elodie.rules import get_rule, load_rules
from generator import build_llm
from run_pipeline import _sut_slug

# The positive run's sample size (see any outputs/*/summary.json), so the negative
# control costs the same number of SUT calls as the run it is compared against.
DEFAULT_MAX_PROMPTS = 10_000


def select_below_threshold(
    records: list[PromptRecord],
    threshold: float,
    rule_ids: list[str],
    *,
    limit: int | None = None,
    stratify: bool = False,
    rules_per_prompt: int = 1,
) -> list[tuple[PromptRecord, list[str]]]:
    """Pick prompts that trigger **no** rule, each paired with its nearest misses.

    A prompt qualifies when every rule under test scores below tau_1 — precisely
    the prompts ``select_prompts`` calls *unassigned* and drops. Each is then
    assigned to the ``rules_per_prompt`` rules it scores highest on (all still
    below tau_1), mirroring the ordinary run's argmax assignment.

    Ordering mirrors it too: strongest-scoring first, so a cap keeps the prompts
    sitting just under the threshold — the hardest negatives, the ones the corpus
    came closest to counting as exploits. ``stratify`` instead fills the cap
    round-robin across the nearest-miss rules, for a sample balanced over rules
    rather than dominated by whichever rules own the closest calls.
    """
    targets = set(rule_ids)

    # (nearest_rule, its_score, record) — the shape _stratified_sample expects.
    assigned: list[tuple[str | None, float, PromptRecord]] = []
    nearest: dict[str, list[str]] = {}
    for rec in records:
        scores = {rid: s for rid, s in rec.rule_scores.items() if rid in targets}
        if not scores or any(s >= threshold for s in scores.values()):
            continue  # unscored, or it triggers something — the positive run's job
        ranked = sorted(scores, key=lambda rid: scores[rid], reverse=True)
        top = ranked[:rules_per_prompt]
        assigned.append((top[0], scores[top[0]], rec))
        nearest[rec.prompt] = top

    if stratify and limit:
        chosen = _stratified_sample(assigned, limit)
    else:
        assigned.sort(key=lambda t: t[1], reverse=True)
        chosen = [rec for _, _, rec in assigned]
        chosen = chosen[:limit] if limit else chosen

    return [(rec, nearest[rec.prompt]) for rec in chosen]


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--rule-id", help="Restrict the rules under test to one.")
    parser.add_argument("--sut", help="SUT type override: gemini | deepseek | local | qwen.")
    parser.add_argument("--sut-model", help="SUT model override (pairs with --sut).")
    parser.add_argument("--output-dir", type=Path, help="Base dir (default: config's outputs/).")
    parser.add_argument("--prompts-file", type=Path, help="Corpus to draw from (default: config).")
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=DEFAULT_MAX_PROMPTS,
        help=f"Cap prompts tested, nearest-miss first (default: {DEFAULT_MAX_PROMPTS}; "
             "0 = every below-threshold prompt in the corpus).",
    )
    parser.add_argument(
        "--stratify",
        action="store_true",
        help="Fill the cap round-robin across nearest-miss rules instead of taking "
             "the globally closest-to-threshold prompts.",
    )
    parser.add_argument(
        "--rules-per-prompt",
        type=int,
        default=1,
        help="How many nearest-miss rules each prompt is tested against (default: 1).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any saved SUT responses and re-query every prompt.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    thresholds = config.thresholds

    if args.sut:
        config.sut["type"] = args.sut
    if args.sut_model:
        config.sut["model"] = args.sut_model

    rules = (
        [get_rule(args.rule_id, config.rules_path)]
        if args.rule_id
        else load_rules(config.rules_path)
    )
    rules_by_id = {r.rule_id: r for r in rules}

    prompts_file = args.prompts_file or config.prompts_path
    if prompts_file is None or not Path(prompts_file).exists():
        parser.error(f"prompt corpus not found: {prompts_file}")

    print(f"Loading corpus {prompts_file}...")
    records = load_prompt_records(prompts_file)
    selected = select_below_threshold(
        records,
        thresholds.exploit,
        list(rules_by_id),
        limit=args.max_prompts or None,
        stratify=args.stratify,
        rules_per_prompt=args.rules_per_prompt,
    )
    if not selected:
        parser.error("no below-threshold prompts found — is tau_1 set sensibly?")

    n_cases = sum(len(rids) for _, rids in selected)
    print(
        f"  {len(records)} prompts in corpus -> {len(selected)} below tau_1="
        f"{thresholds.exploit} on every rule "
        f"({'stratified' if args.stratify else 'nearest-miss first'}) -> {n_cases} cases.\n"
    )

    print(f"Loading oracle ({config.oracle_model})...")
    oracle = DebertaOracle(model_name=config.oracle_model)
    sut = build_llm(config.sut)

    base_dir = args.output_dir or config.output_dir
    out_dir = base_dir / _sut_slug(sut.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "results_below_threshold.csv"

    print(f"SUT: {sut.name}   thresholds: {vars(thresholds)}")
    print(f"Output: {out_csv}\n")

    # -- Phase A: SUT interaction. Its own resume file, so the positive run's
    # -- responses.jsonl (a different prompt set) is never read or written.
    tasks: list[ScoringTask] = [
        (rec.prompt,
         [(rid, rec.rule_scores.get(rid), rec.rule_score_ms.get(rid)) for rid in rids])
        for rec, rids in selected
    ]
    responses = collect_responses(
        tasks,
        sut,
        resume_path=None if args.no_resume else out_dir / ".below_threshold_responses.jsonl",
    )

    # -- Phase B: every gate, timed, threshold-free. The prompt half (Gates 1/2)
    # -- is SUT-independent, so it is cached and the next SUT pays only Gate 3.
    # -- Its own cache file: these (rule, prompt) pairs are disjoint from the
    # -- positive run's, so sharing one buys nothing and would rewrite that file.
    scores = score_responses(
        oracle,
        responses,
        rules_by_id,
        prompt_scores_path=base_dir / "prompt_scores_below_threshold.jsonl",
    )

    # -- Phase C: resolve at the config thresholds and write the single CSV.
    cases_by_rule = resolve_scores(scores, thresholds, rule_ids=list(rules_by_id))
    cases = [c for rule_cases in cases_by_rule.values() for c in rule_cases]
    save_results(cases, out_csv)

    verdicts: dict[str, int] = {}
    for case in cases:
        verdicts[case.verdict.value] = verdicts.get(case.verdict.value, 0) + 1
    print(f"\nWrote {len(cases)} cases -> {out_csv}")
    print(f"  verdicts: {verdicts}")
    for gate, stats in gate_timings(cases).items():
        print(f"  {gate}: {stats['mean_ms']:.1f} ms mean over {stats['n_measured']} measured")


if __name__ == "__main__":
    main()
