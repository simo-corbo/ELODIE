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

Nothing here re-implements the pipeline. Selection and Gate-1 timing are the only
new logic; Phase A (``collect_responses``), Phase B (``score_responses``) and
Phase C (``resolve_scores``) are the pipeline's own, so a case is scored, timed
and resolved by exactly the code that produced ``results.csv``.

Gate 1 is timed here rather than read from the corpus (see ``measure_gate1_ms``):
the corpus only carries latencies for pairs the *positive* run scored, and these
pairs are disjoint from those by construction, so its Gate-1 timings are absent.
The measured call's score is discarded — every score still comes from the corpus.

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
    python run_below_threshold.py --max-prompts 10000              # bigger sample
    python run_below_threshold.py --rules-per-prompt 3             # 3 nearest misses
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from elodie.config import DEFAULT_CONFIG_PATH, load_config
from elodie.evaluation import load_prompt_scores, save_results
from elodie.oracle import DebertaOracle
from elodie.pipeline import (
    ScoringTask,
    collect_responses,
    gate_timings,
    resolve_scores,
    score_responses,
)
from elodie.prompt_dataset import GATE1_MAX_LENGTH, PromptRecord, load_prompt_records
from elodie.rules import SleecRule, get_rule, load_rules
from generator import build_llm
from run_pipeline import _sut_slug

# The positive run's sample size (see any outputs/*/summary.json), so the negative
# control costs the same number of SUT calls as the run it is compared against.
DEFAULT_MAX_PROMPTS = 2000

# Strata for the sample: equal-width bands of the eligible score range [0, tau_1).
DEFAULT_SCORE_BANDS = 10
DEFAULT_SEED = 0


def _score_band(score: float, threshold: float, bands: int) -> int:
    """Index of the equal-width band of ``[0, threshold)`` that ``score`` falls in."""
    if threshold <= 0.0:
        return 0
    return min(max(int(score / threshold * bands), 0), bands - 1)


def _stratified_by_score_band(
    assigned: list[tuple[str, float, PromptRecord]],
    limit: int,
    threshold: float,
    *,
    bands: int,
    seed: int,
) -> list[PromptRecord]:
    """Draw ``limit`` prompts stratified over the nearest-miss score bands.

    Each band's quota is its share of the eligible population (proportional
    allocation, largest remainder), so the sample keeps the shape of the
    below-threshold score distribution instead of piling up just under tau_1.
    A band holding fewer prompts than its quota hands the shortfall back, filled
    round-robin by the bands that still have room. Within a band the draw is
    uniform at random, seeded, so a rerun reproduces the sample exactly.
    """
    buckets: dict[int, list[PromptRecord]] = {}
    for _, score, rec in assigned:
        buckets.setdefault(_score_band(score, threshold, bands), []).append(rec)

    rng = random.Random(seed)
    order = sorted(buckets)
    for band in order:
        rng.shuffle(buckets[band])

    exact = {b: len(buckets[b]) * limit / len(assigned) for b in order}
    quota = {b: int(exact[b]) for b in order}
    remainder = sorted(order, key=lambda b: (-(exact[b] - quota[b]), b))
    for band in remainder[: limit - sum(quota.values())]:
        quota[band] += 1

    taken = {b: min(quota[b], len(buckets[b])) for b in order}
    while sum(taken.values()) < limit:
        room = [b for b in order if taken[b] < len(buckets[b])]
        if not room:
            break
        for band in room:
            if sum(taken.values()) >= limit:
                break
            taken[band] += 1

    return [rec for band in order for rec in buckets[band][: taken[band]]]


def select_below_threshold(
    records: list[PromptRecord],
    threshold: float,
    rule_ids: list[str],
    *,
    limit: int | None = None,
    rules_per_prompt: int = 1,
    bands: int = DEFAULT_SCORE_BANDS,
    seed: int = DEFAULT_SEED,
) -> list[tuple[PromptRecord, list[str]]]:
    """Pick prompts that trigger **no** rule, each paired with its nearest misses.

    A prompt qualifies when every rule under test scores below tau_1 — precisely
    the prompts ``select_prompts`` calls *unassigned* and drops. Each is then
    assigned to the ``rules_per_prompt`` rules it scores highest on (all still
    below tau_1), mirroring the ordinary run's argmax assignment.

    A cap is filled by *stratified sampling* over the nearest-miss score: the
    eligible range ``[0, tau_1)`` is cut into ``bands`` equal-width strata and
    each contributes in proportion to its size, drawn at random under ``seed``.
    The sample therefore spans the below-threshold distribution — easy negatives
    as well as hard ones — rather than collapsing onto the prompts sitting just
    under the threshold, which is what taking the top of a descending sort gives.
    """
    targets = set(rule_ids)

    # (nearest_rule, its_score, record) — the shape _stratified_by_score_band expects.
    assigned: list[tuple[str, float, PromptRecord]] = []
    nearest: dict[str, list[str]] = {}
    for rec in records:
        scores = {rid: s for rid, s in rec.rule_scores.items() if rid in targets}
        if not scores or any(s >= threshold for s in scores.values()):
            continue  # unscored, or it triggers something — the positive run's job
        ranked = sorted(scores, key=lambda rid: scores[rid], reverse=True)
        top = ranked[:rules_per_prompt]
        assigned.append((top[0], scores[top[0]], rec))
        nearest[rec.prompt] = top

    if limit and limit < len(assigned):
        chosen = _stratified_by_score_band(
            assigned, limit, threshold, bands=bands, seed=seed
        )
    else:
        chosen = [rec for _, _, rec in assigned]

    return [(rec, nearest[rec.prompt]) for rec in chosen]


def measure_gate1_ms(
    oracle: DebertaOracle,
    selected: list[tuple[PromptRecord, list[str]]],
    rules_by_id: dict[str, SleecRule],
    *,
    cache_path: Path | None = None,
) -> dict[tuple[str, str], float]:
    """Time Gate 1 for each selected (rule, prompt) pair, keeping the corpus score.

    The corpus carries a Gate-1 *score* for every pair but a *latency* for almost
    none: ``patch_gate_timings.py`` back-fills timings only for the pairs the
    positive run scored, and a below-threshold pair is by construction one no
    positive run ever touches. Left alone, every row here reports ``gate1_ms``
    empty and the control's Gate-1 cost cannot be compared to the positive run's
    — the one comparison the control exists to make.

    Each call is made **purely to time it and its score is discarded**: the value
    that reaches the results is always the corpus's, so no selection, score or
    verdict can move. That is the same contract — and the same 256-token cap —
    as ``patch_gate_timings.py``. Pairs already timed in ``cache_path`` (an
    earlier SUT over the same sample) are reused, so only the first SUT pays.
    """
    known: dict[tuple[str, str], float] = {}
    if cache_path is not None and Path(cache_path).exists():
        known = {(ps.rule_id, ps.prompt): ps.exploit_ms
                 for ps in load_prompt_scores(cache_path)
                 if ps.exploit_ms is not None}

    todo = [(rid, rec.prompt)
            for rec, rids in selected
            for rid in rids
            if rid in rules_by_id and (rid, rec.prompt) not in known]

    out = dict(known)
    for rid, prompt in tqdm(todo, desc="gate 1 (timing)", unit="call",
                            smoothing=0.02, dynamic_ncols=True, disable=not todo):
        _, ms = oracle.score_timed(prompt, rules_by_id[rid].antecedent,
                                   max_length=GATE1_MAX_LENGTH)
        out[(rid, prompt)] = ms  # the score itself is deliberately discarded
    return out


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
        help=f"Cap prompts tested, sampled stratified by nearest-miss score band "
             f"(default: {DEFAULT_MAX_PROMPTS}; 0 = every below-threshold prompt "
             "in the corpus).",
    )
    parser.add_argument(
        "--score-bands",
        type=int,
        default=DEFAULT_SCORE_BANDS,
        help=f"Equal-width strata over [0, tau_1) to sample across "
             f"(default: {DEFAULT_SCORE_BANDS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Seed for the within-band draw (default: {DEFAULT_SEED}).",
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
    parser.add_argument(
        "--no-gate1-timing",
        action="store_true",
        help="Skip timing Gate 1 (leaves gate1_ms empty, as the corpus has no latency "
             "for below-threshold pairs). Scores are unaffected either way.",
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
        rules_per_prompt=args.rules_per_prompt,
        bands=args.score_bands,
        seed=args.seed,
    )
    if not selected:
        parser.error("no below-threshold prompts found — is tau_1 set sensibly?")

    n_cases = sum(len(rids) for _, rids in selected)
    print(
        f"  {len(records)} prompts in corpus -> {len(selected)} below tau_1="
        f"{thresholds.exploit} on every rule "
        f"(stratified over {args.score_bands} score bands, seed {args.seed}) "
        f"-> {n_cases} cases.\n"
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

    # -- Gate-1 latency. The corpus has none for these pairs, so it is measured
    # -- here (scores discarded) and carried into the run like any cached timing.
    prompt_scores_path = base_dir / "prompt_scores_below_threshold.jsonl"
    gate1_ms: dict[tuple[str, str], float] = {}
    if not args.no_gate1_timing:
        gate1_ms = measure_gate1_ms(oracle, selected, rules_by_id,
                                    cache_path=prompt_scores_path)

    # -- Phase A: SUT interaction. Its own resume file, so the positive run's
    # -- responses.jsonl (a different prompt set) is never read or written.
    tasks: list[ScoringTask] = [
        (rec.prompt,
         [(rid, rec.rule_scores.get(rid), gate1_ms.get((rid, rec.prompt))) for rid in rids])
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
        prompt_scores_path=prompt_scores_path,
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
