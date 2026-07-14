"""ELODIE negative control: run the pipeline on prompts that *fail* Gate 1.

The ordinary prompt-centric pipeline (``run_pipeline.py``) tests each corpus
prompt against the rules it **triggers** — the rules whose cached Gate-1 score
P(A|Q) clears tau_1. This script is its mirror image: same corpus, same oracle,
same SUTs, same gates, same timings, but it selects the prompts that clear tau_1
for **no** rule at all, and tests each against the rule it comes *closest* to
triggering (its argmax over rules, still below tau_1 — the near miss).

Every case therefore fails Gate 1 by construction and resolves to
``vacuous_pass``. That is the point: this is the negative control. The
interesting columns are Gate 3 (does the SUT still answer safely when the prompt
was never adversarial?) and the per-gate timings, which are directly comparable
to the ``results.csv`` produced by the ordinary run.

Everything expensive is reused from :mod:`elodie`: prompt selection, the SUT
phase (``collect_responses``, resumable), gate scoring (``score_prompt`` /
``score_response``), and threshold resolution (``CaseScores.resolve``). Only the
selection predicate and the timing instrumentation are new.

Output — one file, the same 18 columns as ``results.csv``, in the SUT's folder:

    outputs/<sut-slug>/results_below_threshold.csv

Two caches sit alongside it so a rerun (or the next SUT) is cheap:

    outputs/.gate_timings/prompt_side_below.jsonl   Gate 1/2 scores + ms, shared
                                                    by every SUT (prompt-side is
                                                    SUT-independent)
    outputs/<sut-slug>/.below_threshold_responses.jsonl   crash-safe SUT resume

Usage:
    python run_below_threshold.py                                  # config's SUT
    python run_below_threshold.py --sut deepseek --sut-model deepseek-ai/DeepSeek-V3.2:fastest
    python run_below_threshold.py --max-prompts 10000 --stratify   # balanced sample
    python run_below_threshold.py --rules-per-prompt 3             # 3 nearest misses
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from elodie.config import DEFAULT_CONFIG_PATH, load_config
from elodie.evaluation import (
    RESULT_COLUMNS,
    CaseScores,
    DefeaterScore,
    PromptScores,
    Thresholds,
    _result_row,
    score_prompt,
    score_response,
)
from elodie.oracle import DebertaOracle, Oracle
from elodie.pipeline import collect_responses
from elodie.prompt_dataset import PromptRecord, load_prompt_records, _stratified_sample
from elodie.rules import SleecRule, get_rule, load_rules
from generator import build_llm
from run_pipeline import _sut_slug

# The ordinary run's default sample size (see any outputs/*/summary.json), kept
# so the negative control costs the same number of SUT calls as the positive one.
DEFAULT_MAX_PROMPTS = 10_000

# results.csv carries a per-gate latency next to each gate's score, plus the
# whole-case oracle total. Same order, so the two CSVs line up column for column.
TIMING_COLUMNS = {
    "gate1_threshold": "gate1_ms",
    "gate2_threshold": "gate2_ms",
    "gate3_b_threshold": "gate3_b_ms",
    "gate3_bd_threshold": "gate3_bd_ms",
}


def _timed_columns() -> list[str]:
    """RESULT_COLUMNS with each gate's ``*_ms`` inserted after its threshold."""
    columns: list[str] = []
    for name in RESULT_COLUMNS:
        columns.append(name)
        if name in TIMING_COLUMNS:
            columns.append(TIMING_COLUMNS[name])
        if name == "gate3_bd_threshold":
            columns.append("oracle_ms")  # sum of the four gate timings
    return columns


RESULT_COLUMNS_TIMED = _timed_columns()


class TimedOracle(Oracle):
    """Oracle decorator that records the wall-clock cost of every scored pair.

    The gates call the oracle from inside ``score_prompt`` / ``score_response``,
    so rather than re-implement them we wrap the oracle and look each call's
    latency back up by its ``(premise, hypothesis)`` pair — which is exactly what
    identifies a gate: Gate 1 is (Q, antecedent), Gate 2 (Q, defeater), Gate 3
    (R, consequent). ``reset()`` between cases keeps the map small.
    """

    def __init__(self, inner: Oracle) -> None:
        self.inner = inner
        self.calls: dict[tuple[str, str], float] = {}

    def score(self, premise: str, hypothesis: str) -> float:
        start = time.perf_counter()
        value = self.inner.score(premise, hypothesis)
        self.calls[(premise, hypothesis)] = (time.perf_counter() - start) * 1000
        return value

    def reset(self) -> None:
        self.calls.clear()

    def ms(self, premise: str, hypothesis: str) -> float | None:
        """Latency of that pair, or None if the gate short-circuited (no call)."""
        return self.calls.get((premise, hypothesis))


# --------------------------------------------------------------------------- #
# Selection: the prompts the ordinary pipeline threw away.
# --------------------------------------------------------------------------- #
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

    A prompt qualifies when every rule under test scores below tau_1 — these are
    precisely the prompts ``select_prompts`` calls *unassigned* and drops. Each
    one is assigned to the ``rules_per_prompt`` rules it scores highest on (still
    all below tau_1), mirroring the ordinary run's argmax assignment.

    Ordering matches the ordinary pipeline's: strongest-triggering first, i.e.
    the prompts sitting just under the threshold — the hardest negatives — unless
    ``stratify`` fills the cap round-robin across the assigned rules instead.
    """
    targets = set(rule_ids)

    # (nearest_rule, its_score, record) — reusing the shape _stratified_sample wants.
    assigned: list[tuple[str | None, float, PromptRecord]] = []
    nearest: dict[str, list[str]] = {}
    for rec in records:
        scores = {rid: s for rid, s in rec.rule_scores.items() if rid in targets}
        if not scores or any(s >= threshold for s in scores.values()):
            continue  # no scores, or it triggers something -> the ordinary run's job
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


# --------------------------------------------------------------------------- #
# Prompt-side cache: Gate 1 + Gate 2 scores *and* timings, shared across SUTs.
# --------------------------------------------------------------------------- #
def _cache_key(rule_id: str, prompt: str) -> str:
    return hashlib.sha1(f"{rule_id}\x00{prompt}".encode()).hexdigest()


def load_prompt_side(path: Path) -> dict[str, dict]:
    """Read the shared prompt-side cache (empty dict when it does not exist)."""
    if not path.exists():
        return {}
    cache: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated final line from an interrupted run
            cache[entry["k"]] = entry
    return cache


def prompt_side_scores(
    oracle: TimedOracle,
    rule: SleecRule,
    prompt: str,
    corpus_exploit: float,
    cache: dict[str, dict],
    cache_file,
) -> tuple[PromptScores, float, float | None]:
    """Gate 1 + Gate 2 for one (rule, prompt), with their latencies.

    Returns ``(PromptScores, gate1_ms, gate2_defeater_entailment_ms)``.

    Gate 1's *score* is the corpus value — the same cached P(A|Q) the ordinary
    pipeline trusts, and the number this prompt was selected on, so it must not
    drift. Its *timing* is a real oracle call made purely to measure the gate,
    which is what the ordinary run recorded too (its results.csv shows a full
    ~43ms Gate-1 latency on every case even though the score came from cache).

    The whole prompt half depends only on Q, never on a SUT response, so it is
    computed once and reused by every subsequent SUT via ``cache``.
    """
    key = _cache_key(rule.rule_id, prompt)
    hit = cache.get(key)
    if hit is not None:
        defeaters = [
            DefeaterScore(d.defeater, cached["entailment"], d.consequent, None)
            for d, cached in zip(rule.defeaters, hit["defeaters"])
        ]
        # Gate 2 reports the *strongest* defeater's latency, matching resolve().
        best = max(hit["defeaters"], key=lambda c: c["entailment"], default=None)
        gate2_ms = best["entailment_ms"] if best else None
        return (
            PromptScores(rule.rule_id, prompt, hit["exploit"], defeaters),
            hit["exploit_ms"],
            gate2_ms,
        )

    # Gate 1: measured for real, scored from the corpus (see docstring).
    oracle.score(prompt, rule.antecedent)
    gate1_ms = oracle.ms(prompt, rule.antecedent) or 0.0

    # Gate 2: score_prompt does the defeater calls; exploit is supplied so it
    # makes no Gate-1 call of its own.
    scores = score_prompt(oracle, rule, prompt, exploit=corpus_exploit)

    defeater_ms = [oracle.ms(prompt, d.defeater) or 0.0 for d in scores.defeaters]
    gate2_ms = (
        max(zip(scores.defeaters, defeater_ms), key=lambda pair: pair[0].entailment)[1]
        if scores.defeaters
        else None
    )

    entry = {
        "k": key,
        "exploit": scores.exploit,
        "exploit_ms": gate1_ms,
        "defeaters": [
            {"entailment": d.entailment, "entailment_ms": ms}
            for d, ms in zip(scores.defeaters, defeater_ms)
        ],
    }
    cache[key] = entry
    cache_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    cache_file.flush()
    return scores, gate1_ms, gate2_ms


def response_side_ms(
    oracle: TimedOracle, rule: SleecRule, case: CaseScores
) -> tuple[float, float | None]:
    """Gate 3 latencies: P(B|R) always, P(B_D|R) for the strongest defeater.

    A gate that short-circuited (empty response or consequent — see
    ``_gate3_score``) made no oracle call and costs 0.0 ms.
    """
    gate3_b_ms = oracle.ms(case.response, rule.consequent) or 0.0

    best = max(case.defeaters, key=lambda d: d.entailment, default=None)
    if best is None or best.consequent is None:
        return gate3_b_ms, None
    return gate3_b_ms, oracle.ms(case.response, best.consequent) or 0.0


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
        help="Ignore the saved SUT responses and re-query every prompt.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    thresholds: Thresholds = config.thresholds

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
        f"({'stratified, ' if args.stratify else ''}nearest-miss first)"
        f" -> {n_cases} cases.\n"
    )

    print(f"Loading oracle ({config.oracle_model})...")
    oracle = TimedOracle(DebertaOracle(model_name=config.oracle_model))
    sut = build_llm(config.sut)

    base_dir = args.output_dir or config.output_dir
    out_dir = base_dir / _sut_slug(sut.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = base_dir / ".gate_timings" / "prompt_side_below.jsonl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "results_below_threshold.csv"

    print(f"SUT: {sut.name}   thresholds: {vars(thresholds)}")
    print(f"Output: {out_csv}\n")

    # -- Phase A: SUT interaction (reused wholesale; resumable across crashes). --
    tasks = [
        (rec.prompt, [(rid, rec.rule_scores.get(rid)) for rid in rids])
        for rec, rids in selected
    ]
    responses = collect_responses(
        tasks,
        sut,
        resume_path=None if args.no_resume else out_dir / ".below_threshold_responses.jsonl",
    )

    # -- Phase B/C: score every gate (timed), then resolve at the thresholds. --
    cache = load_prompt_side(cache_path)
    if cache:
        print(f"  prompt-side: {len(cache)} cached entries reused from {cache_path.name}")

    rows: list[dict] = []
    with cache_path.open("a", encoding="utf-8") as cache_file:
        for rec in tqdm(responses, desc="scoring", unit="prompt"):
            for rule_id in rec.rules:
                rule = rules_by_id.get(rule_id)
                if rule is None:
                    continue
                oracle.reset()

                scores, gate1_ms, gate2_ms = prompt_side_scores(
                    oracle,
                    rule,
                    rec.prompt,
                    rec.exploit.get(rule_id, 0.0),
                    cache,
                    cache_file,
                )
                case = score_response(oracle, rule, scores, rec.response)
                gate3_b_ms, gate3_bd_ms = response_side_ms(oracle, rule, case)

                result = case.resolve(thresholds)
                row = _result_row(result)
                row["gate1_ms"] = gate1_ms
                row["gate2_ms"] = gate2_ms if result.defeater is not None else None
                row["gate3_b_ms"] = gate3_b_ms
                row["gate3_bd_ms"] = (
                    gate3_bd_ms if result.bd_compliance is not None else None
                )
                row["oracle_ms"] = sum(
                    ms
                    for ms in (gate1_ms, row["gate2_ms"], gate3_b_ms, row["gate3_bd_ms"])
                    if ms is not None
                )
                rows.append(row)

    # Grouped by rule, like the ordinary results.csv (which flattens per-rule cases).
    rows.sort(key=lambda r: r["rule_id"])
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS_TIMED)
        writer.writeheader()
        writer.writerows(rows)

    verdicts: dict[str, int] = {}
    for row in rows:
        verdicts[row["verdict"]] = verdicts.get(row["verdict"], 0) + 1
    total_ms = sum(row["oracle_ms"] for row in rows)
    print(f"\nWrote {len(rows)} cases -> {out_csv}")
    print(f"  verdicts: {verdicts}")
    print(f"  oracle: {total_ms / 1000:.1f}s total, {total_ms / len(rows):.1f} ms/case")


if __name__ == "__main__":
    main()
