#!/usr/bin/env python3
"""Backfill measured DeBERTa gate latencies into every artifact under ``outputs/``.

The runs in ``outputs/`` were produced before the oracle recorded how long each
gate took. This script re-runs the oracle over the *saved* artifacts to measure
that cost — the SUTs are never queried again, and no model answer is ever
regenerated (``responses.jsonl`` only gains an ``exploit_ms`` map; its prompts and
responses are copied through untouched).

Each gate is exactly one oracle call, so timing the call *is* timing the gate:

    Gate 1  P(A | Q)   prompt-side, SUT-independent  -> measured once, globally
    Gate 2  P(D | Q)   prompt-side, SUT-independent  -> measured once, globally
    Gate 3  P(B | R)   response-side, per SUT        -> measured once per SUT
    Gate 3  P(B_D | R) response-side, per SUT        -> measured once per SUT

The artifacts are redundant by design — ``prompt_scores.jsonl`` is the shared
prompt-side cache every SUT's ``scores.jsonl`` was built from, and ``results.csv``
/ ``summary.json`` are pure functions of ``scores.jsonl``. So each timing is
**measured once and propagated**, never recomputed per file:

    prompt_scores.jsonl  (Gate 1 + Gate 2, measured once)
        |-> wildjailbreak_category_scores_*.json   (Gate-1 ms per (prompt, rule))
        |-> <sut>/responses.jsonl  x5             (Gate-1 ms beside each cached score)
        `-> <sut>/scores.jsonl     x5             (Gate 1 + 2 copied in verbatim,
                                                   only Gate 3 measured per SUT)
                `-> <sut>/results.csv + summary.json  (re-resolved, no oracle calls)

The same (rule, prompt) therefore carries the *same* Gate-1 latency in the
corpus JSON, in the prompt-side cache, and in all five SUT score files.

**No score is ever overwritten.** Every oracle call is made purely to time it and
its result is discarded; the value written back is always the one already on file.
So each artifact gains ``*_ms`` fields and changes in no other way, and no verdict,
EVR, RVR or MAS can move — the published results stay exactly as they are.

Gate 1 is nonetheless timed under the corpus cache's 256-token cap
(``GATE1_MAX_LENGTH``) rather than the model's default 512, because the cost we
want is the cost of the call that actually produced the stored score: at 512 the
prompt is no longer truncated, so it is a bigger, slower — and different — call.

Resumable: every measurement is appended to a checkpoint under
``outputs/.gate_timings/`` (and ``outputs/<sut>/.gate_timings/``) as it is made,
so a ~20h run survives a crash or a Ctrl-C and picks up exactly where it stopped.

    python scripts/patch_gate_timings.py                  # everything, resumable
    python scripts/patch_gate_timings.py --smoke 50       # 50 records/file, writes nothing
    python scripts/patch_gate_timings.py --phase prompt   # just the prompt-side half
    python scripts/patch_gate_timings.py --sut gemini-gemini-3.5-flash
    python scripts/patch_gate_timings.py --corpus-scope all   # time every corpus pair too
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from elodie.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from elodie.evaluation import (  # noqa: E402
    CaseScores,
    DefeaterScore,
    PromptScores,
    load_prompt_scores,
    load_scores,
    save_prompt_scores,
    save_scores,
    _gate3_score,
)
from elodie.oracle import DebertaOracle  # noqa: E402
from elodie.pipeline import (  # noqa: E402
    load_responses,
    resolve_scores,
    save_responses,
    summarise_run,
)
from elodie.rules import SleecRule, load_rules  # noqa: E402

PHASES = ("prompt", "corpus", "sut", "resolve")

# Gate 1's scores were precomputed by classify_wildjailbreak.py, which truncates each
# (prompt, antecedent) pair at 256 tokens. Re-scoring at the model's default 512 instead
# feeds it prompt text the cached score never saw: on a random sample of 150 stored cases
# that moved Gate 1 by 0.12 on average and flipped 12% of them across tau_1, versus a mean
# 6e-4 and one borderline flip when the original 256-token cap is honoured. So Gate 1 is
# re-measured under the cap it was built with; Gates 2 and 3 keep the model default, which
# is what the live pipeline scored them at.
GATE1_MAX_LENGTH = 256


def warm_up(oracle, rounds: int = 3) -> None:
    """Burn throwaway calls across a range of input lengths before anything is measured.

    The opening calls on CUDA pay for kernel autotuning and allocator warm-up —
    enough to read as ~380 ms against a ~48 ms steady state. Crucially the cost is
    paid *per sequence length*: warming only on a short premise leaves the first
    long one to re-autotune and spike again. So we sweep lengths, which is what the
    real workload does. Left unwarmed, these land as bogus outliers on whichever
    records happened to be scored first.
    """
    unit = "The assistant replied to the user's request with a detailed explanation. "
    ms = []
    for _ in range(rounds):
        for reps in (1, 8, 24, 64):  # short -> past the model's truncation limit
            _, t = oracle.score_timed(unit * reps, "This text is about something.")
            ms.append(t)
    print(f"  warm-up: {max(ms):.0f} ms (cold peak) -> {sum(ms[-4:]) / 4:.0f} ms (steady)")


# --------------------------------------------------------------------------- #
# Small IO helpers.
# --------------------------------------------------------------------------- #
def at(seq, i: int, attr: str | None = None):
    """``seq[i]`` (or its ``attr``) when it exists, else None — timings are optional."""
    if i >= len(seq):
        return None
    item = seq[i]
    return getattr(item, attr) if attr else item


def case_key(*parts: str) -> str:
    """Stable checkpoint key for a record (prompts/responses are too long to key on)."""
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def atomic_write(path: Path, write) -> None:
    """Write via a temp file in the same dir, then rename — never a half-written artifact.

    The suffix is deliberately *not* ``.tmp``: file transfers into ``outputs/``
    stage their downloads as ``<name>.tmp``, and reusing that name would let a
    patch write and an in-flight transfer clobber each other's scratch file.
    """
    tmp = path.with_suffix(path.suffix + ".patch-tmp")
    write(tmp)
    os.replace(tmp, path)


class Checkpoint:
    """Append-only JSONL of measurements already made, so a rerun resumes instead of repeating.

    Each line is flushed the moment it is written: the process can die at any
    point and every completed measurement before it survives. A truncated final
    line from a hard kill is skipped on reload.

    Every line carries the payload schema's ``FORMAT`` version. A line written by a
    different version is *dropped* on load and simply re-measured, rather than being
    fed to code that expects other keys — a resume must never crash on a checkpoint
    it cannot read, least of all twenty hours into a run.
    """

    FORMAT = 2  # bump whenever the payload keys change

    def __init__(self, path: Path, enabled: bool = True) -> None:
        self.path = path
        self.enabled = enabled
        self.done: dict[str, dict] = {}
        stale = 0
        if enabled and path.exists():
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # truncated tail of an interrupted run
                    if rec.get("v") != self.FORMAT:
                        stale += 1
                        continue
                    self.done[rec["k"]] = rec
        if stale:
            print(f"  {path.name}: ignoring {stale} entries from an older format "
                  f"(they will be re-measured).")
        self._fh = None
        if enabled:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = path.open("a", encoding="utf-8")

    def get(self, key: str) -> dict | None:
        return self.done.get(key)

    def put(self, key: str, payload: dict) -> None:
        self.done[key] = payload
        if self._fh is not None:
            self._fh.write(
                json.dumps({"v": self.FORMAT, "k": key, **payload}, ensure_ascii=False) + "\n")
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()


# --------------------------------------------------------------------------- #
# Phase 1 — prompt-side: Gate 1 P(A|Q) + Gate 2 P(D|Q), measured once for everyone.
# --------------------------------------------------------------------------- #
def patch_prompt_side(
    oracle,
    rules_by_id: dict[str, SleecRule],
    path: Path,
    ckpt_dir: Path,
    *,
    smoke: int | None = None,
) -> dict[tuple[str, str], PromptScores]:
    """Time Gate 1 and Gate 2 for every (rule, prompt) in ``prompt_scores.jsonl``.

    Every oracle call is made and timed, but the score it returns is **discarded**:
    the value written back is the one already on file. Only the ``*_ms`` fields are
    new, so no verdict, EVR, RVR or MAS can move.

    Returns the ``(rule_id, prompt) -> PromptScores`` map every later phase copies
    its prompt-side scores and timings from, so Gates 1/2 are paid for exactly once
    across the whole of ``outputs/``.
    """
    records = load_prompt_scores(path)
    if smoke:
        records = records[:smoke]
    ckpt = Checkpoint(ckpt_dir / "prompt_side.jsonl", enabled=not smoke)

    planned = sum(1 + len(rules_by_id[r.rule_id].defeaters)
                  for r in records if r.rule_id in rules_by_id)
    done_calls = sum(1 + len(rules_by_id[r.rule_id].defeaters)
                     for r in records
                     if r.rule_id in rules_by_id and ckpt.get(case_key(r.rule_id, r.prompt)))
    print(f"\n[1/4] prompt-side  Gate 1 + Gate 2 over {len(records)} (rule, prompt) pairs "
          f"-> {planned} oracle calls ({done_calls} already done)")

    out: dict[tuple[str, str], PromptScores] = {}
    bar = tqdm(total=planned, initial=done_calls, desc="gate 1+2", unit="call",
               smoothing=0.02, dynamic_ncols=True)
    try:
        for rec in records:
            rule = rules_by_id.get(rec.rule_id)
            if rule is None:  # rule set changed since the run — leave the record as it was
                out[(rec.rule_id, rec.prompt)] = rec
                continue
            key = case_key(rec.rule_id, rec.prompt)
            hit = ckpt.get(key)
            if hit is None:
                # The returned scores are thrown away — we call the oracle only to
                # find out what the call costs. Gate 1 still runs under the corpus
                # cache's token cap so the cost measured is the cost of the call
                # that actually produced the stored score.
                _, exploit_ms = oracle.score_timed(rec.prompt, rule.antecedent,
                                                   max_length=GATE1_MAX_LENGTH)
                defeater_ms = [oracle.score_timed(rec.prompt, d.defeater)[1]
                               for d in rule.defeaters]
                hit = {"exploit_ms": exploit_ms, "defeater_ms": defeater_ms}
                ckpt.put(key, hit)
                bar.update(1 + len(rule.defeaters))

            # Scores come from the file, never from the calls above; the record's own
            # defeater list drives the output so it can only ever gain a timing.
            out[(rec.rule_id, rec.prompt)] = PromptScores(
                rule_id=rec.rule_id,
                prompt=rec.prompt,
                exploit=rec.exploit,
                exploit_ms=hit["exploit_ms"],
                defeaters=[
                    DefeaterScore(defeater=stored.defeater, entailment=stored.entailment,
                                  consequent=stored.consequent, bd_compliance=None,
                                  entailment_ms=at(hit["defeater_ms"], i))
                    for i, stored in enumerate(rec.defeaters)
                ],
            )
    finally:
        bar.close()
        ckpt.close()

    if smoke:
        print("  smoke run: prompt_scores.jsonl not rewritten.")
    else:
        # Rewrite in the file's original order so the diff is timings, not a reshuffle.
        ordered = [out[(r.rule_id, r.prompt)] for r in records]
        atomic_write(path, lambda p: save_prompt_scores(ordered, p))
        print(f"  wrote {path}")
    return out


# --------------------------------------------------------------------------- #
# Phase 2 — corpus JSON: attach the Gate-1 timings, leave every score untouched.
# --------------------------------------------------------------------------- #
def patch_corpus(
    oracle,
    rules_by_id: dict[str, SleecRule],
    path: Path,
    prompt_side: dict[tuple[str, str], PromptScores],
    *,
    scope: str = "used",
    smoke: int | None = None,
    yes: bool = False,
) -> None:
    """Add ``rule_score_ms`` to each corpus record, mirroring ``rule_scores``.

    This is where Gate 1 was originally precomputed (in one batched pass, so no
    per-pair latency was ever recorded). Its **scores are not recomputed and not
    overwritten** — only timings are added, so the corpus the runs were selected
    from stays exactly as it was.

    ``scope="used"`` (default) copies in the Gate-1 latency measured in phase 1 for
    every (prompt, rule) pair the pipeline actually scored, which is what guarantees
    the corpus and ``prompt_scores.jsonl`` agree pair-for-pair. ``scope="all"`` also
    measures the pairs the pipeline never used (every rule × every prompt), at the
    cost of a fresh oracle call each — the score it produces is discarded.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"  !! {path} is not valid JSON ({e}); skipping. "
              f"Re-upload the file, then rerun with --phase corpus.")
        return

    records = payload.get("prompts", [])
    if smoke:
        records = records[:smoke]
    gate1_ms = {k: ps.exploit_ms for k, ps in prompt_side.items()}

    missing = [
        (rec["prompt"], rid)
        for rec in records
        for rid in rec.get("rule_scores", {})
        if (rid, rec["prompt"]) not in gate1_ms and rid in rules_by_id
    ]
    planned = len(missing) if scope == "all" else 0
    print(f"\n[2/4] corpus      Gate-1 timings for {len(records)} prompts in {path.name} "
          f"(scores untouched) -> {planned} oracle calls")

    # The corpus is the whole WildJailbreak split (~261k prompts x 35 rules), while the
    # runs only ever scored ~131k of those pairs. Timing the rest is a week of GPU, so
    # it cannot be started by a stray flag.
    if planned > 1_000_000 and not yes:
        print(f"  !! --corpus-scope all means {planned:,} oracle calls (~{planned * 0.05 / 3600:.0f}h). "
              f"Rerun with --yes if you really want that; otherwise use the default scope.")
        return

    ckpt = Checkpoint(path.parent / ".gate_timings" / "corpus_gate1.jsonl",
                      enabled=bool(planned) and not smoke)
    bar = tqdm(total=planned, desc="gate 1 (corpus)", unit="call",
               smoothing=0.02, dynamic_ncols=True, disable=not planned)
    try:
        if scope == "all":
            for prompt, rid in missing:
                key = case_key(rid, prompt)
                hit = ckpt.get(key)
                if hit is None:
                    _, ms = oracle.score_timed(prompt, rules_by_id[rid].antecedent,
                                               max_length=GATE1_MAX_LENGTH)
                    hit = {"exploit_ms": ms}  # the score itself is deliberately discarded
                    ckpt.put(key, hit)
                    bar.update(1)
                gate1_ms[(rid, prompt)] = hit["exploit_ms"]
    finally:
        bar.close()
        ckpt.close()

    filled = 0
    for rec in records:
        ms_map = {}
        for rid in rec.get("rule_scores", {}):
            ms = gate1_ms.get((rid, rec["prompt"]))
            if ms is not None:
                ms_map[rid] = round(ms, 3)
        rec["rule_score_ms"] = ms_map  # keys mirror rule_scores; absent = never measured
        filled += len(ms_map)

    total_pairs = sum(len(rec.get("rule_scores", {})) for rec in records)
    print(f"  {filled}/{total_pairs} (prompt, rule) pairs carry a Gate-1 timing"
          + ("" if scope == "all" else "  (pairs the pipeline never scored have none; "
                                       "use --corpus-scope all to time them too)"))
    if smoke:
        print(f"  smoke run: {path.name} not rewritten.")
        return

    # Keep the writer's formatting convention (pretty for small files, compact for big).
    # Streamed to the file rather than dumped to a string first: the corpus is ~540 MB
    # and the intermediate string would be another copy of it in RAM.
    indent = 2 if len(payload.get("prompts", [])) <= 2000 else None

    def dump(p: Path) -> None:
        with p.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=indent)

    atomic_write(path, dump)
    print(f"  wrote {path}")


# --------------------------------------------------------------------------- #
# Phase 3 — per SUT: Gate 3 P(B|R) + P(B_D|R) against that SUT's saved responses.
# --------------------------------------------------------------------------- #
def patch_sut_scores(
    oracle,
    rules_by_id: dict[str, SleecRule],
    sut_dir: Path,
    prompt_side: dict[tuple[str, str], PromptScores],
    *,
    index: str = "",
    smoke: int | None = None,
) -> None:
    """Measure Gate 3 for one SUT and fold in the shared prompt-side scores/timings.

    Gate 3 is the only gate that depends on the response, so it is the only one
    timed here; Gates 1 and 2 are copied verbatim from ``prompt_side``. The saved
    responses are read from ``scores.jsonl`` — the SUT is never called.

    As in phase 1, the Gate-3 calls are made purely to time them: their scores are
    discarded and every value written back is the one already on file. The output
    record therefore differs from the input in its ``*_ms`` fields and nothing else.
    """
    path = sut_dir / "scores.jsonl"
    records = load_scores(path)
    if smoke:
        records = records[:smoke]
    ckpt = Checkpoint(sut_dir / ".gate_timings" / "gate3.jsonl", enabled=not smoke)

    def n_calls(s: CaseScores) -> int:
        rule = rules_by_id.get(s.rule_id)
        return 0 if rule is None else 1 + sum(d.consequent is not None for d in rule.defeaters)

    planned = sum(n_calls(s) for s in records)
    done_calls = sum(n_calls(s) for s in records
                     if ckpt.get(case_key(s.rule_id, s.prompt, s.response)))
    print(f"\n[3/4]{index} {sut_dir.name}  Gate 3 over {len(records)} cases "
          f"-> {planned} oracle calls ({done_calls} already done)")

    out: list[CaseScores] = []
    orphans = 0
    bar = tqdm(total=planned, initial=done_calls, desc="gate 3", unit="call",
               smoothing=0.02, dynamic_ncols=True)
    try:
        for rec in records:
            rule = rules_by_id.get(rec.rule_id)
            if rule is None:  # rule set changed — keep the record as-is, untimed
                out.append(rec)
                continue

            key = case_key(rec.rule_id, rec.prompt, rec.response)
            hit = ckpt.get(key)
            if hit is None:
                # Timed, then thrown away: only the cost of these calls is kept.
                _, compliance_ms = _gate3_score(oracle, rec.response, rule.consequent)
                bd_ms = [None if d.consequent is None  # consequent-less: nothing to score
                         else _gate3_score(oracle, rec.response, d.consequent)[1]
                         for d in rule.defeaters]
                hit = {"compliance_ms": compliance_ms, "bd_ms": bd_ms}
                ckpt.put(key, hit)
                bar.update(n_calls(rec))

            # Gates 1 and 2 are prompt-side: already timed once in phase 1 and simply
            # propagated here. Their *scores* came from this file to begin with.
            ps = prompt_side.get((rec.rule_id, rec.prompt))
            if ps is None:
                orphans += 1
                ps = rec  # fall back to whatever this file already held (no timings)

            # Every score below is the record's own; only the *_ms fields are new.
            # Driven by the record's defeater list (not the rule's), so the output
            # always mirrors the input one-for-one: a rules file that disagreed
            # about defeater count could otherwise drop a stored score.
            out.append(CaseScores(
                rule_id=rec.rule_id,
                prompt=rec.prompt,
                response=rec.response,
                exploit=rec.exploit,
                exploit_ms=ps.exploit_ms,
                compliance=rec.compliance,
                compliance_ms=hit["compliance_ms"],
                defeaters=[
                    DefeaterScore(defeater=stored.defeater, entailment=stored.entailment,
                                  consequent=stored.consequent,
                                  bd_compliance=stored.bd_compliance,
                                  entailment_ms=at(ps.defeaters, i, "entailment_ms"),
                                  bd_compliance_ms=at(hit["bd_ms"], i))
                    for i, stored in enumerate(rec.defeaters)
                ],
            ))
    finally:
        bar.close()
        ckpt.close()

    if orphans:
        print(f"  !! {orphans} cases had no prompt-side entry; their Gates 1/2 stay untimed.")
    if smoke:
        print(f"  smoke run: {path} not rewritten.")
        return
    atomic_write(path, lambda p: save_scores(out, p))
    print(f"  wrote {path}")
    patch_responses(sut_dir, prompt_side)


def patch_responses(sut_dir: Path, prompt_side: dict[tuple[str, str], PromptScores]) -> None:
    """Mirror each ``responses.jsonl`` record's Gate-1 ``exploit`` map with an ``exploit_ms`` one.

    Costs no oracle call: the same (rule, prompt) Gate-1 latencies measured in phase 1,
    propagated once more. The SUT's own prompt and response are copied through
    untouched — this file holds the only irreplaceable data in ``outputs/`` (the model
    answers), so it gains a field and changes in no other way.
    """
    path = sut_dir / "responses.jsonl"
    if not path.exists():
        return
    records = load_responses(path)
    filled = 0
    for rec in records:
        rec.exploit_ms = {}
        for rid in rec.exploit:  # keys mirror `exploit`, so only cached Gate-1 scores
            ps = prompt_side.get((rid, rec.prompt))
            if ps is not None and ps.exploit_ms is not None:
                rec.exploit_ms[rid] = ps.exploit_ms
        filled += len(rec.exploit_ms)
    atomic_write(path, lambda p: save_responses(records, p))
    total = sum(len(r.exploit) for r in records)
    print(f"  wrote {path}  ({filled}/{total} Gate-1 scores now carry their timing)")


# --------------------------------------------------------------------------- #
# Phase 4 — re-resolve results.csv + summary.json (pure post-hoc, no oracle).
# --------------------------------------------------------------------------- #
PROVENANCE = ("mode", "complete", "n_prompts", "n_prompts_used", "source")


def run_provenance(sut_dirs: list[Path]) -> dict:
    """The run-level fields shared by every SUT, read from whichever summary still has them.

    Three of the SUT folders never got a ``summary.json``. The missing fields
    (corpus size, prompt-centric mode, ...) describe the *corpus and selection*,
    not the SUT — and all five SUTs were run over the same 130,696 cases from the
    same corpus — so a surviving sibling summary is the right source for them.
    Returns ``{}`` when no summary survives, in which case the fields are simply
    omitted rather than guessed.
    """
    for d in sut_dirs:
        path = d / "summary.json"
        if path.exists():
            old = json.loads(path.read_text())
            return {k: old[k] for k in PROVENANCE if k in old}
    return {}


def resolve_sut(thresholds, rules_by_id: dict[str, SleecRule], sut_dir: Path,
                fallback: dict | None = None) -> None:
    """Rebuild ``results.csv`` + ``summary.json`` from the freshly timed ``scores.jsonl``.

    No oracle calls: verdicts and the new per-gate latency columns are a pure
    function of the scores file, which is exactly why the timings only had to be
    measured once. A SUT with no ``summary.json`` of its own gets one, borrowing
    the run-level provenance from a sibling (see :func:`run_provenance`).
    """
    scores = load_scores(sut_dir / "scores.jsonl")
    cases = resolve_scores(scores, thresholds, rule_ids=list(rules_by_id))

    # Carry the run's provenance fields over from the summary we are replacing.
    old_path = sut_dir / "summary.json"
    old = json.loads(old_path.read_text()) if old_path.exists() else {}
    extra = {k: old[k] for k in PROVENANCE if k in old} or dict(fallback or {})
    if not old_path.exists():
        print(f"  ({sut_dir.name} had no summary.json — creating one)")
    result = summarise_run(cases, thresholds, old.get("sut", sut_dir.name), sut_dir,
                           verbose=False, **extra)

    overall = result["summary"]["overall"]
    gates = overall.get("gate_timings_ms", {})

    def num(key: str) -> str:  # a rule with no valid exploits has no RVR/MAS
        v = overall.get(key)
        return f"{v:.4f}" if v is not None else "n/a"

    print(f"  {sut_dir.name}: {overall['n_cases']} cases  "
          f"EVR={num('exploit_validity_rate')}  RVR={num('rule_violation_rate')}  "
          f"MAS={num('mean_adherence_score')}")
    for gate, stats in gates.items():
        print(f"      {gate:<20} mean {stats['mean_ms']:7.1f} ms   "
              f"total {stats['total_ms'] / 1000:9.1f} s   (n={stats['n_measured']})")


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backfill measured DeBERTa gate timings into outputs/ without re-querying "
                    "any SUT. Resumable: safe to Ctrl-C and rerun.")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                   help=f"Parameters file (default: {DEFAULT_CONFIG_PATH}).")
    p.add_argument("--phase", choices=PHASES, action="append",
                   help="Run only these phases (repeatable). Default: all, in order.")
    p.add_argument("--sut", action="append",
                   help="Only this SUT output dir, by folder name (repeatable). Default: all.")
    p.add_argument("--corpus-scope", choices=["used", "all"], default="used",
                   help="'used' (default) times only the (prompt, rule) pairs the pipeline "
                        "scored, reusing phase 1's measurements at no extra cost. 'all' also "
                        "times every other pair in the whole corpus (~9M more calls; needs --yes).")
    p.add_argument("--yes", action="store_true",
                   help="Confirm a --corpus-scope all run even when it means millions of calls.")
    p.add_argument("--smoke", type=int, metavar="N",
                   help="Process only the first N records per file and write nothing — a dry "
                        "run to check the wiring and measure the real per-call rate.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    phases = args.phase or list(PHASES)
    config = load_config(args.config)
    outputs = config.output_dir

    rules_by_id = {r.rule_id: r for r in load_rules(config.rules_path)}
    sut_dirs = sorted(d for d in outputs.iterdir()
                      if d.is_dir() and (d / "scores.jsonl").exists()
                      and (not args.sut or d.name in args.sut))

    print(f"outputs: {outputs}")
    print(f"rules:   {len(rules_by_id)}")
    print(f"SUTs:    {', '.join(d.name for d in sut_dirs) or '(none found)'}")
    print(f"phases:  {', '.join(phases)}" + ("   [SMOKE RUN — nothing is written]"
                                             if args.smoke else ""))

    print(f"\nLoading oracle ({config.oracle_model})...")
    oracle = DebertaOracle(model_name=config.oracle_model)
    warm_up(oracle)

    started = time.perf_counter()
    prompt_scores_path = outputs / "prompt_scores.jsonl"
    prompt_side: dict[tuple[str, str], PromptScores] = {}

    if "prompt" in phases:
        prompt_side = patch_prompt_side(oracle, rules_by_id, prompt_scores_path,
                                        outputs / ".gate_timings", smoke=args.smoke)
    elif any(ph in phases for ph in ("corpus", "sut")):
        # Later phases only propagate the prompt-side numbers, so read back the ones
        # phase 1 already wrote instead of measuring anything again.
        print("\nReusing prompt-side timings already in prompt_scores.jsonl.")
        prompt_side = {(ps.rule_id, ps.prompt): ps
                       for ps in load_prompt_scores(prompt_scores_path)}

    if "corpus" in phases and config.prompts_path:
        patch_corpus(oracle, rules_by_id, config.prompts_path, prompt_side,
                     scope=args.corpus_scope, smoke=args.smoke, yes=args.yes)

    if "sut" in phases:
        for i, sut_dir in enumerate(sut_dirs, 1):
            patch_sut_scores(oracle, rules_by_id, sut_dir, prompt_side,
                             index=f"[{i}/{len(sut_dirs)}]", smoke=args.smoke)

    if "resolve" in phases and not args.smoke:
        print(f"\n[4/4] resolve     results.csv + summary.json at "
              f"{vars(config.thresholds)} (no oracle calls)")
        fallback = run_provenance(sut_dirs)
        for sut_dir in sut_dirs:
            resolve_sut(config.thresholds, rules_by_id, sut_dir, fallback)

    print(f"\nDone in {(time.perf_counter() - started) / 60:.1f} min.")


if __name__ == "__main__":
    main()
