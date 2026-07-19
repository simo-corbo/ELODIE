"""Prompt-centric input for ELODIE: real prompts with precomputed Gate-1 scores.

The rule-centric pipeline *generates* adversarial prompts per rule (Phase 1) and
then scores each one through Gate 1 (Q => A) with the oracle. The prompt-centric
pipeline instead reads a fixed corpus of real prompts whose per-rule Gate-1
entailment scores were *already* computed offline by ``classify_wildjailbreak.py``
and cached to JSON. Every prompt therefore arrives with a ``{rule_id: P(A|Q)}``
map, so Gate 1 costs no oracle call at run time: a prompt "triggers" a rule
exactly when that stored score clears tau_1, and we only ever test the (prompt,
rule) pairs a prompt already respects the Gate-1 threshold for.

The entail-mode scores in the cache use the *same formula* as
:meth:`elodie.oracle.DebertaOracle.score` — both are
``sigmoid(logit_entail - logit_contradict)`` for the same (prompt, antecedent)
pair — so reusing them as Gate 1 measures exactly the quantity Gate 1 wants,
not a proxy. They are not guaranteed bit-identical to a live oracle call,
though: ``classify_wildjailbreak.py`` builds the cache under fp16 autocast and
truncates pairs to 256 tokens, whereas a direct ``score`` call runs at the
pipeline's default precision and truncation length, so long prompts can differ
slightly. Regenerate the cache with a different rule set and the ``rule_id`` keys
move with it; the loader only trusts ids it recognises and ignores the rest.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

# Gate 1's cached scores were precomputed by classify_wildjailbreak.py, which truncates
# each (prompt, antecedent) pair at 256 tokens. Anyone *timing* a cached score must call
# under the same cap: at the model's default 512 the prompt is no longer truncated, so it
# is a bigger, slower — and different — call than the one that produced the stored score.
# On a random sample of 150 stored cases, re-scoring at 512 moved Gate 1 by 0.12 on average
# and flipped 12% across tau_1, versus a mean 6e-4 and one borderline flip at 256. Gates 2
# and 3 keep the model default, which is what the live pipeline scored them at.
GATE1_MAX_LENGTH = 256


@dataclass(frozen=True)
class PromptRecord:
    """One corpus prompt with its precomputed per-rule Gate-1 scores.

    ``rule_scores`` is the ``{rule_id: P(A|Q)}`` map straight from the cache,
    keyed by the same ``rule_id`` values used in ``sleec_rules.json``.
    ``rule_score_ms`` mirrors it with the latency of each Gate-1 call, so the cost
    of the precompute is carried into the run rather than lost with it. It is empty
    for a cache written before timings were recorded.
    """

    prompt: str                    # Q
    rule_scores: dict[str, float]  # {rule_id: P(A|Q)} over every scored rule
    data_type: str = "unknown"     # WildJailbreak provenance label, if any
    rule_score_ms: dict[str, float] = field(default_factory=dict)  # {rule_id: Gate-1 ms}

    def triggered_rules(self, threshold: float) -> list[str]:
        """Rule ids whose Gate-1 score clears tau_1, strongest first."""
        hits = [(rid, s) for rid, s in self.rule_scores.items() if s >= threshold]
        hits.sort(key=lambda pair: pair[1], reverse=True)
        return [rid for rid, _ in hits]

    def top_score(self, rule_ids: Iterable[str] | None = None) -> float:
        """Strongest Gate-1 score, optionally restricted to ``rule_ids``."""
        ids = self.rule_scores.keys() if rule_ids is None else rule_ids
        return max((self.rule_scores.get(rid, 0.0) for rid in ids), default=0.0)


def load_prompt_records(path: str | Path) -> list[PromptRecord]:
    """Read every prompt (with its per-rule scores) from a cached scores JSON."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        PromptRecord(
            prompt=rec["prompt"],
            rule_scores={rid: float(s) for rid, s in (rec.get("rule_scores") or {}).items()},
            data_type=rec.get("data_type", "unknown"),
            rule_score_ms={rid: float(ms)
                           for rid, ms in (rec.get("rule_score_ms") or {}).items()},
        )
        for rec in payload.get("prompts", [])
        if rec.get("prompt")
    ]


def select_prompts(
    records: list[PromptRecord],
    threshold: float,
    *,
    rule_ids: Iterable[str] | None = None,
    limit: int | None = None,
    exclude_unassigned: bool = True,
    stratify: bool = False,
) -> list[PromptRecord]:
    """Pick and order the prompts to test, with optional capping and stratification.

    Each prompt is *assigned* to the rule under test it triggers most strongly
    (its argmax over rules clearing tau_1 — the schema's matching step). A prompt
    that triggers no rule under test is **unassigned**.

    * ``rule_ids`` restricts "under test" to a subset (e.g. a single ``--rule-id``).
    * ``exclude_unassigned`` (default) drops unassigned prompts — they would run
      the SUT for zero cases. Set ``False`` to keep them in the returned list
      (e.g. to report dataset coverage); they still produce no cases downstream.
    * ``limit`` caps how many prompts come back. Without ``stratify`` the cap
      keeps the globally strongest-triggering prompts (best exploits first). With
      ``stratify`` the cap is filled **round-robin across assigned rules**, so
      every rule that has any triggering prompt is represented before any rule
      gets a second — a balanced sample across the rules used, rather than one
      dominated by whichever rules own the highest-scoring prompts.
    """
    targets = set(rule_ids) if rule_ids is not None else None

    # (assigned_rule, strongest_score, record); assigned_rule is None if unassigned.
    assigned: list[tuple[str | None, float, PromptRecord]] = []
    for rec in records:
        trig = rec.triggered_rules(threshold)
        if targets is not None:
            trig = [rid for rid in trig if rid in targets]
        if trig:
            assigned.append((trig[0], rec.rule_scores[trig[0]], rec))  # trig[0] = strongest
        elif not exclude_unassigned:
            assigned.append((None, 0.0, rec))

    if stratify and limit:
        return _stratified_sample(assigned, limit)

    assigned.sort(key=lambda t: t[1], reverse=True)
    ordered = [rec for _, _, rec in assigned]
    return ordered[:limit] if limit else ordered


def _stratified_sample(
    assigned: list[tuple[str | None, float, PromptRecord]], limit: int
) -> list[PromptRecord]:
    """Fill ``limit`` round-robin across assigned rules (each bucket strongest-first).

    Deterministic: rules are visited in id order (unassigned, if kept, last) and
    each rule's prompts are drawn highest-score first.
    """
    from collections import deque

    buckets: dict[str | None, list[tuple[float, PromptRecord]]] = {}
    for rule, score, rec in assigned:
        buckets.setdefault(rule, []).append((score, rec))

    queues: dict[str | None, deque[PromptRecord]] = {}
    for rule, items in buckets.items():
        items.sort(key=lambda t: t[0], reverse=True)
        queues[rule] = deque(rec for _, rec in items)

    order = sorted(queues, key=lambda r: (r is None, r or ""))  # unassigned bucket last
    out: list[PromptRecord] = []
    while len(out) < limit and any(queues[r] for r in order):
        for r in order:
            if queues[r]:
                out.append(queues[r].popleft())
                if len(out) >= limit:
                    break
    return out
