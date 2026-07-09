#!/usr/bin/env python3
"""Attribute a SLEEC requirement category to every rule in sleec_rules.json.

SLEEC = Social, Legal, Ethical, Empathetic, Cultural.

For each rule we run the same DeBERTa-v3 NLI oracle the pipeline already uses
(``MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli``) as a zero-shot
classifier. The rule's text is the premise; each SLEEC category is expressed as
a hypothesis. With ``multi_label=True`` the model returns an *independent*
entailment probability per category (they do not sum to 1), which we keep in
full. The highest-scoring category is recorded as the chosen one.

Two fields are added to each rule object:

    "sleec_category": "Social",                      # argmax over the scores
    "sleec_scores": {                                # every score, kept as-is
        "Social": 0.74, "Legal": 0.12, "Ethical": 0.55,
        "Empathetic": 0.08, "Cultural": 0.03
    }

By default the file is rewritten in place (it is git-tracked, so the change is
reviewable/revertible). Use --output to write elsewhere, or --dry-run to only
print the table.

Examples:
    python scripts/classify_sleec_categories.py                 # in place
    python scripts/classify_sleec_categories.py --dry-run       # preview only
    python scripts/classify_sleec_categories.py --output data/sleec_rules.scored.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Reuse the model constant the pipeline already standardises on.
sys.path.insert(0, str(REPO_ROOT / "src"))
try:
    from elodie.oracle import DEFAULT_DEBERTA_MODEL
except Exception:  # pragma: no cover - fallback if src layout changes
    DEFAULT_DEBERTA_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"

# Canonical SLEEC order, kept stable everywhere for reproducible output.
SLEEC_ORDER = ["Social", "Legal", "Ethical", "Empathetic", "Cultural"]

# Each category rendered as a natural-language hypothesis for the NLI model.
# Descriptive hypotheses give a far cleaner entailment signal than bare words.
SLEEC_HYPOTHESES = {
    "Social": (
        "This requirement is about social norms, societal impact, public safety, "
        "or the wellbeing of communities and social groups."
    ),
    "Legal": (
        "This requirement is about laws, legal rights, regulations, statutory "
        "obligations, or legal compliance."
    ),
    "Ethical": (
        "This requirement is about ethical principles, morality, fairness, honesty, "
        "or preventing harm and wrongdoing."
    ),
    "Empathetic": (
        "This requirement is about empathy, compassion, emotional wellbeing, mental "
        "health, or care for how a vulnerable person feels."
    ),
    "Cultural": (
        "This requirement is about cultural values, traditions, religion, identity, "
        "or respect for cultural differences."
    ),
}

# Rule fields concatenated to form the premise (the "text" of the rule). Order
# is descriptive-first so the model sees what the rule is about up front.
DEFAULT_PREMISE_FIELDS = ["category_heading", "policy_reference", "antecedent", "consequent"]


def build_premise(rule: dict, fields: list[str]) -> str:
    parts = [str(rule[f]).strip() for f in fields if rule.get(f)]
    return " ".join(parts)


def pick_device(explicit: str | None) -> int | str:
    """Return a transformers-pipeline device: 0 (cuda) / 'mps' / -1 (cpu)."""
    import torch

    if explicit:
        if explicit == "cpu":
            return -1
        if explicit == "cuda":
            return 0
        return explicit  # e.g. "mps" or "cuda:1"
    if torch.cuda.is_available():
        return 0
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return -1


def load_classifier(model_name: str, device: int | str):
    from transformers import pipeline

    return pipeline(
        "zero-shot-classification",
        model=model_name,
        token=os.getenv("HF_TOKEN"),
        device=device,
    )


def score_rules(classifier, rules: list[dict], premise_fields: list[str], multi_label: bool) -> list[dict]:
    """Attach sleec_category + sleec_scores to each rule; return per-rule rows.

    multi_label=True  -> independent P(entail) per category (may all be high).
    multi_label=False -> the five categories compete via softmax (scores sum to
                         1), forcing a sharper, more discriminative choice.
    """
    premises = [build_premise(r, premise_fields) for r in rules]
    hypotheses = [SLEEC_HYPOTHESES[c] for c in SLEEC_ORDER]
    hyp_to_cat = {SLEEC_HYPOTHESES[c]: c for c in SLEEC_ORDER}

    # One batched call over all rules.
    outputs = classifier(
        premises,
        candidate_labels=hypotheses,
        hypothesis_template="{}",
        multi_label=multi_label,
    )
    if isinstance(outputs, dict):  # single-item convenience form
        outputs = [outputs]

    rows = []
    for rule, out in zip(rules, outputs):
        scores = {hyp_to_cat[lbl]: float(sc) for lbl, sc in zip(out["labels"], out["scores"])}
        ordered = {cat: round(scores[cat], 4) for cat in SLEEC_ORDER}
        chosen = max(ordered, key=ordered.get)
        rule["sleec_category"] = chosen
        rule["sleec_scores"] = ordered
        rows.append({"rule_id": rule.get("rule_id", "?"), "chosen": chosen, "scores": ordered})
    return rows


def print_table(rows: list[dict]) -> None:
    W = 96
    print("=" * W)
    print("SLEEC CATEGORY ATTRIBUTION  (DeBERTa-v3 NLI, independent per-category probability)")
    print("=" * W)
    header = f"{'RULE':<33} {'CHOSEN':<11}" + "".join(f"{c[:5]:>8}" for c in SLEEC_ORDER)
    print(header)
    print("-" * W)
    for r in rows:
        cells = "".join(f"{r['scores'][c]:>8.2f}" for c in SLEEC_ORDER)
        print(f"{r['rule_id']:<33} {r['chosen']:<11}{cells}")
    print("-" * W)

    counts: dict[str, int] = {c: 0 for c in SLEEC_ORDER}
    for r in rows:
        counts[r["chosen"]] += 1
    dist = "   ".join(f"{c}: {counts[c]}" for c in SLEEC_ORDER)
    print(f"Chosen-category distribution over {len(rows)} rules:  {dist}")
    print("=" * W)


def write_json_atomic(doc: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rules", type=Path, default=REPO_ROOT / "data" / "sleec_rules.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the augmented JSON (default: in place over --rules).",
    )
    parser.add_argument("--model", default=DEFAULT_DEBERTA_MODEL)
    parser.add_argument(
        "--device",
        default=None,
        help="Force device: cpu | cuda | mps | cuda:N (default: auto-detect).",
    )
    parser.add_argument(
        "--premise-fields",
        default=",".join(DEFAULT_PREMISE_FIELDS),
        help="Comma-separated rule fields concatenated as the classifier premise.",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "multi"],
        default="single",
        help="single: categories compete (softmax, sum to 1, sharper choice). "
        "multi: independent P(entail) per category. Default: single.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the table but do not write.")
    args = parser.parse_args(argv)

    premise_fields = [f.strip() for f in args.premise_fields.split(",") if f.strip()]
    out_path = args.output or args.rules

    doc = json.loads(args.rules.read_text(encoding="utf-8"))
    rules = doc.get("rules", [])
    if not rules:
        print("No rules found.", file=sys.stderr)
        return 1

    device = pick_device(args.device)
    print(f"Loading '{args.model}' on device={device} ...", file=sys.stderr)
    classifier = load_classifier(args.model, device)

    print(
        f"Scoring {len(rules)} rules against {len(SLEEC_ORDER)} SLEEC categories "
        f"(mode={args.mode}) ...",
        file=sys.stderr,
    )
    rows = score_rules(classifier, rules, premise_fields, multi_label=(args.mode == "multi"))

    print_table(rows)

    if args.dry_run:
        print("\n[dry-run] No file written.", file=sys.stderr)
        return 0

    write_json_atomic(doc, out_path)
    print(f"\nWrote SLEEC fields for {len(rules)} rules -> {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
