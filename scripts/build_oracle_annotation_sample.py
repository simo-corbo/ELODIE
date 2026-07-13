#!/usr/bin/env python3
"""
Build a balanced manual-annotation sample for validating ELODIE's NLI gates.

The script samples premise-hypothesis-score triples from:

  G1     prompt       => antecedent A
  G2     prompt+resp  => defeater D
  G3_C   response     => base consequent C
  G3_CD  response     => defeater consequent C_D

It balances the sample by gate and score band:
  low        [0.00, 0.30)
  borderline [0.30, 0.70]
  high       (0.70, 1.00]

Outputs:
  annotation_sample_master.csv
  annotation_sample_annotator_A.csv
  annotation_sample_annotator_B.csv
  annotation_sampling_report.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCORE_BANDS = {
    "low": (0.0, 0.3),
    "borderline": (0.3, 0.7),
    "high": (0.7, 1.0000001),
}


@dataclass
class RuleInfo:
    rule_id: str
    category_id: str
    antecedent: str
    consequent: str
    defeaters: list[str]
    defeater_consequents: list[str]


def score_band(score: float) -> str | None:
    if 0.0 <= score < 0.3:
        return "low"
    if 0.3 <= score <= 0.7:
        return "borderline"
    if 0.7 < score <= 1.0:
        return "high"
    return None


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_rules(path: Path) -> dict[str, RuleInfo]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rules = {}

    for r in payload.get("rules", []):
        rid = r["rule_id"]
        defeaters = []
        defeater_consequents = []

        for d in r.get("defeaters", []) or []:
            defeater = str(d.get("defeater", "")).strip()
            consequent = str(d.get("consequent", "")).strip()
            if defeater:
                defeaters.append(defeater)
            if consequent:
                defeater_consequents.append(consequent)

        rules[rid] = RuleInfo(
            rule_id=rid,
            category_id=r.get("category_id", ""),
            antecedent=str(r.get("antecedent", "")).strip(),
            consequent=str(r.get("consequent", "")).strip(),
            defeaters=defeaters,
            defeater_consequents=defeater_consequents,
        )

    return rules


def combine_defeaters(rule: RuleInfo) -> str:
    if not rule.defeaters:
        return ""
    if len(rule.defeaters) == 1:
        return rule.defeaters[0]
    return "At least one of the following allowed exception conditions applies: " + " OR ".join(
        f"({d})" for d in rule.defeaters
    )


def combine_defeater_consequents(rule: RuleInfo) -> str:
    if not rule.defeater_consequents:
        return ""
    if len(rule.defeater_consequents) == 1:
        return rule.defeater_consequents[0]
    return "The response satisfies at least one of the following exception-case requirements: " + " OR ".join(
        f"({c})" for c in rule.defeater_consequents
    )


class StratifiedReservoir:
    def __init__(self, n_per_stratum: int, seed: int):
        self.n = n_per_stratum
        self.rng = random.Random(seed)
        self.counts: dict[tuple[str, str], int] = {}
        self.samples: dict[tuple[str, str], list[dict[str, Any]]] = {}

    def add(self, gate: str, band: str, row: dict[str, Any]) -> None:
        key = (gate, band)
        seen = self.counts.get(key, 0) + 1
        self.counts[key] = seen

        bucket = self.samples.setdefault(key, [])
        if len(bucket) < self.n:
            bucket.append(row)
            return

        j = self.rng.randint(1, seen)
        if j <= self.n:
            bucket[j - 1] = row

    def rows(self) -> list[dict[str, Any]]:
        out = []
        for gate in sorted({k[0] for k in self.samples}):
            for band in ["low", "borderline", "high"]:
                out.extend(self.samples.get((gate, band), []))
        return out


def deberta_label(score: float, threshold: float) -> str:
    return "ENTAILED" if score >= threshold else "NOT_ENTAILED"


def make_item(
        *,
        gate: str,
        rule: RuleInfo,
        premise: str,
        hypothesis: str,
        score: float,
        threshold: float,
        source_kind: str,
        source_file: str,
        lut: str = "",
        prompt: str = "",
        response: str = "",
) -> dict[str, Any]:
    band = score_band(score)
    if band is None:
        raise ValueError(f"Invalid score for banding: {score}")

    return {
        "item_id": "",
        "gate": gate,
        "score_band": band,
        "rule_id": rule.rule_id,
        "category_id": rule.category_id,
        "lut": lut,
        "deberta_score": round(score, 6),
        "threshold": threshold,
        "deberta_label": deberta_label(score, threshold),
        "premise": premise,
        "hypothesis": hypothesis,
        "source_kind": source_kind,
        "source_file": source_file,
        "prompt": prompt,
        "response": response,
        "human_label": "",
        "notes": "",
    }


def add_gate1_candidates(
        reservoir: StratifiedReservoir,
        *,
        prompt_scores_path: Path,
        rules: dict[str, RuleInfo],
        threshold: float,
        max_prompts: int | None,
) -> None:
    payload = json.loads(prompt_scores_path.read_text(encoding="utf-8"))
    prompts = payload.get("prompts", [])

    if max_prompts is not None:
        prompts = prompts[:max_prompts]

    for rec in prompts:
        prompt = str(rec.get("prompt", "")).strip()
        if not prompt:
            continue

        rule_scores = rec.get("rule_scores") or {}
        for rid, raw_score in rule_scores.items():
            if rid not in rules:
                continue

            score = parse_float(raw_score)
            if score is None:
                continue

            band = score_band(score)
            if band is None:
                continue

            rule = rules[rid]
            row = make_item(
                gate="G1_ANTECEDENT",
                rule=rule,
                premise=prompt,
                hypothesis=rule.antecedent,
                score=score,
                threshold=threshold,
                source_kind="prompt_scores",
                source_file=str(prompt_scores_path),
                prompt=prompt,
            )
            reservoir.add("G1_ANTECEDENT", band, row)


def iter_result_files(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pat in patterns:
        files.extend(Path(p) for p in glob.glob(pat))
    return sorted(set(files))


def read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def add_result_candidates(
        reservoir: StratifiedReservoir,
        *,
        result_files: list[Path],
        rules: dict[str, RuleInfo],
        threshold: float,
) -> None:
    for file in result_files:
        lut = file.parent.name

        for row in read_csv_rows(file):
            rid = row.get("rule_id", "")
            if rid not in rules:
                continue

            rule = rules[rid]
            prompt = str(row.get("prompt", "")).strip()
            response = str(row.get("response", "")).strip()

            if not response:
                continue

            combined_premise = f"PROMPT:\n{prompt}\n\nRESPONSE:\n{response}"

            # Gate 2: (prompt + response) => D
            if rule.defeaters:
                score = parse_float(row.get("gate2_defeater"))
                if score is not None:
                    band = score_band(score)
                    if band is not None:
                        reservoir.add(
                            "G2_DEFEATER",
                            band,
                            make_item(
                                gate="G2_DEFEATER",
                                rule=rule,
                                premise=combined_premise,
                                hypothesis=combine_defeaters(rule),
                                score=score,
                                threshold=threshold,
                                source_kind="results",
                                source_file=str(file),
                                lut=lut,
                                prompt=prompt,
                                response=response,
                            ),
                        )

            # Gate 3 base: response => C
            score = parse_float(row.get("gate3_b_compliance"))
            if score is not None:
                band = score_band(score)
                if band is not None:
                    reservoir.add(
                        "G3_BASE_CONSEQUENT",
                        band,
                        make_item(
                            gate="G3_BASE_CONSEQUENT",
                            rule=rule,
                            premise=response,
                            hypothesis=rule.consequent,
                            score=score,
                            threshold=threshold,
                            source_kind="results",
                            source_file=str(file),
                            lut=lut,
                            prompt=prompt,
                            response=response,
                        ),
                    )

            # Gate 3 defeater consequent: response => C_D
            if rule.defeater_consequents:
                score = parse_float(row.get("gate3_bd_compliance"))
                if score is not None:
                    band = score_band(score)
                    if band is not None:
                        reservoir.add(
                            "G3_DEFEATER_CONSEQUENT",
                            band,
                            make_item(
                                gate="G3_DEFEATER_CONSEQUENT",
                                rule=rule,
                                premise=response,
                                hypothesis=combine_defeater_consequents(rule),
                                score=score,
                                threshold=threshold,
                                source_kind="results",
                                source_file=str(file),
                                lut=lut,
                                prompt=prompt,
                                response=response,
                            ),
                        )


def assign_item_ids(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = sorted(rows, key=lambda r: (r["gate"], r["score_band"], r["rule_id"], r["lut"], r["deberta_score"]))
    for i, row in enumerate(rows, start=1):
        row["item_id"] = f"ann_{i:04d}"
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, reservoir: StratifiedReservoir, rows: list[dict[str, Any]]) -> None:
    selected_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        selected_counts.setdefault(row["gate"], {})
        selected_counts[row["gate"]][row["score_band"]] = selected_counts[row["gate"]].get(row["score_band"], 0) + 1

    candidate_counts: dict[str, dict[str, int]] = {}
    for (gate, band), n in reservoir.counts.items():
        candidate_counts.setdefault(gate, {})
        candidate_counts[gate][band] = n

    payload = {
        "selected_total": len(rows),
        "candidate_counts": candidate_counts,
        "selected_counts": selected_counts,
        "score_bands": {
            "low": "[0.00, 0.30)",
            "borderline": "[0.30, 0.70]",
            "high": "(0.70, 1.00]",
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build balanced ELODIE NLI annotation sample.")
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--prompt-scores", type=Path, required=True)
    parser.add_argument(
        "--results-glob",
        nargs="+",
        default=["outputs/*/results.revised.csv", "outputs/*/results_revised.csv"],
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/oracle_annotation"))
    parser.add_argument("--n-per-gate-band", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-prompts-for-g1",
        type=int,
        default=None,
        help="Optional cap for debugging; omit for the full prompt-score file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rules = load_rules(args.rules)
    result_files = iter_result_files(args.results_glob)

    if not result_files:
        raise FileNotFoundError(f"No result files matched: {args.results_glob}")

    reservoir = StratifiedReservoir(
        n_per_stratum=args.n_per_gate_band,
        seed=args.seed,
    )

    add_gate1_candidates(
        reservoir,
        prompt_scores_path=args.prompt_scores,
        rules=rules,
        threshold=args.threshold,
        max_prompts=args.max_prompts_for_g1,
    )

    add_result_candidates(
        reservoir,
        result_files=result_files,
        rules=rules,
        threshold=args.threshold,
    )

    rows = assign_item_ids(reservoir.rows())

    master_fields = [
        "item_id",
        "gate",
        "score_band",
        "rule_id",
        "category_id",
        "lut",
        "deberta_score",
        "threshold",
        "deberta_label",
        "premise",
        "hypothesis",
        "source_kind",
        "source_file",
        "prompt",
        "response",
        "human_label",
        "notes",
    ]

    annotator_fields = [
        "item_id",
        "gate",
        "rule_id",
        "category_id",
        "premise",
        "hypothesis",
        "human_label",
        "notes",
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(args.out_dir / "annotation_sample_master.csv", rows, master_fields)
    write_csv(args.out_dir / "annotation_sample_annotator_A.csv", rows, annotator_fields)
    write_csv(args.out_dir / "annotation_sample_annotator_B.csv", rows, annotator_fields)
    write_report(args.out_dir / "annotation_sampling_report.json", reservoir, rows)

    print(f"Wrote {len(rows)} annotation items to {args.out_dir}")
    print(f"Result files used: {len(result_files)}")
    print("Selected items by gate and band:")
    for gate in sorted({r["gate"] for r in rows}):
        counts = {b: sum(1 for r in rows if r["gate"] == gate and r["score_band"] == b) for b in SCORE_BANDS}
        print(f"  {gate}: {counts}")


if __name__ == "__main__":
    main()
