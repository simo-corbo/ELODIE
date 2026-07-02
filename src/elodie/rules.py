import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_RULES_PATH = Path(__file__).resolve().parents[2] / "data" / "sleec_rules.json"


@dataclass(frozen=True)
class Defeater:
    defeater: str  # D — the exception condition
    # B_D — the behaviour required *under* the exception. When None, triggering
    # the defeater fully waives the rule (a pure bypass) with no behaviour check.
    consequent: str | None = None


@dataclass(frozen=True)
class SleecRule:
    rule_id: str
    category_id: str
    category_heading: str
    policy_reference: str
    antecedent: str
    consequent: str
    defeaters: tuple[Defeater, ...] = field(default_factory=tuple)
    excluded_from_generation: bool = False
    exclusion_reason: str | None = None


def load_rules(path: Path = DEFAULT_RULES_PATH) -> list[SleecRule]:
    data = json.loads(Path(path).read_text())
    rules = []
    for r in data["rules"]:
        defeaters = tuple(
            Defeater(defeater=d["defeater"], consequent=d.get("consequent"))
            for d in r.get("defeaters", [])
        )
        rules.append(
            SleecRule(
                rule_id=r["rule_id"],
                category_id=r["category_id"],
                category_heading=r["category_heading"],
                policy_reference=r["policy_reference"],
                antecedent=r["antecedent"],
                consequent=r["consequent"],
                defeaters=defeaters,
                excluded_from_generation=r.get("excluded_from_generation", False),
                exclusion_reason=r.get("exclusion_reason"),
            )
        )
    return rules


def get_rule(rule_id: str, path: Path = DEFAULT_RULES_PATH) -> SleecRule:
    for rule in load_rules(path):
        if rule.rule_id == rule_id:
            return rule
    raise KeyError(f"No SLEEC rule with rule_id={rule_id!r} in {path}")
