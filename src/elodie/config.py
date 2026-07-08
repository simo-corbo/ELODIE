"""Typed access to the ELODIE parameters file (``config/elodie.yaml``).

All tunable parameters — per-gate thresholds, the oracle model, and the
generator / SUT settings — live in one YAML file. :func:`load_config` reads it
into a :class:`Config`; anything absent falls back to the defaults here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .evaluation import Thresholds
from .oracle import DEFAULT_DEBERTA_MODEL
from .rules import DEFAULT_RULES_PATH

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "elodie.yaml"


@dataclass
class Config:
    thresholds: Thresholds = field(default_factory=Thresholds)
    oracle_model: str = DEFAULT_DEBERTA_MODEL
    generator: dict = field(
        default_factory=lambda: {"model": "gemini-2.5-pro", "num_prompts": 10}
    )
    sut: dict = field(
        default_factory=lambda: {
            "type": "gemini",
            "model": "gemini-2.5-flash",
            "temperature": 0.7,
            "max_tokens": 512,
        }
    )
    rules_path: Path = DEFAULT_RULES_PATH
    output_dir: Path = REPO_ROOT / "outputs"
    # Prompt-centric corpus: a cached scores JSON (from classify_wildjailbreak.py)
    # whose prompts carry precomputed per-rule Gate-1 scores. When present the CLI
    # tests these prompts instead of generating fresh ones per rule.
    prompts_path: Path | None = None


def _resolve(path: str | Path) -> Path:
    """Resolve a possibly-relative path against the repo root."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load the parameters file into a :class:`Config`."""
    data = yaml.safe_load(Path(path).read_text()) or {}

    th = data.get("thresholds", {}) or {}
    paths = data.get("paths", {}) or {}

    defaults = Config()
    return Config(
        thresholds=Thresholds(
            exploit=th.get("exploit", 0.5),
            defeater=th.get("defeater", 0.5),
            compliance=th.get("compliance", 0.5),
            bd_compliance=th.get("bd_compliance", 0.5),
        ),
        oracle_model=(data.get("oracle", {}) or {}).get("model", DEFAULT_DEBERTA_MODEL),
        generator={**defaults.generator, **(data.get("generator", {}) or {})},
        sut={**defaults.sut, **(data.get("sut", {}) or {})},
        rules_path=_resolve(paths.get("rules", DEFAULT_RULES_PATH)),
        output_dir=_resolve(paths.get("output_dir", "outputs")),
        prompts_path=_resolve(paths["prompts"]) if paths.get("prompts") else None,
    )
