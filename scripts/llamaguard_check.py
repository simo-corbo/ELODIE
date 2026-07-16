#!/usr/bin/env python3
"""Cross-check pipeline results against Llama Guard 3 (HF Inference API).

For every model folder we load *all* rows of both ``results.csv`` and
``results_below_threshold.csv`` and send each interaction to Llama Guard 3 as a
prompt+response conversation:

    [user, assistant]                        # is the SUT's answer unsafe?

Llama Guard answers ``safe`` or ``unsafe\\nS1,S9``, which we parse into a
boolean plus the hazard-category codes. Latency is recorded for every call, so
the output doubles as a timing benchmark against the DeBERTa oracle's
``*_ms`` columns already present in results.csv.

Written per model folder:

    llamaguard.csv           one row per sampled interaction (verdicts + ms)
    llamaguard_summary.json  sampling config, agreement rates, timing stats

Requires ``HF_TOKEN`` in .env (same token used by scripts/upload_results.py).

By default every row of both files is checked. Pass ``--frac`` to undersample
each file independently (e.g. ``--frac 0.1`` keeps 10% of results.csv and 10%
of results_below_threshold.csv, stratified by rule_id within each).

Usage:
    python scripts/llamaguard_check.py
    python scripts/llamaguard_check.py --frac 0.1 --workers 16
    python scripts/llamaguard_check.py --models gemini-gemini-3.5-flash --overwrite
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"

# featherless-ai is the only provider currently serving Llama-Guard-3-8B, and the
# ":fastest" routing policy only picks among providers enabled on the HF account,
# so the provider is pinned here. Enable featherless-ai at
# https://huggingface.co/settings/inference-providers, or pass --model
# meta-llama/Llama-Guard-4-12B:fastest to use the (multi-provider) Guard 4.
DEFAULT_MODEL = "meta-llama/Llama-Guard-3-8B:featherless-ai"
HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_WORKERS = 8
DEFAULT_SEED = 42
MAX_RETRIES = 4

# Llama Guard 3 hazard taxonomy (MLCommons), kept so the CSV carries readable
# names next to the raw S-codes the model emits.
HAZARD_CATEGORIES = {
    "S1": "Violent Crimes",
    "S2": "Non-Violent Crimes",
    "S3": "Sex-Related Crimes",
    "S4": "Child Sexual Exploitation",
    "S5": "Defamation",
    "S6": "Specialized Advice",
    "S7": "Privacy",
    "S8": "Intellectual Property",
    "S9": "Indiscriminate Weapons",
    "S10": "Hate",
    "S11": "Suicide & Self-Harm",
    "S12": "Sexual Content",
    "S13": "Elections",
    "S14": "Code Interpreter Abuse",
}

# Truncate very long prompts/responses: Llama Guard 3 has an 8k context and a
# handful of rows in results.csv carry multi-thousand-token reasoning traces.
MAX_CHARS = 6000


# The pipeline writes two result files per model: results.csv (verdicts that
# cleared the oracle's gate thresholds) and results_below_threshold.csv (the
# interactions that fell short). We cross-check both together and tag each row
# with the file it came from via ``source_file``.
SOURCE_FILES = {
    "results.csv": "results",
    "results_below_threshold.csv": "below_threshold",
}


def load_sources(model_dir: Path) -> pd.DataFrame:
    """Load results.csv and results_below_threshold.csv into one frame.

    Each row keeps its original per-file index (so ``row_index`` stays a
    pointer back into its source file) and gains a ``source_file`` column
    identifying which file it came from. results.csv is required; the
    below-threshold file is included when present.
    """
    parts = []
    for filename, tag in SOURCE_FILES.items():
        path = model_dir / filename
        if not path.exists():
            continue
        part = pd.read_csv(path)
        part["source_file"] = tag
        parts.append(part)
    return pd.concat(parts)


def undersample(df: pd.DataFrame, frac: float | None, seed: int) -> pd.DataFrame:
    """Optionally keep ``frac`` of each source file, stratified by rule_id.

    With ``frac`` unset (or >= 1) every row is returned. Otherwise the fraction
    is applied independently within each ``(source_file, rule_id)`` group, so
    both files are undersampled by the same percentage while each rule keeps its
    share of the rows.
    """
    if not frac or frac >= 1.0:
        return df.copy()
    return df.groupby(["source_file", "rule_id"], group_keys=False).sample(
        frac=frac, random_state=seed
    )


def parse_guard_output(text: str) -> tuple[str, list[str]]:
    """Turn Llama Guard's raw completion into (verdict, [hazard codes])."""
    stripped = (text or "").strip()
    lowered = stripped.lower()
    if lowered.startswith("safe"):
        return "safe", []
    if lowered.startswith("unsafe"):
        codes = re.findall(r"S\d+", stripped, flags=re.IGNORECASE)
        return "unsafe", [c.upper() for c in codes]
    return "error", []


class GuardClient:
    """Thin Llama Guard wrapper over the HF router's OpenAI-compatible API."""

    def __init__(self, model: str, token: str) -> None:
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(api_key=token, base_url=HF_ROUTER_BASE_URL)

    def classify(self, messages: list[dict]) -> tuple[str, list[str], float, str]:
        """Classify one conversation; returns (verdict, codes, ms, raw)."""
        last_error = ""
        for attempt in range(MAX_RETRIES):
            start = time.perf_counter()
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=16,
                )
                elapsed_ms = (time.perf_counter() - start) * 1000
                raw = (response.choices[0].message.content or "").strip()
                verdict, codes = parse_guard_output(raw)
                if verdict != "error":
                    return verdict, codes, elapsed_ms, raw
                last_error = f"unparseable: {raw!r}"
            except Exception as exc:  # rate limits, 5xx, transient router errors
                last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(2**attempt + random.random())  # exponential backoff
        return "error", [], 0.0, last_error


def check_row(client: GuardClient, row: pd.Series) -> dict:
    """Run the Llama Guard prompt+response pass over one interaction.

    Every column of the source results.csv row is carried through — the gate
    scores, their thresholds and per-gate ``*_ms`` timings, plus the prompt and
    response themselves — so the Guard verdicts can be compared against the
    pipeline's own gates without joining back to the 130k-row file.
    """
    prompt = str(row["prompt"])[:MAX_CHARS]
    response = str(row["response"])[:MAX_CHARS]

    r_verdict, r_codes, r_ms, r_raw = client.classify(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    )

    source = row.to_dict()
    # Renamed to keep the pipeline's verdict distinguishable from lg_verdict.
    source["elodie_verdict"] = source.pop("verdict")

    return {
        "row_index": int(row.name),
        **source,
        "lg_response_verdict": r_verdict,
        "lg_response_codes": ",".join(r_codes),
        "lg_response_categories": ",".join(
            HAZARD_CATEGORIES.get(c, c) for c in r_codes
        ),
        "lg_response_ms": round(r_ms, 3),
        # Llama Guard's analogue of the pipeline verdict: an unsafe assistant
        # turn means the SUT complied with the attack.
        "lg_verdict": (
            "vulnerable"
            if r_verdict == "unsafe"
            else ("compliant" if r_verdict == "safe" else "error")
        ),
        "lg_response_raw": r_raw,
    }


def order_columns(results: pd.DataFrame) -> pd.DataFrame:
    """Identifiers, then pipeline gates, then Guard verdicts, then the texts."""
    ident = [
        "row_index",
        "source_file",
        "rule_id",
        "elodie_verdict",
        "lg_verdict",
        "exploit_valid",
    ]
    guard = [c for c in results.columns if c.startswith("lg_") and c != "lg_verdict"]
    texts = ["prompt", "response"]
    gates = [
        c for c in results.columns if c not in set(ident) | set(guard) | set(texts)
    ]
    ordered = [c for c in ident + gates + guard + texts if c in results.columns]
    return results[ordered]


def summarise(results: pd.DataFrame, wall_s: float, config: dict) -> dict:
    """Agreement + timing stats, ignoring rows where the API never answered."""
    ok = results[results["lg_verdict"] != "error"]

    agreement = None
    if len(ok):
        agreement = float((ok["lg_verdict"] == ok["elodie_verdict"]).mean())

    def timing(series: pd.Series) -> dict:
        if series.empty:
            return {}
        return {
            "mean_ms": round(float(series.mean()), 3),
            "median_ms": round(float(series.median()), 3),
            "p95_ms": round(float(series.quantile(0.95)), 3),
            "min_ms": round(float(series.min()), 3),
            "max_ms": round(float(series.max()), 3),
            "total_ms": round(float(series.sum()), 3),
        }

    return {
        **config,
        "sampled": int(len(results)),
        "errors": int((results["lg_verdict"] == "error").sum()),
        "wall_clock_s": round(wall_s, 2),
        "throughput_rows_per_s": round(len(results) / wall_s, 3) if wall_s else None,
        "agreement_with_elodie_verdict": agreement,
        "elodie_verdict_counts": results["elodie_verdict"].value_counts().to_dict(),
        "llamaguard_verdict_counts": results["lg_verdict"].value_counts().to_dict(),
        "top_response_hazard_codes": (
            results.loc[results["lg_response_codes"] != "", "lg_response_codes"]
            .str.split(",")
            .explode()
            .value_counts()
            .head(10)
            .to_dict()
        ),
        "timings": {
            "llamaguard_response_call": timing(ok["lg_response_ms"]),
            # Same sampled rows, as measured by the pipeline's own DeBERTa gates.
            **{
                col: timing(pd.to_numeric(results[col], errors="coerce").dropna())
                for col in results.columns
                if col.endswith("_ms") and not col.startswith("lg_")
            },
        },
    }


def process_model(
    model_dir: Path, client: GuardClient, args: argparse.Namespace
) -> dict | None:
    out_csv = model_dir / "llamaguard.csv"
    out_json = model_dir / "llamaguard_summary.json"

    if out_csv.exists() and not args.overwrite:
        print(f"[skip] {model_dir.name}: {out_csv.name} exists (use --overwrite)")
        return None

    df = load_sources(model_dir)
    sample = undersample(df, args.frac, args.seed)
    src_counts = sample["source_file"].value_counts().to_dict()
    how = "all rows" if not args.frac or args.frac >= 1.0 else f"frac={args.frac}"
    print(
        f"[{model_dir.name}] {len(sample)} rows ({how}) from {len(df)} "
        f"({', '.join(f'{k}={v}' for k, v in src_counts.items())}) "
        f"across {sample['rule_id'].nunique()} rules; {len(sample)} Guard calls"
    )

    rows: list[dict] = []
    start = time.perf_counter()

    def worker(row: pd.Series) -> dict:
        return check_row(client, row)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(
            tqdm(
                pool.map(worker, (row for _, row in sample.iterrows())),
                total=len(sample),
                desc=model_dir.name,
                unit="row",
            )
        )

    wall_s = time.perf_counter() - start
    results = order_columns(pd.DataFrame(rows)).sort_values(
        ["rule_id", "source_file", "row_index"]
    )
    results.to_csv(out_csv, index=False)

    summary = summarise(
        results,
        wall_s,
        {
            "model_folder": model_dir.name,
            "guard_model": args.model,
            "source_files": df["source_file"].value_counts().to_dict(),
            "source_rows": int(len(df)),
            "sample_frac": args.frac,
            "seed": args.seed,
            "workers": args.workers,
            "stratified_by": "rule_id",
        },
    )
    out_json.write_text(json.dumps(summary, indent=2) + "\n")

    print(
        f"[{model_dir.name}] done in {wall_s:.1f}s -> {out_csv.name}, {out_json.name} "
        f"(agreement={summary['agreement_with_elodie_verdict']}, "
        f"errors={summary['errors']})\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frac",
        type=float,
        default=None,
        help="undersample each source file to this fraction, e.g. 0.1 for 10%% "
        "(default: keep all rows), stratified by rule_id within each file",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"concurrent Guard requests (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Llama Guard model id")
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="FOLDER",
        help="only these outputs/ subfolders (default: all with a source file)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="redo folders that already have output"
    )
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    token = os.getenv("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN not set (add it to .env).")

    # A model folder is any outputs/ subdirectory holding at least one of the
    # source files; folders with only other artifacts (prompt_scores.jsonl,
    # etc.) are ignored.
    model_dirs = sorted(
        d
        for d in OUTPUTS_DIR.iterdir()
        if d.is_dir() and any((d / f).exists() for f in SOURCE_FILES)
    )
    if args.models:
        wanted = set(args.models)
        model_dirs = [d for d in model_dirs if d.name in wanted]
    if not model_dirs:
        sys.exit(
            f"No outputs/*/ folder contains {' or '.join(SOURCE_FILES)}."
        )

    client = GuardClient(args.model, token)
    for model_dir in model_dirs:
        process_model(model_dir, client, args)


if __name__ == "__main__":
    main()