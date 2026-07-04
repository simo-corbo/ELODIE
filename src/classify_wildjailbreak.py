"""Classify each WildJailbreak prompt against the SLEEC rule set with DeBERTa.

Loads the ``allenai/wildjailbreak`` dataset and, for every prompt, uses the same
DeBERTa-v3 NLI oracle that powers ELODIE's Gate 1 (Q => A) to find the SLEEC
rule whose antecedent the prompt most strongly entails -- i.e. the rule the
prompt best *triggers*. It then plots, for each of the dataset's four
``data_type`` categories, the distribution of best-matching SLEEC categories
(or rules, with ``--by rule``).

Run from the ``src/`` directory (so ``elodie`` is importable):

    python classify_wildjailbreak.py --limit 500
    python classify_wildjailbreak.py --limit 0 --by rule      # whole split (slow)

The dataset is gated, so an ``HF_TOKEN`` must be set in the environment (or a
``.env`` file). Outputs a per-prompt CSV and a distribution PNG under
``outputs/wildjailbreak/``.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter, defaultdict
from pathlib import Path

import torch
from dotenv import load_dotenv
from tqdm import tqdm

from elodie.oracle import DEFAULT_DEBERTA_MODEL, DebertaOracle
from elodie.rules import DEFAULT_RULES_PATH, SleecRule, load_rules

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "wildjailbreak"

# Fixed display order for the dataset's four data_type buckets.
DATA_TYPES = ["vanilla_harmful", "vanilla_benign", "adversarial_harmful", "adversarial_benign"]


def effective_prompt(row: dict) -> str:
    """The text to classify: the adversarial prompt if present, else the vanilla one.

    Vanilla rows carry an empty ``adversarial`` string, so this collapses each
    row to the single prompt a user would actually send.
    """
    adversarial = (row.get("adversarial") or "").strip()
    return adversarial or (row.get("vanilla") or "").strip()


def _entailment_indices(config) -> tuple[int, int | None]:
    """(entailment, contradiction) class indices from the model's label map."""
    label2id = {k.lower(): v for k, v in config.label2id.items()}
    return label2id["entailment"], label2id.get("contradiction")


def classify_prompts(
    oracle: DebertaOracle,
    prompts: list[str],
    rules: list[SleecRule],
    batch_size: int,
) -> list[tuple[SleecRule, float]]:
    """Best-matching rule (argmax entailment) and its score for each prompt.

    Every prompt is scored against *every* rule antecedent as an NLI
    (premise=prompt, hypothesis=antecedent) pair; the highest-entailed
    antecedent wins. The entailment probability is softmax over the
    entail/contradiction logits, matching the oracle's ``multi_label=True``
    P(A|Q) used in Gate 1 -- so a prompt's best match here is the rule it would
    most strongly trigger in the pipeline.

    For speed this bypasses the zero-shot *pipeline* (which runs sequentially and
    emits the "use a dataset" warning) and drives the model+tokenizer already
    loaded inside the oracle directly: one dynamically-padded forward pass per
    tile of prompts, batched on the GPU under ``inference_mode`` (+ fp16 autocast
    on CUDA). ``batch_size`` is the target number of (prompt, antecedent) pairs
    per forward pass.
    """
    model = oracle.pipeline.model.eval()
    tokenizer = oracle.pipeline.tokenizer
    device = next(model.parameters()).device
    if device.type != "cuda" and torch.cuda.is_available():
        model.to("cuda")
        device = next(model.parameters()).device
    use_amp = device.type == "cuda"
    ent_idx, con_idx = _entailment_indices(model.config)

    antecedents = [r.antecedent for r in rules]
    n_labels = len(antecedents)
    # How many prompts we can fold into one forward pass while staying near the
    # requested pair budget (each prompt expands into n_labels pairs).
    prompts_per_pass = max(1, batch_size // n_labels)

    matches: list[tuple[SleecRule, float]] = []
    for start in tqdm(
        range(0, len(prompts), prompts_per_pass), desc="classifying", unit="pass"
    ):
        chunk = prompts[start : start + prompts_per_pass]
        premises = [p for p in chunk for _ in range(n_labels)]
        hypotheses = antecedents * len(chunk)

        enc = tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=use_amp):
            logits = model(**enc).logits.float()

        if con_idx is not None:
            probs = logits[:, [ent_idx, con_idx]].softmax(-1)[:, 0]
        else:
            probs = logits.softmax(-1)[:, ent_idx]

        probs = probs.view(len(chunk), n_labels)
        best_idx = probs.argmax(dim=1)
        best_score = probs.gather(1, best_idx.unsqueeze(1)).squeeze(1)
        for i, s in zip(best_idx.tolist(), best_score.tolist()):
            matches.append((rules[i], float(s)))

    return matches


def load_wildjailbreak(config: str, split: str, limit: int, seed: int, token: str | None):
    """Load (and optionally subsample) the WildJailbreak split."""
    from datasets import load_dataset

    ds = load_dataset(
        "allenai/wildjailbreak",
        config,
        split=split,
        delimiter="\t",
        keep_default_na=False,  # required: keeps empty adversarial strings as ""
        token=token,
    )
    if 0 < limit < len(ds):
        ds = ds.shuffle(seed=seed).select(range(limit))
    return ds


def ordered_keys(rules: list[SleecRule], level: str) -> list[str]:
    """Category ids (deduped) or rule ids, in rules-file order, for a stable axis."""
    if level == "rule":
        return [r.rule_id for r in rules]
    seen: list[str] = []
    for r in rules:
        if r.category_id not in seen:
            seen.append(r.category_id)
    return seen


def save_csv(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["data_type", "best_rule_id", "best_category_id", "score", "prompt"]
        )
        writer.writeheader()
        writer.writerows(records)


def plot_distribution(
    counts_by_type: dict[str, Counter],
    keys: list[str],
    level: str,
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    present = [dt for dt in DATA_TYPES if dt in counts_by_type]
    present += [dt for dt in counts_by_type if dt not in present]

    ncols = 2 if len(present) > 1 else 1
    nrows = (len(present) + ncols - 1) // ncols
    height = max(4.0, 0.32 * len(keys)) * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, height), squeeze=False)
    y = range(len(keys))
    colors = plt.cm.tab10(range(len(present)))

    for ax, dt, color in zip(axes.flat, present, colors):
        counts = counts_by_type[dt]
        vals = [counts.get(k, 0) for k in keys]
        ax.barh(list(y), vals, color=color)
        ax.set_yticks(list(y))
        ax.set_yticklabels(keys, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"{dt}  (n={sum(vals)})", fontsize=10)
        ax.set_xlabel("prompts")
        ax.grid(axis="x", alpha=0.3)

    for ax in axes.flat[len(present):]:
        ax.axis("off")

    fig.suptitle(
        f"Best-matching SLEEC {level} distribution by WildJailbreak data_type",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match WildJailbreak prompts to their best-fitting SLEEC rule with DeBERTa."
    )
    parser.add_argument("--config", default="train",
                        help="WildJailbreak dataset config: train | eval (default: train).")
    parser.add_argument("--split", default="train", help="Dataset split (default: train).")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Random sample size; <= 0 uses the entire split (default: -1, all).")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed (default: 0).")
    parser.add_argument("--by", choices=["category", "rule"], default="category",
                        help="Aggregate the distribution by SLEEC category or rule (default: category).")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Target (prompt, antecedent) pairs per DeBERTa forward pass "
                             "(default: 256). Raise it to saturate a larger GPU.")
    parser.add_argument("--model", default=DEFAULT_DEBERTA_MODEL,
                        help=f"NLI model id (default: {DEFAULT_DEBERTA_MODEL}).")
    parser.add_argument("--rules-file", type=Path, default=DEFAULT_RULES_PATH,
                        help="Path to the SLEEC rules JSON file.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory for the CSV and PNG (default: outputs/wildjailbreak).")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        print("Warning: no HF_TOKEN set; loading the gated WildJailbreak dataset will likely fail.")

    rules = load_rules(args.rules_file)
    print(f"Loaded {len(rules)} SLEEC rules.")

    print(f"Loading WildJailbreak ({args.config}/{args.split})...")
    ds = load_wildjailbreak(args.config, args.split, args.limit, args.seed, token)

    rows = list(ds)
    prompts = [effective_prompt(r) for r in rows]
    kept = [(r, p) for r, p in zip(rows, prompts) if p]
    if len(kept) < len(rows):
        print(f"Skipped {len(rows) - len(kept)} rows with no usable prompt text.")
    rows = [r for r, _ in kept]
    prompts = [p for _, p in kept]
    print(f"Classifying {len(prompts)} prompts against {len(rules)} antecedents...")

    print(f"Loading oracle ({args.model})...")
    oracle = DebertaOracle(model_name=args.model, token=token)
    device = next(oracle.pipeline.model.parameters()).device
    if device.type != "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    print(f"Running on {device} (batch_size={args.batch_size} pairs/pass).")
    matches = classify_prompts(oracle, prompts, rules, args.batch_size)

    key = (lambda rule: rule.category_id) if args.by == "category" else (lambda rule: rule.rule_id)
    counts_by_type: dict[str, Counter] = defaultdict(Counter)
    records: list[dict] = []
    for row, prompt, (rule, score) in zip(rows, prompts, matches):
        data_type = row.get("data_type") or "unknown"
        counts_by_type[data_type][key(rule)] += 1
        records.append({
            "data_type": data_type,
            "best_rule_id": rule.rule_id,
            "best_category_id": rule.category_id,
            "score": round(score, 4),
            "prompt": prompt,
        })

    csv_path = args.output_dir / "wildjailbreak_matches.csv"
    png_path = args.output_dir / f"wildjailbreak_{args.by}_distribution.png"
    save_csv(records, csv_path)
    plot_distribution(counts_by_type, ordered_keys(rules, args.by), args.by, png_path)

    print("\n=== Best-match distribution by data_type ===")
    for dt in [d for d in DATA_TYPES if d in counts_by_type] + \
              [d for d in counts_by_type if d not in DATA_TYPES]:
        total = sum(counts_by_type[dt].values())
        top = counts_by_type[dt].most_common(3)
        top_str = ", ".join(f"{k} {n}" for k, n in top)
        print(f"  {dt:<20} n={total:<6} top: {top_str}")

    print(f"\nSaved matches -> {csv_path}")
    print(f"Saved plot    -> {png_path}")


if __name__ == "__main__":
    main()
