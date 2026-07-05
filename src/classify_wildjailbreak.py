"""Classify each WildJailbreak prompt against the SLEEC rule set with DeBERTa.

Loads the ``allenai/wildjailbreak`` dataset and, for every prompt, uses the same
DeBERTa-v3 NLI oracle that powers ELODIE's Gate 1 (Q => A) to score how strongly
the prompt entails each SLEEC rule's antecedent -- i.e. how much it *triggers*
that rule. A category's score is the max over its rules, and the prompt's
closest-matching category is the argmax; if even that best score is below
``--threshold`` the prompt is bucketed as ``Other/None``.

Outputs (under ``outputs/wildjailbreak/``):
  * a CSV with one row per prompt and one column per SLEEC category holding that
    category's score;
  * a histogram of the closest-matching category across the dataset;
  * a separate figure showing how prompts can trigger more than one category
    (distribution of #categories triggered + a category co-occurrence heatmap).

Run from the ``src/`` directory (so ``elodie`` is importable):

    python classify_wildjailbreak.py --limit 500
    python classify_wildjailbreak.py                 # whole split (default)

The dataset is gated, so an ``HF_TOKEN`` must be set in the environment (or a
``.env`` file).
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from pathlib import Path

# Reduce CUDA fragmentation for the many differently-sized batches we run.
# Must be set before torch initialises CUDA, hence before the torch import.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm

from elodie.oracle import DEFAULT_DEBERTA_MODEL, DebertaOracle
from elodie.rules import DEFAULT_RULES_PATH, SleecRule, load_rules

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "wildjailbreak"
OTHER = "Other/None"


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


def _score_chunk(
    model, tokenizer, device, use_amp, chunk, antecedents, ent_idx, con_idx, max_length
) -> torch.Tensor:
    """P(entail) matrix [len(chunk), n_labels] for a chunk of prompts.

    Splits and retries on CUDA OOM so a single very long prompt can never abort
    the run; the recursion bottoms out at one prompt.
    """
    n_labels = len(antecedents)
    premises = [p for p in chunk for _ in range(n_labels)]
    hypotheses = antecedents * len(chunk)
    enc = tokenizer(
        premises, hypotheses, padding=True, truncation=True,
        max_length=max_length, return_tensors="pt",
    ).to(device)
    try:
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=use_amp):
            logits = model(**enc).logits.float()
    except torch.OutOfMemoryError:
        del enc
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if len(chunk) == 1:
            raise
        mid = len(chunk) // 2
        top = _score_chunk(model, tokenizer, device, use_amp, chunk[:mid],
                           antecedents, ent_idx, con_idx, max_length)
        bot = _score_chunk(model, tokenizer, device, use_amp, chunk[mid:],
                           antecedents, ent_idx, con_idx, max_length)
        return torch.cat([top, bot], dim=0)

    if con_idx is not None:
        probs = logits[:, [ent_idx, con_idx]].softmax(-1)[:, 0]
    else:
        probs = logits.softmax(-1)[:, ent_idx]
    return probs.view(len(chunk), n_labels)


def classify_prompts(
    oracle: DebertaOracle,
    prompts: list[str],
    rules: list[SleecRule],
    batch_size: int,
    max_length: int = 256,
) -> np.ndarray:
    """Per-rule entailment scores: an ``[n_prompts, n_rules]`` array of P(A|Q).

    Every prompt is scored against *every* rule antecedent as an NLI
    (premise=prompt, hypothesis=antecedent) pair. The entailment probability is
    softmax over the entail/contradiction logits, matching the oracle's
    ``multi_label=True`` P(A|Q) used in Gate 1. Column ``j`` corresponds to
    ``rules[j]``; rows preserve the input prompt order.

    Fast and memory-safe:
      * bypasses the zero-shot *pipeline* (sequential, emits the "use a dataset"
        warning) and drives the oracle's model+tokenizer directly;
      * **sorts prompts by length** so each padded batch is short and uniform --
        the many short vanilla prompts fly through, and padding waste is minimal;
      * runs under ``inference_mode`` + fp16 autocast on CUDA;
      * **halves and retries any batch that OOMs**, so one pathological prompt
        can't kill the run.

    ``batch_size`` is the target number of (prompt, antecedent) pairs per forward
    pass; ``max_length`` caps the tokenised pair length (prompts are truncated).
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
    prompts_per_pass = max(1, batch_size // n_labels)

    # Order shortest-first so each batch pads to a small, uniform length.
    tokenized = tokenizer(prompts, truncation=True, max_length=max_length)["input_ids"]
    order = sorted(range(len(prompts)), key=lambda i: len(tokenized[i]))

    scores = np.empty((len(prompts), n_labels), dtype=np.float32)
    with tqdm(total=len(prompts), desc="classifying", unit="prompt") as pbar:
        for start in range(0, len(order), prompts_per_pass):
            idx = order[start : start + prompts_per_pass]
            chunk = [prompts[i] for i in idx]
            probs = _score_chunk(
                model, tokenizer, device, use_amp, chunk,
                antecedents, ent_idx, con_idx, max_length,
            )
            scores[idx] = probs.cpu().numpy()
            pbar.update(len(idx))

    return scores


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


def category_order(rules: list[SleecRule]) -> list[str]:
    """Unique category ids in rules-file order, for a stable axis."""
    seen: list[str] = []
    for r in rules:
        if r.category_id not in seen:
            seen.append(r.category_id)
    return seen


def category_scores(
    rule_scores: np.ndarray, rules: list[SleecRule], categories: list[str]
) -> np.ndarray:
    """Collapse per-rule scores to per-category scores (max rule within category).

    Returns an ``[n_prompts, n_categories]`` array; column ``c`` is
    ``categories[c]``.
    """
    index = {c: i for i, c in enumerate(categories)}
    out = np.full((rule_scores.shape[0], len(categories)), -1.0, dtype=np.float32)
    for j, r in enumerate(rules):
        c = index[r.category_id]
        np.maximum(out[:, c], rule_scores[:, j], out=out[:, c])
    return out


def save_scores_csv(
    prompts: list[str],
    data_types: list[str],
    best_labels: list[str],
    n_triggered: np.ndarray,
    cat_scores: np.ndarray,
    categories: list[str],
    path: Path,
) -> None:
    """One row per prompt; one column per category holding that category's score."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["prompt", "data_type", "best_category", "n_triggered"] + \
             [f"score_{c}" for c in categories]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, prompt in enumerate(prompts):
            row = [prompt, data_types[i], best_labels[i], int(n_triggered[i])]
            row += [round(float(v), 4) for v in cat_scores[i]]
            writer.writerow(row)


def plot_category_histogram(
    best_labels: list[str], categories: list[str], threshold: float, path: Path
) -> None:
    """Histogram: x = SLEEC category (closest match), y = number of prompts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = categories + [OTHER]
    counts = Counter(best_labels)
    vals = [counts.get(c, 0) for c in order]

    fig, ax = plt.subplots(figsize=(max(10, 0.9 * len(order)), 6))
    colors = ["tab:gray" if c == OTHER else "tab:blue" for c in order]
    bars = ax.bar(range(len(order)), vals, color=colors)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("prompts (closest-matching category)")
    ax.set_title(
        f"Closest-matching SLEEC category across WildJailbreak  (threshold={threshold})"
    )
    ax.bar_label(bars, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_multitrigger(
    triggered: np.ndarray, categories: list[str], threshold: float, path: Path
) -> None:
    """Separate figure for prompts that trigger more than one category.

    Left: distribution of how many categories each prompt triggers (score >=
    threshold). Right: category co-occurrence heatmap -- cell (i, j) is the
    number of prompts that trigger *both* category i and j; the diagonal is the
    total number of prompts triggering each category.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_trig = triggered.sum(axis=1)
    max_k = int(n_trig.max()) if n_trig.size else 0
    ks = list(range(max_k + 1))
    k_counts = [int((n_trig == k).sum()) for k in ks]

    co = triggered.astype(np.int64).T @ triggered.astype(np.int64)  # [C, C]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    bars = ax1.bar(ks, k_counts, color="tab:purple")
    ax1.set_xlabel("number of categories triggered (score >= threshold)")
    ax1.set_ylabel("prompts")
    ax1.set_title(f"How many categories each prompt triggers  (threshold={threshold})")
    ax1.set_xticks(ks)
    ax1.bar_label(bars, fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    im = ax2.imshow(co, cmap="viridis")
    ax2.set_xticks(range(len(categories)))
    ax2.set_xticklabels(categories, rotation=45, ha="right", fontsize=7)
    ax2.set_yticks(range(len(categories)))
    ax2.set_yticklabels(categories, fontsize=7)
    ax2.set_title("Category co-occurrence (prompts triggering both; diagonal = total)")
    hi = co.max() if co.size else 1
    for i in range(len(categories)):
        for j in range(len(categories)):
            ax2.text(j, i, int(co[i, j]), ha="center", va="center",
                     color="white" if co[i, j] < hi / 2 else "black", fontsize=6)
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    fig.suptitle("Multi-category triggering across WildJailbreak", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match WildJailbreak prompts to their closest SLEEC category with DeBERTa."
    )
    parser.add_argument("--config", default="train",
                        help="WildJailbreak dataset config: train | eval (default: train).")
    parser.add_argument("--split", default="train", help="Dataset split (default: train).")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Random sample size; <= 0 uses the entire split (default: -1, all).")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed (default: 0).")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Min entailment score for a category to count as triggered; "
                             "if the best category is below it the prompt is 'Other/None' "
                             "(default: 0.5).")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Target (prompt, antecedent) pairs per DeBERTa forward pass "
                             "(default: 256). Raise it to saturate a larger GPU; batches that "
                             "OOM are automatically halved and retried.")
    parser.add_argument("--max-length", type=int, default=256,
                        help="Max tokenised pair length; prompts are truncated to fit "
                             "(default: 256). Lower it if you still hit OOM.")
    parser.add_argument("--model", default=DEFAULT_DEBERTA_MODEL,
                        help=f"NLI model id (default: {DEFAULT_DEBERTA_MODEL}).")
    parser.add_argument("--rules-file", type=Path, default=DEFAULT_RULES_PATH,
                        help="Path to the SLEEC rules JSON file.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory for the CSV and PNGs (default: outputs/wildjailbreak).")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        print("Warning: no HF_TOKEN set; loading the gated WildJailbreak dataset will likely fail.")

    rules = load_rules(args.rules_file)
    categories = category_order(rules)
    print(f"Loaded {len(rules)} SLEEC rules across {len(categories)} categories.")

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
    print(f"Running on {device} (batch_size={args.batch_size} pairs/pass, "
          f"max_length={args.max_length}).")

    rule_scores = classify_prompts(oracle, prompts, rules, args.batch_size, args.max_length)
    cat_scores = category_scores(rule_scores, rules, categories)

    best_idx = cat_scores.argmax(axis=1)
    best_score = cat_scores.max(axis=1)
    best_labels = [
        categories[c] if best_score[i] >= args.threshold else OTHER
        for i, c in enumerate(best_idx)
    ]
    triggered = cat_scores >= args.threshold
    n_triggered = triggered.sum(axis=1)
    data_types = [r.get("data_type") or "unknown" for r in rows]

    csv_path = args.output_dir / "wildjailbreak_category_scores.csv"
    hist_path = args.output_dir / "wildjailbreak_category_histogram.png"
    multi_path = args.output_dir / "wildjailbreak_multitrigger.png"
    save_scores_csv(prompts, data_types, best_labels, n_triggered, cat_scores, categories, csv_path)
    plot_category_histogram(best_labels, categories, args.threshold, hist_path)
    plot_multitrigger(triggered, categories, args.threshold, multi_path)

    print(f"\n=== Closest-matching category (threshold={args.threshold}) ===")
    counts = Counter(best_labels)
    for c in categories + [OTHER]:
        print(f"  {c:<28} {counts.get(c, 0)}")
    multi = int((n_triggered >= 2).sum())
    print(f"\nPrompts triggering >= 2 categories: {multi} / {len(prompts)} "
          f"({100 * multi / max(1, len(prompts)):.1f}%)")

    print(f"\nSaved scores    -> {csv_path}")
    print(f"Saved histogram -> {hist_path}")
    print(f"Saved multi-fig -> {multi_path}")


if __name__ == "__main__":
    main()
