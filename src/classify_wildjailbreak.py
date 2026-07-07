"""Classify each WildJailbreak prompt against the SLEEC rule set with DeBERTa.

Loads the ``allenai/wildjailbreak`` dataset and, for every prompt, uses the same
DeBERTa-v3 NLI oracle that powers ELODIE's Gate 1 (Q => A) to score how strongly
the prompt entails each SLEEC rule's antecedent -- i.e. how much it *triggers*
that rule. A category's score is the max over its rules, and the prompt's
closest-matching category is the argmax; if even that best score is below
``--threshold`` the prompt is bucketed as ``Other/None``.

Two scoring modes (``--mode``): ``entail`` scores each rule independently
(multi-label P(entail), so a prompt can trigger several categories); ``classifier``
runs a single softmax over the rules per prompt (zero-shot single-label, scores
sum to 1). Each mode writes its **own** JSON and its **own** full set of plots,
so the two never collide and can be compared side by side (``--mode all`` does
both).

Scores are cached to a per-mode JSON (one record per prompt). Each record holds
the per-category scores and ``rule_scores`` -- a ``{rule_id: score}`` map over every
rule, keyed by the same ``rule_id`` used in ``sleec_rules.json``. The entailed set
is *not* stored: the pipeline derives it dynamically by applying its own Gate-1
threshold τ₁ to these scores, so changing τ₁ never means regenerating the file. The
``--threshold`` flag below is independent and only buckets the analysis plots. On a
later run, every cached JSON present is re-plotted straight from disk (pass
``--force`` to recompute).
Because the threshold-dependent buckets are recomputed from the cached scores,
you can re-plot at a different ``--threshold`` for free.

There is no per-category ground truth: the threshold recommendation uses the
dataset's own ``data_type`` (``*_harmful`` vs ``*_benign``) as a *proxy* label for
whether a prompt should trigger any policy category.

Run from the ``src/`` directory (so ``elodie`` is importable):

    python classify_wildjailbreak.py                       # entail mode, whole split
    python classify_wildjailbreak.py --mode classifier
    python classify_wildjailbreak.py --mode all --limit 500
    python classify_wildjailbreak.py --threshold 0.6       # re-plot every cached JSON

The dataset is gated, so an ``HF_TOKEN`` must be set in the environment (or a
``.env`` file) the first time (when the JSON is generated).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

# Reduce CUDA fragmentation for the many differently-sized batches we run.
# Must be set before torch initialises CUDA, hence before the torch import.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np

from elodie.rules import DEFAULT_RULES_PATH, SleecRule, load_rules

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "wildjailbreak"
OTHER = "Other/None"

# Canonical order + colours for the dataset's four data_type buckets.
DATA_TYPES = ["vanilla_harmful", "vanilla_benign", "adversarial_harmful", "adversarial_benign"]
DT_COLORS = {
    "vanilla_harmful": "tab:red",
    "vanilla_benign": "tab:green",
    "adversarial_harmful": "tab:orange",
    "adversarial_benign": "tab:blue",
}


# --------------------------------------------------------------------------- #
# Dataset + classification (only runs when the scores JSON is absent).
# --------------------------------------------------------------------------- #
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
    model, tokenizer, device, use_amp, chunk, antecedents, ent_idx, con_idx, max_length, mode
):
    """Per-rule score matrix [len(chunk), n_labels] for a chunk of prompts.

    ``mode="entail"`` -> independent P(entail) per rule (multi-label NLI, values
    in [0, 1] that do *not* sum to 1). ``mode="classifier"`` -> a single softmax
    over the rules per prompt (zero-shot single-label classification, a
    distribution that sums to 1). Both use the entail-minus-contradiction logit
    as the per-rule evidence.

    Splits and retries on CUDA OOM so a single very long prompt can never abort
    the run; the recursion bottoms out at one prompt (softmax is per-prompt, so
    splitting rows is safe in both modes).
    """
    import torch

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
                           antecedents, ent_idx, con_idx, max_length, mode)
        bot = _score_chunk(model, tokenizer, device, use_amp, chunk[mid:],
                           antecedents, ent_idx, con_idx, max_length, mode)
        return torch.cat([top, bot], dim=0)

    # Per-(prompt, rule) evidence for entailment.
    if con_idx is not None:
        evidence = logits[:, ent_idx] - logits[:, con_idx]  # log-odds entail vs contradict
    else:
        evidence = logits[:, ent_idx]
    evidence = evidence.view(len(chunk), n_labels)

    if mode == "classifier":
        return evidence.softmax(dim=1)          # distribution over rules, per prompt
    return torch.sigmoid(evidence)              # independent P(entail) per rule


def classify_prompts(
    oracle, prompts: list[str], rules: list[SleecRule], batch_size: int,
    max_length: int = 256, mode: str = "entail",
) -> np.ndarray:
    """Per-rule scores: an ``[n_prompts, n_rules]`` array. Column ``j`` is ``rules[j]``.

    ``mode="entail"`` (default) gives independent P(entail) per rule, matching the
    oracle's ``multi_label=True`` P(A|Q) used in Gate 1 -- good for asking "which
    rules does this prompt trigger" (a prompt may trigger several). ``mode=
    "classifier"`` instead runs a single softmax over all rules per prompt, i.e.
    zero-shot single-label classification: the scores form one distribution that
    sums to 1 and picks a single best rule.

    Fast and memory-safe: bypasses the zero-shot *pipeline* and drives the
    oracle's model+tokenizer directly; **sorts prompts by length** so batches pad
    to a short, uniform length; runs under ``inference_mode`` + fp16 autocast on
    CUDA; and **halves and retries any batch that OOMs** so one pathological
    prompt can't kill the run.
    """
    import torch
    from tqdm import tqdm

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
                antecedents, ent_idx, con_idx, max_length, mode,
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


# --------------------------------------------------------------------------- #
# Aggregation + JSON (the cache).
# --------------------------------------------------------------------------- #
def category_order(rules: list[SleecRule]) -> list[str]:
    """Unique category ids in rules-file order, for a stable axis."""
    seen: list[str] = []
    for r in rules:
        if r.category_id not in seen:
            seen.append(r.category_id)
    return seen


def category_scores(
    rule_scores: np.ndarray, rules: list[SleecRule], categories: list[str], agg: str = "max"
) -> np.ndarray:
    """Collapse per-rule scores to per-category scores.

    ``agg="max"`` (entail mode): a category's score is its strongest rule's
    independent P(entail). ``agg="sum"`` (classifier mode): sum the rule
    probabilities in each category, keeping the per-prompt category distribution
    summing to 1.
    """
    index = {c: i for i, c in enumerate(categories)}
    if agg == "sum":
        out = np.zeros((rule_scores.shape[0], len(categories)), dtype=np.float32)
        for j, r in enumerate(rules):
            out[:, index[r.category_id]] += rule_scores[:, j]
        return out
    out = np.full((rule_scores.shape[0], len(categories)), -1.0, dtype=np.float32)
    for j, r in enumerate(rules):
        c = index[r.category_id]
        np.maximum(out[:, c], rule_scores[:, j], out=out[:, c])
    return out


def save_scores_json(
    prompts, data_types, cat_scores, categories,
    rule_ids, rule_scores, rule_categories, threshold: float, mode: str, path: Path,
) -> None:
    """One record per prompt: the raw per-category and per-rule scores.

    ``rule_scores`` is a ``{rule_id: score}`` map over **every** rule (keys are the
    exact ``rule_id`` values from ``sleec_rules.json``). The entailed set is *not*
    baked in: the pipeline derives it dynamically by applying its own Gate-1
    threshold τ₁ to these scores, so changing τ₁ never means regenerating the file.
    ``best_category`` / ``n_triggered`` are informational analysis snapshots taken at
    ``threshold`` (the ``--threshold`` flag); ``category_scores`` feeds the plots.

    The file is pretty-printed for small samples and written compactly for the
    full split (where indentation would balloon the file).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    best_idx = cat_scores.argmax(axis=1)
    best_score = cat_scores.max(axis=1)

    records = []
    for i, prompt in enumerate(prompts):
        cat_row = cat_scores[i]
        rule_row = rule_scores[i]
        records.append({
            "prompt": prompt,
            "data_type": data_types[i],
            "best_category": categories[int(best_idx[i])] if best_score[i] >= threshold else OTHER,
            "n_triggered": int((cat_row >= threshold).sum()),
            "category_scores": {c: round(float(cat_row[j]), 4)
                                for j, c in enumerate(categories)},
            "rule_scores": {rid: round(float(rule_row[j]), 4)
                            for j, rid in enumerate(rule_ids)},
        })

    payload = {
        "mode": mode,
        "threshold": threshold,
        "categories": categories,
        "rule_ids": list(rule_ids),
        "rule_categories": rule_categories,
        "prompts": records,
    }
    indent = 2 if len(records) <= 2000 else None
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=indent)


def load_scores_json(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    """Read the cached JSON back into (categories, data_types, cat_scores)."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("prompts", [])
    categories = payload.get("categories")
    if categories is None:  # fall back to the keys of the first record's scores
        categories = list(records[0]["category_scores"]) if records else []
    if not categories:
        raise ValueError(f"{path} has no category scores to plot.")
    data_types = [rec.get("data_type", "unknown") for rec in records]
    cat_scores = np.array(
        [[rec["category_scores"][c] for c in categories] for rec in records],
        dtype=np.float32,
    ).reshape(len(records), len(categories))
    return categories, data_types, cat_scores


# --------------------------------------------------------------------------- #
# Plotting.
# --------------------------------------------------------------------------- #
def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def ordered_data_types(data_types: list[str]) -> list[str]:
    present = set(data_types)
    order = [d for d in DATA_TYPES if d in present]
    order += sorted(d for d in present if d not in DATA_TYPES)
    return order


def _dt_colors(dtypes: list[str]):
    plt = _pyplot()
    return [DT_COLORS.get(dt, plt.cm.tab20(j % 20)) for j, dt in enumerate(dtypes)]


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    import matplotlib.pyplot as plt

    plt.close(fig)


def plot_category_histogram(best_labels, categories, threshold, path: Path) -> None:
    """Histogram: x = SLEEC category (closest match), y = number of prompts."""
    plt = _pyplot()
    order = categories + [OTHER]
    counts = Counter(best_labels)
    vals = [counts.get(c, 0) for c in order]

    fig, ax = plt.subplots(figsize=(max(10, 0.9 * len(order)), 6))
    colors = ["tab:gray" if c == OTHER else "tab:blue" for c in order]
    bars = ax.bar(range(len(order)), vals, color=colors)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("prompts (closest-matching category)")
    ax.set_title(f"Closest-matching SLEEC category across WildJailbreak  (threshold={threshold})")
    ax.bar_label(bars, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, path)


def plot_multitrigger(triggered, categories, threshold, path: Path) -> None:
    """Distribution of #categories triggered + a category co-occurrence heatmap."""
    plt = _pyplot()
    n_trig = triggered.sum(axis=1)
    max_k = int(n_trig.max()) if n_trig.size else 0
    ks = list(range(max_k + 1))
    k_counts = [int((n_trig == k).sum()) for k in ks]
    co = triggered.astype(np.int64).T @ triggered.astype(np.int64)

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
    _save(fig, path)


def plot_category_by_datatype(best_labels, data_types, categories, dtypes, path: Path) -> None:
    """Closest-matching category split by data_type: stacked counts + composition."""
    plt = _pyplot()
    order = categories + [OTHER]
    ci = {c: i for i, c in enumerate(order)}
    di = {d: j for j, d in enumerate(dtypes)}
    M = np.zeros((len(order), len(dtypes)), dtype=np.int64)
    for lab, dt in zip(best_labels, data_types):
        M[ci[lab], di[dt]] += 1
    colors = _dt_colors(dtypes)
    x = np.arange(len(order))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    bottom = np.zeros(len(order))
    for j, dt in enumerate(dtypes):
        ax1.bar(x, M[:, j], bottom=bottom, label=dt, color=colors[j])
        bottom += M[:, j]
    ax1.set_xticks(x)
    ax1.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("prompts")
    ax1.set_title("Closest category by data_type (counts)")
    ax1.legend(fontsize=8)

    totals = M.sum(axis=1, keepdims=True)
    frac = np.divide(M, totals, where=totals > 0, out=np.zeros_like(M, dtype=float))
    bottom = np.zeros(len(order))
    for j, dt in enumerate(dtypes):
        ax2.bar(x, frac[:, j], bottom=bottom, label=dt, color=colors[j])
        bottom += frac[:, j]
    ax2.set_xticks(x)
    ax2.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("proportion")
    ax2.set_title("data_type composition within each category")
    ax2.legend(fontsize=8)

    fig.suptitle("Closest-matching category vs data_type", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, path)


def plot_meanscore_heatmap(cat_scores, data_types, categories, dtypes, path: Path) -> None:
    """Heatmap of the mean category score for each data_type."""
    plt = _pyplot()
    dt_arr = np.array(data_types)
    M = np.full((len(dtypes), len(categories)), np.nan)
    for i, dt in enumerate(dtypes):
        mask = dt_arr == dt
        if mask.any():
            M[i] = cat_scores[mask].mean(axis=0)

    fig, ax = plt.subplots(figsize=(max(10, 0.9 * len(categories)), 1.5 + 0.7 * len(dtypes)))
    im = ax.imshow(M, cmap="magma", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(dtypes)))
    ax.set_yticklabels(dtypes, fontsize=9)
    ax.set_title("Mean category score by data_type")
    for i in range(len(dtypes)):
        for j in range(len(categories)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        color="white" if M[i, j] < 0.5 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save(fig, path)


def plot_best_score_hist(best_score, data_types, dtypes, threshold, path: Path) -> None:
    """Distribution of the best (max) category score, overall and per data_type."""
    plt = _pyplot()
    dt_arr = np.array(data_types)
    colors = _dt_colors(dtypes)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.hist(best_score, bins=50, range=(0, 1), color="tab:blue", alpha=0.85)
    ax1.axvline(threshold, color="red", ls="--", label=f"threshold={threshold}")
    ax1.set_xlabel("best category score")
    ax1.set_ylabel("prompts")
    ax1.set_title("Best-match score distribution (all prompts)")
    ax1.legend()

    for dt, c in zip(dtypes, colors):
        vals = best_score[dt_arr == dt]
        if vals.size:
            ax2.hist(vals, bins=50, range=(0, 1), histtype="step", density=True,
                     label=dt, color=c, linewidth=1.6)
    ax2.axvline(threshold, color="red", ls="--")
    ax2.set_xlabel("best category score")
    ax2.set_ylabel("density")
    ax2.set_title("Best-match score by data_type (normalized)")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, path)


def plot_ntriggered_by_datatype(n_triggered, data_types, dtypes, path: Path) -> None:
    """Row-normalized heatmap: within each data_type, share of prompts by #triggered."""
    plt = _pyplot()
    dt_arr = np.array(data_types)
    max_k = int(n_triggered.max()) if n_triggered.size else 0
    ks = list(range(max_k + 1))
    M = np.zeros((len(dtypes), len(ks)))
    for i, dt in enumerate(dtypes):
        vals = n_triggered[dt_arr == dt]
        if vals.size:
            for k in ks:
                M[i, k] = (vals == k).mean()

    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(ks) + 3), 1.5 + 0.7 * len(dtypes)))
    im = ax.imshow(M, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels(ks)
    ax.set_yticks(range(len(dtypes)))
    ax.set_yticklabels(dtypes, fontsize=9)
    ax.set_xlabel("# categories triggered")
    ax.set_title("Triggered-count distribution within each data_type (row-normalized)")
    for i in range(len(dtypes)):
        for k in ks:
            ax.text(k, i, f"{M[i, k]:.2f}", ha="center", va="center",
                    color="white" if M[i, k] < 0.5 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save(fig, path)


def plot_score_correlation(cat_scores, categories, path: Path) -> None:
    """Correlation between category scores (which categories co-vary)."""
    plt = _pyplot()
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(cat_scores.T)
    corr = np.nan_to_num(corr)  # zero-variance categories -> undefined; show as 0
    fig, ax = plt.subplots(figsize=(max(9, 0.95 * len(categories)),
                                    max(8, 0.85 * len(categories))))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories, fontsize=7)
    ax.set_title("Correlation between category scores")
    for i in range(len(categories)):
        for j in range(len(categories)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(corr[i, j]) > 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save(fig, path)


def plot_category_score_box(cat_scores, categories, threshold, path: Path) -> None:
    """Per-category score distribution across all prompts (box plots)."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(max(10, 0.75 * len(categories)), 7))
    ax.boxplot([cat_scores[:, j] for j in range(len(categories))], showfliers=False)
    ax.set_xticks(range(1, len(categories) + 1))
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=8)
    ax.axhline(threshold, color="red", ls="--", label=f"threshold={threshold}")
    ax.set_ylabel("score")
    ax.set_title("Per-category score distribution (all prompts)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, path)


def recommend_threshold(best_score: np.ndarray, data_types: list[str]):
    """Threshold that best separates harmful from benign prompts (Youden's J).

    Uses the top category score as a "harmfulness" score and the ``data_type``
    label (harmful vs benign) as ground truth. Returns ``(t, tpr, fpr, j)`` or
    ``None`` when the split lacks both harmful and benign rows.
    """
    harmful = np.array(["harmful" in d for d in data_types])
    benign = np.array(["benign" in d for d in data_types])
    if not harmful.any() or not benign.any():
        return None
    best = None
    for t in np.linspace(0.01, 0.99, 99):
        flagged = best_score >= t
        tpr = float((flagged & harmful).sum()) / int(harmful.sum())
        fpr = float((flagged & benign).sum()) / int(benign.sum())
        j = tpr - fpr
        if best is None or j > best[3]:
            best = (float(t), tpr, fpr, j)
    return best


def plot_threshold_sweep(cat_scores, data_types, path: Path, recommended=None) -> None:
    """How the threshold affects prompt coverage and harmful/benign separation."""
    plt = _pyplot()
    best_score = cat_scores.max(axis=1)
    grid = np.linspace(0, 1, 101)
    other = [float((best_score < t).mean()) * 100 for t in grid]
    ge1 = [float((best_score >= t).mean()) * 100 for t in grid]
    ge2 = [float(((cat_scores >= t).sum(axis=1) >= 2).mean()) * 100 for t in grid]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(grid, other, label="Other/None", color="tab:gray")
    ax1.plot(grid, ge1, label=">= 1 categoria", color="tab:blue")
    ax1.plot(grid, ge2, label=">= 2 categorie", color="tab:purple")
    if recommended is not None:
        ax1.axvline(recommended, color="red", ls="--",
                    label=f"consigliata = {recommended:.2f}")
    ax1.set_xlabel("soglia")
    ax1.set_ylabel("% prompt")
    ax1.set_title("Copertura dei prompt al variare della soglia")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    harmful = np.array(["harmful" in d for d in data_types])
    benign = np.array(["benign" in d for d in data_types])
    if harmful.any() and benign.any():
        tpr = [float((best_score[harmful] >= t).mean()) * 100 for t in grid]
        fpr = [float((best_score[benign] >= t).mean()) * 100 for t in grid]
        j = np.array(tpr) - np.array(fpr)
        ax2.plot(grid, tpr, label="dannosi segnalati (TPR)", color="tab:red")
        ax2.plot(grid, fpr, label="benigni segnalati (FPR)", color="tab:green")
        ax2.plot(grid, j, label="Youden J = TPR - FPR", color="tab:blue", ls=":")
        if recommended is not None:
            ax2.axvline(recommended, color="red", ls="--")
        ax2.set_title("Separazione dannosi vs benigni al variare della soglia\n"
                      "(etichetta proxy = data_type harmful/benign)")
    else:
        ax2.text(0.5, 0.5, "data_type harmful/benign non disponibili",
                 ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Separazione non disponibile")
    ax2.set_xlabel("soglia")
    ax2.set_ylabel("%")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    fig.suptitle("Scelta della soglia", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, path)


# One-line Italian description per plot, printed to the terminal to share.
PLOT_DESCRIPTIONS_IT = {
    "category_histogram":
        "Istogramma della categoria SLEEC più vicina a ogni prompt; la barra grigia 'Other/None' raccoglie i prompt sotto la soglia.",
    "multitrigger":
        "A sinistra quante categorie attiva ciascun prompt, a destra la matrice di co-occorrenza tra categorie attivate insieme.",
    "category_by_datatype":
        "Categoria più vicina suddivisa per data_type, sia in conteggi assoluti sia in composizione percentuale.",
    "meanscore_heatmap":
        "Heatmap del punteggio medio di ciascuna categoria per ogni data_type (dannoso/benigno, vanilla/adversarial).",
    "bestscore_hist":
        "Distribuzione del punteggio della categoria migliore, complessiva e per data_type, con la soglia evidenziata.",
    "ntriggered_by_datatype":
        "Heatmap normalizzata per riga di quante categorie vengono attivate all'interno di ciascun data_type.",
    "score_correlation":
        "Matrice di correlazione tra i punteggi delle categorie, per capire quali categorie variano insieme.",
    "category_score_box":
        "Box plot della distribuzione dei punteggi per ciascuna categoria su tutti i prompt.",
    "threshold_sweep":
        "Effetto della soglia sulla copertura dei prompt e sulla separazione tra prompt dannosi e benigni, con la soglia consigliata.",
}


def generate_plots(
    cat_scores: np.ndarray,
    categories: list[str],
    data_types: list[str],
    threshold: float,
    out_dir: Path,
    mode: str = "entail",
) -> list[Path]:
    """Derive threshold-dependent buckets from cached scores and write every plot."""
    best_idx = cat_scores.argmax(axis=1)
    best_score = cat_scores.max(axis=1)
    best_labels = [
        categories[c] if best_score[i] >= threshold else OTHER
        for i, c in enumerate(best_idx)
    ]
    triggered = cat_scores >= threshold
    n_triggered = triggered.sum(axis=1)
    dtypes = ordered_data_types(data_types)

    rec = recommend_threshold(best_score, data_types)

    def out(name: str) -> Path:
        return out_dir / f"wildjailbreak_{name}_{mode}.png"

    paths = {
        "category_histogram": out("category_histogram"),
        "multitrigger": out("multitrigger"),
        "category_by_datatype": out("category_by_datatype"),
        "meanscore_heatmap": out("meanscore_by_datatype"),
        "bestscore_hist": out("bestscore_hist"),
        "ntriggered_by_datatype": out("ntriggered_by_datatype"),
        "score_correlation": out("score_correlation"),
        "category_score_box": out("category_score_box"),
        "threshold_sweep": out("threshold_sweep"),
    }
    plot_category_histogram(best_labels, categories, threshold, paths["category_histogram"])
    plot_multitrigger(triggered, categories, threshold, paths["multitrigger"])
    plot_category_by_datatype(best_labels, data_types, categories, dtypes,
                              paths["category_by_datatype"])
    plot_meanscore_heatmap(cat_scores, data_types, categories, dtypes,
                           paths["meanscore_heatmap"])
    plot_best_score_hist(best_score, data_types, dtypes, threshold, paths["bestscore_hist"])
    plot_ntriggered_by_datatype(n_triggered, data_types, dtypes,
                                paths["ntriggered_by_datatype"])
    plot_score_correlation(cat_scores, categories, paths["score_correlation"])
    plot_category_score_box(cat_scores, categories, threshold, paths["category_score_box"])
    plot_threshold_sweep(cat_scores, data_types, paths["threshold_sweep"],
                         rec[0] if rec else None)

    _print_summary(best_labels, n_triggered, categories, data_types, dtypes, threshold)
    _print_threshold_advice(threshold, rec)
    _print_descriptions(paths)
    return list(paths.values())


def _print_threshold_advice(threshold, rec) -> None:
    print(f"\n=== Threshold choice (currently using {threshold}) ===")
    if rec is None:
        print("  No harmful/benign data_type labels available -> cannot recommend a threshold.")
        return
    t, tpr, fpr, j = rec
    print("  There is no per-category ground truth; the only label we have is WildJailbreak's")
    print("  own data_type (vanilla/adversarial _harmful vs _benign), used here as a proxy for")
    print("  'should this prompt trigger any policy category'. TPR = share of harmful prompts")
    print("  flagged (best score >= t); FPR = share of benign prompts flagged.")
    print(f"  Recommended threshold (max Youden's J = TPR - FPR): {t:.2f}")
    print(f"    at {t:.2f}: harmful flagged {tpr * 100:.1f}%, benign flagged {fpr * 100:.1f}%, "
          f"J = {j:.2f}")
    print("    Lower it to catch more harmful prompts (more benign false positives);")
    print("    raise it for higher precision. See the threshold-sweep plot.")


def _print_descriptions(paths: dict[str, Path]) -> None:
    print("\n=== Descrizione dei grafici (IT) ===")
    for key, path in paths.items():
        desc = PLOT_DESCRIPTIONS_IT.get(key, "")
        print(f"  {path.name}\n    {desc}")


def _print_summary(best_labels, n_triggered, categories, data_types, dtypes, threshold) -> None:
    print(f"\n=== Closest-matching category (threshold={threshold}) ===")
    counts = Counter(best_labels)
    for c in categories + [OTHER]:
        print(f"  {c:<28} {counts.get(c, 0)}")
    n = len(best_labels)
    multi = int((n_triggered >= 2).sum())
    none = int((n_triggered == 0).sum())
    print(f"\nTrigger >= 2 categories: {multi}/{n} ({100 * multi / max(1, n):.1f}%)   "
          f"trigger none: {none}/{n} ({100 * none / max(1, n):.1f}%)")
    dt_counts = Counter(data_types)
    print("data_type counts: " + ", ".join(f"{d}={dt_counts.get(d, 0)}" for d in dtypes))


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    from elodie.oracle import DEFAULT_DEBERTA_MODEL

    parser = argparse.ArgumentParser(
        description="Match WildJailbreak prompts to their closest SLEEC category with DeBERTa, "
                    "cache the scores, and plot the analysis (re-plots from cache if present)."
    )
    parser.add_argument("--config", default="train",
                        help="WildJailbreak dataset config: train | eval (default: train).")
    parser.add_argument("--split", default="train", help="Dataset split (default: train).")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Random sample size; <= 0 uses the entire split (default: -1, all).")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed (default: 0).")
    parser.add_argument("--mode", choices=["entail", "classifier", "all"], default="entail",
                        help="Scoring mode: 'entail' = independent per-rule P(entail) (multi-label, "
                             "supports multi-trigger analysis); 'classifier' = single softmax over "
                             "rules per prompt (zero-shot single-label, scores sum to 1); 'all' does "
                             "both. Each mode has its own JSON + plots (default: entail).")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Min score for a category to count as triggered; below it the best "
                             "category becomes 'Other/None' (default: 0.5). Applied when plotting, "
                             "so it can be changed without re-running the model.")
    parser.add_argument("--force", action="store_true",
                        help="Recompute scores with DeBERTa even if the cached JSON exists.")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Target (prompt, antecedent) pairs per DeBERTa forward pass "
                             "(default: 256). OOM batches are halved and retried.")
    parser.add_argument("--max-length", type=int, default=256,
                        help="Max tokenised pair length; prompts are truncated (default: 256).")
    parser.add_argument("--model", default=DEFAULT_DEBERTA_MODEL,
                        help=f"NLI model id (default: {DEFAULT_DEBERTA_MODEL}).")
    parser.add_argument("--rules-file", type=Path, default=DEFAULT_RULES_PATH,
                        help="Path to the SLEEC rules JSON file.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory for the JSON and PNGs (default: outputs/wildjailbreak).")
    return parser.parse_args()


def scores_json_path(out_dir: Path, mode: str) -> Path:
    return out_dir / f"wildjailbreak_category_scores_{mode}.json"


def existing_score_jsons(out_dir: Path) -> dict[str, Path]:
    """Map mode -> cached scores JSON for every mode present in ``out_dir``."""
    found: dict[str, Path] = {}
    prefix = "wildjailbreak_category_scores_"
    for p in sorted(out_dir.glob(f"{prefix}*.json")):
        found[p.stem[len(prefix):]] = p
    return found


def compute_scores(args, mode: str):
    """Run the DeBERTa classification for one mode.

    Returns (categories, data_types, cat_scores, prompts, rule_ids, rule_scores,
    rule_categories) — the last three carrying the per-rule detail embedded in
    the JSON cache.
    """
    from dotenv import load_dotenv

    from elodie.oracle import DebertaOracle

    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        print("Warning: no HF_TOKEN set; loading the gated WildJailbreak dataset will likely fail.")

    rules = load_rules(args.rules_file)
    categories = category_order(rules)
    rule_ids = [r.rule_id for r in rules]
    rule_categories = {r.rule_id: r.category_id for r in rules}
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
    data_types = [r.get("data_type") or "unknown" for r in rows]
    print(f"Classifying {len(prompts)} prompts against {len(rules)} antecedents (mode={mode})...")

    print(f"Loading oracle ({args.model})...")
    oracle = DebertaOracle(model_name=args.model, token=token)
    device = next(oracle.pipeline.model.parameters()).device
    print(f"Running on {device} (batch_size={args.batch_size} pairs/pass, "
          f"max_length={args.max_length}).")
    rule_scores = classify_prompts(oracle, prompts, rules, args.batch_size, args.max_length, mode)
    agg = "sum" if mode == "classifier" else "max"
    cat_scores = category_scores(rule_scores, rules, categories, agg)
    return categories, data_types, cat_scores, prompts, rule_ids, rule_scores, rule_categories


def main() -> None:
    args = parse_args()
    out = args.output_dir
    target_modes = ["entail", "classifier"] if args.mode == "all" else [args.mode]

    # 1) Ensure a scores JSON exists for each requested mode (compute only if missing/forced).
    #    Every rule's raw score is stored; the pipeline applies its own τ₁ at read time.
    cached = existing_score_jsons(out)
    for mode in target_modes:
        if mode in cached and not args.force:
            print(f"Found cached {mode} scores -> {cached[mode]} (use --force to recompute).")
            continue
        (categories, data_types, cat_scores, prompts,
         rule_ids, rule_scores, rule_categories) = compute_scores(args, mode)
        path = scores_json_path(out, mode)
        save_scores_json(prompts, data_types, cat_scores, categories,
                         rule_ids, rule_scores, rule_categories, args.threshold, mode, path)
        print(f"Saved {mode} scores -> {path}")

    # 2) Plot every scores JSON present (each mode gets its own full set of plots).
    cached = existing_score_jsons(out)
    if not cached:
        print("No score JSON files found; nothing to plot.")
        return
    for mode, json_path in cached.items():
        print(f"\n########## mode: {mode}  ({json_path.name}) ##########")
        categories, data_types, cat_scores = load_scores_json(json_path)
        plot_paths = generate_plots(cat_scores, categories, data_types, args.threshold, out, mode)
        print(f"\nSaved {len(plot_paths)} plots for mode '{mode}':")
        for p in plot_paths:
            print(f"  {p.name}")


if __name__ == "__main__":
    main()
