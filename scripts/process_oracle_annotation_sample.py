#!/usr/bin/env python3
"""
Analyze ELODIE oracle-validation annotations.

Inputs:
  - annotation_sample_master.csv
  - annotation_sample_annotator_A.csv
  - annotation_sample_annotator_B.csv

The annotator files must contain:
  item_id, human_label, notes

Allowed human labels:
  ENTAILED
  NOT_ENTAILED
  UNCLEAR

The master file must contain:
  item_id, gate, score_band, rule_id, deberta_score, threshold, deberta_label

Outputs:
  - oracle_validation_joined.csv
  - oracle_validation_summary_by_gate.csv
  - oracle_validation_summary_by_gate_band.csv
  - oracle_validation_report.txt

Usage:
  python analyze_oracle_annotations.py \
    --master outputs/oracle_annotation/annotation_sample_master.csv \
    --annotator-a outputs/oracle_annotation/annotation_sample_annotator_A.csv \
    --annotator-b outputs/oracle_annotation/annotation_sample_annotator_B.csv \
    --out-dir outputs/oracle_annotation/analysis
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

VALID_LABELS = {"ENTAILED", "NOT_ENTAILED", "UNCLEAR"}


def add_deberta_label_at_threshold(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = df.copy()
    out["threshold_eval"] = threshold
    out["deberta_label"] = np.where(
        out["deberta_score"] >= threshold,
        "ENTAILED",
        "NOT_ENTAILED",
    )
    return out


def threshold_sweep(df: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows = []

    for tau in thresholds:
        tmp = add_deberta_label_at_threshold(df, tau)

        overall = compute_metrics(tmp)
        overall["gate"] = "ALL"
        overall["threshold_eval"] = tau
        rows.append(overall)

        by_gate = summarize_by(tmp, ["gate"])
        for _, row in by_gate.iterrows():
            d = row.to_dict()
            d["threshold_eval"] = tau
            rows.append(d)

    return pd.DataFrame(rows)


def best_thresholds(sweep_df: pd.DataFrame, metric: str = "accuracy") -> pd.DataFrame:
    usable = sweep_df.dropna(subset=[metric]).copy()
    usable = usable.sort_values(
        ["gate", metric, "precision_entailment", "f1_entailment"],
        ascending=[True, False, False, False],
    )
    return usable.groupby("gate", as_index=False).head(1)


def normalize_label(value: object) -> str:
    if value is None or pd.isna(value):
        return ""

    text = str(value).strip().upper()
    text = text.replace("-", "_").replace(" ", "_")

    aliases = {
        "E": "ENTAILED",
        "YES": "ENTAILED",
        "Y": "ENTAILED",
        "TRUE": "ENTAILED",
        "ENTAIL": "ENTAILED",
        "ENTAILED": "ENTAILED",

        "N": "NOT_ENTAILED",
        "NO": "NOT_ENTAILED",
        "FALSE": "NOT_ENTAILED",
        "NOT_ENTAIL": "NOT_ENTAILED",
        "NOT_ENTAILED": "NOT_ENTAILED",
        "NOTENTAILED": "NOT_ENTAILED",
        "DOES_NOT_ENTAIL": "NOT_ENTAILED",
        "DOESNT_ENTAIL": "NOT_ENTAILED",

        "U": "UNCLEAR",
        "UNSURE": "UNCLEAR",
        "AMBIGUOUS": "UNCLEAR",
        "UNCLEAR": "UNCLEAR",
        "CANNOT_TELL": "UNCLEAR",
        "CAN_NOT_TELL": "UNCLEAR",
    }

    return aliases.get(text, text)


def load_annotator(path: Path, suffix: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "item_id" not in df.columns:
        raise ValueError(f"{path} is missing item_id")
    if "human_label" not in df.columns:
        raise ValueError(f"{path} is missing human_label")

    out = df[["item_id", "human_label"]].copy()
    out[f"human_label_{suffix}"] = out["human_label"].map(normalize_label)
    out = out.drop(columns=["human_label"])

    if "notes" in df.columns:
        out[f"notes_{suffix}"] = df["notes"].fillna("").astype(str)

    bad = sorted(set(out[f"human_label_{suffix}"]) - VALID_LABELS - {""})
    if bad:
        raise ValueError(f"{path} has invalid labels: {bad}")

    return out


def load_data(master: Path, annotator_a: Path, annotator_b: Path) -> pd.DataFrame:
    m = pd.read_csv(master)

    required = {
        "item_id",
        "gate",
        "score_band",
        "rule_id",
        "deberta_score",
        "threshold",
        "deberta_label",
    }
    missing = required - set(m.columns)
    if missing:
        raise ValueError(f"Master file is missing columns: {sorted(missing)}")

    a = load_annotator(annotator_a, "A")
    b = load_annotator(annotator_b, "B")

    df = m.merge(a, on="item_id", how="left").merge(b, on="item_id", how="left")

    df["human_label_A"] = df["human_label_A"].fillna("")
    df["human_label_B"] = df["human_label_B"].fillna("")

    missing_a = int((df["human_label_A"] == "").sum())
    missing_b = int((df["human_label_B"] == "").sum())
    if missing_a or missing_b:
        print(f"WARNING: missing labels: A={missing_a}, B={missing_b}")

    df["deberta_label"] = df["deberta_label"].map(normalize_label)

    if "deberta_score" in df.columns:
        df["deberta_score"] = pd.to_numeric(df["deberta_score"], errors="coerce")
    if "threshold" in df.columns:
        df["threshold"] = pd.to_numeric(df["threshold"], errors="coerce")

    return df


def majority_label(a: str, b: str, unclear_policy: str) -> str:
    labels = [x for x in [a, b] if x]

    if not labels:
        return ""

    if len(labels) == 1:
        lab = labels[0]
        if lab == "UNCLEAR":
            return "NOT_ENTAILED" if unclear_policy == "negative" else "UNCLEAR"
        return lab

    if a == b:
        if a == "UNCLEAR":
            return "NOT_ENTAILED" if unclear_policy == "negative" else "UNCLEAR"
        return a

    if unclear_policy == "exclude":
        return "DISAGREEMENT"

    if unclear_policy == "negative":
        mapped = ["NOT_ENTAILED" if x == "UNCLEAR" else x for x in [a, b]]
        if mapped[0] == mapped[1]:
            return mapped[0]
        return "DISAGREEMENT"

    raise ValueError(f"Unknown unclear_policy: {unclear_policy}")


def add_human_gold(df: pd.DataFrame, unclear_policy: str) -> pd.DataFrame:
    out = df.copy()
    out["human_gold"] = [
        majority_label(a, b, unclear_policy)
        for a, b in zip(out["human_label_A"], out["human_label_B"])
    ]
    out["annotators_agree_exact"] = out["human_label_A"] == out["human_label_B"]
    return out


def binary_from_label(label: str) -> int:
    if label == "ENTAILED":
        return 1
    if label == "NOT_ENTAILED":
        return 0
    raise ValueError(f"Cannot convert to binary: {label}")


def compute_metrics(df: pd.DataFrame) -> dict:
    usable = df[df["human_gold"].isin(["ENTAILED", "NOT_ENTAILED"])].copy()

    n = len(df)
    n_usable = len(usable)

    if n_usable == 0:
        return {
            "n_items": n,
            "n_usable": 0,
            "accuracy": np.nan,
            "precision_entailment": np.nan,
            "recall_entailment": np.nan,
            "f1_entailment": np.nan,
            "false_positive_rate": np.nan,
            "false_negative_rate": np.nan,
            "deberta_positive_rate": np.nan,
            "human_positive_rate": np.nan,
        }

    y_true = usable["human_gold"].map(binary_from_label).to_numpy()
    y_pred = usable["deberta_label"].map(binary_from_label).to_numpy()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "n_items": n,
        "n_usable": n_usable,
        "n_unclear_or_disagreement": n - n_usable,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_entailment": precision_score(y_true, y_pred, zero_division=0),
        "recall_entailment": recall_score(y_true, y_pred, zero_division=0),
        "f1_entailment": f1_score(y_true, y_pred, zero_division=0),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else np.nan,
        "false_negative_rate": fn / (fn + tp) if (fn + tp) else np.nan,
        "deberta_positive_rate": float(y_pred.mean()),
        "human_positive_rate": float(y_true.mean()),
    }


def summarize_by(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in df.groupby(cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: val for col, val in zip(cols, key)}
        row.update(compute_metrics(group))
        rows.append(row)
    return pd.DataFrame(rows)


def fmt(x: object, digits: int = 3) -> str:
    if x is None or pd.isna(x):
        return "--"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.{digits}f}"
    return str(x)


def fmt_latex_float(x: object, digits: int = 3) -> str:
    if x is None or pd.isna(x):
        return "--"
    return f"{float(x):.{digits}f}"


def fmt_latex_int(x: object) -> str:
    if x is None or pd.isna(x):
        return "--"
    return str(int(x))


def threshold_table_for_latex(best_df: pd.DataFrame, caption: str, label: str) -> str:
    cols = [
        "gate",
        "threshold_eval",
        "n_usable",
        "accuracy",
        "precision_entailment",
        "recall_entailment",
        "f1_entailment",
        "false_positive_rate",
        "false_negative_rate",
    ]

    table = best_df[cols].copy()

    table["gate"] = table["gate"].replace({
        "ALL": "All",
        "G1_ANTECEDENT": "G1 antecedent",
        "G2_DEFEATER": "G2 defeater",
        "G3_BASE_CONSEQUENT": "G3 base consequent",
        "G3_DEFEATER_CONSEQUENT": "G3 defeater consequent",
    })

    table = table.rename(columns={
        "gate": "Gate",
        "threshold_eval": "$\\tau$",
        "n_usable": "$N$",
        "accuracy": "Acc.",
        "precision_entailment": "Prec.",
        "recall_entailment": "Rec.",
        "f1_entailment": "$F_1$",
        "false_positive_rate": "FPR",
        "false_negative_rate": "FNR",
    })

    table["$\\tau$"] = table["$\\tau$"].map(lambda x: fmt_latex_float(x, 2))
    table["$N$"] = table["$N$"].map(fmt_latex_int)

    for col in ["Acc.", "Prec.", "Rec.", "$F_1$", "FPR", "FNR"]:
        table[col] = table[col].map(lambda x: fmt_latex_float(x, 3))

    return table.to_latex(
        index=False,
        escape=False,
        column_format="lrrrrrrrr",
        caption=caption,
        label=label,
    )


def latex_threshold_sweep_by_gate(sweep_df: pd.DataFrame) -> str:
    table = sweep_df[sweep_df["gate"] != "ALL"].copy()

    cols = [
        "threshold_eval",
        "gate",
        "n_usable",
        "accuracy",
        "precision_entailment",
        "recall_entailment",
        "f1_entailment",
        "false_positive_rate",
        "false_negative_rate",
    ]

    table = table[cols].copy()

    table["gate"] = table["gate"].replace({
        "G1_ANTECEDENT": "G1 antecedent",
        "G2_DEFEATER": "G2 defeater",
        "G3_BASE_CONSEQUENT": "G3 base consequent",
        "G3_DEFEATER_CONSEQUENT": "G3 defeater consequent",
    })

    table = table.rename(columns={
        "threshold_eval": "$\\tau$",
        "gate": "Gate",
        "n_usable": "$N$",
        "accuracy": "Acc.",
        "precision_entailment": "Prec.",
        "recall_entailment": "Rec.",
        "f1_entailment": "$F_1$",
        "false_positive_rate": "FPR",
        "false_negative_rate": "FNR",
    })

    table["$\\tau$"] = table["$\\tau$"].map(lambda x: "--" if pd.isna(x) else f"{x:.2f}")
    table["$N$"] = table["$N$"].map(lambda x: "--" if pd.isna(x) else f"{int(x)}")

    for col in ["Acc.", "Prec.", "Rec.", "$F_1$", "FPR", "FNR"]:
        table[col] = table[col].map(lambda x: "--" if pd.isna(x) else f"{x:.2f}")

    table = table.sort_values(["$\\tau$", "Gate"])

    return table.to_latex(
        index=False,
        escape=False,
        column_format="llrrrrrrr",
        caption="DeBERTa oracle agreement with human annotations by gate across entailment thresholds.",
        label="tab:oracle-threshold-sweep-by-gate",
    )

def latex_threshold_sweep_by_gate_grouped(sweep_df: pd.DataFrame) -> str:
    table = sweep_df[sweep_df["gate"] != "ALL"].copy()

    cols = [
        "gate",
        "threshold_eval",
        "n_usable",
        "accuracy",
        "precision_entailment",
        "recall_entailment",
        "f1_entailment",
        "false_positive_rate",
        "false_negative_rate",
    ]

    table = table[cols].copy()

    gate_order = [
        "G1_ANTECEDENT",
        "G2_DEFEATER",
        "G3_BASE_CONSEQUENT",
        "G3_DEFEATER_CONSEQUENT",
    ]

    gate_names = {
        "G1_ANTECEDENT": "G1 antecedent",
        "G2_DEFEATER": "G2 defeater",
        "G3_BASE_CONSEQUENT": "G3 base consequent",
        "G3_DEFEATER_CONSEQUENT": "G3 defeater consequent",
    }

    table["gate_order"] = table["gate"].map({g: i for i, g in enumerate(gate_order)})
    table = table.sort_values(["gate_order", "threshold_eval"])
    table["gate"] = table["gate"].replace(gate_names)

    table = table.rename(columns={
        "gate": "Gate",
        "threshold_eval": "$\\tau$",
        "n_usable": "$N$",
        "accuracy": "Acc.",
        "precision_entailment": "Prec.",
        "recall_entailment": "Rec.",
        "f1_entailment": "$F_1$",
        "false_positive_rate": "FPR",
        "false_negative_rate": "FNR",
    })

    table = table.drop(columns=["gate_order"])

    table["$\\tau$"] = table["$\\tau$"].map(
        lambda x: "--" if pd.isna(x) else f"{x:.2f}"
    )
    table["$N$"] = table["$N$"].map(
        lambda x: "--" if pd.isna(x) else f"{int(x)}"
    )

    for col in ["Acc.", "Prec.", "Rec.", "$F_1$", "FPR", "FNR"]:
        table[col] = table[col].map(
            lambda x: "--" if pd.isna(x) else f"{x:.2f}"
        )

    return table.to_latex(
        index=False,
        escape=False,
        column_format="llrrrrrrr",
        caption="DeBERTa oracle agreement with human annotations by gate across entailment thresholds.",
        label="tab:oracle-threshold-sweep-by-gate",
    )


def build_report(df: pd.DataFrame, summary_all: dict, summary_gate: pd.DataFrame, summary_band: pd.DataFrame) -> str:
    lines = []

    lines.append("ELODIE ORACLE VALIDATION REPORT")
    lines.append("=" * 80)
    lines.append("")

    complete = df[(df["human_label_A"] != "") & (df["human_label_B"] != "")]
    if len(complete):
        kappa_3way = cohen_kappa_score(
            complete["human_label_A"],
            complete["human_label_B"],
            labels=["ENTAILED", "NOT_ENTAILED", "UNCLEAR"],
        )

        a_bin = complete["human_label_A"].replace({"UNCLEAR": "NOT_ENTAILED"})
        b_bin = complete["human_label_B"].replace({"UNCLEAR": "NOT_ENTAILED"})
        kappa_binary_unclear_negative = cohen_kappa_score(
            a_bin,
            b_bin,
            labels=["ENTAILED", "NOT_ENTAILED"],
        )

        exact_agreement = (complete["human_label_A"] == complete["human_label_B"]).mean()
    else:
        kappa_3way = np.nan
        kappa_binary_unclear_negative = np.nan
        exact_agreement = np.nan

    lines.append("Inter-annotator agreement")
    lines.append("-" * 80)
    lines.append(f"Items with both labels: {len(complete)}")
    lines.append(f"Raw exact agreement:    {fmt(exact_agreement)}")
    lines.append(f"Cohen's kappa, 3-way:   {fmt(kappa_3way)}")
    lines.append(f"Cohen's kappa, binary:  {fmt(kappa_binary_unclear_negative)}")
    lines.append("  Binary kappa maps UNCLEAR to NOT_ENTAILED.")
    lines.append("")

    lines.append("DeBERTa vs human gold, overall")
    lines.append("-" * 80)
    for k in [
        "n_items",
        "n_usable",
        "n_unclear_or_disagreement",
        "accuracy",
        "precision_entailment",
        "recall_entailment",
        "f1_entailment",
        "false_positive_rate",
        "false_negative_rate",
        "deberta_positive_rate",
        "human_positive_rate",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    ]:
        lines.append(f"{k:<32} {fmt(summary_all.get(k))}")
    lines.append("")

    lines.append("DeBERTa vs human gold, by gate")
    lines.append("-" * 80)
    if not summary_gate.empty:
        cols = [
            "gate",
            "n_usable",
            "accuracy",
            "precision_entailment",
            "recall_entailment",
            "f1_entailment",
            "false_positive_rate",
            "false_negative_rate",
        ]
        lines.append(summary_gate[cols].to_string(index=False, formatters={
            c: (lambda x: fmt(x)) for c in cols if c != "gate"
        }))
    lines.append("")

    lines.append("DeBERTa vs human gold, by gate and score band")
    lines.append("-" * 80)
    if not summary_band.empty:
        cols = [
            "gate",
            "score_band",
            "n_usable",
            "accuracy",
            "precision_entailment",
            "recall_entailment",
            "f1_entailment",
        ]
        lines.append(summary_band[cols].to_string(index=False, formatters={
            c: (lambda x: fmt(x)) for c in cols if c not in {"gate", "score_band"}
        }))

    return "\n".join(lines)


def latex_threshold_sweep_accuracy_by_gate(sweep_df: pd.DataFrame) -> str:
    table = sweep_df[sweep_df["gate"] != "ALL"].copy()

    table["gate"] = table["gate"].replace({
        "G1_ANTECEDENT": "G1",
        "G2_DEFEATER": "G2",
        "G3_BASE_CONSEQUENT": "G3-C",
        "G3_DEFEATER_CONSEQUENT": "G3-CD",
    })

    pivot = table.pivot_table(
        index="threshold_eval",
        columns="gate",
        values=["accuracy", "precision_entailment", "f1_entailment"],
        aggfunc="first",
    )

    pivot.columns = [
        f"{metric}_{gate}"
        for metric, gate in pivot.columns
    ]

    pivot = pivot.reset_index()

    rename = {
        "threshold_eval": "$\\tau$",
        "accuracy_G1": "G1 Acc.",
        "accuracy_G2": "G2 Acc.",
        "accuracy_G3-C": "G3-C Acc.",
        "accuracy_G3-CD": "G3-CD Acc.",
        "precision_entailment_G1": "G1 Prec.",
        "precision_entailment_G2": "G2 Prec.",
        "precision_entailment_G3-C": "G3-C Prec.",
        "precision_entailment_G3-CD": "G3-CD Prec.",
        "f1_entailment_G1": "G1 $F_1$",
        "f1_entailment_G2": "G2 $F_1$",
        "f1_entailment_G3-C": "G3-C $F_1$",
        "f1_entailment_G3-CD": "G3-CD $F_1$",
    }

    pivot = pivot.rename(columns=rename)

    preferred_cols = [
        "$\\tau$",
        "G1 Acc.", "G2 Acc.", "G3-C Acc.", "G3-CD Acc.",
        "G1 Prec.", "G2 Prec.", "G3-C Prec.", "G3-CD Prec.",
        "G1 $F_1$", "G2 $F_1$", "G3-C $F_1$", "G3-CD $F_1$",
    ]

    pivot = pivot[[c for c in preferred_cols if c in pivot.columns]]

    pivot["$\\tau$"] = pivot["$\\tau$"].map(lambda x: f"{x:.2f}")

    for col in pivot.columns:
        if col != "$\\tau$":
            pivot[col] = pivot[col].map(lambda x: "--" if pd.isna(x) else f"{x:.3f}")

    return pivot.to_latex(
        index=False,
        escape=False,
        column_format="l" + "r" * (len(pivot.columns) - 1),
        caption="Threshold sensitivity of DeBERTa oracle agreement by gate.",
        label="tab:oracle-threshold-sensitivity-compact",
    )


def main() -> None:
    master_path = Path("../plots/oracle_annotation/annotation_sample_master.csv")
    annotator_a_path = Path("../plots/oracle_annotation/annotation_sample_annotator_A.csv")
    annotator_b_path = Path("../plots/oracle_annotation/annotation_sample_annotator_B.csv")
    output_path = Path("../plots/oracle_annotation/")
    unclear_policy = "negative"

    df = load_data(master_path, annotator_a_path, annotator_b_path)
    df = add_human_gold(df, unclear_policy=unclear_policy)

    summary_all = compute_metrics(df)
    summary_gate = summarize_by(df, ["gate"]).sort_values("gate")
    summary_band = summarize_by(df, ["gate", "score_band"]).sort_values(["gate", "score_band"])

    thresholds = [round(x, 2) for x in np.arange(0.5, 1.00, 0.1)]

    sweep = threshold_sweep(df, thresholds)
    best_by_accuracy = best_thresholds(sweep, metric="accuracy")
    best_by_f1 = best_thresholds(sweep, metric="f1_entailment")
    best_by_precision = best_thresholds(sweep, metric="precision_entailment")

    joined_path = output_path / "oracle_validation_joined.csv"
    gate_path = output_path / "oracle_validation_summary_by_gate.csv"
    band_path = output_path / "oracle_validation_summary_by_gate_band.csv"
    json_path = output_path / "oracle_validation_summary.json"
    report_path = output_path / "oracle_validation_report.txt"

    sweep_path = output_path / "oracle_validation_threshold_sweep.csv"
    best_acc_path = output_path / "oracle_validation_best_threshold_by_accuracy.csv"
    best_f1_path = output_path / "oracle_validation_best_threshold_by_f1.csv"
    best_precision_path = output_path / "oracle_validation_best_threshold_by_precision.csv"

    sweep.to_csv(sweep_path, index=False)
    best_by_accuracy.to_csv(best_acc_path, index=False)
    best_by_f1.to_csv(best_f1_path, index=False)
    best_by_precision.to_csv(best_precision_path, index=False)

    df.to_csv(joined_path, index=False)
    summary_gate.to_csv(gate_path, index=False)
    summary_band.to_csv(band_path, index=False)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "unclear_policy": unclear_policy,
                "overall": summary_all,
            },
            f,
            indent=2,
            default=lambda x: None if pd.isna(x) else x,
        )

    report = build_report(df, summary_all, summary_gate, summary_band)
    report_path.write_text(report, encoding="utf-8")

    print(report)
    print("")
    print(f"Wrote: {joined_path}")
    print(f"Wrote: {gate_path}")
    print(f"Wrote: {band_path}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {report_path}")

    print("\nBest thresholds by accuracy")
    print(
        best_by_accuracy[
            [
                "gate",
                "threshold_eval",
                "n_usable",
                "accuracy",
                "precision_entailment",
                "recall_entailment",
                "f1_entailment",
                "false_positive_rate",
                "false_negative_rate",
            ]
        ].to_string(index=False)
    )

    print("\nBest thresholds by F1")
    print(
        best_by_f1[
            [
                "gate",
                "threshold_eval",
                "n_usable",
                "accuracy",
                "precision_entailment",
                "recall_entailment",
                "f1_entailment",
                "false_positive_rate",
                "false_negative_rate",
            ]
        ].to_string(index=False)
    )

    print(f"\nWrote: {sweep_path}")
    print(f"Wrote: {best_acc_path}")
    print(f"Wrote: {best_f1_path}")
    print(f"Wrote: {best_precision_path}")

    latex_best_accuracy = threshold_table_for_latex(
        best_by_accuracy,
        caption="Best NLI thresholds by accuracy on the manually annotated oracle-validation sample.",
        label="tab:oracle-best-threshold-accuracy",
    )

    latex_best_f1 = threshold_table_for_latex(
        best_by_f1,
        caption="Best NLI thresholds by entailment $F_1$ on the manually annotated oracle-validation sample.",
        label="tab:oracle-best-threshold-f1",
    )

    print("\nLaTeX table: best thresholds by accuracy")
    print(latex_best_accuracy)

    print("\nLaTeX table: best thresholds by F1")
    print(latex_best_f1)

    latex_gate_threshold_sweep = latex_threshold_sweep_by_gate(sweep)
    print(latex_gate_threshold_sweep)
    latex_compact_gate_threshold_sweep = latex_threshold_sweep_accuracy_by_gate(sweep)
    print(latex_compact_gate_threshold_sweep)
    latex_gate_threshold_sweep = latex_threshold_sweep_by_gate_grouped(sweep)
    print(latex_gate_threshold_sweep)


if __name__ == "__main__":
    main()
