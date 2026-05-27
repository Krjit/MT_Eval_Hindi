"""
scripts/run_results.py
───────────────────────
produce:
  Component-wise Precision / Recall / F1
  Per-category Precision / Recall / F1 / Support
  % Good-Score samples before and after adjustment
  Pearson correlation with P/NP semantic labels
  Pearson correlation with human DA scores

Requires a results CSV produced by evaluate/run_evaluation.py that contains:
  ref, mt, label, category, d_rule, d_contra, d_para, d_score,
  COMET, BERTScore, BLEURT, BLEU, ChrF, ChrF++,
  COMET_adj, BERTScore_adj, BLEURT_adj, BLEU_adj, ChrF_adj, ChrF++_adj

Usage:
    python scripts/run_results.py \
        --results_csv results/evaluated_testset.csv \
        --human_csv  data/human_da_scores_test_data.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from scipy.stats import pearsonr  # type: ignore
from sklearn.metrics import classification_report, precision_recall_fscore_support  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

METRICS = ["COMET", "BERTScore", "BLEURT", "BLEU", "ChrF", "ChrF++"]
TAU = 0.6

SIMILAR_CATS = {"Word_synm", "Mixd_lang", "Negt_anto", "Identical", "Fluent", "Default_similar"}
DISSIMILAR_CATS = {
    "Anto_flip", "Negt_flip", "Gend_flip", "Sing_plul", "Tens_chng",
    "Word_ordr", "Word_rplc", "Add_extra", "Omission", "Neutral", "Default_dissimilar",
}


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

def _header(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def _mean_metric_score(df: pd.DataFrame, col: str) -> float:
    return df[col].mean() if col in df.columns else float("nan")


def _good_score_pct(scores: list[float], label: str) -> float:
    """
    Percentage of samples that receive the correct score direction.
    P group  : score > TAU  is "good"
    NP group : score ≤ TAU  is "good"
    """
    arr = np.array(scores)
    if label == "P":
        return float((arr > TAU).mean() * 100)
    else:
        return float((arr <= TAU).mean() * 100)


# ─────────────────────────────────────────────────────────────
# Component-wise performance
# ─────────────────────────────────────────────────────────────

def component_wise(df: pd.DataFrame) -> None:
    """
    Produce Precision / Recall / F1 for each pipeline component.

    We derive binary predictions from each decision score:
      d ≥ 0.5 → P (1)   d < 0.5 → NP (0)
    Then compare to the ground-truth label (P=1, NP=0).
    """
    _header("Component-wise Performance (Precision / Recall / F1)")

    y_true = (df["label"].str.strip() == "P").astype(int).tolist()

    components = {
        "Rule-Based":           df["d_rule"].apply(lambda x: 1 if x == 1 else 0),
        "Contra. Detection":    (1 - df["d_contra"]).apply(lambda x: 0 if x >= 0.5 else 1),
        "Paraphrase Detection": df["d_para"].apply(lambda x: 1 if x >= 0.5 else 0),
        "Full Framework":       df["d_score"].apply(lambda x: 1 if x >= 0.5 else 0),
    }

    # Rule + Contradiction: cascade — use d_rule where available, else contra
    def rule_contra(row):
        if row["d_rule"] is not None and not (isinstance(row["d_rule"], float) and np.isnan(row["d_rule"])):
            return 1 if row["d_rule"] == 1 else 0
        return 0 if (1 - row["d_contra"]) >= 0.5 else 1

    def rule_para(row):
        if row["d_rule"] is not None and not (isinstance(row["d_rule"], float) and np.isnan(row["d_rule"])):
            return 1 if row["d_rule"] == 1 else 0
        return 1 if row["d_para"] >= 0.5 else 0

    components["Rule + Contradiction"] = df.apply(rule_contra, axis=1)
    components["Rule + Paraphrase"] = df.apply(rule_para, axis=1)

    order = [
        "Rule-Based", "Contra. Detection", "Paraphrase Detection",
        "Rule + Contradiction", "Rule + Paraphrase", "Full Framework",
    ]

    print(f"\n{'Component':<25} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print("-" * 48)
    for name in order:
        preds = components[name].tolist()
        p, r, f, _ = precision_recall_fscore_support(
            y_true, preds, average="macro", zero_division=0
        )
        print(f"{name:<25} {p:>6.2f} {r:>6.2f} {f:>6.2f}")


# ─────────────────────────────────────────────────────────────
# Per-category performance
# ─────────────────────────────────────────────────────────────

def per_category(df: pd.DataFrame) -> None:
    _header("Per-Category Precision / Recall / F1 / Support")

    y_true = (df["label"].str.strip() == "P").astype(int)
    y_pred = (df["d_score"] >= 0.5).astype(int)

    categories = sorted(df["category"].unique())

    print(f"\n{'Category':<22} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Supp':>6}  Group")
    print("-" * 60)

    for cat in categories:
        mask = df["category"] == cat
        if mask.sum() == 0:
            continue
        yt = y_true[mask].tolist()
        yp = y_pred[mask].tolist()
        p, r, f, _ = precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)
        group = "P" if cat in SIMILAR_CATS else "NP"
        print(f"{cat:<22} {p:>6.2f} {r:>6.2f} {f:>6.2f} {mask.sum():>6}  {group}")

    # Macro average
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    print("-" * 60)
    print(f"{'Macro Avg':<22} {p:>6.2f} {r:>6.2f} {f:>6.2f}")


# ─────────────────────────────────────────────────────────────
#  Good-score % before and after adjustment
# ─────────────────────────────────────────────────────────────

def good_score_pct(df: pd.DataFrame) -> None:
    _header("% Good-Score Samples Before and After Adjustment")

    p_mask = df["label"].str.strip() == "P"
    np_mask = df["label"].str.strip() == "NP"

    print(f"\n{'Metric':<12} {'Orig P%':>9} {'Orig NP%':>9} {'Adj P%':>9} {'Adj NP%':>9}")
    print("-" * 55)

    for m in METRICS:
        if m not in df.columns:
            continue
        adj_col = f"{m}_adj"
        orig_p = _good_score_pct(df.loc[p_mask, m].tolist(), "P")
        orig_np = _good_score_pct(df.loc[np_mask, m].tolist(), "NP")
        adj_p = _good_score_pct(df.loc[p_mask, adj_col].tolist(), "P") if adj_col in df.columns else float("nan")
        adj_np = _good_score_pct(df.loc[np_mask, adj_col].tolist(), "NP") if adj_col in df.columns else float("nan")
        print(f"{m:<12} {orig_p:>9.1f} {orig_np:>9.1f} {adj_p:>9.1f} {adj_np:>9.1f}")


# ─────────────────────────────────────────────────────────────
#  Pearson correlation with P/NP labels
# ─────────────────────────────────────────────────────────────

def pearson_label(df: pd.DataFrame) -> None:
    _header("Pearson Correlation with Semantic Labels (P=1 / NP=0)")

    label_numeric = (df["label"].str.strip() == "P").astype(float)
    p_mask = df["label"].str.strip() == "P"
    np_mask = df["label"].str.strip() == "NP"

    print(f"\n{'Metric':<12} {'Orig Corr':>10} {'Adj Corr':>10} {'Orig MG':>9} {'Adj MG':>9}")
    print("-" * 58)

    for m in METRICS:
        if m not in df.columns:
            continue
        adj_col = f"{m}_adj"

        orig_r, _ = pearsonr(df[m].fillna(0), label_numeric)
        adj_r = float("nan")
        if adj_col in df.columns:
            adj_r, _ = pearsonr(df[adj_col].fillna(0), label_numeric)

        orig_mg = df.loc[p_mask, m].mean() - df.loc[np_mask, m].mean()
        adj_mg = (
            df.loc[p_mask, adj_col].mean() - df.loc[np_mask, adj_col].mean()
            if adj_col in df.columns else float("nan")
        )
        print(f"{m:<12} {orig_r:>10.2f} {adj_r:>10.2f} {orig_mg:>9.2f} {adj_mg:>9.2f}")


# ─────────────────────────────────────────────────────────────
# Pearson correlation with human DA scores
# ─────────────────────────────────────────────────────────────

def human_correlation(df: pd.DataFrame, human_csv: str) -> None:
    _header("Pearson Correlation with Human DA Scores (0–25)")

    human_df = pd.read_csv(human_csv)
    if "human_score" not in human_df.columns:
        logger.warning("Human CSV must have a 'human_score' column")
        return

    merged = df.merge(human_df[["ref", "mt", "human_score"]], on=["ref", "mt"], how="inner")
    logger.info("Matched %d samples for human correlation", len(merged))

    da = merged["human_score"].astype(float)

    print(f"\n{'Metric':<12} {'Orig r':>8} {'Adj r':>8}")
    print("-" * 32)
    for m in METRICS:
        if m not in merged.columns:
            continue
        adj_col = f"{m}_adj"
        orig_r, _ = pearsonr(merged[m].fillna(0), da)
        adj_r = float("nan")
        if adj_col in merged.columns:
            adj_r, _ = pearsonr(merged[adj_col].fillna(0), da)
        print(f"{m:<12} {orig_r:>8.2f} {adj_r:>8.2f}")


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    if not os.path.exists(args.results_csv):
        logger.error("Results CSV not found: %s", args.results_csv)
        sys.exit(1)

    df = pd.read_csv(args.results_csv)
    logger.info("Loaded %d rows from %s", len(df), args.results_csv)

    component_wise(df)
    per_category(df)
    good_score_pct(df)
    pearson_label(df)

    if args.human_csv and os.path.exists(args.human_csv):
        human_correlation(df, args.human_csv)
    else:
        print("\nProvide --human_csv to include human DA correlation")

    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Produce results")
    p.add_argument("--results_csv", required=True,
                   help="Output CSV from evaluate/run_evaluation.py")
    p.add_argument("--human_csv", default=None,
                   help="Optional CSV with columns [ref, mt, human_score]")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
