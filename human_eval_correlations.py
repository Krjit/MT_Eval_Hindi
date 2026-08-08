"""
Computes, from a results CSV that already contains precomputed automatic
metric scores (COMET, BERT[Score], BLEURT, BLEU, ChrF, ChrF++) alongside
MQM-derived Human_scores and a `model` column identifying the MT system:

  - Table 1 style: segment-level Pearson (rho) and Kendall-tau (tau)
    correlation between each metric and Human_scores, over every row.

  - Table 2 style: system-level Pearson (rho) and Kendall-tau (tau)
    correlation, computed by first averaging each metric (and
    Human_scores) per MT system, then correlating those per-system means
    across systems (following Louis & Nenkova, 2013, as cited in the
    paper for the system-level evaluation).

Usage:
    python human_eval_correlations.py \
        --input six_eval_score_Indic_MT_MQM_data_hin.csv \
        --out-table1 table1_segment_level.csv \
        --out-table2 table2_system_level.csv
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr

METRIC_COLUMNS = ["COMET", "BERT", "BLEURT", "BLEU", "ChrF", "ChrF++", "GEMBA_MQM"]
HUMAN_COLUMN = "Human_scores" #"Computed_scores"
MODEL_COLUMN = "model"


def correlate(x, y):
    """Pearson and Kendall-tau between two equal-length numeric series,
    dropping any row where either value is NaN."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]

    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan")

    rho, _ = pearsonr(x, y)
    tau, _ = kendalltau(x, y)
    return round(float(rho), 3), round(float(tau), 3)


def segment_level_table(df: pd.DataFrame) -> pd.DataFrame:
    """Table 1 style: one row per metric, correlated against Human_scores
    over all 1400 segments."""
    rows = []
    for metric in METRIC_COLUMNS:
        rho, tau = correlate(df[metric], df[HUMAN_COLUMN])
        rows.append({"metric": metric, "pearson": rho, "kendall_tau": tau})
    return pd.DataFrame(rows).sort_values("pearson", ascending=False).reset_index(drop=True)


def system_level_table(df: pd.DataFrame) -> pd.DataFrame:
    """Table 2 style: average each metric and Human_scores per MT system,
    then correlate those per-system means across systems."""
    per_system = df.groupby(MODEL_COLUMN)[METRIC_COLUMNS + [HUMAN_COLUMN]].mean()

    rows = []
    for metric in METRIC_COLUMNS:
        rho, tau = correlate(per_system[metric], per_system[HUMAN_COLUMN])
        rows.append({"metric": metric, "pearson": rho, "kendall_tau": tau})
    table = pd.DataFrame(rows).sort_values("pearson", ascending=False).reset_index(drop=True)
    return table, per_system


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="six_eval_score_Indic_MT_MQM_data_hin.csv")
    parser.add_argument("--out-table1", default="table1_segment_level.csv")
    parser.add_argument("--out-table2", default="table2_system_level.csv")
    parser.add_argument("--out-system-means", default="system_level_means.csv",
                         help="Also save the intermediate per-system mean scores used for Table 2")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    missing = [c for c in METRIC_COLUMNS + [HUMAN_COLUMN, MODEL_COLUMN] if c not in df.columns]
    if missing:
        raise SystemExit(f"Input CSV is missing expected column(s): {missing}")

    print(f"Loaded {len(df)} segments across {df[MODEL_COLUMN].nunique()} systems "
          f"({', '.join(sorted(df[MODEL_COLUMN].unique()))}).")

    # --- Table 1: segment-level -----------------------------------------------------
    table1 = segment_level_table(df)
    table1.to_csv(args.out_table1, index=False)
    print("\n=== Table 1 - Segment-level correlations (n=%d) ===" % len(df))
    print(table1.to_string(index=False))
    print(f"Saved to {args.out_table1}")

    # --- Table 2: system-level -------------------------------------------------------
    table2, per_system_means = system_level_table(df)
    table2.to_csv(args.out_table2, index=False)
    per_system_means.round(3).to_csv(args.out_system_means)
    print("\n=== Table 2 - System-level correlations (n=%d systems) ===" % df[MODEL_COLUMN].nunique())
    print(table2.to_string(index=False))
    print(f"Saved to {args.out_table2}")
    print(f"Per-system mean scores saved to {args.out_system_means}")


if __name__ == "__main__":
    main()
