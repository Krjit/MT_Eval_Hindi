#!/usr/bin/env python3
"""Score combined Hindi JSONL data for Table 3-style COMET correlations."""

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_JSONL = [
    PROJECT_DIR/"test_human_indic_comet.jsonl",
]
BASELINE_MODELS = {
    "COMET_DA": "Unbabel/wmt22-comet-da"
}


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for row_id, line in enumerate(handle):
            item = json.loads(line)
            rows.append({
                "source_file": path.name,
                "row_id": row_id,
                "src": str(item["src"]).strip(),
                "mt": str(item["translation"]).strip(),
                "ref": str(item["ref"]).strip(),
                "human_da": float(item["da_norm_score"]),
                "human_mqm": float(item["mqm_norm_score"]),
                "adequacy_score": float(item["adequacy_score"]),
                "fluency_score": float(item["fluency_score"]),
                "full_score": float(item["full_score"]),
                'human_scores': float(item["Human_scores"]),
                "model": item.get("model") or item.get("system"),
            })
    return pd.DataFrame(rows)


def load_combined(jsonl_files, dedupe=True):
    df = pd.concat([read_jsonl(path) for path in jsonl_files], ignore_index=True)
    if dedupe:
        df = df.drop_duplicates(subset=["src", "mt", "ref"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("No rows loaded from JSONL input.")
    return df


def comet_records(df):
    return df[["src", "mt", "ref"]].to_dict(orient="records")


def predict_scores(model, records, batch_size, gpus):
    try:
        output = model.predict(records, batch_size=batch_size, gpus=gpus)
    except TypeError:
        output = model.predict(records, batch_size=batch_size)
    return [float(score) for score in output.scores]


def load_comet_model(model_or_checkpoint):
    from comet import download_model, load_from_checkpoint

    model_path = Path(model_or_checkpoint)
    if model_path.exists():
        ensure_hparams_for_checkpoint(model_path)
        return load_from_checkpoint(str(model_path))

    downloaded_path = download_model(model_or_checkpoint)
    return load_from_checkpoint(downloaded_path)


def ensure_hparams_for_checkpoint(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    run_dir = checkpoint_path.parents[1]
    expected = run_dir / "hparams.yaml"
    if expected.exists():
        return

    candidates = sorted(
        (run_dir / "lightning_logs").glob("version_*/hparams.yaml"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return

    shutil.copy2(candidates[0], expected)
    print(f"Copied hparams for COMET loading: {expected}")


def best_checkpoint(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    if not checkpoints and (checkpoint_dir / "checkpoints").is_dir():
        checkpoints = list((checkpoint_dir / "checkpoints").glob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No .ckpt files found in {checkpoint_dir}")

    def sort_key(path):
        match = re.search(r"val_kendall[=.-](-?\d+(?:\.\d+)?)", path.name)
        if match:
            return (1, float(match.group(1)), path.stat().st_mtime)
        return (0, path.stat().st_mtime, path.stat().st_mtime)

    return max(checkpoints, key=sort_key)


def score_oof_indic(df, checkpoints_dir, batch_size, gpus):
    checkpoints_dir = Path(checkpoints_dir)
    if not checkpoints_dir.exists() and (PROJECT_DIR / checkpoints_dir).exists():
        checkpoints_dir = PROJECT_DIR / checkpoints_dir

    scores = pd.Series(np.nan, index=df.index, dtype=float)
    key_to_indices = {}
    for idx, row in df.iterrows():
        key = (row["src"], row["mt"], row["ref"])
        key_to_indices.setdefault(key, []).append(idx)

    for fold in range(1, 4):
        checkpoint = best_checkpoint(checkpoints_dir / f"fold{fold}")
        val_file = PROJECT_DIR / "folds" / f"fold{fold}_val.csv"
        val_df = pd.read_csv(val_file)
        model = load_comet_model(checkpoint)
        fold_scores = predict_scores(model, comet_records(val_df), batch_size, gpus)

        assigned = 0
        for row, score in zip(val_df.itertuples(index=False), fold_scores):
            for idx in key_to_indices.get((row.src, row.mt, row.ref), []):
                scores.loc[idx] = score
                assigned += 1
        print(f"Scored fold {fold} with {checkpoint} ({assigned} rows assigned).")

    missing = scores.isna().sum()
    if missing:
        raise ValueError(
            f"{missing} rows did not receive out-of-fold IndicCOMET scores. "
            "Use the same dedupe setting used when folds were created."
        )
    return scores.tolist()


def parse_named_path(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Expected NAME=PATH, for example IndicCOMETMQM=runs/indiccomet_mqm_comet_mqm"
        )
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Metric name before '=' cannot be empty.")
    return name, Path(path)


def correlate(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan"), len(x)
    pearson, _ = pearsonr(x, y)
    kendall, _ = kendalltau(x, y)
    return round(float(pearson), 4), round(float(kendall), 4), len(x)


def correlation_table(df, metrics, human_column):
    rows = []
    for metric in metrics:
        pearson, kendall, n = correlate(df[metric], df[human_column])
        rows.append({
            "level": "segment",
            "metric": metric,
            "human": human_column,
            "n": n,
            "pearson": pearson,
            "kendall_tau": kendall,
        })

    if "model" in df.columns and df["model"].notna().any():
        per_system = df.groupby("model")[metrics + [human_column]].mean()
        for metric in metrics:
            pearson, kendall, n = correlate(per_system[metric], per_system[human_column])
            rows.append({
                "level": "system",
                "metric": metric,
                "human": human_column,
                "n": n,
                "pearson": pearson,
                "kendall_tau": kendall,
            })

    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jsonl",
        nargs="+",
        type=Path,
        default=DEFAULT_JSONL,
        help="JSONL file(s) to combine and score.",
    )
    parser.add_argument(
        "--out-scores",
        type=Path,
        default=PROJECT_DIR / "results" / "combined_table3_scores.csv",
    )
    parser.add_argument(
        "--out-corr",
        type=Path,
        default=PROJECT_DIR / "results" / "combined_table3_correlations.csv",
    )
    parser.add_argument("--human-column", default="human_scores")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=sorted(BASELINE_MODELS),
        default=sorted(BASELINE_MODELS),
        help="Pretrained COMET baselines to score. Table 3 uses COMET_DA and COMET_MQM.",
    )
    parser.add_argument(
        "--indic-checkpoint",
        type=Path,
        help="Single trained IndicCOMET checkpoint to score every row.",
    )
    parser.add_argument(
        "--indic-run",
        action="append",
        type=parse_named_path,
        default=[],
        metavar="NAME=PATH",
        help=(
            "Out-of-fold IndicCOMET run to score. Can be repeated, e.g. "
            "IndicCOMETXLM=runs/indiccomet_mqm_xlm."
        ),
    )
    parser.add_argument(
        "--oof",
        action="store_true",
        help="Use fold1/fold2/fold3 checkpoints for out-of-fold IndicCOMET scores.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=PROJECT_DIR / "runs",
        help="Directory containing fold1/fold2/fold3 run or checkpoint subdirectories.",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep duplicate src/mt/ref rows instead of matching the training dedupe.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = load_combined(args.jsonl)#, dedupe=not args.keep_duplicates)
    args.out_scores.parent.mkdir(parents=True, exist_ok=True)
    args.out_corr.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(df)} rows.")

    records = comet_records(df)
    metrics = []
    for metric in args.baselines:
        comet_model = load_comet_model(BASELINE_MODELS[metric])
        df[metric] = predict_scores(
            comet_model,
            records,
            args.batch_size,
            args.gpus,
        )
        metrics.append(metric)

    if args.oof:
        print("Using the best checkpoint from the trained folds...")
        
        # Automatically pick the best checkpoint among fold1, fold2 and fold3
        best_ckpt = None
        best_score = -1e9
        for fold in range(1, 4):
            ckpt = best_checkpoint(args.checkpoints_dir / f"fold{fold}")
            m = re.search(r"val_kendall[=.-](-?\d+(?:\.\d+)?)", ckpt.name)
            score = float(m.group(1)) if m else -1e9
            if score > best_score:
                best_score = score
                best_ckpt = ckpt
        
        print(f"Using checkpoint: {best_ckpt}")
        indic_model = load_comet_model(best_ckpt)
        df["IndicCOMET_DA"] = predict_scores(
            indic_model,
            records,
            args.batch_size,
            args.gpus,
        )
        metrics.append("IndicCOMET_DA")
        
    elif args.indic_checkpoint:
        indic_model = load_comet_model(args.indic_checkpoint)
        df["IndicCOMET_DA"] = predict_scores(
            indic_model,
            records,
            args.batch_size,
            args.gpus,
        )
        metrics.append("IndicCOMET_DA")
        
    elif not args.indic_run:
        print("No IndicCOMET checkpoint/run provided; only COMET baselines will be scored.")

    corr = correlation_table(df, metrics, args.human_column)
    df.to_csv(args.out_scores, index=False)
    corr.to_csv(args.out_corr, index=False)

    print(f"Saved scores: {args.out_scores}")
    print(f"Saved correlations: {args.out_corr}")
    print(corr.to_string(index=False))


if __name__ == "__main__":
    main()
