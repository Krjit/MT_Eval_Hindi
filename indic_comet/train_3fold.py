#!/usr/bin/env python3
"""Prepare and train IndicCOMET-DA Hindi with 3-fold cross validation."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold

SEED = 42
N_SPLITS = 3
INIT_MODELS = {
    "comet-da": "Unbabel/wmt22-comet-da",
}

PROJECT_DIR = Path(__file__).resolve().parent
CFG_DIR = PROJECT_DIR / "configs"
GEN_CFG = CFG_DIR / "generated"
FOLD_DIR = PROJECT_DIR / "folds"
RUN_DIR = PROJECT_DIR / "runs"

BASE_MODEL_CFG = CFG_DIR / "regression_model.yaml"
TRAINER_CFG = CFG_DIR / "trainer.yaml"
EARLY_STOPPING_CFG = CFG_DIR / "early_stopping.yaml"
MODEL_CHECKPOINT_CFG = CFG_DIR / "model_checkpoint.yaml"

DEFAULT_DATA_FILE = PROJECT_DIR / "train_human_indic_comet.jsonl"

np.random.seed(SEED)


def load_yaml(file):
    with open(file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj, file):
    with open(file, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            obj,
            f,
            sort_keys=False,
            allow_unicode=True,
        )


def read_jsonl(file, score_field):
    rows = []

    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)

            rows.append({
                "src": x["Source"].strip(),
                "mt": x["Translation"].strip(),
                "ref": x["Reference"].strip(),
                "score": float(x[score_field]),
            })

    return pd.DataFrame(rows)


def ensure_directories():
    for directory in [GEN_CFG, FOLD_DIR, RUN_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def load_dataset(score_field, data_file):
    data_file = Path(data_file)

    if not data_file.exists():
        raise FileNotFoundError(f"Missing Hindi data file: {data_file}")

    df = read_jsonl(data_file, score_field)

    df = df.drop_duplicates(
        subset=["src", "mt", "ref"],
    ).reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No rows found in {data_file}.")

    print(f"Data file     : {data_file}")
    print(f"Total Samples : {len(df)}")
    print(f"Unique Sources: {df.src.nunique()}")
    print(f"Training target: {score_field}")

    return df


def save_fold(train_df, val_df, fold):
    train_file = FOLD_DIR / f"fold{fold}_train.csv"
    val_file = FOLD_DIR / f"fold{fold}_val.csv"

    train_df.to_csv(
        train_file,
        index=False,
    )

    val_df.to_csv(
        val_file,
        index=False,
    )

    print(
        f"Fold {fold}: "
        f"{len(train_df)} train | "
        f"{len(val_df)} val"
    )


def prepare_folds(df):
    if df.src.nunique() < N_SPLITS:
        raise ValueError(
            f"Need at least {N_SPLITS} unique sources for GroupKFold, "
            f"found {df.src.nunique()}."
        )

    splitter = GroupKFold(
        n_splits=N_SPLITS,
    )

    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(df, groups=df.src),
        start=1,
    ):

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        save_fold(
            train_df,
            val_df,
            fold,
        )


def abs_posix(path):
    return path.resolve().as_posix()


def build_fold_config(
    fold,
    load_from_checkpoint=None,
    devices=None,
    strategy=None,
    run_name="indiccomet_mqm_xlm",
):
    cfg = load_yaml(BASE_MODEL_CFG)

    train_file = FOLD_DIR / f"fold{fold}_train.csv"
    val_file = FOLD_DIR / f"fold{fold}_val.csv"
    run_dir = RUN_DIR / run_name / f"fold{fold}"
    ckpt_dir = run_dir / "checkpoints"

    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    metric_args = cfg["regression_metric"]["init_args"]
    metric_args["train_data"] = [abs_posix(train_file)]
    metric_args["validation_data"] = [abs_posix(val_file)]

    trainer_cfg = load_yaml(TRAINER_CFG)
    trainer_cfg["init_args"]["default_root_dir"] = abs_posix(run_dir)

    if devices is not None:
        trainer_cfg["init_args"]["devices"] = devices
        trainer_cfg["init_args"]["use_distributed_sampler"] = devices > 1
    if strategy is not None:
        trainer_cfg["init_args"]["strategy"] = strategy
    elif trainer_cfg["init_args"].get("devices") == 1:
        trainer_cfg["init_args"].pop("strategy", None)

    checkpoint_cfg = load_yaml(MODEL_CHECKPOINT_CFG)
    checkpoint_cfg["init_args"]["dirpath"] = abs_posix(ckpt_dir)

    cfg["trainer"] = trainer_cfg
    cfg["early_stopping"] = load_yaml(EARLY_STOPPING_CFG)
    cfg["model_checkpoint"] = checkpoint_cfg
    if load_from_checkpoint is not None:
        cfg["load_from_checkpoint"] = abs_posix(load_from_checkpoint)

    out_dir = GEN_CFG / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"fold{fold}.yaml"
    save_yaml(cfg, out_file)
    print(f"Saved config: {out_file}")
    return out_file


def generate_configs(
    load_from_checkpoint=None,
    devices=None,
    strategy=None,
    run_name="indiccomet_mqm_xlm",
):
    return [
        build_fold_config(
            fold,
            load_from_checkpoint,
            devices,
            strategy,
            run_name,
        )
        for fold in range(1, N_SPLITS + 1)
    ]


def sync_hparams_for_comet_loader(run_dir):
    run_dir = Path(run_dir)
    expected = run_dir / "hparams.yaml"
    if expected.exists():
        return

    candidates = sorted(
        (run_dir / "lightning_logs").glob("version_*/hparams.yaml"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        shutil.copy2(candidates[0], expected)
        print(f"Copied hparams for COMET loading: {expected}")


def resolve_init_checkpoint(init, load_from_checkpoint):
    if load_from_checkpoint is not None:
        checkpoint = Path(load_from_checkpoint)
        if checkpoint.exists():
            return checkpoint.resolve()

        from comet import download_model

        return Path(download_model(load_from_checkpoint)).resolve()

    model_name = INIT_MODELS[init]
    if model_name is None:
        return None

    from comet import download_model
    return Path(download_model(model_name)).resolve()


def run_training(config_files, run_name):
    comet_train = shutil.which("comet-train")
    if comet_train is None:
        raise RuntimeError(
            "Could not find 'comet-train' on PATH. Install COMET or activate the "
            "environment that provides the COMET training CLI."
        )

    for fold, config_file in enumerate(config_files, start=1):
        print(f"\nStarting fold {fold}/{N_SPLITS}")
        subprocess.run(
            [
                comet_train,
                "--cfg",
                abs_posix(config_file),
                "--seed_everything",
                str(SEED),
            ],
            cwd=PROJECT_DIR,
            check=True,
        )
        sync_hparams_for_comet_loader(RUN_DIR / run_name / f"fold{fold}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare and train IndicCOMET Hindi 3-fold CV.",
    )
    parser.add_argument(
        "--data-file",
        default=str(DEFAULT_DATA_FILE),
        help=(
            "Path to the cleaned single JSONL file (e.g. train_indic_comet.jsonl) "
            "containing all Hindi rows to be split into 3 folds."
        ),
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Create folds and generated YAML configs, but do not run comet-train.",
    )
    parser.add_argument(
        "--skip-folds",
        action="store_true",
        help="Reuse existing CSV files in folds/ and only regenerate YAML configs.",
    )
    parser.add_argument(
        "--load-from-checkpoint",
        help=(
            "Initialize every fold from this checkpoint path or Hugging Face "
            "COMET model name. Overrides --init."
        ),
    )
    parser.add_argument(
        "--init",
        choices=sorted(INIT_MODELS),
        default="xlm",
        help=(
            "Initial weights variant for Table 3-style experiments: xlm, "
            "comet-da, or comet-mqm."
        ),
    )
    parser.add_argument(
        "--score-field",
        default="Human_scores",
        choices=[
            "Human_scores",
            "mqm_norm_score",
            "da_norm_score",
            "adequacy_score",
            "fluency_score",
            "full_score",
        ],
        help=(
            "JSONL human score used as COMET training target. Table 3-style "
            "IndicCOMET uses the MQM dataset, so Human_scores is the default."
        ),
    )
    parser.add_argument(
        "--run-name",
        help=(
            "Folder name under runs/ and configs/generated/. Defaults to "
            "indiccomet_<score-field>_<init>."
        ),
    )
    parser.add_argument(
        "--devices",
        type=int,
        help=(
            "Number of GPU devices for Lightning. Default comes from "
            "configs/trainer.yaml; this project uses 1 to avoid Python 3.12 "
            "Hydra/DDP launcher crashes."
        ),
    )
    parser.add_argument(
        "--strategy",
        help=(
            "Optional Lightning strategy override. Leave unset for single-GPU "
            "training; use ddp only in an environment with compatible Hydra."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_directories()
    run_name = args.run_name
    if run_name is None:
        score_name = args.score_field.replace("_norm_score", "").replace("_score", "")
        init_name = args.init.replace("-", "_")
        run_name = f"indiccomet_{score_name}_{init_name}"

    if not args.skip_folds:
        df = load_dataset(args.score_field, args.data_file)
        prepare_folds(df)

    init_checkpoint = resolve_init_checkpoint(args.init, args.load_from_checkpoint)
    config_files = generate_configs(
        init_checkpoint,
        args.devices,
        args.strategy,
        run_name,
    )

    if args.prepare_only:
        print("\nPreparation complete. Training was skipped.")
        return

    run_training(config_files, run_name)


if __name__ == "__main__":
    main()
