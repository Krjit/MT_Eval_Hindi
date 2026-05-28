"""
run_evaluation.py
───────────────────────────
Compute raw MT evaluation metrics (COMET, BERTScore, BLEURT, BLEU, ChrF, ChrF++)
for all samples in the test set, then run the full adjustment framework.

Outputs a CSV with original and adjusted scores for every metric. 

Usage:
    python run_evaluation.py \
        --test_csv     data/Testset_All_Mixed_Hindi_Balanced.csv \
        --contra_model models/contra_detect_best.pt \
        --word2idx     models/hindi_cwe_word2idx300d.pkl \
        --cwe_emb      models/hindi_cwe_finetuned_emb_300d.pt \
        --para_model   models/para_detect_best.pth \
        --antonym_csv  data/eng-hin_antonym_pairs.csv \
        --output_csv   results/evaluated_testset.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  Metric computation helpers
# ─────────────────────────────────────────────────────────────

def compute_comet(df: pd.DataFrame, model_name: str, device: torch.device) -> list[float]:
    from comet import download_model, load_from_checkpoint  # type: ignore
    logger.info("Computing COMET …")
    model_path = download_model(model_name)
    comet_model = load_from_checkpoint(model_path)
    data = [
        {"src": row["src"], "mt": row["mt"], "ref": row["ref"]}
        for _, row in df.iterrows()
    ]
    gpus = 1 if device.type == "cuda" else 0
    scores = comet_model.predict(data, batch_size=16, gpus=gpus).scores
    return [round(float(s), 4) for s in scores]


def compute_bertscore(df: pd.DataFrame, device: torch.device) -> list[float]:
    from bert_score import score as bscore  # type: ignore
    logger.info("Computing BERTScore …")
    _, _, F1 = bscore(
        df["mt"].tolist(), df["ref"].tolist(),
        lang="hi", verbose=False, device=str(device),
    )
    return [round(f.item(), 4) for f in F1]


def compute_bleurt(df: pd.DataFrame, model_name: str, device: torch.device) -> list[float]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
    logger.info("Computing BLEURT …")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    mdl.eval()

    scores = []
    for mt, ref in tqdm(zip(df["mt"], df["ref"]), total=len(df), desc="BLEURT"):
        inputs = tok(mt, ref, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            s = mdl(**inputs).logits.squeeze().cpu().item()
        scores.append(round(max(0.0, min(1.0, float(s))), 4))
    return scores


def compute_sacrebleu_metrics(df: pd.DataFrame) -> tuple[list[float], list[float], list[float]]:
    import sacrebleu  # type: ignore
    logger.info("Computing BLEU / ChrF / ChrF++ …")
    bleu_scores, chrf_scores, chrfpp_scores = [], [], []
    for mt, ref in zip(df["mt"], df["ref"]):
        bleu_scores.append(round(sacrebleu.sentence_bleu(mt, [ref]).score / 100, 4))
        chrf_scores.append(round(sacrebleu.sentence_chrf(mt, [ref]).score / 100, 4))
        chrfpp_scores.append(round(
            sacrebleu.sentence_chrf(mt, [ref], word_order=2).score / 100, 4
        ))
    return bleu_scores, chrf_scores, chrfpp_scores


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    df = pd.read_csv(args.test_csv)
    logger.info("Loaded %d samples from %s", len(df), args.test_csv)

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    # ── Compute raw metrics ───────────────────────────────────
    if args.skip_metrics and os.path.exists(args.output_csv):
        logger.info("--skip_metrics set; loading pre-computed scores from %s", args.output_csv)
        df = pd.read_csv(args.output_csv)
    else:
        df["COMET"] = compute_comet(df, args.comet_model, device)
        df["BERTScore"] = compute_bertscore(df, device)
        df["BLEURT"] = compute_bleurt(df, args.bleurt_model, device)
        bleu, chrf, chrfpp = compute_sacrebleu_metrics(df)
        df["BLEU"] = bleu
        df["ChrF"] = chrf
        df["ChrF++"] = chrfpp

        # Save intermediate result
        df.to_csv(args.output_csv, index=False)
        logger.info("Raw metric scores saved to %s", args.output_csv)

    # ── Run framework ─────────────────────────────────────────
    from src.framework import MTEvalFramework  # noqa: E402

    fw = MTEvalFramework(
        antonym_csv=args.antonym_csv,
        contra_model_path=args.contra_model,
        word2idx_path=args.word2idx,
        cwe_emb_path=args.cwe_emb,
        para_model_path=args.para_model,
        tau=args.tau,
        device=str(device),
    )

    logger.info("Running full framework …")
    result_df = fw.evaluate(df)

    result_df.to_csv(args.output_csv, index=False)
    logger.info("Adjusted scores saved to %s", args.output_csv)

    # ── Quick summary ─────────────────────────────────────────
    metrics = ["COMET", "BERTScore", "BLEURT", "BLEU", "ChrF", "ChrF++"]
    print("\n" + "=" * 60)
    print(f"{'Metric':<12} {'Orig Mean':>10} {'Adj Mean':>10}")
    print("-" * 60)
    for m in metrics:
        if m in result_df.columns and f"{m}_adj" in result_df.columns:
            orig_mean = result_df[m].mean()
            adj_mean = result_df[f"{m}_adj"].mean()
            print(f"{m:<12} {orig_mean:>10.4f} {adj_mean:>10.4f}")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate MT metrics and run adjustment framework")
    p.add_argument("--test_csv", required=True)
    p.add_argument("--contra_model", required=True)
    p.add_argument("--word2idx", required=True)
    p.add_argument("--cwe_emb", required=True)
    p.add_argument("--para_model", required=True)
    p.add_argument("--antonym_csv", default="data/eng-hin_antonym_pairs.csv")
    p.add_argument("--output_csv", default="results/evaluated_testset.csv")
    p.add_argument("--comet_model", default="Unbabel/wmt22-comet-da")
    p.add_argument("--bleurt_model", default="Elron/bleurt-large-512")
    p.add_argument("--tau", type=float, default=0.6)
    p.add_argument("--skip_metrics", action="store_true",
                   help="Skip metric computation and load from --output_csv if it exists")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
