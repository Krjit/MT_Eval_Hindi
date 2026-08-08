"""
inference.py — single-sample inference (raw + adjusted metric scores).

Usage:
    python inference.py --src "..." --ref "..." --mt "..." \
        --contra_model models/contra_detect_best.pt \
        --word2idx models/hindi_cwe_word2idx300d.pkl \
        --cwe_emb models/hindi_cwe_finetuned_emb_300d.pt \
        --para_model models/para_detect_best.pth
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import torch

from run_evaluation import (
    compute_comet, compute_bertscore, compute_bleurt, compute_sacrebleu_metrics,
)
from src.framework import MTEvalFramework

METRICS = ["COMET", "BERTScore", "BLEURT", "BLEU", "ChrF", "ChrF++"]


def run_single_inference(
    src: str, ref: str, mt: str,
    contra_model: str, word2idx: str, cwe_emb: str, para_model: str,
    antonym_csv: str, tau: float = 0.6,
    comet_model: str = "Unbabel/wmt22-comet-da",
    bleurt_model: str = "Elron/bleurt-large-512",
    device: str | None = None,
) -> dict:
    """Compute raw metrics + run the step-wise framework on one (src, ref, mt) triple."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    df = pd.DataFrame([{"src": src, "ref": ref, "mt": mt}])

    bleu, chrf, chrfpp = compute_sacrebleu_metrics(df)
    raw = {
        "COMET": compute_comet(df, comet_model, dev)[0],
        "BERTScore": compute_bertscore(df, dev)[0],
        "BLEURT": compute_bleurt(df, bleurt_model, dev)[0],
        "BLEU": bleu[0], "ChrF": chrf[0], "ChrF++": chrfpp[0],
    }

    fw = MTEvalFramework(
        antonym_csv=antonym_csv, contra_model_path=contra_model,
        word2idx_path=word2idx, cwe_emb_path=cwe_emb,
        para_model_path=para_model, tau=tau, device=str(dev),
    )
    result = fw.adjust_single(mt, ref, raw)
    stage, d_score = result.pop("stage_resolved"), result.pop("d_score")

    return {"raw_scores": raw, "adjusted_scores": result, "stage_resolved": stage, "d_score": d_score}


def print_report(src: str, ref: str, mt: str, result: dict) -> None:
    raw, adj = result["raw_scores"], result["adjusted_scores"]
    print(f"\nSource: {src}\nReference: {ref}\nTranslation: {mt}")
    print(f"Stage resolved: {result['stage_resolved']}  |  Decision score: {result['d_score']:.4f}\n")
    print(f"{'Metric':<12}{'Orig':>8}{'Adj':>8}  Direction")
    for m in METRICS:
        o, a = raw[m], adj[m]
        arrow = "↑ reward" if a > o else "↓ penalty" if a < o else "= unchanged"
        print(f"{m:<12}{o:>8.4f}{a:>8.4f}  {arrow}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--mt", required=True)
    p.add_argument("--contra_model", required=True)
    p.add_argument("--word2idx", required=True)
    p.add_argument("--cwe_emb", required=True)
    p.add_argument("--para_model", required=True)
    p.add_argument("--antonym_csv", default="data/eng-hin_antonym_pairs.csv")
    p.add_argument("--comet_model", default="Unbabel/wmt22-comet-da")
    p.add_argument("--bleurt_model", default="Elron/bleurt-large-512")
    p.add_argument("--tau", type=float, default=0.6)
    args = p.parse_args()

    result = run_single_inference(
        args.src, args.ref, args.mt,
        args.contra_model, args.word2idx, args.cwe_emb, args.para_model,
        args.antonym_csv, args.tau, args.comet_model, args.bleurt_model,
    )
    print_report(args.src, args.ref, args.mt, result)


if __name__ == "__main__":
    main()
