"""
src/framework.py
─────────────────
Full multi-step MT evaluation pipeline.

Pipeline per sample:
  1. Rule-Based  → d_rule ∈ {0, 1, None}
  2. Contradiction Detection (if d_rule is None) → d_contra ∈ [0,1]
  3. Paraphrase Detection  → d_para ∈ [0,1]
  4. Score Adjustment Layer → adjusted metric scores

Usage example:
    from src.framework import MTEvalFramework

    fw = MTEvalFramework(
        antonym_csv="data/eng-hin_antonym_pairs.csv",
        contra_model_path="models/contra_detect_best.pt",
        word2idx_path="models/hindi_cwe_word2idx300d.pkl",
        cwe_emb_path="models/hindi_cwe_finetuned_emb_300d.pt",
        para_model_path="models/para_detect_best.pth",
    )
    results = fw.evaluate(df)  # df has columns: src, ref, mt, + metric columns
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from .contradiction_detection import ContraDetector
from .paraphrase_detection import ParaDetector
from .rule_based import apply_rule_batch, load_antonym_dict
from .score_adjustment import METRICS, adjust_scores_batch, select_decision

logger = logging.getLogger(__name__)

REQUIRED_COLS = {"src", "ref", "mt"}


class MTEvalFramework:
    """
    Orchestrates the three-stage hybrid evaluation pipeline.

    Parameters
    ----------
    antonym_csv        : path to CSV with columns [word_hi, antonym_hi]
    contra_model_path  : path to trained ContraDetectCNN checkpoint (.pt)
    word2idx_path      : path to CWE vocabulary pickle (.pkl)
    cwe_emb_path       : path to CWE embedding matrix (.pt)
    para_model_path    : path to trained SiameseCNN checkpoint (.pth)
    tau                : score adjustment threshold (default 0.6)
    device             : 'cuda' | 'cpu' | None (auto-detect)
    """

    def __init__(
        self,
        antonym_csv: str,
        contra_model_path: str,
        word2idx_path: str,
        cwe_emb_path: str,
        para_model_path: str,
        tau: float = 0.6,
        device: Optional[str] = None,
    ) -> None:
        self.tau = tau
        self.antonym_dict = load_antonym_dict(antonym_csv)

        logger.info("Loading contradiction detection model …")
        self.contra_detector = ContraDetector(
            model_path=contra_model_path,
            word2idx_path=word2idx_path,
            cwe_emb_path=cwe_emb_path,
            device=device,
        )

        logger.info("Loading paraphrase detection model …")
        self.para_detector = ParaDetector(
            model_path=para_model_path,
            antonym_dict=self.antonym_dict,
            device=device,
        )

    # ──────────────────────────────────────────────────────────
    #  Core per-sample prediction
    # ──────────────────────────────────────────────────────────

    def predict_decision(self, mt: str, ref: str) -> tuple[Optional[int], float, float, float]:
        """
        Run the three pipeline stages for a single (mt, ref) pair.

        Returns
        -------
        d_rule   : Optional[int]  (0, 1, or None)
        d_contra : float          non-contradiction probability
        d_para   : float          paraphrase probability
        d_final  : float          unified decision score used for adjustment
        """
        d_rule = apply_rule_batch([mt], [ref], self.antonym_dict)[0]

        d_contra = self.contra_detector.predict([mt], [ref])[0]
        d_para = self.para_detector.predict([mt], [ref])[0]

        d_final = select_decision(d_rule, d_contra, d_para)
        return d_rule, d_contra, d_para, d_final

    # ──────────────────────────────────────────────────────────
    #  Batch evaluation
    # ──────────────────────────────────────────────────────────

    def evaluate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full framework on a DataFrame and return an enriched copy.

        Input DataFrame columns (required):
            src, ref, mt
        Optional metric columns (if present, adjusted scores are appended):
            COMET, BERTScore, BLEURT, BLEU, ChrF, ChrF++

        Added columns:
            d_rule, d_contra, d_para, d_score
            <Metric>_adj  for each metric column found in the DataFrame
        """
        df = df.copy()
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")

        mt_list = df["mt"].tolist()
        ref_list = df["ref"].tolist()

        # ── Stage 1: Rule-based ───────────────────────────────
        logger.info("Stage 1: applying rule-based checks …")
        d_rules = apply_rule_batch(mt_list, ref_list, self.antonym_dict)

        # ── Stage 2: Contradiction detection ─────────────────
        logger.info("Stage 2: running contradiction detection …")
        d_contras = self.contra_detector.predict(mt_list, ref_list)

        # ── Stage 3: Paraphrase detection ────────────────────
        logger.info("Stage 3: running paraphrase detection …")
        d_paras = self.para_detector.predict(mt_list, ref_list)

        # ── Combine signals ───────────────────────────────────
        d_finals = [
            select_decision(dr, dc, dp)
            for dr, dc, dp in zip(d_rules, d_contras, d_paras)
        ]

        df["d_rule"] = d_rules
        df["d_contra"] = [round(v, 4) for v in d_contras]
        df["d_para"] = [round(v, 4) for v in d_paras]
        df["d_score"] = [round(v, 4) for v in d_finals]

        # ── Score adjustment ──────────────────────────────────
        present_metrics = [m for m in METRICS if m in df.columns]
        if present_metrics:
            logger.info("Adjusting scores for: %s", present_metrics)
            raw_scores = {m: df[m].tolist() for m in present_metrics}
            adj_scores = adjust_scores_batch(raw_scores, d_finals, self.tau)
            for m, vals in adj_scores.items():
                df[f"{m}_adj"] = [round(v, 4) for v in vals]
        else:
            logger.warning(
                "No metric columns (%s) found in DataFrame; skipping score adjustment.",
                METRICS,
            )

        return df

    # ──────────────────────────────────────────────────────────
    #  Single-sample convenience method
    # ──────────────────────────────────────────────────────────

    def adjust_single(
        self,
        mt: str,
        ref: str,
        metric_scores: dict[str, float],
    ) -> dict[str, float]:
        """
        Adjust metric scores for a single (mt, ref) pair.

        Parameters
        ----------
        mt, ref        : Hindi strings
        metric_scores  : dict  metric_name → original score

        Returns
        -------
        dict  metric_name → adjusted score
        """
        _, _, _, d_final = self.predict_decision(mt, ref)
        return {
            m: round(
                __import__("src.score_adjustment", fromlist=["adjust_score"])
                .adjust_score(s, d_final, self.tau),
                4,
            )
            for m, s in metric_scores.items()
        }
