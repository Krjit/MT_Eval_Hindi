"""
src/score_adjustment.py
────────────────────────
Score Adjustment Layer.

Given a metric score s ∈ [0,1] and a decision signal d ∈ [0,1]:
    s' = τ + (τ − s)·d        if d ≥ 0.5 AND s < τ   (reward under-scored paraphrase)
       = τ − (s − τ)·(1 − d)  if d < 0.5 AND s > τ   (penalise over-scored contradiction)
       = s                     otherwise               (no change)
    s' = clip(s', 0, 1)

τ = 0.6 (default).
"""

from __future__ import annotations

METRICS = ["COMET", "BERTScore", "BLEURT", "BLEU", "ChrF", "ChrF++"]


def adjust_score(s: float, d: float, tau: float = 0.6) -> float:
    """
    Adjust a single metric score using the hybrid decision signal.
    Parameters
    ----------
    s   : float  — original metric score ∈ [0, 1]
    d   : float  — decision score ∈ [0, 1];
                    d ≥ 0.5  → evidence of paraphrase (reward)
                    d <  0.5 → evidence of contradiction (penalty)
    tau : float  — threshold for "good" score (default 0.6)

    Returns
    -------
    float — adjusted score clipped to [0, 1]
    """
    if d >= 0.5 and s < tau:
        # Under-scored paraphrase: lift toward / beyond tau
        s_prime = tau + (tau - s) * d
    elif d < 0.5 and s > tau:
        # Over-scored contradiction: pull down toward / below tau
        s_prime = tau - (s - tau) * (1.0 - d)
    else:
        s_prime = s

    return max(0.0, min(1.0, s_prime))


def adjust_scores_batch(
    scores: dict[str, list[float]],
    decisions: list[float],
    tau: float = 0.6,
) -> dict[str, list[float]]:
    """
    Apply score adjustment to multiple metrics at once.

    Parameters
    ----------
    scores    : dict  metric_name → list of scores (N,)
    decisions : list  decision scores (N,)
    tau       : float threshold

    Returns
    -------
    dict  metric_name → list of adjusted scores (N,)
    """
    adjusted: dict[str, list[float]] = {}
    for metric, metric_scores in scores.items():
        adjusted[metric] = [
            adjust_score(s, d, tau)
            for s, d in zip(metric_scores, decisions)
        ]
    return adjusted


def select_decision(
    d_rule: int | None,
    d_contra: float,
    d_para: float,
) -> float:
    """
    Cascade decision selection logic.

    Priority:
      1. Rule-based (if not None)  → d ∈ {0.0, 1.0}
      2. Contradiction detection   → d = 1 − p_contra  (low = contradiction)
      3. Paraphrase detection      → d = p_para         (high = paraphrase)

    The caller is responsible for running only the stages it needs,
    but this function implements the fall-through logic cleanly.
    """
    if d_rule is not None:
        return float(d_rule)
    # Use contradiction signal as primary neural signal
    # d_contra is already expressed as "non-contradiction probability"
    # blend with paraphrase if available
    if d_rule = None and d_contra > 0.5:
        return d_para
    
    return d_contra
