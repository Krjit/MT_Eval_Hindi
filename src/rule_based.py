"""
src/rule_based.py
─────────────────
Rule-based negation & antonym detector for Hindi MT evaluation.

Algorithm (Negation and Antonym Penalty-Reward):
  - If negation counts differ between mt and ref → penalty (d=0)
  - If an antonym of a ref word appears in mt:
      • If one sentence has the antonym adjacent to a negation → reward (d=1)
      • If both sentences contain a cross-antonym pair             → reward (d=1)
      • Otherwise                                                  → penalty (d=0)
  - If no negation/antonym trigger is found → None (pass to next stage)
"""

from __future__ import annotations

import logging
from typing import Optional

import nltk
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
#  Hindi negation vocabulary
# ─────────────────────────────────────────────────────────────
NEGATION_WORDS: set[str] = {
    "नहीं", "न", "मत", "ना", "नहि", "बिना",
    "अभी नहीं", "कभी नहीं", "कहीं नहीं",
    "कोई नहीं", "कुछ नहीं", "न जाने", "हरगिज़ नहीं",
    "अभी तक नहीं", "बिलकुल नहीं", "कभी भी नहीं",
    "नकार", "इन्कार", "अस्वीकृत", "अस्वीकार",
    "वर्जित", "रद्द", "नहीं किया", "नहीं होगा",
    "नहीं था", "नहीं है", "ना ही",
}

# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase word-tokenise a Hindi sentence."""
    try:
        return nltk.word_tokenize(text.lower())
    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        return nltk.word_tokenize(text.lower())


def _lemmatize(word: str) -> str:
    """
    Attempt lemmatisation via IndicNLP; fall back to the original form.
    IndicNLP is optional — if not installed the word is returned unchanged.
    """
    try:
        from indicnlp.morph import unsupervised_morph  # type: ignore
        analyzer = unsupervised_morph.UnsupervisedMorphAnalyzer("hi")
        forms = analyzer.morph_analyze_word(word)
        return forms[0] if forms else word
    except Exception:
        return word


def _count_negations(tokens: list[str]) -> int:
    """Count how many tokens belong to the Hindi negation vocabulary."""
    return sum(1 for t in tokens if t in NEGATION_WORDS)


def _has_neg_adjacent(
    word: str,
    antonym: str,
    mt_tokens: list[str],
    ref_tokens: list[str],
    window: int = 2,
) -> bool:
    """
    Return True iff *exactly one* of (mt, ref) has a negation word
    within `window` tokens of `word` or `antonym`.
    Implements HasNegAdjacent from the paper.
    """

    def adjacent_neg(target: str, tokens: list[str]) -> bool:
        if target not in tokens:
            return False
        idx = tokens.index(target)
        context = tokens[max(idx - window, 0): idx] + tokens[idx + 1: idx + window + 1]
        return any(t in NEGATION_WORDS for t in context)

    mt_has = adjacent_neg(antonym, mt_tokens) or adjacent_neg(word, mt_tokens)
    ref_has = adjacent_neg(word, ref_tokens) or adjacent_neg(antonym, ref_tokens)
    return mt_has != ref_has  # True only when exactly one side is negated


def _has_cross_antonyms(word: str, mt_tokens: list[str], ref_tokens: list[str],
                         hindi_antonyms: dict[str, list[str]]) -> bool:
    """
    Return True iff both mt and ref contain a known antonym pair for `word`.
    Implements HasCrossAntonyms from the paper.
    """
    antonyms = hindi_antonyms.get(word, [])
    for antonym in antonyms:
        if antonym in mt_tokens and antonym in ref_tokens:
            reverse_antonyms = hindi_antonyms.get(antonym, [])
            if any(ra in ref_tokens for ra in reverse_antonyms):
                return True
    return False


# ─────────────────────────────────────────────────────────────
#  Antonym dictionary loader
# ─────────────────────────────────────────────────────────────

def load_antonym_dict(csv_path: str) -> dict[str, list[str]]:
    """
    Load a CSV with columns [word_hi, antonym_hi] and return
    a bidirectional dictionary: word → [antonyms].
    """
    df = pd.read_csv(csv_path)
    antonyms: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        w = str(row["word_hi"]).strip()
        a = str(row["antonym_hi"]).strip()
        antonyms.setdefault(w, []).append(a)
        antonyms.setdefault(a, []).append(w)
    logger.info("Loaded %d antonym entries from %s", len(antonyms), csv_path)
    return antonyms


# ─────────────────────────────────────────────────────────────
#  Core rule-based detector
# ─────────────────────────────────────────────────────────────

def compute_rule_decision(
    mt: str,
    ref: str,
    hindi_antonyms: dict[str, list[str]],
) -> Optional[int]:
    """
    Negation-and-Antonym Penalty-Reward algorithm.

    Parameters
    ----------
    mt : str
        Machine-translated text (Hindi).
    ref : str
        Reference translation (Hindi).
    hindi_antonyms : dict
        Bidirectional antonym dictionary {word: [antonym, ...]}.

    Returns
    -------
    int | None
        0   → penalty  (translation is semantically wrong)
        1   → reward   (translation is semantically equivalent via negation-antonym)
        None → no trigger found; pass to next pipeline stage
    """
    mt_tokens = _tokenize(mt)
    ref_tokens = _tokenize(ref)

    d_rule: Optional[int] = None

    # ── Step 1: compare negation counts ──────────────────────
    n_mt = _count_negations(mt_tokens)
    n_ref = _count_negations(ref_tokens)

    if n_mt != n_ref:
        d_rule = 0  # polarity mismatch → penalty

    # ── Step 2: check antonym substitutions ──────────────────
    for word, antonyms in hindi_antonyms.items():
        word_lem = _lemmatize(word)
        for antonym in antonyms:
            antonym_lem = _lemmatize(antonym)

            word_in_ref = (word in ref_tokens) or (word_lem in ref_tokens)
            anto_in_mt = (antonym in mt_tokens) or (antonym_lem in mt_tokens)

            if word_in_ref and anto_in_mt:
                if n_mt == n_ref:
                    # Antonym used without compensating negation → penalty
                    d_rule = 0
                elif _has_neg_adjacent(word, antonym, mt_tokens, ref_tokens):
                    # One side has negation near the antonym → semantic equivalence
                    d_rule = 1
                if _has_cross_antonyms(word, mt_tokens, ref_tokens, hindi_antonyms):
                    # Both sentences contain a known reciprocal antonym pair
                    d_rule = 1

    return d_rule


# ─────────────────────────────────────────────────────────────
#  Batch-level helper (used by framework.py)
# ─────────────────────────────────────────────────────────────

def apply_rule_batch(
    mt_list: list[str],
    ref_list: list[str],
    hindi_antonyms: dict[str, list[str]],
) -> list[Optional[int]]:
    """
    Apply compute_rule_decision over parallel lists of mt and ref strings.

    Returns
    -------
    list of Optional[int]  — same length as input lists
    """
    return [
        compute_rule_decision(mt, ref, hindi_antonyms)
        for mt, ref in zip(mt_list, ref_list)
    ]
