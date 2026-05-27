"""
src/contradiction_detection.py
────────────────────────────────
Contradiction detection using Contradiction-specific Word Embeddings (CWE) and CNN-based classifier.

Architecture:
  • Embedding layer initialised from IndicBERT-derived CWE vectors
  • Two Conv1d branches: one for full sentences, one for unaligned phrase tokens
  • Sentence representations added; phrase representations added
  • Concatenated with 3 shallow features → Linear(1003, 2) → softmax
"""

from __future__ import annotations

import logging
import pickle
from typing import Optional

import nltk
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
#  Hindi negation tokens (used for shallow features)
# ─────────────────────────────────────────────────────────────
NEG_TOKS: set[str] = {
    "नहीं", "ना", "मत", "न",
    "अभी तक नहीं", "कभी नहीं", "बिलकुल नहीं", "कभी भी नहीं",
    "नकार", "इन्कार", "अस्वीकृत", "अस्वीकार",
    "वर्जित", "रद्द", "बिना",
    "नहीं किया", "नहीं होगा", "नहीं था", "नहीं है", "ना ही",
}

MAX_LEN = 70  # fixed sequence length for CWE encoding


# ─────────────────────────────────────────────────────────────
#  Vocabulary / encoding utilities
# ─────────────────────────────────────────────────────────────

def load_vocab(pkl_path: str) -> tuple[dict[str, int], int, int]:
    """
    Load the CWE word-to-index mapping from a pickle file.

    Returns
    -------
    word2idx, pad_idx, unk_idx
    """
    with open(pkl_path, "rb") as f:
        word2idx: dict[str, int] = pickle.load(f)
    pad_idx = word2idx["<PAD>"]
    unk_idx = word2idx["<UNK>"]
    logger.info("Loaded vocab with %d entries from %s", len(word2idx), pkl_path)
    return word2idx, pad_idx, unk_idx


def _tokenize(sentence: str) -> list[str]:
    try:
        return nltk.word_tokenize(sentence.lower())
    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        return nltk.word_tokenize(sentence.lower())


def encode_sentence(
    tokens: list[str],
    word2idx: dict[str, int],
    pad_idx: int,
    unk_idx: int,
    max_len: int = MAX_LEN,
) -> list[int]:
    """Convert tokens → fixed-length index list (pad / truncate to max_len)."""
    idxs = [word2idx.get(t, unk_idx) for t in tokens]
    if len(idxs) >= max_len:
        return idxs[:max_len]
    return idxs + [pad_idx] * (max_len - len(idxs))


def get_overlapping_and_unaligned(
    tokens1: list[str], tokens2: list[str]
) -> tuple[list[int], list[int], list[str], list[str]]:
    """
    Identify overlapping tokens between two sentences and return
    their positions plus the unaligned remainder.

    Returns
    -------
    overlap_idxs1, overlap_idxs2, unaligned_tokens1, unaligned_tokens2
    """
    s1, s2 = set(tokens1), set(tokens2)
    overlap = s1 & s2
    oi1 = [i for i, t in enumerate(tokens1) if t in overlap]
    oi2 = [i for i, t in enumerate(tokens2) if t in overlap]
    un1 = [t for t in tokens1 if t not in overlap]
    un2 = [t for t in tokens2 if t not in overlap]
    return oi1, oi2, un1, un2


def compute_shallow_features(
    tokens1: list[str], tokens2: list[str]
) -> list[float]:
    """
    Compute 3 shallow features used by the CWE classifier.

    Returns [neg_parity, word_order_diff, unaligned_count]
    """
    # 1) Negation parity feature (1.0 if odd combined negation count)
    neg_count = sum(t in NEG_TOKS for t in tokens1) + sum(t in NEG_TOKS for t in tokens2)
    f_neg = float(neg_count % 2)

    # 2) Average positional displacement of overlapping tokens
    _, _, un1, un2 = get_overlapping_and_unaligned(tokens1, tokens2)
    oi1 = [i for i, t in enumerate(tokens1) if t in (set(tokens1) & set(tokens2))]
    oi2 = [i for i, t in enumerate(tokens2) if t in (set(tokens1) & set(tokens2))]
    if oi1 and oi2 and len(oi1) == len(oi2):
        f_wod = float(np.mean([abs(a - b) for a, b in zip(oi1, oi2)]))
    else:
        f_wod = 0.0

    # 3) Mean unaligned word count
    f_unaligned = float((len(un1) + len(un2)) / 2.0)

    return [f_neg, f_wod, f_unaligned]


# ─────────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────────

class ContraDataset(Dataset):
    """
    PyTorch Dataset for the contradiction detection model.

    Each sample provides:
        s1, s2       — encoded sentence index sequences  (MAX_LEN,)
        u1, u2       — encoded unaligned phrase sequences (MAX_LEN,)
        shallow_feats — 3 shallow features
        label        — 0 = non-contradiction (P), 1 = contradiction (NP)
    """

    def __init__(
        self,
        pairs: list[tuple[str, str]],
        labels: list[int],
        word2idx: dict[str, int],
        pad_idx: int,
        unk_idx: int,
        max_len: int = MAX_LEN,
    ) -> None:
        self.max_len = max_len
        s1_idxs, s2_idxs, u1_idxs, u2_idxs, shallow, lbls = [], [], [], [], [], []

        for (sent1, sent2), label in zip(pairs, labels):
            t1 = _tokenize(sent1)
            t2 = _tokenize(sent2)
            _, _, un1, un2 = get_overlapping_and_unaligned(t1, t2)

            s1_idxs.append(encode_sentence(t1, word2idx, pad_idx, unk_idx, max_len))
            s2_idxs.append(encode_sentence(t2, word2idx, pad_idx, unk_idx, max_len))
            u1_idxs.append(encode_sentence(un1, word2idx, pad_idx, unk_idx, max_len))
            u2_idxs.append(encode_sentence(un2, word2idx, pad_idx, unk_idx, max_len))
            shallow.append(compute_shallow_features(t1, t2))
            lbls.append(label)

        self.s1 = torch.tensor(s1_idxs, dtype=torch.long)
        self.s2 = torch.tensor(s2_idxs, dtype=torch.long)
        self.u1 = torch.tensor(u1_idxs, dtype=torch.long)
        self.u2 = torch.tensor(u2_idxs, dtype=torch.long)
        self.sf = torch.tensor(shallow, dtype=torch.float)
        self.lbl = torch.tensor(lbls, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.lbl)

    def __getitem__(self, idx: int):
        return self.s1[idx], self.s2[idx], self.u1[idx], self.u2[idx], self.sf[idx], self.lbl[idx]


# ─────────────────────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────────────────────

class ContraDetectCNN(nn.Module):
    """
    CWE-based CNN for contradiction detection.

    Architecture:
      • Embedding(vocab, 300) initialised from CWE vectors
      • Conv1d(300 → conv_out, kernel=window_size=70): sentence branch
      • Conv1d(300 → conv_out, kernel=window_size=70): phrase branch
      • sent_rel  = tanh(conv(e1)) + tanh(conv(e2))      (conv_out,)
      • phrase_rel = tanh(conv_un(eu1)) + tanh(conv_un(eu2)) (conv_out,)
      • concat [sent_rel ; phrase_rel ; shallow_feats]   (2*conv_out+3,)
      • Linear(2*conv_out+3, 2) → softmax
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 300,
        conv_out: int = 500,
        window_size: int = MAX_LEN,
        init_emb: Optional[torch.Tensor] = None,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        if init_emb is not None:
            self.embed.weight.data.copy_(init_emb)

        # Sentence-level and phrase-level Conv1d branches
        self.conv = nn.Conv1d(emb_dim, conv_out, kernel_size=window_size, bias=True)
        self.conv_un = nn.Conv1d(emb_dim, conv_out, kernel_size=window_size, bias=True)

        # Final feed-forward classifier head (hidden layers 256 → 128 → 2)
        self.ffn = nn.Sequential(
            nn.Linear(conv_out * 2 + 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(
        self,
        s1_idxs: torch.Tensor,
        s2_idxs: torch.Tensor,
        un1_idxs: torch.Tensor,
        un2_idxs: torch.Tensor,
        shallow_feats: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        s1_idxs, s2_idxs   : (B, 70) long
        un1_idxs, un2_idxs : (B, 70) long
        shallow_feats       : (B, 3)  float

        Returns
        -------
        logits : (B, 2)
        """
        # Sentence branch
        e1 = self.embed(s1_idxs).transpose(1, 2)   # (B, emb_dim, 70)
        e2 = self.embed(s2_idxs).transpose(1, 2)
        c1 = torch.tanh(self.conv(e1).squeeze(2))   # (B, conv_out)
        c2 = torch.tanh(self.conv(e2).squeeze(2))
        sent_rel = c1 + c2                           # (B, conv_out)

        # Phrase branch
        eu1 = self.embed(un1_idxs).transpose(1, 2)
        eu2 = self.embed(un2_idxs).transpose(1, 2)
        cu1 = torch.tanh(self.conv_un(eu1).squeeze(2))
        cu2 = torch.tanh(self.conv_un(eu2).squeeze(2))
        phrase_rel = cu1 + cu2                       # (B, conv_out)

        # Concat and classify
        combined = torch.cat([sent_rel, phrase_rel, shallow_feats], dim=1)
        return self.ffn(combined)                    # (B, 2)


# ─────────────────────────────────────────────────────────────
#  Inference helper
# ─────────────────────────────────────────────────────────────

class ContraDetector:
    """
    Lightweight wrapper for batch inference.

    Loads the trained ContraDetectCNN and returns contradiction
    probability (class 1 probability) for each (mt, ref) pair.
    """

    def __init__(
        self,
        model_path: str,
        word2idx_path: str,
        cwe_emb_path: str,
        device: Optional[str] = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.word2idx, self.pad_idx, self.unk_idx = load_vocab(word2idx_path)

        init_emb = torch.load(cwe_emb_path, map_location="cpu")
        vocab_size = init_emb.size(0)

        self.model = ContraDetectCNN(
            vocab_size=vocab_size,
            init_emb=init_emb,
            padding_idx=self.pad_idx,
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        logger.info("Loaded ContraDetectCNN from %s", model_path)

    @torch.no_grad()
    def predict(self, mt_list: list[str], ref_list: list[str]) -> list[float]:
        """
        Return contradiction probability in [0,1] for each pair.
        High value → likely contradiction → penalty during score adjustment.
        """
        probs = []
        for mt, ref in zip(mt_list, ref_list):
            t1, t2 = _tokenize(mt), _tokenize(ref)
            _, _, un1, un2 = get_overlapping_and_unaligned(t1, t2)

            s1 = torch.tensor([encode_sentence(t1, self.word2idx, self.pad_idx, self.unk_idx)],
                               dtype=torch.long, device=self.device)
            s2 = torch.tensor([encode_sentence(t2, self.word2idx, self.pad_idx, self.unk_idx)],
                               dtype=torch.long, device=self.device)
            u1 = torch.tensor([encode_sentence(un1, self.word2idx, self.pad_idx, self.unk_idx)],
                               dtype=torch.long, device=self.device)
            u2 = torch.tensor([encode_sentence(un2, self.word2idx, self.pad_idx, self.unk_idx)],
                               dtype=torch.long, device=self.device)
            sf = torch.tensor([compute_shallow_features(t1, t2)],
                               dtype=torch.float, device=self.device)

            logits = self.model(s1, s2, u1, u2, sf)          # (1, 2)
            prob_contra = torch.softmax(logits, dim=-1)[0, 1].item()
            # Convert to decision score: low value → contradiction → penalty
            probs.append(1.0 - prob_contra)  # d_contra: high → non-contradiction

        return probs
