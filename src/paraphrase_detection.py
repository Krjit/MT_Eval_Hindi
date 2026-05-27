"""
src/paraphrase_detection.py
────────────────────────────
Paraphrase detection using a Siamese Sentence-BERT + CNN encoder.

Architecture:
  • Dual SBERT (paraphrase-multilingual-MiniLM-L12-v2) encoders share weights
  • CNNEncoder: Conv2D with filters of sizes 3, 4, 5 → max-over-time pooling
  • Absolute embedding difference + cosine similarity
  • Augmented with 4 shallow lexical features (overlap, edit-dist, neg-diff, antonym ratio)
  • Dense head: 300+1+4 → 128 → 1 → sigmoid (paraphrase probability)
"""

from __future__ import annotations

import logging
from typing import Optional

import editdistance
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────
SBERT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MAX_LEN = 64

HINDI_NEG_WORDS: set[str] = {
    "नहीं", "ना", "मत", "न", "अभी तक नहीं", "कभी नहीं",
    "बिलकुल नहीं", "कभी भी नहीं", "इन्कार", "वर्जित",
    "रद्द", "बिना", "नहीं किया", "नहीं होगा", "नहीं था",
    "नहीं है", "ना ही",
}


# ─────────────────────────────────────────────────────────────
#  Shallow features
# ─────────────────────────────────────────────────────────────

def extract_shallow_features(
    s1_tokens: list[str],
    s2_tokens: list[str],
    antonym_dict: dict[str, list[str]],
) -> list[float]:
    """
    Compute 4 lexical shallow features for a sentence pair.

    1. Word-overlap ratio
    2. Levenshtein distance ratio (character level)
    3. Negation mismatch flag (0/1)
    4. Antonym co-occurrence ratio
    """
    # 1) Word-overlap ratio
    overlap = len(set(s1_tokens) & set(s2_tokens)) / max(len(s1_tokens), len(s2_tokens), 1)

    # 2) Character-level Levenshtein similarity
    str1, str2 = " ".join(s1_tokens), " ".join(s2_tokens)
    ed = editdistance.eval(str1, str2)
    ed_score = 1.0 - ed / max(len(str1), len(str2), 1)

    # 3) Negation mismatch
    neg1 = any(w in HINDI_NEG_WORDS for w in s1_tokens)
    neg2 = any(w in HINDI_NEG_WORDS for w in s2_tokens)
    neg_diff = float(neg1 != neg2)

    # 4) Antonym co-occurrence ratio
    ant_pairs = 0
    for w in s1_tokens:
        for ant in antonym_dict.get(w, []):
            if ant in s2_tokens:
                ant_pairs += 1
                break
    ant_ratio = ant_pairs / max(len(s1_tokens), 1)

    return [overlap, ed_score, neg_diff, ant_ratio]


# ─────────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────────

class ParaDataset(Dataset):
    """
    PyTorch Dataset for the paraphrase detection model.

    Each sample provides tokenised SBERT inputs for sentence 1 and 2,
    4 shallow features, and a binary label (1 = paraphrase, 0 = non-paraphrase).
    """

    def __init__(
        self,
        pairs: list[tuple[str, str]],
        labels: list[int],
        tokenizer: AutoTokenizer,
        antonym_dict: dict[str, list[str]],
        max_len: int = MAX_LEN,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.antonym_dict = antonym_dict
        self.pairs = pairs
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        s1, s2 = self.pairs[idx]
        label = self.labels[idx]

        enc1 = self.tokenizer(
            s1.strip(), padding="max_length", truncation=True,
            max_length=self.max_len, return_tensors="pt",
        )
        enc2 = self.tokenizer(
            s2.strip(), padding="max_length", truncation=True,
            max_length=self.max_len, return_tensors="pt",
        )
        # Squeeze the leading batch dimension added by return_tensors="pt"
        enc1 = {k: v.squeeze(0) for k, v in enc1.items()}
        enc2 = {k: v.squeeze(0) for k, v in enc2.items()}

        shallow = extract_shallow_features(s1.split(), s2.split(), self.antonym_dict)
        shallow_t = torch.tensor(shallow, dtype=torch.float)

        return enc1, enc2, shallow_t, torch.tensor(label, dtype=torch.float)


# ─────────────────────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────────────────────

class CNNEncoder(nn.Module):
    """
    1-D CNN encoder operating on SBERT last-hidden-state embeddings.
    Applies convolutions with kernel sizes 3, 4, 5 and max-over-time pooling.
    Output dim = num_filters × len(kernel_sizes).
    """

    def __init__(
        self,
        embed_dim: int,
        num_filters: int = 100,
        kernel_sizes: list[int] = (3, 4, 5),
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_dim)) for k in kernel_sizes
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, D)  SBERT last-hidden-state

        Returns
        -------
        (B, num_filters * len(kernel_sizes))
        """
        x = x.unsqueeze(1)                          # (B, 1, T, D)
        pooled = []
        for conv in self.convs:
            h = F.relu(conv(x)).squeeze(3)           # (B, F, T-k+1)
            h = F.max_pool1d(h, h.size(2)).squeeze(2)  # (B, F)
            pooled.append(h)
        return torch.cat(pooled, dim=1)             # (B, F*3)


class SiameseCNN(nn.Module):
    """
    Siamese SBERT + CNN paraphrase detection model.

    input_dim  = num_filters × 3 (CNN output)
    classifier = Linear(input_dim + 1 cosine + 4 shallow, 512) → 128 → 1
    """

    def __init__(
        self,
        bert: AutoModel,
        embed_dim: int,
        num_filters: int = 100,
        kernel_sizes: tuple[int, ...] = (3, 4, 5),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.bert = bert
        self.encoder = CNNEncoder(embed_dim, num_filters, list(kernel_sizes))
        cnn_out = num_filters * len(kernel_sizes)   # 300 by default

        self.classifier = nn.Sequential(
            nn.Linear(cnn_out + 1 + 4, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def _encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.bert(**batch).last_hidden_state  # (B, T, D)

    def forward(
        self,
        s1_batch: dict[str, torch.Tensor],
        s2_batch: dict[str, torch.Tensor],
        shallow_feats: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns raw logits (B,). Apply sigmoid for probabilities.
        """
        v1 = self.encoder(self._encode(s1_batch))   # (B, cnn_out)
        v2 = self.encoder(self._encode(s2_batch))

        abs_diff = torch.abs(v1 - v2)               # (B, cnn_out)
        cos_sim = F.cosine_similarity(v1, v2).unsqueeze(1)  # (B, 1)

        combined = torch.cat([abs_diff, cos_sim, shallow_feats], dim=1)
        return self.classifier(combined).squeeze(1)  # (B,)


# ─────────────────────────────────────────────────────────────
#  Inference helper
# ─────────────────────────────────────────────────────────────

class ParaDetector:
    """
    Lightweight wrapper for batch inference with the trained SiameseCNN.

    Returns paraphrase probability (d_para) for each (mt, ref) pair.
    High d_para → likely paraphrase → reward during score adjustment.
    """

    def __init__(
        self,
        model_path: str,
        antonym_dict: dict[str, list[str]],
        sbert_model: str = SBERT_MODEL,
        device: Optional[str] = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.antonym_dict = antonym_dict

        self.tokenizer = AutoTokenizer.from_pretrained(sbert_model)
        bert = AutoModel.from_pretrained(sbert_model)
        embed_dim = bert.config.hidden_size

        self.model = SiameseCNN(bert, embed_dim).to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.eval()
        logger.info("Loaded SiameseCNN from %s", model_path)

    @torch.no_grad()
    def predict(self, mt_list: list[str], ref_list: list[str]) -> list[float]:
        """
        Return paraphrase probability in [0,1] for each pair.
        """
        probs = []
        for mt, ref in zip(mt_list, ref_list):
            enc1 = self.tokenizer(
                mt.strip(), padding="max_length", truncation=True,
                max_length=MAX_LEN, return_tensors="pt",
            )
            enc2 = self.tokenizer(
                ref.strip(), padding="max_length", truncation=True,
                max_length=MAX_LEN, return_tensors="pt",
            )
            s1b = {k: v.to(self.device) for k, v in enc1.items()}
            s2b = {k: v.to(self.device) for k, v in enc2.items()}

            shallow = extract_shallow_features(mt.split(), ref.split(), self.antonym_dict)
            sf = torch.tensor([shallow], dtype=torch.float, device=self.device)

            logit = self.model(s1b, s2b, sf)
            prob = torch.sigmoid(logit).item()
            probs.append(prob)

        return probs
