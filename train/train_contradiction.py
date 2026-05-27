"""
train/train_contradiction.py
─────────────────────────────
Train the CWE-based ContraDetectCNN model.

Usage:
    python train/train_contradiction.py \
        --train_csv data/Trainset_All_Mixed_Hindi_Balanced.csv \
        --trial_csv data/Trialset_All_Mixed_Hindi_Balanced.csv \
        --word2idx  models/hindi_cwe_word2idx300d.pkl \
        --cwe_emb   models/hindi_cwe_finetuned_emb_300d.pt \
        --output    models/contra_detect_best.pt \
        --epochs 20 --batch_size 128 --lr 1e-4
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.contradiction_detection import (
    ContraDataset,
    ContraDetectCNN,
    load_vocab,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────

def load_pairs_labels(csv_path: str, is_test: bool = False):
    """
    Load (ref, mt) pairs and binary labels from a CSV.

    Columns expected: src, ref, mt, model, category, label
    label: 'P' → 0 (non-contradiction), 'NP' → 1 (contradiction)
    """
    df = pd.read_csv(csv_path)
    pairs, labels = [], []
    for _, row in df.iterrows():
        ref = str(row["ref"]).strip()
        mt = str(row["mt"]).strip()
        lbl = 0 if str(row["label"]).strip() == "P" else 1
        pairs.append((ref, mt))
        labels.append(lbl)
    return pairs, labels


# ─────────────────────────────────────────────────────────────
#  Training helpers
# ─────────────────────────────────────────────────────────────

def evaluate_accuracy(loader: DataLoader, model: nn.Module, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for s1, s2, u1, u2, sf, lbl in loader:
            s1, s2, u1, u2 = s1.to(device), s2.to(device), u1.to(device), u2.to(device)
            sf, lbl = sf.to(device), lbl.to(device)
            preds = model(s1, s2, u1, u2, sf).argmax(dim=1)
            correct += (preds == lbl).sum().item()
            total += lbl.size(0)
    return correct / total if total else 0.0


def evaluate_full(loader: DataLoader, model: nn.Module, device: torch.device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for s1, s2, u1, u2, sf, lbl in loader:
            s1, s2, u1, u2 = s1.to(device), s2.to(device), u1.to(device), u2.to(device)
            sf, lbl = sf.to(device), lbl.to(device)
            logits = model(s1, s2, u1, u2, sf)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbl.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return all_labels, all_preds, all_probs


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ── Vocab & embeddings ────────────────────────────────────
    word2idx, pad_idx, unk_idx = load_vocab(args.word2idx)
    init_emb = torch.load(args.cwe_emb, map_location="cpu")
    vocab_size = init_emb.size(0)

    # ── Data ──────────────────────────────────────────────────
    train_pairs, train_labels = load_pairs_labels(args.train_csv)
    trial_pairs, trial_labels = load_pairs_labels(args.trial_csv)

    logger.info("Train: %d | Trial: %d", len(train_pairs), len(trial_pairs))

    train_ds = ContraDataset(train_pairs, train_labels, word2idx, pad_idx, unk_idx)
    trial_ds = ContraDataset(trial_pairs, trial_labels, word2idx, pad_idx, unk_idx)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)
    trial_dl = DataLoader(trial_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # ── Model ─────────────────────────────────────────────────
    model = ContraDetectCNN(
        vocab_size=vocab_size,
        emb_dim=300,
        conv_out=500,
        init_emb=init_emb,
        padding_idx=pad_idx,
    ).to(device)

    # Class-balanced loss
    from collections import Counter
    counts = Counter(train_labels)
    weights = torch.tensor(
        [len(train_labels) / counts[i] for i in range(2)], dtype=torch.float
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ── Training loop ─────────────────────────────────────────
    best_acc = 0.0
    patience_counter = 0
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}"):
            s1, s2, u1, u2, sf, lbl = batch
            s1, s2, u1, u2 = s1.to(device), s2.to(device), u1.to(device), u2.to(device)
            sf, lbl = sf.to(device), lbl.to(device)

            optimizer.zero_grad()
            logits = model(s1, s2, u1, u2, sf)
            loss = criterion(logits, lbl)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dl)
        trial_acc = evaluate_accuracy(trial_dl, model, device)
        logger.info("Epoch %d | Loss=%.4f | Trial Acc=%.4f", epoch, avg_loss, trial_acc)

        if trial_acc > best_acc:
            best_acc = trial_acc
            patience_counter = 0
            torch.save(model.state_dict(), args.output)
            logger.info("  ✓ New best (%.4f) — saved to %s", best_acc, args.output)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("Early stopping after %d epochs without improvement.", epoch)
                break

    # ── Final test evaluation ─────────────────────────────────
    if args.test_csv:
        test_pairs, test_labels = load_pairs_labels(args.test_csv, is_test=True)
        test_ds = ContraDataset(test_pairs, test_labels, word2idx, pad_idx, unk_idx)
        test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        model.load_state_dict(torch.load(args.output, map_location=device))
        y_true, y_pred, _ = evaluate_full(test_dl, model, device)
        print("\n=== Test Set Classification Report ===")
        print(classification_report(y_true, y_pred, target_names=["P", "NP"], digits=4))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train contradiction detection model")
    p.add_argument("--train_csv", required=True)
    p.add_argument("--trial_csv", required=True)
    p.add_argument("--test_csv", default=None)
    p.add_argument("--word2idx", required=True, help="CWE vocab pkl path")
    p.add_argument("--cwe_emb", required=True, help="CWE embedding .pt path")
    p.add_argument("--output", default="models/contra_detect_best.pt")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=3)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
