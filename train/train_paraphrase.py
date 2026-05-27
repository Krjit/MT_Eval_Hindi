"""
train/train_paraphrase.py
──────────────────────────
Train the Siamese SBERT + CNN paraphrase detection model.

Training strategy:
  - Phase 1 (epochs 1 → bert_unfreeze_epoch): BERT frozen; only CNN + classifier trained
  - Phase 2 (remaining epochs):               BERT unfrozen with very small lr

Usage:
    python train/train_paraphrase.py \
        --train_csv data/Trainset_All_Mixed_Hindi_Balanced.csv \
        --trial_csv data/Trialset_All_Mixed_Hindi_Balanced.csv \
        --test_csv  data/Testset_All_Mixed_Hindi_Balanced.csv \
        --output    models/para_detect_best.pth \
        --epochs 15 --batch_size 64 --lr 5e-5
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
from transformers import AutoModel, AutoTokenizer

from src.paraphrase_detection import ParaDataset, SiameseCNN
from src.rule_based import load_antonym_dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────

def load_train_pairs(csv_path: str) -> tuple[list[tuple[str, str]], list[int]]:
    """Train CSV format: ref, mt, label  (no header or with header)."""
    df = pd.read_csv(csv_path)
    pairs, labels = [], []
    for _, row in df.iterrows():
        s1, s2 = str(row.iloc[0]).strip(), str(row.iloc[1]).strip()
        label_str = str(row.iloc[2]).strip()
        pairs.append((s1, s2))
        labels.append(1 if label_str == "P" else 0)
    return pairs, labels


def load_test_pairs(csv_path: str) -> tuple[list[tuple[str, str]], list[int]]:
    """Test CSV format: src, ref, mt, model, category, label."""
    df = pd.read_csv(csv_path)
    pairs, labels = [], []
    for _, row in df.iterrows():
        ref, mt = str(row["ref"]).strip(), str(row["mt"]).strip()
        pairs.append((ref, mt))
        labels.append(1 if str(row["label"]).strip() == "P" else 0)
    return pairs, labels


# ─────────────────────────────────────────────────────────────
#  Training helpers
# ─────────────────────────────────────────────────────────────

def evaluate(loader: DataLoader, model: nn.Module, device: torch.device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for s1b, s2b, sf, lbls in loader:
            s1b = {k: v.to(device) for k, v in s1b.items()}
            s2b = {k: v.to(device) for k, v in s2b.items()}
            sf = sf.to(device)
            logits = model(s1b, s2b, sf).squeeze()
            preds = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(lbls.numpy())
    return all_labels, all_preds


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ── Antonym dict ──────────────────────────────────────────
    antonym_dict: dict = {}
    if args.antonym_csv and os.path.exists(args.antonym_csv):
        antonym_dict = load_antonym_dict(args.antonym_csv)

    # ── SBERT backbone ────────────────────────────────────────
    sbert_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    tokenizer = AutoTokenizer.from_pretrained(sbert_name)
    bert = AutoModel.from_pretrained(sbert_name)
    embed_dim = bert.config.hidden_size

    # ── Data ──────────────────────────────────────────────────
    # train_pairs, train_labels = load_train_pairs(args.train_csv)
    # test_pairs, test_labels = load_test_pairs(args.test_csv)
    # logger.info("Train: %d | Test: %d", len(train_pairs), len(test_pairs))
    
    train_df = pd.read_csv(args.train_csv)
    trial_df = pd.read_csv(args.trial_csv)
    merged_train_df = pd.concat([train_df, trial_df], ignore_index=True)
    temp_train_csv = "merged_train_tmp.csv"
    merged_train_df.to_csv(temp_train_csv, index=False)

    train_pairs, train_labels = load_train_pairs(temp_train_csv)
    test_pairs, test_labels = load_test_pairs(args.test_csv)
    logger.info("Train: %d | Test: %d", len(train_pairs), len(test_pairs))
    
    train_ds = ParaDataset(train_pairs, train_labels, tokenizer, antonym_dict)
    test_ds = ParaDataset(test_pairs, test_labels, tokenizer, antonym_dict)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    

    # ── Model ─────────────────────────────────────────────────
    # Freeze BERT for phase 1
    for param in bert.parameters():
        param.requires_grad = False

    model = SiameseCNN(bert, embed_dim).to(device)

    optimizer = optim.Adam(
        [
            {"params": model.encoder.parameters(), "lr": args.lr},
            {"params": model.classifier.parameters(), "lr": args.lr},
        ],
        weight_decay=1e-3,
    )

    num_pos = sum(train_labels)
    num_neg = len(train_labels) - num_pos
    pos_weight = torch.tensor([max(1.0, num_neg / num_pos)]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_f1 = 0.0
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    for epoch in range(1, args.epochs + 1):

        # Phase 2: unfreeze BERT
        if epoch == args.bert_unfreeze_epoch + 1:
            logger.info("Unfreezing BERT parameters …")
            for param in bert.parameters():
                param.requires_grad = True
            optimizer.add_param_group({"params": bert.parameters(), "lr": 1e-5})

        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for s1b, s2b, sf, lbls in tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}"):
            s1b = {k: v.to(device) for k, v in s1b.items()}
            s2b = {k: v.to(device) for k, v in s2b.items()}
            sf = sf.to(device)
            lbls = lbls.to(device)

            logits = model(s1b, s2b, sf).squeeze()
            loss = loss_fn(logits, lbls)

            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        train_acc = correct / total
        avg_loss = total_loss / len(train_dl)
        logger.info("Epoch %d | Loss=%.4f | Train Acc=%.4f", epoch, avg_loss, train_acc)

        # Validation F1
        y_true, y_pred = evaluate(test_dl, model, device)
        from sklearn.metrics import f1_score
        val_f1 = f1_score(y_true, y_pred, average="macro")
        logger.info("  Val macro-F1=%.4f", val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), args.output)
            logger.info("  ✓ New best (F1=%.4f) — saved to %s", best_f1, args.output)

    # ── Final evaluation ──────────────────────────────────────
    model.load_state_dict(torch.load(args.output, map_location=device))
    y_true, y_pred = evaluate(test_dl, model, device)
    print("\n=== Test Set Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=["NP", "P"], digits=4))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train paraphrase detection model")
    p.add_argument("--train_csv", required=True)
    p.add_argument("--trial_csv", required=True)
    p.add_argument("--test_csv", required=True)
    p.add_argument("--antonym_csv", default=None, help="Optional CSV with word_hi, antonym_hi")
    p.add_argument("--output", default="models/para_detect_best.pth")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--bert_unfreeze_epoch", type=int, default=5,
                   help="BERT layers are frozen until this epoch (then unfrozen)")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
