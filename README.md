# Refining Machine Translation Evaluation via Rule-Guided Analysis and Adjustments

A hybrid MT evaluation framework for Hindi that combines rule-based checks, contradiction detection, and paraphrase analysis to refine scores from standard automatic metrics (COMET, BERTScore, BLEURT, BLEU, ChrF, ChrF++).

## Abstract

Automatic evaluation metrics for Machine Translation (MT) often rely on surface-level similarity and can fail to capture semantic adequacy--particularly for phenomena such as negation, antonymy, and paraphrase variation. Although some meta-evaluation studies have been done with European languages but their approaches may not always be suitable for Indian languages. In this paper, we introduce a principled multi-step framework that refines MT evaluation by sequentially applying rule-based checks, contradiction detection, and paraphrase analysis, with a focus on Hindi language. Our method comprises explicit categories like Multidimensional Quality Metrics (MQM) error typology using linguistic rules, detects implicit semantic opposition through contradiction-specific embedding, and estimates paraphrase likelihood via dual sentence encoders with CNN, and a set of shallow lexical features to estimate paraphrase likelihood. The outputs of these components are used to adjust existing metric scores and nudge them toward more faithful judgments. We experimented with 14,214 Hindi samples combined across four datasets. The framework consistently improves six standard metrics (COMET, BERTScore, BLEURT, BLEU, ChrF, ChrF++), demonstrating strong performance across 17 fine-grained translation categories. We hope that our analysis will facilitate further research on Indic MT evaluation.

---

## Overview

Standard MT metrics fail to reliably capture semantic errors like negation flips, antonym substitutions, and paraphrase variations. This framework applies three sequential modules on top of existing metric scores:

1. **Rule-Based Module** — detects negation/antonym flips with explicit Hindi linguistic rules  
2. **Contradiction Detection** — CWE-based CNN that identifies semantic polarity reversals  
3. **Paraphrase Detection** — Siamese SBERT + CNN that identifies meaning-preserving variations  

A unified **Score Adjustment Layer** then nudges existing metric scores toward more faithful judgements.

```
Source + Reference + MT + Metric Score
           │
    ┌──────▼───────┐
    │ Rule-Based   │ ──► d_rule ∈ {0, 1, None}
    └──────┬───────┘
           │ None (not resolved)
    ┌──────▼───────────┐
    │ Contradiction     │ ──► d_contra ∈ [0,1]
    │ Detection (CWE)   │
    └──────┬────────────┘
           │ Non-contradiction
    ┌──────▼───────────┐
    │ Paraphrase        │ ──► d_para ∈ [0,1]
    │ Detection (SBERT) │
    └──────┬────────────┘
           │
    ┌──────▼──────────────┐
    │ Score Adjustment    │ ──► s' (adjusted score)
    └─────────────────────┘
```

---

## Project Structure

```
mt_eval_hindi/
├── cwe_hindi/
│   ├── build_cwe_corpus.py         # Build paraphrase & contradiction pairs .txt files for CWE
│   ├── trans_to_cwe_contra_hi.py   # Translate the cwe contradiction pairs to hindi
│   ├── trans_to_cwe_para_hi.py     # Translate the cwe paraphrase pairs to hindi
│   └── cwe_setup.py                # First step to setup the CWE with vocub and emb-matrix checkpoint
├── data/                           # CSV data files 
├── models/                         # Saved model checkpoints
├── src/
│   ├── rule_based.py               # Negation & antonym rule detector
│   ├── contradiction_detection.py  # CWE-based CNN model + data utils
│   ├── paraphrase_detection.py     # Siamese SBERT+CNN model + data utils
│   ├── score_adjustment.py         # Score adjustment formula
│   └── framework.py                # Full pipeline orchestrator
├── train/
│   ├── train_contradiction.py      # Train the contradiction detection model
│   └── train_paraphrase.py         # Train the paraphrase detection model
├── run_evaluation.py               # Compute metrics + run full framework
├── run_results.py                  # Reproduce paper tables (Tables 5-9)
├── configs.yaml                    # Hyperparameters and paths
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone <repo>
cd mt_eval_hindi
pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab
```

---

## Data

Place the following files in `data/`:

| File | Description |
|------|-------------|
| `Trainset_All_Mixed_Hindi_Balanced.csv` | Training split (13,654 samples) |
| `Trialset_All_Mixed_Hindi_Balanced.csv` | Validation split |
| `Testset_All_Mixed_Hindi_Balanced.csv` | Test split (560 samples) |

CSV columns: `src, ref, mt, model, category, label`  
`label` ∈ `{P, NP}` — Paraphrase / Non-Paraphrase

## For the contradiction model:
1. Download the [ParaNMT-50M dataset](https://drive.google.com/file/d/1rbF3daJjCsa1-fu2GANeJd2FBXos1ugD/view?usp=sharing)
2. Use cwe_hindi/build_cwe_corpus.py to make the cwe paraphrase and contradiction corpus
3. Translate the above generated corpus to hindi using the trans_to_cwe_....py files
4. Make cwe vocab and emb mat ckpt using the file cwe_hindi/cwe_setup.py
    - `hindi_cwe_word2idx300d.pkl` — CWE vocabulary (word→index map)
    - `hindi_cwe_finetuned_emb_300d.pt` — Pre-trained CWE embedding matrix (vocab×300)
    And Place both under `models/`.

---

## Training

### 1. Train Contradiction Detection Model
```bash
python train/train_contradiction.py \
  --train_csv data/Trainset_All_Mixed_Hindi_Balanced.csv \
  --trial_csv data/Trialset_All_Mixed_Hindi_Balanced.csv \
  --word2idx models/hindi_cwe_word2idx300d.pkl \
  --cwe_emb  models/hindi_cwe_finetuned_emb_300d.pt \
  --output_model models/contra_detect_best.pt \
  --epochs 20 --batch_size 128 --lr 1e-4
```

### 2. Train Paraphrase Detection Model
```bash
python train/train_paraphrase.py \
  --train_csv data/Trainset_All_Mixed_Hindi_Balanced.csv \
  --test_csv  data/Testset_All_Mixed_Hindi_Balanced.csv \
  --output_model models/para_detect_best.pth \
  --epochs 15 --batch_size 64 --lr 5e-5
```

---

## Evaluation & Results

### Run full framework evaluation
```bash
python evaluate/run_evaluation.py \
  --test_csv  data/Testset_All_Mixed_Hindi_Balanced.csv \
  --contra_model models/contra_detect_best.pt \
  --word2idx     models/hindi_cwe_word2idx300d.pkl \
  --cwe_emb      models/hindi_cwe_finetuned_emb_300d.pt \
  --para_model   models/para_detect_best.pth \
  --output_csv   results/evaluated_testset.csv
```

### Reproduce paper tables (Tables 5–9)
```bash
python scripts/run_results.py \
  --results_csv results/evaluated_testset.csv
```

---

## Score Adjustment Formula
For a metric score `s ∈ [0,1]` and decision signal `d ∈ [0,1]` with threshold `τ = 0.6`:

```
s' = τ + (τ − s)·d    if d ≥ 0.5 and s < τ   (reward: lift under-scored paraphrases)
   = τ − (s − τ)·(1−d) if d < 0.5 and s > τ   (penalty: lower over-scored contradictions)
   = s                  otherwise               (no change)
s' = clip(s', 0, 1)
```

---

## Fine-Grained Categories (17 total)
| Group | Category | Description |
|-------|----------|-------------|
| **P** (similar) | `Word_synm` | Synonym substitution |
| P | `Mixd_lang` | Mixed-language or script variation |
| P | `Negt_anto` | Negation + antonym logical equivalence |
| P | `Identical` | Exact or near-exact match |
| P | `Fluent` | Fluent reformulation |
| P | `Default_similar` | Other meaning-preserving variation |
| **NP** (dissimilar) | `Anto_flip` | Antonym substitution flips meaning |
| NP | `Negt_flip` | Negation added/removed flips polarity |
| NP | `Gend_flip` | Wrong gender agreement |
| NP | `Sing_plul` | Number agreement error |
| NP | `Tens_chng` | Tense inconsistency |
| NP | `Word_ordr` | Disruptive word-order change |
| NP | `Word_rplc` | Wrong word/named-entity replacement |
| NP | `Add_extra` | Extra information added |
| NP | `Omission` | Key information omitted |
| NP | `Neutral` | Content does not align at all |
| NP | `Default_dissimilar` | Other meaning-altering variation |

---
