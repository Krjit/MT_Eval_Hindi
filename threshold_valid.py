def _threshold_accuracies(df: pd.DataFrame, tau: float) -> tuple[float, float, float]:
    """
    Compute P Acc, NP Acc, and Balanced Acc for a given threshold τ
 
    For each sample, the "mean raw score" is the average of all six
    metric scores available in the DataFrame.
 
        P Acc  = (P samples where mean_score >  τ) / total P samples
        NP Acc = (NP samples where mean_score ≤ τ) / total NP samples
        Bal Acc = (P Acc + NP Acc) / 2
    """
    METRICS = ["COMET", "BERTScore", "BLEURT", "BLEU", "ChrF", "ChrF++"]
    available = [m for m in METRICS if m in df.columns]
    if not available:
        raise ValueError(
            f"No metric columns found in DataFrame. Expected one of: {METRICS}"
        )
 
    # Mean across all available metrics for each sample (raw/original scores)
    df = df.copy()
    df["_mean_score"] = df[available].mean(axis=1)
 
    p_mask  = df["label"].str.strip() == "P"
    np_mask = df["label"].str.strip() == "NP"
 
    n_p  = p_mask.sum()
    n_np = np_mask.sum()
 
    p_acc  = (df.loc[p_mask,  "_mean_score"] >  tau).sum() / n_p  if n_p  > 0 else 0.0
    np_acc = (df.loc[np_mask, "_mean_score"] <= tau).sum() / n_np if n_np > 0 else 0.0
    bal_acc = (p_acc + np_acc) / 2.0
 
    return float(p_acc), float(np_acc), float(bal_acc)

def threshold_validation(train_df: pd.DataFrame) -> None:
    """
    Threshold validation on the TRAINING set.
 
    Sweeps τ ∈ {0.5, 0.6, 0.7} and reports P Acc, NP Acc, Bal. Acc
    for each value. Bold row (τ = 0.6) has the highest Bal. Acc per the paper.
 
    The paper computes this on the training distribution to justify
    the choice of τ = 0.6 as the score adjustment threshold.
    """
    _header("Threshold Validation on Training Set (τ ∈ {0.5, 0.6, 0.7})")
 
    print(f"\n  {'Setting':<12} {'P Acc':>8} {'NP Acc':>8} {'Bal. Acc':>10}")
    print("  " + "-" * 42)
 
    best_bal = -1.0
    best_tau = None
    rows = []
 
    for tau in [0.5, 0.6, 0.7]:
        p_acc, np_acc, bal_acc = _threshold_accuracies(train_df, tau)
        rows.append((tau, p_acc, np_acc, bal_acc))
        if bal_acc > best_bal:
            best_bal = bal_acc
            best_tau = tau
 
    for tau, p_acc, np_acc, bal_acc in rows:
        marker = "  ← best" if tau == best_tau else ""
        bold   = "**" if tau == best_tau else "  "
        print(f"  {bold}τ = {tau}{bold}  {p_acc:>8.2f} {np_acc:>8.2f} {bal_acc:>10.2f}{marker}")
 
    print(f"\n  Chosen threshold: τ = {best_tau}  (Bal. Acc = {best_bal:.2f})") 

def main():
    df_train = pd.read_csv('data/Trainset_All_Mixed_Hindi_Balanced.csv')
    df_trial = pd.read_csv('data/Trialset_All_Mixed_Hindi_Balanced.csv')

    df = pd.concat([df_train, df_trial], ignore_index=True)
    threshold_validation(df)
    
if __name__ == "__main__":
    main()
