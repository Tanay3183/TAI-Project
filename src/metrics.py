import os
import json
import pandas as pd
from sklearn.metrics import roc_auc_score
from datasets import load_from_disk

def compute_subgroup_metrics(df, label_col="labels", prob_col="M0_prob", identity_col="identity"):
    """Compute subgroup AUC and BPSN AUC."""
    
    # 1. Subgroup AUC
    subgroup_df = df[df[identity_col] == 1]
    if len(subgroup_df[label_col].unique()) < 2:
        subgroup_auc = None
    else:
        subgroup_auc = roc_auc_score(subgroup_df[label_col], subgroup_df[prob_col])
        
    # 2. BPSN AUC (Background Positive, Subgroup Negative)
    # Background Positive: Toxic examples NOT mentioning the identity
    # Subgroup Negative: Benign examples mentioning the identity
    bp = df[(df[label_col] == 1) & (df[identity_col] == 0)]
    sn = df[(df[label_col] == 0) & (df[identity_col] == 1)]
    
    bpsn_df = pd.concat([bp, sn])
    
    if len(bpsn_df[label_col].unique()) < 2:
        bpsn_auc = None
    else:
        bpsn_auc = roc_auc_score(bpsn_df[label_col], bpsn_df[prob_col])
        
    return subgroup_auc, bpsn_auc

def audit_model(data_dir="data", prob_col="M0_prob", output_file="results/M0_baseline/audit.json"):
    print(f"Loading test dataset to audit {prob_col}...")
    
    dataset_path = os.path.join(data_dir, f"test_with_{prob_col}s")
    if not os.path.exists(dataset_path):
        print(f"Could not find {dataset_path}. Did you run the training script first?")
        return
        
    dataset = load_from_disk(dataset_path)
    df = dataset.to_pandas()
    
    identities = [
        "male_mask", "female_mask", "black_mask", "white_mask", 
        "muslim_mask", "christian_mask", "jewish_mask", "homosexual_bisexual_mask"
    ]
    
    results = []
    
    # Check if prob_col exists
    if prob_col not in df.columns:
        print(f"Error: Column {prob_col} not found in the dataset.")
        return
        
    for identity in identities:
        display_name = identity.replace("_mask", "")
        # Subgroup Negative Count
        subgroup_negatives = len(df[(df["labels"] == 0) & (df[identity] == 1)])
        
        # We need at least 1 negative to even compute something, usually >= 100 is needed for stability
        if subgroup_negatives < 1:  
            results.append({
                "Identity": display_name,
                "Subgroup AUC": None,
                "BPSN AUC": None,
                "Notes": f"Skipped ({subgroup_negatives} negatives)"
            })
            continue
            
        subgroup_auc, bpsn_auc = compute_subgroup_metrics(df, identity_col=identity, prob_col=prob_col)
        
        results.append({
            "Identity": display_name,
            "Subgroup AUC": subgroup_auc,
            "BPSN AUC": bpsn_auc,
            "Notes": "OK"
        })
        
    results_df = pd.DataFrame(results)
    print("\n--- Fairness Audit Results ---")
    print(results_df.to_string(index=False))
    
    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results_df.to_json(output_file, orient="records", indent=4)
    print(f"\nAudit saved to {output_file}")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prob_col", type=str, default="M0_prob")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    if args.output is None:
        model_name = args.prob_col.replace("_prob", "") if "_prob" in args.prob_col else args.prob_col
        args.output = f"results/{model_name}/audit.json"
        
    audit_model(prob_col=args.prob_col, output_file=args.output)
