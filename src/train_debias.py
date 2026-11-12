import os
import sys
# Ensure src can be imported whether run from root or src/
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import torch
import torch.nn.functional as F
import json
import argparse
import numpy as np
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import roc_auc_score

def compute_mimic_loss(predictions, labels, subgroups=None, power=5.0):
    """
    Kaggle 1st Place Metric-Mimic Loss.
    predictions: primary toxicity logits [batch_size]
    labels: binary toxicity labels [batch_size]
    subgroups: binary indicator matrix of identity presence [batch_size, num_identities]
    """
    bce = F.binary_cross_entropy_with_logits(predictions, labels.float(), reduction="none")
    
    if subgroups is None or subgroups.sum() == 0:
        return bce.mean()
        
    labels_binary = (labels.unsqueeze(-1) >= 0.5)
    subgroups_bool = subgroups.bool()
    
    # Define boolean masks for sub-slices
    subgroup_pos = subgroups_bool & labels_binary
    subgroup_neg = subgroups_bool & ~labels_binary
    background_pos = ~subgroups_bool & labels_binary
    background_neg = ~subgroups_bool & ~labels_binary
    
    bpsn_mask = (background_pos | subgroup_neg).float()
    bnsp_mask = (background_neg | subgroup_pos).float()
    sg_mask = subgroups_bool.float()
    
    bce_col = bce.unsqueeze(-1)
    eps = 1e-6
    
    # Per-identity slice losses
    sb_loss = (bce_col * sg_mask).sum(0) / sg_mask.sum(0).clamp(min=1.0)
    bpsn_loss = (bce_col * bpsn_mask).sum(0) / bpsn_mask.sum(0).clamp(min=1.0)
    bnsp_loss = (bce_col * bnsp_mask).sum(0) / bnsp_mask.sum(0).clamp(min=1.0)
    
    # Power-mean aggregation (power=5.0 focuses gradients on hard slices)
    sb_valid = sg_mask.sum(0) > 0
    bpsn_valid = bpsn_mask.sum(0) > 0
    bnsp_valid = bnsp_mask.sum(0) > 0
    
    def power_mean(tensor, valid_mask):
        valid = tensor[valid_mask]
        if valid.numel() == 0:
            return bce.mean()
        return (valid + eps).pow(power).mean().pow(1.0 / power)
        
    l_sb = power_mean(sb_loss, sb_valid)
    l_bpsn = power_mean(bpsn_loss, bpsn_valid)
    l_bnsp = power_mean(bnsp_loss, bnsp_valid)
    l_overall = bce.mean()
    
    return (l_overall + l_sb + l_bpsn + l_bnsp) / 4.0

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Primary toxicity is logit 0
    if logits.ndim > 1 and logits.shape[1] > 1:
        logits = logits[:, 0]
        labels = labels[:, 0]
    else:
        logits = logits.squeeze()
        labels = labels.squeeze()
        
    probs = 1 / (1 + np.exp(-logits))
    if len(np.unique(labels)) < 2:
        return {"roc_auc": 0.0}
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.0
    return {"roc_auc": auc}

class CLPTrainer(Trainer):
    def __init__(self, clp_lambda=1.0, use_mimic_loss=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clp_lambda = clp_lambda
        self.use_mimic_loss = use_mimic_loss
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Extract counterfactual inputs if they exist
        cf_input_ids = inputs.pop("cf_input_ids", None)
        cf_attention_mask = inputs.pop("cf_attention_mask", None)
        
        # Extract subgroup masks for mimic loss
        subgroup_masks = inputs.pop("subgroup_masks", None)
        if subgroup_masks is not None and not isinstance(subgroup_masks, torch.Tensor):
            subgroup_masks = torch.tensor(subgroup_masks, dtype=torch.float32, device=model.device)
        elif subgroup_masks is not None:
            subgroup_masks = subgroup_masks.to(model.device).float()
            
        # Clean text columns if present
        for key in ["text", "cf_text"]:
            inputs.pop(key, None)
            
        labels = inputs.get("labels") # [batch_size, num_labels]
        
        # Standard forward pass
        outputs = model(**inputs)
        logits = outputs.logits # [batch_size, num_labels]
        
        # Multi-task disentanglement
        primary_logits = logits[:, 0] if logits.ndim > 1 and logits.shape[1] > 1 else logits.squeeze()
        primary_labels = labels[:, 0] if labels.ndim > 1 and labels.shape[1] > 1 else labels.squeeze()
        
        if self.use_mimic_loss and subgroup_masks is not None:
            primary_loss = compute_mimic_loss(primary_logits, primary_labels, subgroup_masks)
        else:
            primary_loss = F.binary_cross_entropy_with_logits(primary_logits, primary_labels.float())
            
        # Auxiliary subtypes loss if num_labels > 1
        if logits.ndim > 1 and logits.shape[1] > 1:
            aux_logits = logits[:, 1:]
            aux_labels = labels[:, 1:].float()
            aux_loss = F.binary_cross_entropy_with_logits(aux_logits, aux_labels)
        else:
            aux_loss = 0.0
            
        loss = primary_loss + 0.2 * aux_loss
        
        # Counterfactual CDA / CLP
        if cf_input_ids is not None:
            cf_outputs = model(input_ids=cf_input_ids, attention_mask=cf_attention_mask)
            cf_logits = cf_outputs.logits
            cf_primary_logits = cf_logits[:, 0] if cf_logits.ndim > 1 and cf_logits.shape[1] > 1 else cf_logits.squeeze()
            
            # CDA loss on twin
            if self.use_mimic_loss and subgroup_masks is not None:
                cf_loss = compute_mimic_loss(cf_primary_logits, primary_labels, subgroup_masks)
            else:
                cf_loss = F.binary_cross_entropy_with_logits(cf_primary_logits, primary_labels.float())
                
            loss = (loss + cf_loss) / 2.0
            
            # CLP penalty if lambda > 0
            if self.clp_lambda > 0:
                probs = torch.sigmoid(primary_logits)
                cf_probs = torch.sigmoid(cf_primary_logits)
                clp_penalty = torch.mean((probs - cf_probs) ** 2)
                loss = loss + (self.clp_lambda * 5.0) * clp_penalty
                
        return (loss, outputs) if return_outputs else loss

def train_debias(
    model_type="M2", 
    model_name="distilbert-base-uncased",
    data_dir="data", 
    output_dir="results", 
    use_mimic_loss=True,
    seed=42, 
    debug=False
):
    print(f"Starting {model_type} Debiasing Pipeline (Backbone: {model_name})...")
    output_dir = os.path.join(output_dir, model_type)
    
    print("Loading datasets...")
    if model_type in ["M2", "M3"]:
        train_ds = load_from_disk(os.path.join(data_dir, "train_cda"))
    else:
        train_ds = load_from_disk(os.path.join(data_dir, "train"))
        
    val_ds = load_from_disk(os.path.join(data_dir, "validation"))
    test_ds = load_from_disk(os.path.join(data_dir, "test"))
    
    # Determine if auxiliary subtypes targets are present
    has_aux = "aux_labels" in train_ds.column_names
    num_labels = 7 if has_aux else 1
    print(f"Multi-task heads: {num_labels} output logits (Primary Toxicity + {num_labels - 1} Auxiliary Subtypes).")
    
    # Keep columns for training
    base_cols = ["input_ids", "attention_mask", "labels"]
    if has_aux:
        base_cols.append("aux_labels")
    if "subgroup_masks" in train_ds.column_names:
        base_cols.append("subgroup_masks")
        
    if model_type in ["M2", "M3"]:
        keep_cols = base_cols + ["cf_input_ids", "cf_attention_mask"]
    else:
        keep_cols = base_cols
        
    train_ds_filtered = train_ds.select_columns(keep_cols)
    val_ds_filtered = val_ds.select_columns(base_cols)
    test_ds_filtered = test_ds.select_columns(base_cols)
    
    print(f"Loading {model_name} with num_labels={num_labels}...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.config.problem_type = "multi_label_classification"

    def format_labels(examples):
        if has_aux:
            combined = []
            for lab, aux in zip(examples["labels"], examples["aux_labels"]):
                combined.append([float(lab)] + [float(a) for a in aux])
            return {"labels": combined}
        else:
            return {"labels": [[float(l)] for l in examples["labels"]]}

    print("Formatting multi-task target vectors...")
    train_ds_filtered = train_ds_filtered.map(format_labels, batched=True)
    val_ds_filtered = val_ds_filtered.map(format_labels, batched=True)
    test_ds_filtered = test_ds_filtered.map(format_labels, batched=True)

    if debug:
        print("Running in DEBUG mode. Minimal epochs/steps.")
        num_epochs = 1
        eval_steps = 10
    else:
        num_epochs = 2
        eval_steps = 500
        
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=8 if model_type == "M3" else 16,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=16,
        fp16=True,
        gradient_checkpointing=True,
        seed=seed,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        logging_steps=eval_steps // 2 if debug else 100,
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        report_to="none",
        remove_unused_columns=False
    )
    
    if model_type in ["M2", "M3"]:
        trainer = CLPTrainer(
            clp_lambda=0.5 if model_type == "M3" else 0.0,
            use_mimic_loss=use_mimic_loss,
            model=model,
            args=training_args,
            train_dataset=train_ds_filtered,
            eval_dataset=val_ds_filtered,
            compute_metrics=compute_metrics
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds_filtered,
            eval_dataset=val_ds_filtered,
            compute_metrics=compute_metrics
        )
    
    print(f"Training {model_type}...")
    trainer.train()
    
    final_model_path = os.path.join(output_dir, "final_model")
    print(f"Saving final {model_type} model to {final_model_path}...")
    trainer.save_model(final_model_path)
    
    print("Evaluating on test set...")
    test_results = trainer.evaluate(test_ds_filtered)
    
    print("Generating test-set probabilities (Primary Toxicity)...")
    predictions = trainer.predict(test_ds_filtered)
    raw_preds = predictions.predictions
    if raw_preds.ndim > 1 and raw_preds.shape[1] > 1:
        primary_logits = raw_preds[:, 0]
    else:
        primary_logits = raw_preds.squeeze()
    probs = (1 / (1 + np.exp(-primary_logits))).squeeze().tolist()
    
    prob_col_name = f"{model_type}_prob"
    
    try:
        test_base = load_from_disk(os.path.join(data_dir, "test_with_M0_probs"))
        if len(test_base) != len(probs):
            print("Warning: test_with_M0_probs size mismatch. Using fresh test dataset.")
            test_base = load_from_disk(os.path.join(data_dir, "test"))
    except FileNotFoundError:
        test_base = load_from_disk(os.path.join(data_dir, "test"))
        
    if prob_col_name in test_base.column_names:
        test_base = test_base.remove_columns(prob_col_name)
    test_base = test_base.add_column(prob_col_name, probs)
    
    save_path = os.path.join(data_dir, f"test_with_{model_type}_probs")
    test_base.save_to_disk(save_path)
    print(f"Saved predictions to {save_path}")
    
    with open(os.path.join(output_dir, "test_metrics.json"), "w") as f:
        json.dump(test_results, f, indent=4)
        
    # Fit LEACE Concept Erasure post-hoc on validation set
    try:
        from src.leace_posthoc import fit_and_save_leace
        print("\n--- Fitting Post-Hoc LEACE Concept Erasure (Inference Safety Net) ---")
        fit_and_save_leace(model_dir=final_model_path, data_dir=data_dir)
    except Exception as e:
        print(f"Note: LEACE fitting encountered notice: {e}")
    
    print(f"\n{model_type} pipeline complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="M2", choices=["M1", "M2", "M3"])
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="Backbone transformer model")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing dataset splits")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save model")
    parser.add_argument("--no_mimic", action="store_true", help="Disable Kaggle 1st Place Mimic Loss")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    
    train_debias(
        model_type=args.model, 
        model_name=args.model_name,
        data_dir=args.data_dir, 
        output_dir=args.output_dir, 
        use_mimic_loss=not args.no_mimic,
        debug=args.debug
    )
