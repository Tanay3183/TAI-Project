import os
import torch
import json
import argparse
import numpy as np
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import roc_auc_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    if probs.ndim > 1 and probs.shape[1] > 1:
        probs = probs[:, 1]
    else:
        probs = probs.squeeze()
    labels = labels.squeeze()
    if len(np.unique(labels)) < 2:
        return {"roc_auc": 0.0}
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.0
    return {"roc_auc": auc}

class CLPTrainer(Trainer):
    def __init__(self, clp_lambda=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clp_lambda = clp_lambda
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Extract counterfactual inputs if they exist
        cf_input_ids = inputs.pop("cf_input_ids", None)
        cf_attention_mask = inputs.pop("cf_attention_mask", None)
        
        # We also need to remove text columns that might crash the model
        for key in ["text", "cf_text"]:
            inputs.pop(key, None)
        
        # Standard forward pass
        outputs = model(**inputs)
        loss = outputs.loss
        logits = outputs.logits
        
        if cf_input_ids is not None and self.clp_lambda > 0:
            # Counterfactual forward pass (CDA: We pass labels to calculate cf_loss)
            cf_outputs = model(input_ids=cf_input_ids, attention_mask=cf_attention_mask, labels=inputs.get("labels"))
            cf_loss = cf_outputs.loss
            cf_logits = cf_outputs.logits
            
            # CDA: Average the standard BCE loss with the counterfactual BCE loss
            loss = (loss + cf_loss) / 2.0
            
            # Sigmoid for BCE probs
            probs = torch.sigmoid(logits)
            cf_probs = torch.sigmoid(cf_logits)
            
            # Counterfactual Logit Pairing (CLP) penalty
            # Multiply by a stronger lambda (e.g., 5.0) because MSE on probs is very small
            clp_penalty = torch.mean((probs - cf_probs) ** 2)
            loss = loss + (self.clp_lambda * 5.0) * clp_penalty
            
        return (loss, outputs) if return_outputs else loss

def train_debias(model_type="M3", data_dir="data", output_dir="results", seed=42, debug=False):
    print(f"Starting {model_type} Debiasing Pipeline...")
    output_dir = os.path.join(output_dir, model_type)
    
    print("Loading datasets...")
    # For M2 and M3, we use the CDA augmented dataset
    if model_type in ["M2", "M3"]:
        train_ds = load_from_disk(os.path.join(data_dir, "train_cda"))
    else:
        train_ds = load_from_disk(os.path.join(data_dir, "train"))
        
    val_ds = load_from_disk(os.path.join(data_dir, "validation"))
    test_ds = load_from_disk(os.path.join(data_dir, "test"))
    
    # Keep only features for the model
    # M2 and M3 require the cf_ columns to be passed to the trainer
    if model_type in ["M2", "M3"]:
        keep_cols = ["input_ids", "attention_mask", "labels", "cf_input_ids", "cf_attention_mask"]
    else:
        keep_cols = ["input_ids", "attention_mask", "labels"]
        
    train_ds_filtered = train_ds.select_columns(keep_cols)
    val_ds_filtered = val_ds.select_columns(["input_ids", "attention_mask", "labels"])
    test_ds_filtered = test_ds.select_columns(["input_ids", "attention_mask", "labels"])
    
    print("Loading distilbert-base-uncased...")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=1)
    model.config.problem_type = "multi_label_classification"

    def format_labels(examples):
        return {"labels": [[float(l)] for l in examples["labels"]]}

    print("Formatting labels...")
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
        per_device_train_batch_size=8 if model_type == "M3" else 16, # Reduced batch size to prevent OOM
        gradient_accumulation_steps=2, # Keep effective batch size same
        per_device_eval_batch_size=16, # Reduced eval batch size
        fp16=True,
        gradient_checkpointing=True, # Saves VRAM at the cost of slight compute overhead
        seed=seed,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        logging_steps=eval_steps // 2 if debug else 100,
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        report_to="none",
        remove_unused_columns=False  # CRITICAL: Prevents Trainer from dropping cf_input_ids
    )
    
    # M2 and M3 both use the CLPTrainer to process the counterfactual columns in parallel.
    # M2 is pure CDA (lambda = 0.0), M3 is CDA + CLP (lambda = 0.5 to prevent accuracy destruction)
    if model_type in ["M2", "M3"]:
        trainer = CLPTrainer(
            clp_lambda=0.5 if model_type == "M3" else 0.0,
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
    
    print(f"Saving final {model_type} model...")
    trainer.save_model(os.path.join(output_dir, "final_model"))
    
    print("Evaluating on test set...")
    test_results = trainer.evaluate(test_ds_filtered)
    
    print("Generating test-set probabilities...")
    predictions = trainer.predict(test_ds_filtered)
    probs = (1 / (1 + np.exp(-predictions.predictions))).squeeze().tolist()
    
    prob_col_name = f"{model_type}_prob"
    
    # Attempt to accumulate with M0 probabilities, but fallback to fresh test dataset if there's a size mismatch
    try:
        test_base = load_from_disk(os.path.join(data_dir, "test_with_M0_probs"))
        if len(test_base) != len(probs):
            print("Warning: test_with_M0_probs has a size mismatch. Using fresh test dataset instead.")
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
    
    print(f"{model_type} training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="M3", choices=["M1", "M2", "M3"])
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    train_debias(model_type=args.model, debug=args.debug)
