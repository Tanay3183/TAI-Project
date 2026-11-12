import os
import argparse
import torch
import json
import numpy as np
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import roc_auc_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
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

def train_baseline(data_dir="data", output_dir="results/M0_baseline", seed=42, debug=False):
    print("Loading prepared datasets...")
    train_ds = load_from_disk(os.path.join(data_dir, "train"))
    val_ds = load_from_disk(os.path.join(data_dir, "validation"))
    test_ds = load_from_disk(os.path.join(data_dir, "test"))
    
    # Keep only features for the model
    train_ds_filtered = train_ds.select_columns(["input_ids", "attention_mask", "labels"])
    val_ds_filtered = val_ds.select_columns(["input_ids", "attention_mask", "labels"])
    test_ds_filtered = test_ds.select_columns(["input_ids", "attention_mask", "labels"])
    
    print("Loading distilbert-base-uncased...")
    # num_labels=1 for BCE loss regression formulation (which corresponds to logits + sigmoid + BCE)
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", 
        num_labels=1
    )
    
    # Ensure problem_type uses BCEWithLogitsLoss in Hugging Face Trainer
    model.config.problem_type = "multi_label_classification"

    # HF expects float lists for multi_label_classification
    def format_labels(examples):
        return {"labels": [[float(l)] for l in examples["labels"]]}

    print("Formatting labels for BCE loss...")
    train_ds_filtered = train_ds_filtered.map(format_labels, batched=True)
    val_ds_filtered = val_ds_filtered.map(format_labels, batched=True)
    test_ds_filtered = test_ds_filtered.map(format_labels, batched=True)

    # Shared hyperparameters from PDF: lr 2e-5 (AdamW), epochs 2-3, max_length 128, batch 32, fp16, seed 42.
    # Colab ops rules: checkpoint to Google Drive every ~500 steps.
    
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
        per_device_train_batch_size=16, # Reduced to prevent OOM
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
        report_to="none"  # Keep Colab clean
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds_filtered,
        eval_dataset=val_ds_filtered,
        compute_metrics=compute_metrics
    )
    
    print("Starting training M0 (Baseline)...")
    trainer.train()
    
    print(f"Saving final model to {output_dir}...")
    trainer.save_model(os.path.join(output_dir, "final_model"))
    
    print("Evaluating on test set...")
    test_results = trainer.evaluate(test_ds_filtered)
    print(f"Test Results: {test_results}")
    
    # Save test-set probabilities
    print("Generating test-set probabilities...")
    predictions = trainer.predict(test_ds_filtered)
    raw_preds = predictions.predictions
    if raw_preds.ndim > 1 and raw_preds.shape[1] > 1:
        primary_logits = raw_preds[:, 0]
    else:
        primary_logits = raw_preds.squeeze()
    probs = (1 / (1 + np.exp(-primary_logits))).squeeze().tolist()
    
    # Add predicted probabilities back to the original test dataset for later auditing
    print("Saving test-set probabilities to disk...")
    # Check if 'M0_prob' column already exists in case we run multiple times
    if "M0_prob" in test_ds.column_names:
        test_ds = test_ds.remove_columns("M0_prob")
    test_ds = test_ds.add_column("M0_prob", probs)
    
    # Save the updated test set
    test_ds.save_to_disk(os.path.join(data_dir, "test_with_M0_probs"))
    
    # Save metrics JSON
    with open(os.path.join(output_dir, "test_metrics.json"), "w") as f:
        json.dump(test_results, f, indent=4)
    
    print("M0 Baseline training and evaluation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing dataset splits")
    parser.add_argument("--output_dir", type=str, default="results/M0_baseline", help="Directory to save model")
    parser.add_argument("--debug", action="store_true", help="Run with minimal steps for quick testing")
    args = parser.parse_args()
    train_baseline(data_dir=args.data_dir, output_dir=args.output_dir, debug=args.debug)
