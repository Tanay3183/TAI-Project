import os
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

def prepare_data(seed=42, train_size=200000, val_size=10000, test_size=10000, save_dir="data", debug=False):
    if debug:
        print("Running in DEBUG mode. Using tiny dataset sizes.")
        train_size, val_size, test_size = 100, 20, 20
        
    print("Loading Jigsaw Civil Comments dataset...")
    # Load dataset (Hugging Face's google/civil_comments lacks the identity columns, so we use keyword fallbacks)
    dataset = load_dataset("google/civil_comments")
    
    # Target identity columns from the PDF
    identity_cols = [
        "male", "female", "black", "white", "muslim", 
        "christian", "jewish", "homosexual_bisexual"
    ]
    
    # Keyword fallbacks for all identities since the Hugging Face dataset lacks the Kaggle identity columns
    keywords = {
        "male": ["male", "man", "men", "boy", "boys", "guy", "guys"],
        "female": ["female", "woman", "women", "girl", "girls", "lady", "ladies"],
        "black": ["black", "african american"],
        "white": ["white", "caucasian"],
        "muslim": ["muslim", "islam", "islamic"],
        "christian": ["christian", "catholic", "protestant", "christianity"],
        "jewish": ["jewish", "jew", "judaism"],
        "homosexual_bisexual": ["gay", "lesbian", "homosexual", "bisexual", "queer"]
    }
    
    def process_batch(batch):
        # Binarize toxicity (toxic = toxicity >= 0.5)
        batch["labels"] = [1 if t >= 0.5 else 0 for t in batch["toxicity"]]
        
        # Binarize identity masks using keyword fallbacks
        for col in identity_cols:
            col_mask = []
            for text in batch["text"]:
                text_lower = text.lower()
                # Check for word boundaries to avoid matching "black" in "blackboard"
                import re
                matched = False
                for kw in keywords[col]:
                    if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                        matched = True
                        break
                col_mask.append(1 if matched else 0)
            batch[col + "_mask"] = col_mask
            
        return batch

    print("Processing and binarizing labels & masks (with keyword fallbacks)...")
    # Map function to binarize labels and create identity masks
    dataset = dataset.map(process_batch, batched=True, batch_size=1000)
    
    # Subsample datasets
    print(f"Subsampling (train: {train_size}, val: {val_size}, test: {test_size})...")
    train_ds = dataset["train"].shuffle(seed=seed).select(range(train_size))
    val_ds = dataset["validation"].shuffle(seed=seed).select(range(val_size))
    test_ds = dataset["test"].shuffle(seed=seed).select(range(test_size))
    
    print("Tokenizing using distilbert-base-uncased...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    
    train_ds = train_ds.map(tokenize_function, batched=True)
    val_ds = val_ds.map(tokenize_function, batched=True)
    test_ds = test_ds.map(tokenize_function, batched=True)
    
    # Save to disk
    print(f"Saving to {save_dir}...")
    os.makedirs(save_dir, exist_ok=True)
    train_ds.save_to_disk(os.path.join(save_dir, "train"))
    val_ds.save_to_disk(os.path.join(save_dir, "validation"))
    test_ds.save_to_disk(os.path.join(save_dir, "test"))
    
    print("Data preparation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Run with tiny dataset for quick testing")
    args = parser.parse_args()
    prepare_data(debug=args.debug)
