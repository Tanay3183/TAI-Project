import os
import re
import argparse
import random
import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer

# Target identity groups
KEYWORDS = {
    "male": ["male", "man", "men", "boy", "boys", "guy", "guys"],
    "female": ["female", "woman", "women", "girl", "girls", "lady", "ladies"],
    "black": ["black", "african american"],
    "white": ["white", "caucasian"],
    "muslim": ["muslim", "islam", "islamic"],
    "christian": ["christian", "catholic", "protestant", "christianity"],
    "jewish": ["jewish", "jew", "judaism"],
    "homosexual_bisexual": ["gay", "lesbian", "homosexual", "bisexual", "queer"]
}

IDENTITY_COLS = list(KEYWORDS.keys())
AUX_COLS = ["severe_toxicity", "obscene", "threat", "insult", "identity_attack", "sexual_explicit"]

PATTERNS = {
    k: re.compile(r'\b(' + '|'.join(re.escape(w) for w in v) + r')\b', re.IGNORECASE)
    for k, v in KEYWORDS.items()
}

RARE_IDENTITIES = {"jewish", "homosexual_bisexual", "muslim", "black"}

def prepare_identity_dataset(
    save_dir="data_identity",
    model_name="distilbert-base-uncased",
    train_size=100000,
    val_size=10000,
    test_size=10000,
    identity_ratio=0.35,
    seed=42,
    debug=False
):
    random.seed(seed)
    
    if debug:
        print("Running in DEBUG mode with tiny dataset...")
        train_size, val_size, test_size = 200, 50, 50

    print("Loading full Jigsaw Civil Comments dataset (~1.8M comments)...")
    raw_dataset = load_dataset("google/civil_comments", split="train")
    
    total_needed = train_size + val_size + test_size
    print(f"Curating dataset (Total needed: {total_needed} rows: train={train_size}, val={val_size}, test={test_size})...")
    print(f"Target composition: ~{int(identity_ratio*100)}% identity-bearing, ~{int((1-identity_ratio)*100)}% background.")

    rare_indices = []
    common_identity_indices = []
    background_toxic_indices = []
    background_benign_indices = []

    print("Scanning comments to find all identity mentions and balanced background samples...")
    # Scan through dataset in memory-efficient batches
    max_scan = 2000 if debug else len(raw_dataset)
    batch_size = 50000
    
    for start_idx in range(0, max_scan, batch_size):
        end_idx = min(start_idx + batch_size, max_scan)
        batch = raw_dataset[start_idx:end_idx]
        
        for offset, (text, tox) in enumerate(zip(batch["text"], batch["toxicity"])):
            idx = start_idx + offset
            matched_rare = False
            matched_common = False
            
            for id_name, pattern in PATTERNS.items():
                if pattern.search(text):
                    if id_name in RARE_IDENTITIES:
                        matched_rare = True
                    else:
                        matched_common = True
            
            if matched_rare:
                rare_indices.append(idx)
            elif matched_common:
                common_identity_indices.append(idx)
            else:
                if tox >= 0.5:
                    background_toxic_indices.append(idx)
                else:
                    if len(background_benign_indices) < 50000:
                        background_benign_indices.append(idx)

    print(f"\nScan results from full dataset:")
    print(f"  - Comments with rare identities (jewish, lgbtq+, muslim, black): {len(rare_indices)}")
    print(f"  - Comments with other identities (male, female, white, christian): {len(common_identity_indices)}")
    print(f"  - Background toxic comments (no identity): {len(background_toxic_indices)}")
    print(f"  - Background benign comments (no identity): {len(background_benign_indices)}")

    # Shuffle categories
    random.shuffle(rare_indices)
    random.shuffle(common_identity_indices)
    random.shuffle(background_toxic_indices)
    random.shuffle(background_benign_indices)

    # Compose balanced pool:
    # 1. Take rare identities first
    # 2. Fill remaining identity quota with common identities
    # 3. Fill the rest with background (both toxic and benign)
    identity_target = int(total_needed * identity_ratio)
    background_target = total_needed - identity_target

    selected_rare = rare_indices
    needed_common = max(0, identity_target - len(selected_rare))
    selected_common = common_identity_indices[:needed_common]
    selected_identity = (selected_rare + selected_common)[:identity_target]

    bg_toxic_count = min(len(background_toxic_indices), background_target // 2)
    bg_benign_count = min(len(background_benign_indices), background_target - bg_toxic_count)
    selected_background = (
        background_toxic_indices[:bg_toxic_count] + 
        background_benign_indices[:bg_benign_count]
    )

    combined_pool = selected_identity + selected_background
    random.shuffle(combined_pool)

    # Trim to total_needed
    combined_pool = combined_pool[:total_needed]
    print(f"\nFinal curated pool size: {len(combined_pool)} (Identity-bearing: ~{len(selected_identity)}, Background: ~{len(selected_background)})")

    # Split into train, val, test
    train_indices = combined_pool[:train_size]
    val_indices = combined_pool[train_size:train_size + val_size]
    test_indices = combined_pool[train_size + val_size:train_size + val_size + test_size]

    print(f"Split sizes: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

    train_ds = raw_dataset.select(train_indices)
    val_ds = raw_dataset.select(val_indices)
    test_ds = raw_dataset.select(test_indices)

    # Process batch function to add labels, auxiliary subtypes, and mask columns
    def process_batch(batch):
        batch["labels"] = [1 if t >= 0.5 else 0 for t in batch["toxicity"]]
        
        # Auxiliary multi-task targets: 6 subtypes binarized at >= 0.5
        aux_targets = []
        for i in range(len(batch["toxicity"])):
            sub_vec = [1.0 if batch[col][i] >= 0.5 else 0.0 for col in AUX_COLS]
            aux_targets.append(sub_vec)
        batch["aux_labels"] = aux_targets
        
        # Identity masks (8 groups)
        subgroup_masks = []
        for text in batch["text"]:
            mask_vec = [1 if PATTERNS[col].search(text) else 0 for col in IDENTITY_COLS]
            subgroup_masks.append(mask_vec)
            
        for idx, col in enumerate(IDENTITY_COLS):
            batch[col + "_mask"] = [m[idx] for m in subgroup_masks]
        batch["subgroup_masks"] = subgroup_masks
        return batch

    print("\nAdding labels, auxiliary multi-task targets, and identity masks...")
    train_ds = train_ds.map(process_batch, batched=True, batch_size=1000)
    val_ds = val_ds.map(process_batch, batched=True, batch_size=1000)
    test_ds = test_ds.map(process_batch, batched=True, batch_size=1000)

    print(f"Tokenizing with Head+Tail truncation (64+64) using {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def head_tail_tokenize(examples):
        batch_ids = tokenizer(examples["text"], truncation=False, padding=False)["input_ids"]
        
        input_ids_list = []
        attention_mask_list = []
        max_len = 128
        head_len = 63
        tail_len = 63
        
        cls_id = tokenizer.cls_token_id or tokenizer.bos_token_id or 101
        sep_id = tokenizer.sep_token_id or tokenizer.eos_token_id or 102
        pad_id = tokenizer.pad_token_id or 0
        
        for ids in batch_ids:
            core_ids = [t for t in ids if t not in (cls_id, sep_id, pad_id)]
            if len(core_ids) <= (max_len - 2):
                seq = [cls_id] + core_ids + [sep_id]
                pad_count = max_len - len(seq)
                input_ids = seq + [pad_id] * pad_count
                attention_mask = [1] * len(seq) + [0] * pad_count
            else:
                head = core_ids[:head_len]
                tail = core_ids[-tail_len:]
                seq = [cls_id] + head + tail + [sep_id]
                input_ids = seq[:max_len]
                attention_mask = [1] * len(input_ids)
                
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            
        return {"input_ids": input_ids_list, "attention_mask": attention_mask_list}

    train_ds = train_ds.map(head_tail_tokenize, batched=True, batch_size=1000)
    val_ds = val_ds.map(head_tail_tokenize, batched=True, batch_size=1000)
    test_ds = test_ds.map(head_tail_tokenize, batched=True, batch_size=1000)

    print(f"\nSaving identity-enriched dataset to '{save_dir}'...")
    os.makedirs(save_dir, exist_ok=True)
    train_ds.save_to_disk(os.path.join(save_dir, "train"))
    val_ds.save_to_disk(os.path.join(save_dir, "validation"))
    test_ds.save_to_disk(os.path.join(save_dir, "test"))

    print("\nIdentity dataset preparation complete!")
    print(f"Saved splits at: {save_dir}/train, {save_dir}/validation, {save_dir}/test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an identity-enriched dataset from Jigsaw Civil Comments")
    parser.add_argument("--save_dir", type=str, default="data_identity", help="Directory to save the new dataset")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="Backbone tokenizer/model name")
    parser.add_argument("--train_size", type=int, default=100000, help="Number of training samples")
    parser.add_argument("--val_size", type=int, default=10000, help="Number of validation samples")
    parser.add_argument("--test_size", type=int, default=10000, help="Number of test samples")
    parser.add_argument("--identity_ratio", type=float, default=0.35, help="Proportion of identity-bearing samples (e.g. 0.35 for 35%)")
    parser.add_argument("--debug", action="store_true", help="Run quick debug test with tiny sample size")
    args = parser.parse_args()

    prepare_identity_dataset(
        save_dir=args.save_dir,
        model_name=args.model_name,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        identity_ratio=args.identity_ratio,
        debug=args.debug
    )
