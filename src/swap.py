import re
import os
from datasets import load_from_disk
from transformers import AutoTokenizer

# Symmetrical swap dictionary to ensure we can swap in both directions
SWAP_DICT = {
    # Gender
    "male": "female", "female": "male",
    "males": "females", "females": "males",
    "man": "woman", "woman": "man",
    "men": "women", "women": "men",
    "boy": "girl", "girl": "boy",
    "boys": "girls", "girls": "boys",
    "guy": "girl",
    "he": "she", "she": "he",
    "his": "hers", "hers": "his",
    "him": "her", "her": "him",
    
    # Race
    "black": "white", "white": "black",
    "african american": "white",
    
    # Sexuality
    "gay": "straight", "straight": "gay",
    "lesbian": "straight",
    "homosexual": "heterosexual", "heterosexual": "homosexual",
    
    # Religion (rotating through)
    "muslim": "christian",
    "islam": "christianity",
    "christian": "muslim",
    "jewish": "christian",
    "jew": "christian"
}

def create_single_pass_regex(swap_dict):
    """
    Creates a single regex that matches any of the keys in the swap dictionary
    using word boundaries \b to ensure we don't swap substrings (e.g., 'mankind').
    """
    # Sort keys by length descending to match longest terms first (e.g., "african american" before "african")
    sorted_keys = sorted(swap_dict.keys(), key=len, reverse=True)
    pattern = r'\b(' + '|'.join(re.escape(k) for k in sorted_keys) + r')\b'
    return re.compile(pattern, re.IGNORECASE)

SWAP_REGEX = create_single_pass_regex(SWAP_DICT)

def swap_identities(text):
    """
    Swaps identity terms in a string in a single pass.
    """
    def replacer(match):
        word = match.group(1)
        # Preserve original casing if possible
        swapped = SWAP_DICT[word.lower()]
        if word.istitle():
            return swapped.title()
        elif word.isupper():
            return swapped.upper()
        return swapped

    return SWAP_REGEX.sub(replacer, text)

def run_tests():
    """Unit tests to ensure naive replacement bugs don't happen."""
    cases = [
        ("I am a proud Black woman.", "I am a proud White man."),
        ("He is a gay rights advocate.", "She is a straight rights advocate."),
        ("This affects mankind.", "This affects mankind."), # 'man' shouldn't swap here due to \b
        ("Muslims and Christians.", "Christians and Muslims.") # Assuming simple lower casing check works
    ]
    
    passed = True
    for original, expected in cases:
        result = swap_identities(original)
        # Check lowercased comparison since our simple dictionary might not perfectly handle plural casing
        if result.lower() != expected.lower():
            print(f"FAIL: '{original}' -> '{result}' (Expected: '{expected}')")
            passed = False
    
    if passed:
        print("All swap unit tests passed successfully!")
    else:
        print("Some unit tests failed. Check regex logic.")
        
def augment_dataset(data_dir="data", model_name="distilbert-base-uncased"):
    """
    Loads the train dataset, generates counterfactual text for each row, 
    tokenizes it into `cf_input_ids` with Head+Tail truncation, and saves the augmented dataset.
    """
    print("Loading train dataset for CDA augmentation...")
    train_ds = load_from_disk(os.path.join(data_dir, "train"))
    
    print(f"Initializing Tokenizer ({model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    max_len = 128
    head_len = 63
    tail_len = 63
    cls_id = tokenizer.cls_token_id or tokenizer.bos_token_id or 101
    sep_id = tokenizer.sep_token_id or tokenizer.eos_token_id or 102
    pad_id = tokenizer.pad_token_id or 0

    def add_cf_columns(batch):
        # 1. Swap the text
        cf_texts = [swap_identities(text) for text in batch["text"]]
        batch["cf_text"] = cf_texts
        
        # 2. Tokenize the swapped text using Head+Tail truncation
        batch_ids = tokenizer(cf_texts, truncation=False, padding=False)["input_ids"]
        cf_input_ids = []
        cf_attention_mask = []
        
        for ids in batch_ids:
            core_ids = [t for t in ids if t not in (cls_id, sep_id, pad_id)]
            if len(core_ids) <= (max_len - 2):
                seq = [cls_id] + core_ids + [sep_id]
                pad_count = max_len - len(seq)
                cf_input_ids.append(seq + [pad_id] * pad_count)
                cf_attention_mask.append([1] * len(seq) + [0] * pad_count)
            else:
                head = core_ids[:head_len]
                tail = core_ids[-tail_len:]
                seq = [cls_id] + head + tail + [sep_id]
                inp = seq[:max_len]
                cf_input_ids.append(inp)
                cf_attention_mask.append([1] * len(inp))
                
        batch["cf_input_ids"] = cf_input_ids
        batch["cf_attention_mask"] = cf_attention_mask
        
        return batch

    print("Generating counterfactual twins and tokenizing `cf_input_ids`...")
    # Generate the twin columns
    augmented_ds = train_ds.map(add_cf_columns, batched=True, batch_size=1000)
    
    # Print custom examples to demonstrate the swapping logic clearly
    print("\n--- Example Swaps ---")
    custom_examples = [
        "You are a proud gay person.",
        "Females are not mean't to grow outside kitchen.",
        "Are you a muslim or hindu?"
    ]
    for text in custom_examples:
        swapped = swap_identities(text)
        print(f"Original: {text}")
        print(f"Swapped:  {swapped}\n")
    print("---------------------\n")
    
    save_path = os.path.join(data_dir, "train_cda")
    print(f"Saving augmented dataset to {save_path}...")
    augmented_ds.save_to_disk(save_path)
    print("Phase 4 Augmentation complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run unit tests")
    parser.add_argument("--augment", action="store_true", help="Augment train dataset")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing dataset splits")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="Backbone tokenizer name")
    args = parser.parse_args()
    
    if args.test or not args.augment:
        run_tests()
    if args.augment:
        augment_dataset(data_dir=args.data_dir, model_name=args.model_name)
