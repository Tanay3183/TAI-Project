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
        
def augment_dataset(data_dir="data"):
    """
    Loads the train dataset, generates counterfactual text for each row, 
    tokenizes it into `cf_input_ids`, and saves the augmented dataset.
    """
    print("Loading train dataset for CDA augmentation...")
    train_ds = load_from_disk(os.path.join(data_dir, "train"))
    
    print("Initializing Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    def add_cf_columns(batch):
        # 1. Swap the text
        cf_texts = [swap_identities(text) for text in batch["text"]]
        batch["cf_text"] = cf_texts
        
        # 2. Tokenize the swapped text
        tokens = tokenizer(cf_texts, padding="max_length", truncation=True, max_length=128)
        batch["cf_input_ids"] = tokens["input_ids"]
        batch["cf_attention_mask"] = tokens["attention_mask"]
        
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
    args = parser.parse_args()
    
    if args.test or not args.augment:
        run_tests()
    if args.augment:
        augment_dataset()
