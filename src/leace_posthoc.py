import os
import argparse
import torch
import torch.nn as nn
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from concept_erasure import LeaceEraser

class LeaceDebiasedModel(nn.Module):
    """
    Wraps a trained Transformer classifier and applies post-hoc closed-form LEACE 
    concept erasure to scrub residual demographic/identity signals from [CLS] embeddings.
    """
    def __init__(self, base_model, eraser):
        super().__init__()
        self.base_model = base_model
        self.eraser = eraser
        self.config = base_model.config

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Extract base backbone depending on architecture (DistilBERT or RoBERTa/BERT)
        if hasattr(self.base_model, "distilbert"):
            backbone = self.base_model.distilbert
            classifier = self.base_model.classifier
            pre_classifier = getattr(self.base_model, "pre_classifier", nn.Identity())
            dropout = getattr(self.base_model, "dropout", nn.Identity())
            
            outputs = backbone(input_ids=input_ids, attention_mask=attention_mask)
            cls_repr = outputs.last_hidden_state[:, 0, :]
            
            # Apply LEACE projection to scrub linear identity correlations
            scrubbed_cls = self.eraser(cls_repr)
            
            pooled = pre_classifier(scrubbed_cls)
            pooled = nn.ReLU()(pooled)
            pooled = dropout(pooled)
            logits = classifier(pooled)
            
        elif hasattr(self.base_model, "roberta"):
            backbone = self.base_model.roberta
            classifier = self.base_model.classifier
            
            outputs = backbone(input_ids=input_ids, attention_mask=attention_mask)
            cls_repr = outputs.last_hidden_state[:, 0, :]
            
            # Apply LEACE projection
            scrubbed_cls = self.eraser(cls_repr)
            
            # RoBERTa classification head
            x = classifier.dense(scrubbed_cls)
            x = torch.tanh(x)
            x = classifier.dropout(x)
            logits = classifier.out_proj(x)
        else:
            # Fallback direct forward pass if unknown backbone
            logits = self.base_model(input_ids=input_ids, attention_mask=attention_mask).logits
            
        return logits

def extract_embeddings_and_identities(model, dataset, device, max_samples=5000):
    """
    Extracts [CLS] representations and corresponding 8-dim identity masks 
    from a calibration/validation set.
    """
    model.eval()
    embeddings = []
    identities = []
    
    identity_cols = [
        "male_mask", "female_mask", "black_mask", "white_mask", 
        "muslim_mask", "christian_mask", "jewish_mask", "homosexual_bisexual_mask"
    ]
    
    n_samples = min(len(dataset), max_samples)
    subset = dataset.select(range(n_samples))
    batch_size = 32
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = subset[i:i + batch_size]
            input_ids = torch.tensor(batch["input_ids"]).to(device)
            attention_mask = torch.tensor(batch["attention_mask"]).to(device)
            
            if hasattr(model, "distilbert"):
                outputs = model.distilbert(input_ids=input_ids, attention_mask=attention_mask)
            elif hasattr(model, "roberta"):
                outputs = model.roberta(input_ids=input_ids, attention_mask=attention_mask)
            elif hasattr(model, "bert"):
                outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                
            cls_vec = outputs.last_hidden_state[:, 0, :].cpu()
            embeddings.append(cls_vec)
            
            # Identity masks
            mask_batch = []
            for j in range(len(batch["input_ids"])):
                row_mask = [float(batch[col][j]) if col in batch else 0.0 for col in identity_cols]
                mask_batch.append(row_mask)
            identities.append(torch.tensor(mask_batch))
            
    embeddings = torch.cat(embeddings, dim=0)
    identities = torch.cat(identities, dim=0)
    return embeddings, identities

def fit_and_save_leace(model_dir, data_dir="data_identity", output_file=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading trained model from {model_dir} on {device}...")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    
    val_path = os.path.join(data_dir, "validation")
    print(f"Loading calibration dataset from {val_path}...")
    val_ds = load_from_disk(val_path)
    
    print("Extracting hidden embeddings & identity targets...")
    X, Z = extract_embeddings_and_identities(model, val_ds, device)
    
    print(f"Fitting closed-form LEACE Eraser (Embeddings: {X.shape}, Target Identities: {Z.shape})...")
    eraser = LeaceEraser.fit(X, Z)
    
    if output_file is None:
        output_file = os.path.join(model_dir, "leace_eraser.pt")
        
    torch.save(eraser, output_file)
    print(f"LEACE Concept Eraser successfully fitted and saved to: {output_file}")
    return eraser

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit closed-form LEACE Concept Erasure on trained model")
    parser.add_argument("--model_dir", type=str, default="results_identity/M2/final_model", help="Path to trained model")
    parser.add_argument("--data_dir", type=str, default="data_identity", help="Directory containing dataset splits")
    parser.add_argument("--output", type=str, default=None, help="Path to save leace_eraser.pt")
    args = parser.parse_args()
    
    fit_and_save_leace(model_dir=args.model_dir, data_dir=args.data_dir, output_file=args.output)

