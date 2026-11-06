import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ToxicityEvaluator:
    def __init__(self, m0_path, debiased_path):
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        print("Loading M0 (Baseline) model...")
        self.m0 = AutoModelForSequenceClassification.from_pretrained(m0_path)
        self.m0.eval()
        
        print("Loading Debiased model...")
        self.debiased = AutoModelForSequenceClassification.from_pretrained(debiased_path)
        self.debiased.eval()
        
    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        
        with torch.no_grad():
            # M0 Prediction
            m0_outputs = self.m0(**inputs)
            m0_prob = torch.sigmoid(m0_outputs.logits).item()
            
            # Debiased Prediction
            deb_outputs = self.debiased(**inputs)
            deb_prob = torch.sigmoid(deb_outputs.logits).item()
            
        return m0_prob, deb_prob
