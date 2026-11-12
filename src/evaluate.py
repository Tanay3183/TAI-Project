import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ToxicityEvaluator:
    def __init__(self, m0_path, debiased_path, model_name="distilbert-base-uncased", use_leace=True):
        print(f"Loading tokenizer ({model_name})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print("Loading M0 (Baseline) model...")
        self.m0 = AutoModelForSequenceClassification.from_pretrained(m0_path)
        self.m0.eval()
        
        print("Loading Debiased model...")
        self.debiased = AutoModelForSequenceClassification.from_pretrained(debiased_path)
        self.debiased.eval()
        
        self.use_leace = False
        leace_path = os.path.join(debiased_path, "leace_eraser.pt")
        if use_leace and os.path.exists(leace_path):
            try:
                from src.leace_posthoc import LeaceDebiasedModel
                eraser = torch.load(leace_path, map_location="cpu", weights_only=False)
                self.debiased_leace = LeaceDebiasedModel(self.debiased, eraser)
                self.debiased_leace.eval()
                self.use_leace = True
                print("Loaded LEACE Concept Eraser for debiased inference.")
            except Exception as e:
                print(f"Notice: Could not attach LEACE wrapper: {e}")
                
    def _tokenize(self, text):
        # Head + Tail Truncation
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        max_len = 128
        cls_id = self.tokenizer.cls_token_id or 101
        sep_id = self.tokenizer.sep_token_id or 102
        
        if len(tokens) <= (max_len - 2):
            input_ids = [cls_id] + tokens + [sep_id]
        else:
            head = tokens[:63]
            tail = tokens[-63:]
            input_ids = [cls_id] + head + tail + [sep_id]
            
        pad_len = max_len - len(input_ids)
        mask = [1] * len(input_ids) + [0] * pad_len
        input_ids = input_ids + [self.tokenizer.pad_token_id or 0] * pad_len
        
        return {
            "input_ids": torch.tensor([input_ids]),
            "attention_mask": torch.tensor([mask])
        }

    def predict(self, text):
        inputs = self._tokenize(text)
        
        with torch.no_grad():
            # M0 Prediction
            m0_outputs = self.m0(**inputs)
            m0_logits = m0_outputs.logits
            if m0_logits.ndim > 1 and m0_logits.shape[1] > 1:
                m0_logits = m0_logits[:, 0]
            m0_prob = torch.sigmoid(m0_logits).squeeze().item()
            
            # Debiased Prediction
            if self.use_leace:
                deb_logits = self.debiased_leace(**inputs)
            else:
                deb_outputs = self.debiased(**inputs)
                deb_logits = deb_outputs.logits
                
            if deb_logits.ndim > 1 and deb_logits.shape[1] > 1:
                deb_logits = deb_logits[:, 0]
            deb_prob = torch.sigmoid(deb_logits).squeeze().item()
            
        return m0_prob, deb_prob
