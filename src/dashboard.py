import sys
import os

# Ensure the src module can be imported
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import gradio as gr
from src.evaluate import ToxicityEvaluator

import pandas as pd

# Point to the saved models (auto-detect results_identity or results)
m0_path = os.path.join("results_identity", "M0_baseline", "final_model")
if not os.path.exists(m0_path):
    m0_path = os.path.join("results", "M0_baseline", "final_model")

debiased_path = os.path.join("results_identity", "M2", "final_model") 
if not os.path.exists(debiased_path):
    debiased_path = os.path.join("results", "M2", "final_model") 

print("Initializing Evaluator (this might take a few seconds to load both models into memory)...")
evaluator = ToxicityEvaluator(m0_path, debiased_path)

def analyze_texts(text_block):
    # Split by newlines and remove empty lines
    sentences = [line.strip() for line in text_block.split('\n') if line.strip()]
    
    if not sentences:
        return pd.DataFrame(columns=["Sentence", "Baseline (M0) Toxicity", "Debiased (M2) Toxicity"])
        
    results = []
    for sentence in sentences:
        m0_prob, deb_prob = evaluator.predict(sentence)
        results.append({
            "Sentence": sentence,
            "Baseline (M0) Toxicity": f"{m0_prob:.2%}",
            "Debiased (M2) Toxicity": f"{deb_prob:.2%}"
        })
        
    return pd.DataFrame(results)

demo = gr.Interface(
    fn=analyze_texts,
    inputs=gr.Textbox(lines=10, placeholder="Enter sentences here (one per line)...", label="Input Sentences"),
    outputs=gr.Dataframe(label="Toxicity Predictions", headers=["Sentence", "Baseline (M0) Toxicity", "Debiased (M2) Toxicity"]),
    title="Toxicity Classifier: Bias Mitigation Dashboard",
    description="Compare how the Baseline model and the Debiased model score exact same sentences side-by-side. Enter multiple sentences (one per line) to process them in batch!",
    examples=[
        ["I am a proud gay man.\nFemales are not meant to grow outside kitchen.\nAre you a muslim or hindu?\nYou are an absolute idiot and I hate you.\nSome people in the black community feel ignored."]
    ]
)

if __name__ == "__main__":
    print("\nLaunching Dashboard...")
    demo.launch(inbrowser=True)
