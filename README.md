# Mitigating Identity Bias in NLP Toxicity Classifiers

This project aims to reproduce, measure, and fix the "identity mention shortcut" bias in standard fine-tuned DistilBERT toxicity models using the Jigsaw Civil Comments dataset.

## Setup

1. Create a virtual environment and install dependencies:
   ```bash
   pip install torch transformers datasets scikit-learn matplotlib accelerate
   ```

2. To prepare data:
   ```bash
   python src/data_prep.py
   ```

## Google Colab Instructions

1. Upload the entire project directory to Google Drive.
2. Open `notebooks/Colab_Runner.ipynb` in Google Colab.
3. Mount Google Drive and run the cells to execute training scripts (`src/train_baseline.py`, `src/train_debias.py`, etc.).
