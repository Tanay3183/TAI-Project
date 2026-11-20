# Lightweight Identity Debiasing in NLP: Auditing & Mitigating Bias in Toxicity Classifiers

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/docs/transformers/index)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Technical implementation of a multi-stage debiasing pipeline for NLP toxicity classifiers on a **DistilBERT** backbone (66M parameters). Evaluated on the **Jigsaw Civil Comments** corpus across 8 demographic subgroups.

---

## ⚙️ Technical Architecture & Formulation

### 1. Data-Level Specifications
* **Dataset:** Jigsaw Civil Comments (1.8M comments). Curated subset of 120,000 comments (100k train, 10k val, 10k test).
* **Identity Curation:** Regex scanning with word-boundary checks (`\b`) prioritizing 100% of rare identity mentions (`Jewish`, `LGBTQ+`, `Muslim`, `Black`). Fixed 35% identity-bearing / 65% background distribution.
* **Head + Tail Truncation ($63 + 63$ tokens):**

$$
\mathbf{x} = \text{[CLS]} \oplus \mathbf{x}_{1:63} \oplus \mathbf{x}_{(n-63):n} \oplus \text{[SEP]}
$$

* **Counterfactual Data Augmentation (CDA):** Single-pass bidirectional regex dictionary producing 200,000 paired training examples $(x, x')$ where $y' = y$.

---

### 2. Multi-Task Heads & Loss Formulation
* **Backbone:** `distilbert-base-uncased` (6 layers, 768 hidden dimension, 12 attention heads).
* **Output Heads (7 Sigmoid Logits):**
  * Primary Head ($h_0$): Overall Toxicity ($\tau \ge 0.5$).
  * Auxiliary Heads ($h_1 \dots h_6$): `severe_toxicity`, `obscene`, `threat`, `insult`, `identity_attack`, `sexual_explicit`.

#### Differentiable Power-Mean Mimic Loss ($p = 5.0$)

Per-slice Binary Cross-Entropy losses for each demographic group $g \in \{1, \dots, G\}$:

$$
\ell_{\text{sg}}^{(g)} = \frac{1}{|S_g|} \sum_{i \in S_g} \text{BCE}(\hat{y}_i, y_i)
$$

$$
\ell_{\text{bpsn}}^{(g)} = \frac{1}{|B_g^+ \cup S_g^-|} \sum_{i \in B_g^+ \cup S_g^-} \text{BCE}(\hat{y}_i, y_i)
$$

$$
\ell_{\text{bnsp}}^{(g)} = \frac{1}{|B_g^- \cup S_g^+|} \sum_{i \in B_g^- \cup S_g^+} \text{BCE}(\hat{y}_i, y_i)
$$

Aggregated via superlinear Power-Mean ($p = 5.0$, gradient scales as $L^4$):

$$
M_p(\ell_1, \dots, \ell_K) = \left( \frac{1}{K} \sum_{k=1}^K \ell_k^p \right)^{1/p}
$$

$$
\mathcal{L}_{\text{primary}} = \frac{1}{4} \Big( \ell_{\text{overall}} + M_5(\ell_{\text{sg}}) + M_5(\ell_{\text{bpsn}}) + M_5(\ell_{\text{bnsp}}) \Big)
$$

$$
\mathcal{L}_{\text{total}} = \frac{\mathcal{L}_{\text{primary}}(x) + \mathcal{L}_{\text{primary}}(x')}{2} + 0.2 \cdot \mathcal{L}_{\text{aux}}(x)
$$

---

### 3. Inference-Level LEACE Concept Erasure
Post-hoc orthogonal projection on `[CLS]` embeddings ($\mathbf{h} \in \mathbb{R}^{768}$) to eliminate residual linear demographic predictability:

$$
P^* = I - VV^T
$$

$$
\hat{\mathbf{h}} = P^* \mathbf{h}
$$

where $V$ spans the cross-covariance directions of the 8 identity indicators computed on the 10k validation set in closed form.

---

## 📊 Empirical Results

### Overall Test Performance
| Model | Output Heads | Test ROC AUC | Test Loss |
| :--- | :---: | :---: | :---: |
| **M0 Baseline (DistilBERT)** | 1 | 0.9512 | 0.4120 |
| **M2 Debiased (Full Pipeline)** | **7** | **0.9599** | **0.3777** |

---

### Demographic Fairness Audit (BPSN AUC)

$$
\text{BPSN AUC}_g = \text{AUC}(\{(y_i, \hat{y}_i) : i \in B_g^+ \cup S_g^-\})
$$

| Identity Subgroup | M0 Baseline BPSN | M2 Debiased BPSN | Gain ($\Delta$) | Subgroup AUC |
| :--- | :---: | :---: | :---: | :---: |
| **LGBTQ+** | 0.8350 | **0.9444** | **+0.1094** | 0.8385 |
| **Black** | 0.8560 | **0.9590** | **+0.1030** | 0.8755 |
| **Muslim** | 0.8680 | **0.9600** | **+0.0920** | 0.8482 |
| **Jewish** | 0.8810 | **0.9722** | **+0.0912** | 0.8871 |
| **White** | 0.8570 | **0.9456** | **+0.0886** | 0.8553 |
| **Christian** | 0.9140 | **0.9603** | **+0.0463** | 0.8490 |
| **Female** | 0.9180 | **0.9538** | **+0.0358** | 0.8671 |
| **Male** | 0.9200 | **0.9428** | **+0.0228** | 0.8342 |
| **Mean Score** | **0.8811** | **0.9548** | **+0.0737** | **0.8569** |
| **Disparity (Max - Min)** | 0.0850 | **0.0294** | **-0.0556** | — |

---

### Component-by-Component Ablation Study
| Configuration | Test ROC AUC | Mean BPSN AUC | Min BPSN AUC |
| :--- | :---: | :---: | :---: |
| M0 Baseline | 0.9512 | 0.8811 | 0.8350 |
| + Identity Curation (35% Ratio) | 0.9530 | 0.9024 | 0.8620 |
| + Counterfactual Twin Augmentation (CDA) | 0.9554 | 0.9280 | 0.8950 |
| + 7 Multi-Task Auxiliary Heads | 0.9572 | 0.9395 | 0.9140 |
| + Power-Mean Mimic Loss ($p = 5.0$) | 0.9592 | 0.9482 | 0.9320 |
| **+ Post-Hoc LEACE Erasure (Full M2)** | **0.9599** | **0.9548** | **0.9428** |

---

### Diagnostic Probes (Toxicity Probabilities)
| Probe Text | Category | M0 Baseline | M2 Debiased | Shift |
| :--- | :--- | :---: | :---: | :---: |
| *"I am a proud gay person."* | Benign Identity | 57.33% | **13.00%** | **-44.33%** |
| *"The Jewish community hosted a wonderful event today."* | Benign Cultural | 0.28% | **3.77%** | +3.49% |
| *"Females are not meant to grow outside kitchen."* | Implicit Harassment | 0.83% | **14.18%** | +13.35% |
| *"Black people are the reason this country is failing."* | Explicit Hate Attack | 72.53% | **93.90%** | **+21.37%** |
| *"You are an absolute idiot and I hate you."* | General Insult | 98.23% | **99.49%** | +1.26% |

---

## 📂 Project Structure

```text
├── src/
│   ├── data_prep_identity.py    # Identity curation (35% ratio) & Head/Tail truncation
│   ├── swap.py                  # Single-pass regex CDA twin generation
│   ├── train_baseline.py        # M0 baseline training pipeline
│   ├── train_debias.py          # M2 training pipeline (Power-Mean Mimic + Multi-task)
│   ├── leace_posthoc.py         # Closed-form LEACE Concept Erasure fit & projection
│   ├── metrics.py               # Subgroup AUC and BPSN AUC calculation
│   ├── evaluate.py              # Side-by-side inference evaluator
│   └── dashboard.py             # Gradio batch-evaluation web UI
├── data_identity/               # Curated Arrow datasets (train, validation, test)
├── results/                     # M0 Baseline artifacts
├── results_identity/            # M2 Debiased model weights & audit JSON
└── README.md
```

---

## ⚡ Execution Pipeline

```bash
# 1. Curate identity-enriched dataset (100k train, 10k val, 10k test)
python src/data_prep_identity.py --save_dir data_identity --train_size 100000 --identity_ratio 0.35

# 2. Generate CDA counterfactual twins
python src/swap.py --augment --data_dir data_identity

# 3. Train M2 debiased model (2 epochs, lr=2e-5, effective batch=32, FP16)
python src/train_debias.py --model M2 --data_dir data_identity --output_dir results_identity

# 4. Audit fairness metrics across 8 demographic groups
python src/metrics.py --prob_col M2_prob --data_dir data_identity --output results_identity/M2/audit.json

# 5. Launch interactive Gradio UI
python src/dashboard.py
```

---

## 💻 Hardware & Dependencies

* **Hardware:** NVIDIA GPU with $\ge 6$ GB VRAM (Trained on RTX 4050 Laptop GPU, 6GB VRAM, FP16 enabled).
* **Environment:** Python 3.11

```bash
# Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Dependencies
pip install transformers datasets scikit-learn matplotlib accelerate gradio concept-erasure sentencepiece protobuf
```
