# NASA C-MAPSS — Remaining Useful Life (RUL) Prediction

> Comparative study of Classical Machine Learning vs. Deep Learning for predictive maintenance on turbofan engines using the NASA C-MAPSS benchmark dataset.

**Universidad LEAD · BCD-6210 Advanced Data Mining · IC 2026**  
**Advisor:** Dr. Juan Murillo-Morera

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Project Structure](#3-project-structure)
4. [Pipeline](#4-pipeline)
5. [Models & Results](#5-models--results)
6. [Streamlit App — How to Use](#6-streamlit-app--how-to-use)
7. [Installation](#7-installation)
8. [Notebooks](#8-notebooks)
9. [SHAP Interpretability](#9-shap-interpretability)
10. [Paper](#10-paper)

---

## 1. Project Overview

This project builds an end-to-end predictive maintenance system that estimates the **Remaining Useful Life (RUL)** of aircraft turbofan engines. The goal is to predict, at any given cycle, how many cycles remain before a motor reaches failure — enabling proactive, data-driven maintenance decisions.

**Key contributions:**

- Full EDA pipeline: distributions, degradation curves, correlations, RUL evolution, and 3-D condition analysis.
- Standardized preprocessing with operating-condition normalization and piece-wise linear RUL labeling (capped at `MAX_RUL = 125` cycles).
- Classical ML baseline: Linear Regression, SVR, Random Forest, XGBoost — with binary failure classification.
- Deep Learning suite: LSTM, GRU, TCN, and Transformer — all hyperparameter-tuned with **Optuna (800 trials each)**.
- **Ensemble DL** combining all four architectures (best overall performance).
- SHAP analysis to identify the most informative sensors across models.
- Interactive **Streamlit dashboard** for real-time RUL prediction from uploaded test files.

---

## 2. Dataset

The [NASA C-MAPSS dataset](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository) (Commercial Modular Aero-Propulsion System Simulation) contains run-to-failure time series from simulated turbofan engines under varying operating conditions and fault modes.

| Sub-dataset | Train units | Test units | Op. conditions | Fault modes |
|-------------|-------------|------------|----------------|-------------|
| FD001       | 100         | 100        | 1              | 1           |
| FD002       | 260         | 259        | 6              | 1           |
| FD003       | 100         | 100        | 1              | 2           |
| FD004       | 249         | 248        | 6              | 2           |

**Columns:** `unit_id`, `cycle`, 3 operational settings, 21 sensor readings.

**Dropped sensors** (near-zero variance): `sensor_1`, `sensor_5`, `sensor_6`, `sensor_10`, `sensor_16`, `sensor_18`, `sensor_19`.

**Key sensors used:**
| Sensor | Physical meaning |
|--------|-----------------|
| sensor_2  | Total temperature at fan inlet (°R) |
| sensor_3  | Total temperature at LPC outlet (°R) |
| sensor_4  | Total temperature at HPC outlet (°R) |
| sensor_7  | Total pressure at HPC outlet (psia) |
| sensor_8  | Physical fan speed (rpm) |
| sensor_9  | Physical core speed (rpm) |
| sensor_11 | Static pressure at HPC outlet (psia) — **top predictor** |
| sensor_12 | Ratio of fuel flow to Ps30 |
| sensor_13 | Corrected fan speed (rpm) |
| sensor_14 | Corrected core speed (rpm) |
| sensor_15 | Bypass ratio |
| sensor_17 | Bleed enthalpy |
| sensor_20 | Hpt coolant bleed |
| sensor_21 | Lpt coolant bleed |

Raw data files go in `data/raw/`:
```
train_FD001.txt  test_FD001.txt  RUL_FD001.txt
train_FD002.txt  test_FD002.txt  RUL_FD002.txt
train_FD003.txt  test_FD003.txt  RUL_FD003.txt
train_FD004.txt  test_FD004.txt  RUL_FD004.txt
```

---

## 3. Project Structure

```
NASA_C-MAPSS_Data/
├── app/
│   └── app.py                     # Streamlit dashboard
├── data/
│   ├── raw/                       # NASA C-MAPSS txt files (not tracked)
│   └── processed/                 # Parquet files + optuna_results.json
├── models/
│   ├── *.pkl                      # Classical ML models (sklearn/XGBoost)
│   └── *_optuna_FD00X.pt          # Deep Learning weights (PyTorch)
├── notebooks/
│   ├── eda_basico.py
│   ├── eda_distribuciones.py
│   ├── eda_degradaciones.py
│   ├── eda_correlaciones.py
│   ├── eda_rul.py
│   ├── eda_3d.py
│   ├── preprocessing_pipeline.py
│   ├── classical_ml_pipeline.py
│   ├── deep_learning_pipeline.py
│   └── shap-pipeline.py
├── paper/
│   └── figures/
│       ├── *.png                  # All publication-quality figures
│       ├── documento/
│       │   └── NASA_C_MAPSS_RUL_PREDICTION.pdf   # Full academic paper
│       └── notebooks/             # Notebook-level figure subfolders
├── src/
│   ├── config.py                  # Global constants and paths
│   ├── data_loader.py             # Dataset loading utilities
│   ├── preprocessing.py           # Operating-condition normalization
│   ├── feature_engineering.py     # Rolling stats, trend features
│   ├── shap_analysis.py           # SHAP computation helpers
│   ├── visualization.py           # Reusable plotting functions
│   ├── utils.py                   # Misc utilities
│   └── models/
│       ├── classical.py           # sklearn / XGBoost wrappers
│       ├── deep_learning.py       # FlexibleModel (LSTM/GRU/TCN/Transformer)
│       ├── train.py               # Training loop with early stopping
│       └── evaluate.py            # Metrics: RMSE, MAE, R², NASA Score
├── requirements.txt
└── setup.py
```

---

## 4. Pipeline

```
Raw .txt files
    │
    ▼
[1] EDA
    ├── Basic stats & sensor distributions
    ├── Degradation curves per unit
    ├── Sensor correlations with RUL
    ├── RUL distribution analysis
    └── 3-D operating condition clusters
    │
    ▼
[2] Preprocessing
    ├── Operating-condition normalization (z-score per condition cluster)
    ├── Piece-wise linear RUL labeling (cap at MAX_RUL = 125)
    ├── Drop constant / low-variance sensors
    └── MinMax scaling → Parquet export
    │
    ▼
[3] Classical ML (FD001)
    ├── Linear Regression
    ├── SVR (RBF kernel)
    ├── Random Forest (reg + clf)
    └── XGBoost (reg + clf)
    │
    ▼
[4] Deep Learning (FD001–FD004)
    ├── Sequence length = 80 cycles
    ├── LSTM  ─┐
    ├── GRU   ─┤ → Optuna (800 trials each)
    ├── TCN   ─┤    hidden_size, num_layers, dropout,
    └── Transformer─┘    d_model, nhead, lr, batch_size
    │
    ▼
[5] Ensemble DL
    └── Mean of LSTM + GRU + TCN + Transformer predictions
    │
    ▼
[6] SHAP Analysis
    ├── Feature importance (XGBoost + Random Forest)
    └── Beeswarm / waterfall / dependence plots
    │
    ▼
[7] Streamlit App
    └── Upload test file → select model → real-time RUL predictions
```

---

## 5. Models & Results

### 5.1 Classical ML — FD001

| Model            | RMSE  | MAE   | R²    | NASA Score |
|------------------|-------|-------|-------|------------|
| Linear Regression| 21.91 | 17.61 | 0.722 | 1318.9     |
| SVR              | 19.61 | 13.82 | 0.777 | 1624.9     |
| Random Forest    | 17.92 | 12.92 | 0.814 | **886.4**  |
| XGBoost          | 18.31 | 13.39 | 0.806 | 1058.1     |

**Binary classification (critical if RUL ≤ 30 cycles):**

| Model         | Accuracy | Precision | Recall | F1   | AUC-ROC |
|---------------|----------|-----------|--------|------|---------|
| Random Forest | 0.910    | 0.900     | 0.720  | 0.800| **0.983**|
| XGBoost       | 0.910    | 0.900     | 0.720  | 0.800| 0.980   |

### 5.2 Deep Learning — All datasets

| Dataset | LSTM  | GRU   | TCN   | Transformer | **Ensemble** |
|---------|-------|-------|-------|-------------|--------------|
| FD001   | 14.24 | 14.89 | 14.39 | 13.16       | **11.97**    |
| FD002   | 12.80 | 14.86 | 13.43 | 14.28       | **11.95**    |
| FD003   | 15.56 | 13.71 | 16.82 | 13.34       | **13.46**    |
| FD004   | 17.92 | 18.66 | 21.48 | 17.77       | **17.32**    |

*Values shown are RMSE (cycles). Lower is better.*

**Best overall:** Ensemble DL on FD002 — RMSE = **11.95**, R² = **0.906**

---

## 6. Streamlit App — How to Use

### Quick start

```bash
# From the project root
streamlit run app/app.py
```

The app runs at `http://localhost:8501`.

### Sidebar configuration

| Control | Options | Description |
|---------|---------|-------------|
| Sub-dataset | FD001 · FD002 · FD003 · FD004 | Selects which trained models to use |
| Prediction mode | ML Clasico · Deep Learning · Ensemble DL | Selects the inference backend |
| Modelo (ML) | linear_regression · svr · random_forest · xgboost | Only shown in ML mode |
| Modelo DL | lstm · gru · tcn · transformer | Only shown in single-DL mode |

### Tabs

| Tab | Content |
|-----|---------|
| 🔮 Prediccion | Upload a test file and get RUL predictions for every engine |
| 📊 ML Clasico | Embedded results table + bar chart for classical models |
| 🧠 Deep Learning | RMSE/R² table + heatmap across all datasets and architectures |
| 🔍 SHAP | Top sensor importances for XGBoost and Random Forest |
| ℹ️ Acerca de | Project summary, key metrics, course information |

### Prediction tab — step by step

1. **Select sub-dataset** in the sidebar (e.g., `FD001`).
2. **Choose prediction mode:**
   - *ML Clasico (rapido)* — instant inference using sklearn/XGBoost models.
   - *Deep Learning (preciso)* — ~5 s, runs one PyTorch model.
   - *Ensemble DL (mejor)* — ~15 s, averages all four DL models.
3. **Upload the test file** (`test_FD001.txt`, space-separated, no header).
4. The app automatically loads `data/processed/test_FD001.parquet` for ML mode or normalizes the uploaded file on-the-fly for DL mode.
5. If `data/raw/RUL_FD001.txt` is present, the app shows:
   - Full metrics: RMSE, MAE, accuracy, confusion matrix.
   - Scatter plot predicted vs. real RUL.
   - False-negative and false-positive tables.
6. **Download** the full prediction table as CSV.

### Input file format

The test file must be space-separated with **26 columns** and no header:

```
unit_id  cycle  op1  op2  op3  s1  s2  ...  s21
1        1      -0.0  -0.0  100.0  518.67  ...
1        2      ...
```

This matches the original NASA C-MAPSS format exactly.

### Prediction modes comparison

| Mode | Speed | Accuracy | Use case |
|------|-------|----------|----------|
| ML Clasico | ⚡ < 1 s | Good (R² ≈ 0.81) | Quick sanity check |
| Deep Learning | 🔄 ~5 s | Very good (R² ≈ 0.89) | Single-model precision |
| Ensemble DL | 🔄 ~15 s | Best (R² ≈ 0.91) | Production / paper results |

---

## 7. Installation

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0 (CPU or CUDA)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/smartinezabarproyectos/NASA_C-MAPSS_RUL_PREDICTION.git
cd NASA_C-MAPSS_RUL_PREDICTION

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install PyTorch (CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 5. Place the NASA C-MAPSS raw files in data/raw/
#    (train_FD00X.txt, test_FD00X.txt, RUL_FD00X.txt)

# 6. Run the preprocessing pipeline to generate processed parquets
python notebooks/preprocessing_pipeline.py

# 7. Launch the app
streamlit run app/app.py
```

### Run the full ML/DL training (optional)

```bash
# Classical ML (FD001)
python notebooks/classical_ml_pipeline.py

# Deep Learning (all datasets, ~hours with Optuna)
python notebooks/deep_learning_pipeline.py

# SHAP analysis
python notebooks/shap-pipeline.py
```

> **Note:** Pre-trained model weights (`.pt`, `.pkl`) and processed parquets are already included in the repository so you can skip training and go straight to the app.

---

## 8. Notebooks

Each notebook is a self-contained Python script that produces figures saved to `paper/figures/`.

| File | Description |
|------|-------------|
| `eda_basico.py` | Basic dataset statistics, missing values, unit counts |
| `eda_distribuciones.py` | Sensor distribution histograms and KDE plots |
| `eda_degradaciones.py` | Degradation trajectories per unit and per sensor |
| `eda_correlaciones.py` | Pearson/Spearman correlation heatmaps with RUL |
| `eda_rul.py` | RUL distribution, cap effect, early vs. late lifecycle analysis |
| `eda_3d.py` | 3-D PCA / operating-condition cluster visualization |
| `preprocessing_pipeline.py` | Full preprocessing: normalize → scale → export parquet |
| `classical_ml_pipeline.py` | Train, tune (GridSearchCV), evaluate, and save classical models |
| `deep_learning_pipeline.py` | Optuna hyperparameter search + final training for LSTM/GRU/TCN/Transformer |
| `shap-pipeline.py` | SHAP values, beeswarm, waterfall, and dependence plots |

---

## 9. SHAP Interpretability

SHAP (SHapley Additive exPlanations) was applied to XGBoost and Random Forest to explain which sensors drive RUL predictions.

**Top 5 sensors by mean |SHAP| value:**

| Rank | Sensor | Physical meaning | SHAP XGB | SHAP RF |
|------|--------|-----------------|----------|---------|
| 1 | sensor_11 | Static pressure at HPC outlet | 9.85 | 15.83 |
| 2 | sensor_9  | Physical core speed (rpm)     | 7.14 | 8.91  |
| 3 | sensor_4  | Total temperature at HPC outlet| 5.88 | 4.73  |
| 4 | sensor_14 | Corrected core speed (rpm)    | 4.31 | 2.10  |
| 5 | sensor_12 | Fuel flow / Ps30 ratio        | 4.11 | 3.60  |

**Key finding:** `sensor_11` (HPC static pressure) is the dominant degradation indicator, consistent across both XGBoost and Random Forest — which confirms the finding is algorithm-agnostic and physically meaningful (HPC pressure drops as blade wear increases).

---

## 10. Paper

The full academic paper is available in:

```
paper/figures/documento/NASA_C_MAPSS_RUL_PREDICTION.pdf
```

It covers the complete methodology, experimental setup, results analysis, and conclusions following standard academic structure (Abstract, Introduction, Related Work, Methodology, Experiments, Results, Discussion, Conclusion).

---

## Configuration Reference

All global constants are defined in `src/config.py`:

| Constant | Value | Meaning |
|----------|-------|---------|
| `MAX_RUL` | 125 | Piece-wise linear RUL cap (cycles) |
| `CLASSIFICATION_W` | 30 | Binary failure threshold (cycles) |
| `SEQUENCE_LENGTH` | 80 | Input window for DL models (cycles) |
| `BATCH_SIZE` | 64 | Training batch size |
| `EPOCHS` | 100 | Max training epochs |
| `EARLY_STOPPING_PATIENCE` | 10 | Early stopping patience |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data processing | pandas, NumPy, pyarrow |
| Classical ML | scikit-learn, XGBoost |
| Deep Learning | PyTorch |
| Hyperparameter tuning | Optuna |
| Interpretability | SHAP |
| Visualization | Plotly, Matplotlib, Seaborn |
| Web app | Streamlit |
| Packaging | setuptools |
