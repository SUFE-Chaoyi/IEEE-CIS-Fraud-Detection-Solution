
#  IEEE-CIS Advanced Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![XGBoost](https://img.shields.io/badge/Model-XGBoost%20%7C%20LightGBM-green) ![Status](https://img.shields.io/badge/Status-Completed-success) ![AUC](https://img.shields.io/badge/OOF%20AUC-0.9550-orange)

An industrial-level fraud detection pipeline capable of handling large-scale transaction data with severe class imbalance (3.5% positive rate). This solution implements advanced feature engineering, dynamic model ensemble, and business-oriented threshold optimization strategies.

> **Key Achievement**: Improved AUC from **0.9267** (Baseline) to **0.9550** (Final Ensemble), reducing potential monthly financial loss by **34%** through cost-sensitive learning.

---

##  Business Performance & Key Metrics

Instead of just pursuing high AUC, this project focuses on **Cost Minimization** and **Recall Maximization** under strict FPR constraints.

| Metric | XGB Baseline | **Final Solution (Ensemble)** | Improvement |
| :--- | :--- | :--- | :--- |
| **ROC-AUC** | 0.9267 | **0.9550** |  **+0.0283** |
| **Recall (@FPR $\le$ 5%)** | 76.2% | **87.4%** |  **+11.2%** |
| **Est. Monthly Loss** | \$1,050,200 | **\$694,900** | **Savings: \$355k** |
| **Model Stability (Fold 0)**| 0.8952 | **0.9308** | Stronger anti-drift capability |

*Note: The cost simulation assumes \$10 administrative cost per False Positive and \$1000 loss per False Negative.*

---

##  Solution Architecture

### 1. Preprocessing & Memory Optimization
- Implemented `reduce_mem_usage` to downcast data types, reducing memory footprint by **50%+**.
- Standardized time-delta columns (`D1`-`D15`) to remove the effect of the increasing time axis, making the model robust to future data.

### 2. Feature Engineering (The "Magic")
Feature engineering was the main driver of performance improvement.
- **UID Construction**: Created a virtual user ID using `card1 + addr1 + D1`.
- **Aggregations**: Calculated `mean`, `std`, and `count` of transaction amounts and `C` features grouped by `UID`.
    - *Insight*: High-frequency transactions by the same UID in a short period are strong indicators of fraud (Shapley values confirmed `uid_C13_ct` as the top feature).
- **Frequency Encoding**: Applied to high-cardinality categorical features.

### 3. Model Training & Validation
- **Time-Consistent Validation**: Used `GroupKFold` based on Months (`DT_M`) to simulate the production environment (predicting future months based on past data) and prevent time leakage.
- **Dynamic Ensemble**:
    - Trained **XGBoost** and **LightGBM** separately for each fold.
    - Implemented a **Dynamic Weight Search** (Grid Search from 0.0 to 1.0) to find the optimal blending ratio for each time slice.
    - *Result*: LightGBM dominated in early months, while XGBoost contributed significantly in later months (e.g., Fold 4), proving the ensemble's robustness.

### 4. Business Strategy Layer
Post-processing strategies to translate probabilities into decisions:
- **Strategy 1 (Min Cost)**: Dynamic threshold search to minimize `FP * Cost_FP + FN * Cost_FN`.
- **Strategy 2 (FPR Constraint)**: Hard constraint to keep False Positive Rate below 5% while maximizing Recall (Selected for final deployment).

---

##  Project Structure

```text
.
├── src/
│   ├── __init__.py
│   ├── config.py           # Hyperparameters & Paths
│   ├── data.py             # Data loading & Memory reduction
│   ├── features2.py        # Feature Engineering (Base + Magic Features)
│   ├── model_xgb.py        # XGBoost Wrapper
│   ├── model_lgbm.py       # LightGBM Wrapper
│   ├── cv.py               # Time-based Cross-Validation Split
│   ├── evaluation.py       # AUC, Cost Matrix, Confusion Matrix
│   └── train2.py           # Main Pipeline Entry Point
├── outputs/
│   ├── oof/                # Out-of-fold predictions
│   └── meta/               # Feature lists used for training
├── .gitignore
└── README.md
```

---

##  How to Run

### Prerequisites
- Python 3.10+
- XGBoost, LightGBM, Pandas, NumPy, Scikit-Learn, SHAP

### Installation
```bash
git clone https://github.com/YOUR_USERNAME/IEEE-CIS-Fraud-Detection-Solution.git
cd IEEE-CIS-Fraud-Detection-Solution
pip install -r requirements.txt  # (Create a requirements.txt if needed)
```

### Run the Pipeline
The entire training, validation, and inference process is encapsulated in `src/train2.py`.

```bash
# Run from the project root
python -m src.train2
```
This command will:
1. Load raw data.
2. Generate base and aggregated features.
3. Train XGBoost and LightGBM models using 6-fold Time-Split CV.
4. Perform dynamic ensemble weighting.
5. Output evaluation logs (AUC, Cost, SHAP values) to the console.
6. Save OOF predictions to `outputs/oof/`.

---

##  Tech Stack Details

- **Core Models**: `xgboost`, `lightgbm`, `sklearn.ensemble.IsolationForest`
- **Interpretability**: `shap` (TreeExplainer)
- **Data Manipulation**: `pandas`, `numpy`
- **Validation Strategy**: Time-series split (GroupKFold by Month)

