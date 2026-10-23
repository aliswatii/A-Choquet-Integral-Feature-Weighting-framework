# CI-FWM-Balanced

### A Choquet-Integral Feature-Weighted, Class-Balanced Gradient Boosting Framework for Diabetes Risk Assessment

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![LightGBM](https://img.shields.io/badge/Classifier-LightGBM-brightgreen)](https://lightgbm.readthedocs.io/)
[![Optuna](https://img.shields.io/badge/HPO-Optuna-orange)](https://optuna.org/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-purple)](https://github.com/shap/shap)

> 📄 Official implementation accompanying the paper *"CI-FWM-Balanced: A Choquet-Integral Feature-Weighted, Class-Balanced Gradient Boosting Framework for Diabetes Risk Assessment"*.

---

## 📌 Overview

CI-FWM-Balanced is a diabetes risk-assessment pipeline built around three components, each validated and isolated rather than reported only as a single end-to-end number:

1. **CI-FWM** — a Choquet-integral Feature Weighting Matrix that aggregates six feature-selection techniques (Pearson correlation, mutual information, ANOVA F-score, chi-square, L1-logistic coefficients, Random Forest importance) using a 2-additive fuzzy measure. Unlike a simple weighted sum, it can represent **redundancy and complementarity** between techniques (Propositions 1–2 in the paper), and provably reduces to the standard linear weighting scheme when no interaction is present.
2. **SMOTE-RUS class balancing** — a combined oversampling/undersampling stage with a proven exact 50/50 balancing guarantee (Lemma 2), applied strictly within each training fold to avoid data leakage.
3. **Optuna-tuned LightGBM** — a gradient-boosted tree classifier tuned via Tree-structured Parzen Estimator search over the balanced, CI-FWM-weighted features.

Four post-hoc modules operate on the trained model's outputs without altering it: **SHAP attribution**, **isotonic calibration**, **split-conformal prediction**, and **K-means-based risk-category thresholding** (Low / Moderate / High / Very High).

The complete, architecturally unchanged pipeline is evaluated on **two independent datasets**:

| Dataset | Task | Accuracy | AUC |
|---|---|---|---|
| Diabetes Prediction Dataset (DPD, 96,146 patients) | Binary classification | **96.92%** | **0.9760** |
| Health & Lifestyle Dataset (HLDDP, 100,000 patients) | 4-class risk stratification | **78.84%** | **0.9466** (macro OvR) |

---

## ⚠️ A Note on Diagnostic-Criterion Leakage

Several public diabetes datasets include **HbA1c** and **blood glucose** as predictors — but these are the clinical diagnostic criteria for diabetes itself, not prospective risk factors. Using them as inputs can make a model look like it is "predicting" diabetes when it is largely detecting the diagnostic threshold. This repository handles the issue **per-dataset, explicitly**:

- **DPD**: HbA1c and glucose are retained as predictors, because the diabetes label here is an independently recorded clinical diagnosis rather than a threshold computed directly from either column. Their outsized contribution is explicitly measured (not assumed) via SHAP and incremental feature selection (see `notebooks/04_feature_attribution.ipynb`).
- **HLDDP**: HbA1c is **excluded** from the predictor set, because the 4-class risk label in this dataset is directly thresholded from HbA1c — including it would leak the label into its own inputs.

See Section 2.4 and Section 4.1 of the paper for the full discussion.

---

## 🗂️ Repository Struct

---

## 📊 Datasets

Both datasets are public and available on Kaggle. Download and place them under `data/raw/` before running any pipeline stage.

| Dataset | Source | Patients | Features | Task |
|---|---|---|---|---|
| Diabetes Prediction Dataset (DPD) | [Kaggle](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) | 96,146 (after dedup) | 8 | Binary |
| Health & Lifestyle Data for Diabetes Prediction (HLDDP) | [Kaggle](https://www.kaggle.com/datasets/alamshihab075/health-and-lifestyle-data-for-diabetes-prediction) | 100,000 | 21 (→ 18 PCA components) | 4-class |

> **Note:** 3,854 exact duplicate rows (3.9%) are removed from the raw DPD prior to splitting, to prevent train/test leakage.

---

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/CI-FWM-Balanced.git
cd CI-FWM-Balanced

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
lightgbm>=4.0
optuna>=3.0
imbalanced-learn>=0.11
shap>=0.44
scikit-learn>=1.3
scipy
pandas
numpy
matplotlib
seaborn
@article{riaz2026cifwmbalanced,
  title     = {CI-FWM-Balanced: A Choquet-Integral Feature-Weighted, Class-Balanced
               Gradient Boosting Framework for Diabetes Risk Assessment},
  author    = {Riaz, Muhammad and Zeb, Aurang and Ali, Haider and
               Durrany, Muhibullah Mahboob and Hou, Muzhou},
  journal   = {TBD},
  year      = {2026}
}
