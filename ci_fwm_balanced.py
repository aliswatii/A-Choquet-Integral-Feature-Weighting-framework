"""
CI-FWM-Balanced
================
A Choquet-Integral Feature-Weighted, Class-Balanced Gradient Boosting
Framework for Diabetes Risk Assessment.

This module implements every stage described in the paper's Methodology
(Section 3) and the evaluation machinery used in Experiments and Results /
Ablation Studies (Sections 4-5):

    3.3  Data preprocessing (standardization, dataset-conditional PCA)
    3.4  CI-FWM: Choquet-integral feature weighting   (Algorithm 1)
    3.5  SMOTE-RUS class balancing                    (Algorithm 2, Lemma 2)
    3.6  Optuna-tuned LightGBM classification          (Algorithm 3)
    3.7  Post-hoc modules: SHAP, isotonic calibration, conformal prediction
    3.8  Risk-category thresholding (K-means on predicted probabilities)
    3.9  Cross-dataset validation design
    3.10 Statistical testing (Nadeau-Bengio corrected paired t-test)
    4    Evaluation metrics
    5    Ablation / sensitivity-sweep helpers

Dependencies
------------
    numpy, scipy, pandas, scikit-learn, imbalanced-learn, lightgbm, optuna, shap

    pip install numpy scipy pandas scikit-learn imbalanced-learn lightgbm optuna shap

Everything here is a faithful, runnable re-implementation of the paper's
equations and algorithms. It is written against generic (X, y) arrays, so it
is not tied to the DPD / HLDDP datasets specifically -- plug in any tabular
classification dataset.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as student_t
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize


# ======================================================================
# Section 3.3 -- Data preprocessing
# ======================================================================

@dataclass
class Preprocessor:
    """Standardization (Eq. 1) + dataset-conditional PCA (Eq. 2).

    PCA is fit on the *training* partition only; validation/calibration/
    test partitions are transformed with those frozen training statistics,
    exactly as specified in Section 3.3.
    """

    use_pca: bool = False
    variance_threshold: float = 0.95
    n_components_: Optional[int] = None

    def __post_init__(self):
        self._scaler = StandardScaler()
        self._components: Optional[np.ndarray] = None  # V_K, shape (m, K)

    def fit(self, X_train: np.ndarray) -> "Preprocessor":
        # Eq. (1): z_ij = (x_ij - mu_j) / sigma_j, fit on the training fold only
        self._scaler.fit(X_train)
        if self.use_pca:
            Z = self._scaler.transform(X_train)
            # Eq. (2): Y = X_s V_K, V_K = top-K eigenvectors of Sigma = X_s^T X_s / (s-1)
            s = Z.shape[0]
            cov = (Z.T @ Z) / (s - 1)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            eigvals, eigvecs = eigvals[order], eigvecs[:, order]
            cum_var = np.cumsum(eigvals) / np.sum(eigvals)
            K = int(np.searchsorted(cum_var, self.variance_threshold) + 1)
            K = max(1, min(K, eigvecs.shape[1]))
            self.n_components_ = K
            self._components = eigvecs[:, :K]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = self._scaler.transform(X)
        if self.use_pca:
            if self._components is None:
                raise RuntimeError("Preprocessor must be fit before transform().")
            return Z @ self._components
        return Z

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        return self.fit(X_train).transform(X_train)


# ======================================================================
# Section 3.4 -- CI-FWM: Choquet-integral feature weighting (Algorithm 1)
# ======================================================================

from sklearn.feature_selection import mutual_info_classif, f_classif, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def _rank_scores(scores: np.ndarray) -> np.ndarray:
    """Rank features by descending score; rank 1 = most important."""
    order = np.argsort(-scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def default_feature_selection_techniques(X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    """The six feature-selection techniques T = {T1, ..., T6} named in Section 3.2:
    Pearson correlation, mutual information, ANOVA F-score, chi-square,
    L1-penalized logistic-regression coefficient magnitude, and Random Forest
    importance. Each returns a per-feature rank vector (1 = most important).
    """
    m = X.shape[1]
    techniques: Dict[str, np.ndarray] = {}

    # 1. Pearson correlation with the label
    pear = np.array([abs(np.corrcoef(X[:, j], y)[0, 1]) if np.std(X[:, j]) > 0 else 0.0
                      for j in range(m)])
    techniques["pearson"] = _rank_scores(np.nan_to_num(pear))

    # 2. Mutual information
    mi = mutual_info_classif(X, y, random_state=0)
    techniques["mutual_info"] = _rank_scores(mi)

    # 3. ANOVA F-score
    f_vals, _ = f_classif(X, y)
    techniques["anova_f"] = _rank_scores(np.nan_to_num(f_vals))

    # 4. chi-square (requires non-negative features; shift to be safe)
    X_nonneg = X - X.min(axis=0, keepdims=True) + 1e-9
    chi2_vals, _ = chi2(X_nonneg, y)
    techniques["chi_square"] = _rank_scores(np.nan_to_num(chi2_vals))

    # 5. L1-penalized logistic-regression coefficient magnitude
    lr = LogisticRegression(penalty="l1", solver="liblinear", max_iter=2000, random_state=0)
    lr.fit(X, y)
    coef = np.abs(lr.coef_).sum(axis=0) if lr.coef_.ndim > 1 else np.abs(lr.coef_[0])
    techniques["l1_logreg"] = _rank_scores(coef)

    # 6. Random Forest importance
    rf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    rf.fit(X, y)
    techniques["random_forest"] = _rank_scores(rf.feature_importances_)

    return techniques


@dataclass
class CIFWM:
    """Choquet-integral Feature Weighting Matrix (Algorithm 1).

    Fits a 2-additive fuzzy measure over the set of feature-selection
    techniques and produces a final, normalized per-feature weight vector w
    (Eq. 7) via the closed-form Choquet weight of Proposition 1 (Eq. 6).
    """

    n_bootstrap: int = 200          # B
    top_l: int = 10                 # L: size of each technique's top-ranked set
    singleton_fraction: float = 0.7  # the 0.7 in a_i = 0.7 * rho_i (Section 3.4)
    epsilon: float = 1e-6
    random_state: int = 0

    weights_: Optional[np.ndarray] = field(default=None, init=False)
    reliability_: Optional[Dict[str, float]] = field(default=None, init=False)
    singleton_masses_: Optional[Dict[str, float]] = field(default=None, init=False)
    pairwise_masses_: Optional[Dict[Tuple[str, str], float]] = field(default=None, init=False)

    # ---- Eq. (3): rank-dependent contribution psi_r ----
    @staticmethod
    def _psi(m: int) -> np.ndarray:
        r = np.arange(1, m + 1)
        psi = 2.0 * (m - r + 1) / (m * (m + 1))
        return psi  # sums to 1 over r = 1..m

    # ---- Eq. (4): bootstrap-stability reliability rho_i ----
    def _reliability(self, X: np.ndarray, y: np.ndarray,
                      technique_fn) -> Dict[str, float]:
        rng = np.random.RandomState(self.random_state)
        n = X.shape[0]
        base_top_sets: Dict[str, List[set]] = {name: [] for name in technique_fn(X, y)}

        for b in range(self.n_bootstrap):
            idx = rng.randint(0, n, size=n)  # bootstrap resample
            Xb, yb = X[idx], y[idx]
            if len(np.unique(yb)) < 2:
                continue
            ranks_b = technique_fn(Xb, yb)
            for name, ranks in ranks_b.items():
                top_set = set(np.argsort(ranks)[: self.top_l].tolist())
                base_top_sets[name].append(top_set)

        eta: Dict[str, float] = {}
        for name, sets_list in base_top_sets.items():
            B = len(sets_list)
            if B < 2:
                eta[name] = 1.0
                continue
            sims = []
            for a, b in itertools.combinations(range(B), 2):
                Fa, Fb = sets_list[a], sets_list[b]
                union = len(Fa | Fb)
                inter = len(Fa & Fb)
                sims.append(inter / union if union else 0.0)
            # Eq. (4): eta_i = (2 / (B(B-1))) * sum_{a<b} Jaccard(F_a, F_b)
            eta[name] = (2.0 / (B * (B - 1))) * sum(sims) if B > 1 else 1.0

        total = sum(eta.values())
        rho = {name: (val / total if total > 0 else 1.0 / len(eta)) for name, val in eta.items()}
        return rho

    def fit(self, X: np.ndarray, y: np.ndarray, technique_fn=None) -> "CIFWM":
        technique_fn = technique_fn or default_feature_selection_techniques
        m = X.shape[1]
        psi = self._psi(m)

        ranks = technique_fn(X, y)                 # {technique_name: rank vector}
        rho = self._reliability(X, y, technique_fn)  # Eq. (4)
        self.reliability_ = rho

        # Eq. (3)+(4): per-technique, per-feature WAM contribution vector
        # phi_i[p] = psi_{rank_i(p)}
        names = list(ranks.keys())
        phi = {name: psi[ranks[name] - 1] for name in names}  # rank is 1-indexed

        # --- Singleton masses: a_i = 0.7 * rho_i (Section 3.4) ---
        singleton = {name: self.singleton_fraction * rho[name] for name in names}
        self.singleton_masses_ = singleton
        singleton_budget_used = sum(singleton.values())
        pairwise_budget = max(0.0, 1.0 - singleton_budget_used)

        # --- Pairwise masses: proportional to disagreement = 1 - |Spearman rho| ---
        disagreement: Dict[Tuple[str, str], float] = {}
        for i, j in itertools.combinations(names, 2):
            corr, _ = spearmanr(phi[i], phi[j])
            corr = 0.0 if np.isnan(corr) else corr
            disagreement[(i, j)] = 1.0 - abs(corr)
        total_disagreement = sum(disagreement.values())
        pairwise = {
            key: (pairwise_budget * (val / total_disagreement) if total_disagreement > 0 else 0.0)
            for key, val in disagreement.items()
        }
        self.pairwise_masses_ = pairwise

        # --- Proposition 1 closed-form Choquet weight per technique, then
        #     aggregate into a single per-feature Choquet weight (Eq. 6/7). ---
        # We apply the Choquet integral over the *techniques* as the "features"
        # of the aggregation (the six rank-contribution vectors phi_i are the
        # x_i inputs of Definition 3), separately for each original feature p.
        a_single = np.array([singleton[name] for name in names])

        def pairwise_mass(i_name, j_name):
            key = (i_name, j_name) if (i_name, j_name) in pairwise else (j_name, i_name)
            return pairwise.get(key, 0.0)

        feature_weights = np.zeros(m)
        for p in range(m):
            x = np.array([phi[name][p] for name in names])  # technique contributions for feature p
            order = np.argsort(x)  # ascending, per Definition 3 (sigma)
            # Proposition 1 closed form: w_{sigma(i)} = a_{sigma(i)} + sum_{l: rank(l) > i} a_{sigma(i), sigma(l)}
            total = 0.0
            for pos, tech_idx in enumerate(order):
                later = order[pos + 1:]
                pw = sum(pairwise_mass(names[tech_idx], names[l]) for l in later)
                w_i = a_single[tech_idx] + pw
                total += x[tech_idx] * w_i
            feature_weights[p] = total

        # Eq. (7): normalize to a valid weight distribution
        w = (feature_weights + self.epsilon) / (feature_weights.sum() + m * self.epsilon)
        self.weights_ = w
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the fitted, fixed feature weights (never refit at inference)."""
        if self.weights_ is None:
            raise RuntimeError("CIFWM must be fit before transform().")
        return X * self.weights_[np.newaxis, :]

    def fit_transform(self, X: np.ndarray, y: np.ndarray, technique_fn=None) -> np.ndarray:
        return self.fit(X, y, technique_fn).transform(X)


# ======================================================================
# Section 3.5 -- SMOTE-RUS class balancing (Algorithm 2, Lemma 2)
# ======================================================================

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def smote_rus_balance(
    X_train: np.ndarray,
    y_train: np.ndarray,
    tau_s: float = 1.0,
    tau_r: float = 1.0,
    k_neighbors: int = 5,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Algorithm 2: SMOTE-RUS training-partition balancing.

    Applies SMOTE oversampling to a target minority ratio tau_s, then
    random under-sampling to a target ratio tau_r. By Lemma 2, tau_s =
    tau_r = 1 gives an exact, deterministic 50/50 balance regardless of the
    original imbalance ratio n0/n1.

    Must only ever be called on a fold's *training* partition -- never on
    validation, calibration, or test data (Section 3.5).
    """
    classes, counts = np.unique(y_train, return_counts=True)
    if len(classes) != 2:
        raise ValueError("smote_rus_balance expects a binary label vector; "
                          "for multi-class, balance one-vs-rest or call per class pair.")
    n0, n1 = counts.max(), counts.min()
    majority_label = classes[np.argmax(counts)]
    minority_label = classes[np.argmin(counts)]

    # Step 1: SMOTE to raise minority count toward floor(tau_s * n0)
    target_minority = max(n1, math.floor(tau_s * n0))
    smote_strategy = {minority_label: int(target_minority), majority_label: int(n0)}
    sm = SMOTE(sampling_strategy=smote_strategy, k_neighbors=min(k_neighbors, n1 - 1),
               random_state=random_state)
    X_s, y_s = sm.fit_resample(X_train, y_train)

    # Step 2: RUS to shrink majority count toward floor(n1' / tau_r)
    n1_prime = target_minority
    target_majority = max(n1_prime, math.floor(n1_prime / tau_r)) if tau_r > 0 else n0
    target_majority = min(n0, target_majority)
    rus_strategy = {majority_label: int(target_majority), minority_label: int(n1_prime)}
    rus = RandomUnderSampler(sampling_strategy=rus_strategy, random_state=random_state)
    X_bal, y_bal = rus.fit_resample(X_s, y_s)

    return X_bal, y_bal


# ======================================================================
# Section 3.6 -- Optuna-tuned LightGBM classification (Algorithm 3)
# ======================================================================

import lightgbm as lgb
import optuna
from sklearn.metrics import recall_score as _recall_score


def _lightgbm_search_space(trial: "optuna.Trial") -> dict:
    return {
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }


def optuna_tune_lightgbm(
    X_bal: np.ndarray,
    y_bal: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 100,
    multiclass: bool = False,
    n_classes: int = 2,
    random_state: int = 0,
    class_weight: Optional[str] = None,
) -> Tuple[lgb.LGBMClassifier, dict]:
    """Algorithm 3: end-to-end hyperparameter search + final fit.

    Optuna's default sampler is a Tree-structured Parzen Estimator (TPE),
    which is precisely the l(theta)/g(theta) ratio of Eq. (8): it models
    "good" vs "bad" trials relative to a quantile of observed objective
    values and samples new trials favoring the good-trial density. The
    objective itself is validation accuracy (binary) or macro-averaged
    recall (severely imbalanced multi-class), evaluated on the untouched,
    non-resampled validation partition X_val/y_val -- never on X_bal/y_bal.
    """

    objective_name = "multi_logloss" if multiclass else "binary_logloss"

    def objective(trial: "optuna.Trial") -> float:
        params = _lightgbm_search_space(trial)
        params.update({
            "objective": "multiclass" if multiclass else "binary",
            "num_class": n_classes if multiclass else None,
            "random_state": random_state,
            "n_estimators": 2000,
            "verbosity": -1,
            "class_weight": class_weight,
        })
        params = {k: v for k, v in params.items() if v is not None}

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_bal, y_bal,
            eval_set=[(X_val, y_val)],
            eval_metric=objective_name,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        y_pred = model.predict(X_val)
        if multiclass:
            # macro-averaged recall for severely imbalanced multi-class settings
            return _recall_score(y_val, y_pred, average="macro")
        return accuracy_score(y_val, y_pred)

    sampler = optuna.samplers.TPESampler(seed=random_state)  # implements Eq. (8)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = dict(study.best_params)
    best_params.update({
        "objective": "multiclass" if multiclass else "binary",
        "num_class": n_classes if multiclass else None,
        "random_state": random_state,
        "n_estimators": 2000,
        "verbosity": -1,
        "class_weight": class_weight,
    })
    best_params = {k: v for k, v in best_params.items() if v is not None}

    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(
        X_bal, y_bal,
        eval_set=[(X_val, y_val)],
        eval_metric=objective_name,
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    return final_model, study.best_params


# ======================================================================
# Section 3.7 -- Post-hoc analysis modules
# ======================================================================

def shap_attribution(model: lgb.LGBMClassifier, X: np.ndarray):
    """Exact TreeExplainer SHAP values for a trained LightGBM model."""
    import shap
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(X)


def fit_isotonic_calibration(model, X_cal: np.ndarray, y_cal: np.ndarray) -> IsotonicRegression:
    """Pool-adjacent-violators isotonic calibration on a dedicated calibration
    partition, kept separate from training/validation/test (Section 3.7)."""
    raw_probs = model.predict_proba(X_cal)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_probs, y_cal)
    return iso


def apply_isotonic_calibration(iso: IsotonicRegression, model, X: np.ndarray) -> np.ndarray:
    raw_probs = model.predict_proba(X)[:, 1]
    return iso.transform(raw_probs)


def conformal_prediction_sets(
    model, X_cal: np.ndarray, y_cal: np.ndarray, X_test: np.ndarray, alpha: float = 0.1
) -> List[set]:
    """Split-conformal Adaptive Prediction Sets (APS) with marginal coverage
    1 - alpha, using a dedicated calibration partition (Section 3.7)."""
    cal_probs = model.predict_proba(X_cal)
    n_classes = cal_probs.shape[1]

    def aps_score(probs_row, true_label):
        order = np.argsort(-probs_row)
        cum = np.cumsum(probs_row[order])
        rank = np.where(order == true_label)[0][0]
        return cum[rank]

    scores = np.array([aps_score(cal_probs[i], y_cal[i]) for i in range(len(y_cal))])
    n = len(scores)
    qlevel = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    qhat = np.quantile(scores, qlevel)

    test_probs = model.predict_proba(X_test)
    pred_sets: List[set] = []
    for row in test_probs:
        order = np.argsort(-row)
        cum = np.cumsum(row[order])
        cutoff = np.searchsorted(cum, qhat) + 1
        pred_sets.append(set(order[:cutoff].tolist()))
    return pred_sets


# ======================================================================
# Section 3.8 -- Risk category thresholding
# ======================================================================

RISK_LABELS = ["Low", "Moderate", "High", "Very High"]


@dataclass
class RiskCategoryThresholder:
    """K-means (k=4) on validation-partition predicted probabilities, with
    thresholds set as midpoints between sorted cluster centers (Eq. 10).
    Fit on the validation partition only -- never on test, to avoid
    threshold leakage (Section 3.8)."""

    k: int = 4
    random_state: int = 0
    thresholds_: Optional[np.ndarray] = field(default=None, init=False)
    centers_: Optional[np.ndarray] = field(default=None, init=False)

    def fit(self, y_prob_val: np.ndarray) -> "RiskCategoryThresholder":
        km = KMeans(n_clusters=self.k, random_state=self.random_state, n_init=10)
        km.fit(y_prob_val.reshape(-1, 1))
        centers = np.sort(km.cluster_centers_.ravel())
        self.centers_ = centers
        # Eq. (10): Th_j = (c_j + c_{j+1}) / 2, j = 1, 2, 3
        self.thresholds_ = np.array([(centers[j] + centers[j + 1]) / 2 for j in range(self.k - 1)])
        return self

    def categorize(self, y_prob: np.ndarray) -> np.ndarray:
        if self.thresholds_ is None:
            raise RuntimeError("RiskCategoryThresholder must be fit before categorize().")
        th1, th2, th3 = self.thresholds_
        cats = np.full(y_prob.shape, "", dtype=object)
        cats[y_prob <= th1] = RISK_LABELS[0]
        cats[(y_prob > th1) & (y_prob <= th2)] = RISK_LABELS[1]
        cats[(y_prob > th2) & (y_prob <= th3)] = RISK_LABELS[2]
        cats[y_prob > th3] = RISK_LABELS[3]
        return cats


# ======================================================================
# Section 3.10 -- Statistical testing (Nadeau-Bengio corrected paired t-test)
# ======================================================================

def nadeau_bengio_test(diffs: Sequence[float], n_test_over_n_train: float = 0.25) -> Tuple[float, float]:
    """Eq. (11): corrected standard error for k non-independent CV folds.

    diffs: per-fold performance differences between two methods (A - B).
    n_test_over_n_train: ratio of test-set size to training-set size per
        fold (1/(k-1) for standard k-fold CV).

    Returns (t_statistic, two_sided_p_value).
    """
    d = np.asarray(diffs, dtype=float)
    k = len(d)
    d_bar = d.mean()
    sd = d.std(ddof=1)
    # Eq. (11): SE = s_d * sqrt(1/k + n_test/n_train)
    se = sd * math.sqrt(1.0 / k + n_test_over_n_train)
    if se == 0:
        return float("inf") if d_bar != 0 else 0.0, 0.0 if d_bar != 0 else 1.0
    t_stat = d_bar / se
    p_value = 2 * (1 - student_t.cdf(abs(t_stat), df=k - 1))
    return t_stat, p_value


# ======================================================================
# Section 4 -- Evaluation metrics
# ======================================================================

def evaluate_binary(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if y_prob is not None:
        out["auc"] = roc_auc_score(y_true, y_prob)
    return out


def evaluate_multiclass(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None,
                         n_classes: Optional[int] = None) -> Dict[str, float]:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro"),
        "recall": recall_score(y_true, y_pred, average="macro"),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if y_prob is not None:
        n_classes = n_classes or y_prob.shape[1]
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        # macro-averaged one-vs-rest AUC (Section 4, Evaluation metrics)
        aucs = [roc_auc_score(y_bin[:, c], y_prob[:, c]) for c in range(n_classes)]
        out["auc"] = float(np.mean(aucs))
    return out


# ======================================================================
# End-to-end pipeline (Algorithm 3 composed with Sections 3.3-3.8)
# ======================================================================

@dataclass
class CIFWMBalancedPipeline(BaseEstimator, ClassifierMixin):
    """Composes Eq. (9): y_hat = f_theta( C_w( P_K( S(x) ) ) ).

    S = standardization, P_K = dataset-conditional PCA, C_w = CI-FWM feature
    weighting, f_theta = the Optuna-tuned LightGBM classifier. SMOTE-RUS
    balances only the training fold used to fit f_theta -- it never appears
    in this composition because it is never applied at inference.
    """

    use_pca: bool = False
    rebalance: bool = True       # False reproduces Table 5's "None (class-weighted loss only)"
    tau_s: float = 1.0
    tau_r: float = 1.0
    singleton_fraction: float = 0.7  # CIFWM's a_i = singleton_fraction * rho_i (relates to beta: the
                                      # pairwise-mass budget beta = 1 - singleton_fraction)
    n_bootstrap: int = 200
    n_optuna_trials: int = 100
    multiclass: bool = False
    n_classes: int = 2
    random_state: int = 0
    class_weight: Optional[str] = None  # set to "balanced" by fit() when rebalance=False

    def fit(self, X_train, y_train, X_val, y_val):
        self.preprocessor_ = Preprocessor(use_pca=self.use_pca)
        Z_train = self.preprocessor_.fit_transform(X_train)
        Z_val = self.preprocessor_.transform(X_val)

        self.cifwm_ = CIFWM(
            singleton_fraction=self.singleton_fraction,
            n_bootstrap=self.n_bootstrap,
            random_state=self.random_state,
        )
        W_train = self.cifwm_.fit_transform(Z_train, y_train)
        W_val = self.cifwm_.transform(Z_val)

        if self.rebalance:
            X_bal, y_bal = smote_rus_balance(
                W_train, y_train, tau_s=self.tau_s, tau_r=self.tau_r,
                random_state=self.random_state,
            )
            self._fit_class_weight = None
        else:
            # Table 5's "None" condition: no SMOTE-RUS at all, class-weighted
            # loss only, everything else in the pipeline unchanged.
            X_bal, y_bal = W_train, y_train
            self._fit_class_weight = "balanced"

        self.model_, self.best_params_ = optuna_tune_lightgbm(
            X_bal, y_bal, W_val, y_val,
            n_trials=self.n_optuna_trials,
            multiclass=self.multiclass,
            n_classes=self.n_classes,
            random_state=self.random_state,
            class_weight=self._fit_class_weight,
        )
        return self

    def _transform(self, X):
        return self.cifwm_.transform(self.preprocessor_.transform(X))

    def predict(self, X):
        return self.model_.predict(self._transform(X))

    def predict_proba(self, X):
        return self.model_.predict_proba(self._transform(X))


# ======================================================================
# Section 4.1 / 5 -- 5-fold CV driver, baselines, ablation, sensitivity
# ======================================================================

def run_5fold_cv(
    X: np.ndarray, y: np.ndarray,
    pipeline_kwargs: Optional[dict] = None,
    n_splits: int = 5,
    val_fraction: float = 0.2,
    random_state: int = 0,
) -> pd.DataFrame:
    """5-fold stratified cross-validation of CIFWMBalancedPipeline, matching
    Section 4.1's protocol: each training fold is further split into an
    inner train/validation partition so that Optuna tuning and risk-category
    thresholding never touch the held-out test fold."""
    pipeline_kwargs = pipeline_kwargs or {}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_trainval, y_trainval = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        n_val = int(len(X_trainval) * val_fraction)
        rng = np.random.RandomState(random_state + fold)
        perm = rng.permutation(len(X_trainval))
        val_idx, train_idx2 = perm[:n_val], perm[n_val:]
        X_train, y_train = X_trainval[train_idx2], y_trainval[train_idx2]
        X_val, y_val = X_trainval[val_idx], y_trainval[val_idx]

        pipe = CIFWMBalancedPipeline(random_state=random_state, **pipeline_kwargs)
        pipe.fit(X_train, y_train, X_val, y_val)

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)

        if pipeline_kwargs.get("multiclass"):
            metrics = evaluate_multiclass(y_test, y_pred, y_prob,
                                           n_classes=pipeline_kwargs.get("n_classes"))
        else:
            metrics = evaluate_binary(y_test, y_pred, y_prob[:, 1])

        metrics["fold"] = fold
        rows.append(metrics)

    return pd.DataFrame(rows)


def ablation_rebalancing_strategy(X: np.ndarray, y: np.ndarray, **cv_kwargs) -> pd.DataFrame:
    """Table 5: None (class-weighted loss only) vs. Partial (tau=0.5) vs.
    Full (tau=1.0) rebalancing, everything else held fixed."""
    results = []
    conditions = [
        ("none", {"rebalance": False}),
        ("partial", {"rebalance": True, "tau_s": 0.5, "tau_r": 0.5}),
        ("full", {"rebalance": True, "tau_s": 1.0, "tau_r": 1.0}),
    ]
    for label, kwargs in conditions:
        df = run_5fold_cv(X, y, pipeline_kwargs=kwargs, **cv_kwargs)
        df["condition"] = label
        results.append(df)
    return pd.concat(results, ignore_index=True)


def sensitivity_sweep_beta(X: np.ndarray, y: np.ndarray, betas: Sequence[float],
                            **cv_kwargs) -> pd.DataFrame:
    """Table 6: sensitivity of CI-FWM-Balanced to the pairwise-mass budget
    beta, exposed here as CIFWM.singleton_fraction = 1 - beta (beta=0
    recovers the WAM baseline per Proposition 2, since singleton_fraction=1.0
    leaves no budget for pairwise masses). The paper uses a single
    train/validation/test split per grid point rather than 5-fold CV; pass
    n_splits=1 via cv_kwargs if you want that exact protocol, or leave the
    5-fold default for a more robust (but non-paper-identical) sweep."""
    rows = []
    for beta in betas:
        singleton_fraction = 1.0 - beta
        df = run_5fold_cv(
            X, y,
            pipeline_kwargs={"singleton_fraction": singleton_fraction},
            **cv_kwargs,
        )
        df["beta"] = beta
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def sensitivity_sweep_tau_s(X: np.ndarray, y: np.ndarray, tau_s_values: Sequence[float],
                             tau_r_fixed: float = 1.0, **cv_kwargs) -> pd.DataFrame:
    """Table 7: sensitivity to the SMOTE minority target ratio tau_s, with
    tau_r held fixed at 1.0 and beta at its default."""
    rows = []
    for tau_s in tau_s_values:
        df = run_5fold_cv(X, y, pipeline_kwargs={"tau_s": tau_s, "tau_r": tau_r_fixed}, **cv_kwargs)
        df["tau_s"] = tau_s
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


# ======================================================================
# Usage example
# ======================================================================

if __name__ == "__main__":
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=2000, n_features=8, n_informative=6, weights=[0.9, 0.1],
        random_state=0,
    )

    results = run_5fold_cv(X, y, pipeline_kwargs={"n_optuna_trials": 15}, random_state=0)
    print(results)
    print(results.drop(columns=["fold"]).mean())
