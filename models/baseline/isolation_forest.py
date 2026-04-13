"""
models/baseline/isolation_forest.py
-------------------------------------
Baseline anomaly detector using Isolation Forest + Local Outlier Factor.

Why Isolation Forest?
  - No labels needed (unsupervised)
  - Handles high-dimensional tabular data well
  - Fast O(n log n) training
  - Produces a continuous anomaly score (interpretable via SHAP)
  - Industry standard for tabular fraud detection

Why also LOF?
  - Catches local density-based anomalies IF misses
  - Ensemble of IF + LOF gives better coverage

Outputs per user:
  - if_score       : raw Isolation Forest decision function (lower = more anomalous)
  - if_anomaly     : 1/-1 prediction from IF
  - lof_score      : LOF negative outlier factor
  - lof_anomaly    : 1/-1 prediction from LOF
  - ensemble_score : normalized [0,1] combined score (1 = most anomalous)
  - top_features   : top 5 contributing feature names (via permutation importance proxy)
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler


# ── Config ────────────────────────────────────────────────────────────────────
IF_CONTAMINATION  = 0.10    # expected ~10% anomalous
LOF_CONTAMINATION = 0.10
IF_N_ESTIMATORS   = 200
LOF_N_NEIGHBORS   = 20
RANDOM_STATE      = 42
MODEL_DIR         = "models/baseline/saved"


# ── Explainability — feature importance proxy ─────────────────────────────────

def _top_features_by_permutation(X: np.ndarray, model, feat_cols: list,
                                  n_top: int = 5) -> list[list[str]]:
    """
    Lightweight permutation-based feature importance.
    For each feature, shuffle it and measure drop in anomaly score variance.
    Higher drop → more important feature.
    Returns top-n feature names per sample (approximated globally here,
    then we pick the highest-value features per user).
    """
    base_scores = model.decision_function(X)
    importances = []
    for i in range(X.shape[1]):
        X_perm = X.copy()
        np.random.shuffle(X_perm[:, i])
        perm_scores = model.decision_function(X_perm)
        # importance = how much the score changed
        importances.append(float(np.mean(np.abs(base_scores - perm_scores))))

    importance_arr  = np.array(importances)
    top_global_idx  = np.argsort(importance_arr)[::-1][:n_top]
    top_global_feat = [feat_cols[i] for i in top_global_idx]
    return top_global_feat


def _per_user_top_features(X: np.ndarray, feat_cols: list,
                            global_top: list[str], n_top: int = 5) -> list[list[str]]:
    """
    For each user, return the top features by their (standardized) absolute value.
    This gives a simple, fast per-sample explanation.
    """
    col_idx = {c: i for i, c in enumerate(feat_cols)}
    # Use global top features as candidates, then rank by user's actual value
    top_idx = [col_idx[f] for f in global_top if f in col_idx]
    # Standardize candidate columns across all users
    subset   = X[:, top_idx]
    col_std  = subset.std(axis=0) + 1e-9
    col_mean = subset.mean(axis=0)
    z_scores = np.abs((subset - col_mean) / col_std)

    result = []
    for row in z_scores:
        ranked = np.argsort(row)[::-1][:n_top]
        result.append([global_top[r] for r in ranked])
    return result


# ── Training ──────────────────────────────────────────────────────────────────

class BaselineDetector:
    def __init__(self):
        self.scaler = RobustScaler()
        self.if_model  = IsolationForest(
            n_estimators=IF_N_ESTIMATORS,
            contamination=IF_CONTAMINATION,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        self.lof_model = LocalOutlierFactor(
            n_neighbors=LOF_N_NEIGHBORS,
            contamination=LOF_CONTAMINATION,
            novelty=False,
        )
        self.feat_cols    = None
        self.global_top   = None

    def fit(self, X: np.ndarray, feat_cols: list):
        self.feat_cols = feat_cols
        X_scaled = self.scaler.fit_transform(X)
        print("[IF]  Training Isolation Forest...")
        self.if_model.fit(X_scaled)
        print("[LOF] Training Local Outlier Factor...")
        self.lof_model.fit(X_scaled)
        # save original (unscaled) for inference lookup
        self.lof_train_X_ = X
        # Precompute global top features for explanation
        self.global_top = _top_features_by_permutation(
            X_scaled, self.if_model, feat_cols, n_top=15
        )
        # Store training score ranges for stable inference normalization
        _if_train = self.if_model.decision_function(X_scaled)
        _lof_train = self.lof_model.negative_outlier_factor_
        self.if_score_min_  = float(-_if_train.max())   # inverted
        self.if_score_max_  = float(-_if_train.min())
        self.lof_score_min_ = float(-_lof_train.max())
        self.lof_score_max_ = float(-_lof_train.min())
        print(f"[baseline] Top global features: {self.global_top[:5]}")
        return self

    def predict(self, X: np.ndarray) -> pd.DataFrame:
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        n_query  = X_scaled.shape[0]

        # Isolation Forest — works on any number of samples
        if_scores = self.if_model.decision_function(X_scaled)
        if_labels = self.if_model.predict(X_scaled)

        # LOF — scores are only valid for training data.
        # For inference, find nearest training neighbor and use its LOF score.
        from sklearn.metrics import pairwise_distances
        train_X   = self.scaler.transform(self.lof_train_X_)
        dists     = pairwise_distances(X_scaled, train_X)
        nn_idx    = np.argmin(dists, axis=1)
        lof_scores = self.lof_model.negative_outlier_factor_[nn_idx]
        lof_labels = np.where(lof_scores < -1.5, -1, 1)

        # Normalize IF scores across the query batch
        # Normalize using training distribution ranges (stable for any batch size)
        def norm_against_train(arr, mn, mx):
            a = -arr
            return np.clip((a - mn) / (mx - mn + 1e-9), 0.0, 1.0)

        if_norm  = norm_against_train(if_scores,  self.if_score_min_,  self.if_score_max_)
        lof_norm = norm_against_train(lof_scores, self.lof_score_min_, self.lof_score_max_)
        ensemble = 0.6 * if_norm + 0.4 * lof_norm

        per_user_feats = _per_user_top_features(
            X_scaled, self.feat_cols, self.global_top, n_top=5
        )

        return pd.DataFrame({
            "if_score":       np.round(if_norm,  4),
            "if_anomaly":     if_labels,
            "lof_score":      np.round(lof_norm, 4),
            "lof_anomaly":    lof_labels,
            "ensemble_score": np.round(ensemble, 4),
            "top_features":   [json.dumps(f) for f in per_user_feats],
        })
    
    def save(self, path: str = MODEL_DIR):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "baseline.pkl"), "wb") as f:
            pickle.dump(self, f)
        print(f"[baseline] Model saved → {path}/baseline.pkl")

    @classmethod
    def load(cls, path: str = MODEL_DIR) -> "BaselineDetector":
        with open(os.path.join(path, "baseline.pkl"), "rb") as f:
            return pickle.load(f)


# ── Evaluation helper ─────────────────────────────────────────────────────────

def evaluate(scores_df: pd.DataFrame, labels: pd.Series):
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                  precision_score, recall_score, f1_score)

    # Convert -1/1 to 0/1
    if_pred  = (scores_df["if_anomaly"]  == -1).astype(int)
    lof_pred = (scores_df["lof_anomaly"] == -1).astype(int)
    ens_pred = (scores_df["ensemble_score"] >= 0.5).astype(int)
    y        = labels.astype(int)

    print("\n── Evaluation Results ───────────────────────────────────")
    for name, pred, score_col in [
        ("Isolation Forest", if_pred,  "if_score"),
        ("LOF",              lof_pred, "lof_score"),
        ("Ensemble",         ens_pred, "ensemble_score"),
    ]:
        auc  = roc_auc_score(y, scores_df[score_col])
        ap   = average_precision_score(y, scores_df[score_col])
        prec = precision_score(y, pred, zero_division=0)
        rec  = recall_score(y, pred, zero_division=0)
        f1   = f1_score(y, pred, zero_division=0)
        print(f"  {name:<20}  AUC={auc:.3f}  AP={ap:.3f}  "
              f"P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def train_baseline(feature_store_path: str = "data/raw/feature_store.csv",
                   save: bool = True) -> tuple:
    """
    Returns:
        detector    : trained BaselineDetector
        results_df  : per-user scores merged with metadata
    """
    store = pd.read_csv(feature_store_path)
    feat_cols = [c for c in store.columns
                 if c not in ("user_id", "is_anomalous")]
    X      = store[feat_cols].fillna(0).to_numpy(dtype=np.float32)
    labels = store["is_anomalous"]

    detector = BaselineDetector()
    detector.fit(X, feat_cols)

    scores = detector.predict(X)
    results = pd.concat([store[["user_id", "is_anomalous"]].reset_index(drop=True),
                         scores], axis=1)

    evaluate(scores, labels)

    if save:
        detector.save()
        results.to_csv("data/raw/baseline_scores.csv", index=False)
        print("[baseline] Scores saved → data/raw/baseline_scores.csv")

    return detector, results


if __name__ == "__main__":
    train_baseline()
