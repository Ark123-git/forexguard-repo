"""
tracking/mlflow_logger.py
--------------------------
Logs model training runs to MLflow for experiment tracking.

Usage:
    from tracking.mlflow_logger import log_baseline_run, log_ae_run
    log_baseline_run(results_df, labels)
"""

import os

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

TRACKING_URI  = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
EXPERIMENT    = "forexguard-anomaly-detection"


def _get_client():
    if not MLFLOW_AVAILABLE:
        return None
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    return mlflow


def log_baseline_run(scores_df, labels, params: dict = None):
    client = _get_client()
    if client is None:
        print("[tracking] MLflow not installed, skipping logging.")
        return

    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    import pandas as pd

    y      = labels.astype(int)
    params = params or {}

    with client.start_run(run_name="isolation_forest_lof_ensemble"):
        client.log_params({
            "model_type":      "IsolationForest+LOF",
            "contamination":   0.10,
            "n_estimators":    200,
            "lof_n_neighbors": 20,
            **params,
        })
        for name, score_col, pred_col in [
            ("if",  "if_score",       "if_anomaly"),
            ("lof", "lof_score",      "lof_anomaly"),
            ("ens", "ensemble_score", "if_anomaly"),
        ]:
            try:
                auc = roc_auc_score(y, scores_df[score_col])
                ap  = average_precision_score(y, scores_df[score_col])
                pred = (scores_df[pred_col] == -1).astype(int) \
                       if pred_col != "lof_anomaly" else (scores_df[pred_col] == -1).astype(int)
                f1  = f1_score(y, pred, zero_division=0)
                client.log_metrics({
                    f"{name}_auc": auc,
                    f"{name}_ap":  ap,
                    f"{name}_f1":  f1,
                })
            except Exception:
                pass
        print(f"[tracking] Logged baseline run to MLflow experiment '{EXPERIMENT}'")


def log_ae_run(scores_df, labels, params: dict = None):
    client = _get_client()
    if client is None:
        print("[tracking] MLflow not installed, skipping logging.")
        return

    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    params = params or {}
    y = labels.astype(int)

    with client.start_run(run_name="mlp_autoencoder"):
        client.log_params({
            "model_type":  "MLPAutoencoder",
            "hidden_dim":  64,
            "latent_dim":  16,
            "epochs":      80,
            **params,
        })
        try:
            auc = roc_auc_score(y, scores_df["lstm_score"])
            ap  = average_precision_score(y, scores_df["lstm_score"])
            f1  = f1_score(y, scores_df["lstm_anomaly"], zero_division=0)
            client.log_metrics({"auc": auc, "ap": ap, "f1": f1})
        except Exception:
            pass
        print(f"[tracking] Logged AE run to MLflow experiment '{EXPERIMENT}'")
