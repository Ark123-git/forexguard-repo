"""
train_all.py
------------
Master script — runs the full pipeline end to end:
  1. Generate data
  2. Build feature store
  3. Train baseline model (Isolation Forest + LOF)
  4. Train advanced model (MLP Autoencoder)
  5. Generate batch alerts
  6. Print comparison report

Usage:
    cd forexguard
    python train_all.py
"""

import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))

def section(title):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")

if __name__ == "__main__":
    t0 = time.time()

    # ── Step 1: Data generation ───────────────────────────────────
    section("STEP 1 — Synthetic Data Generation")
    os.makedirs("data/raw", exist_ok=True)
    from data.generator.portal_events  import generate_portal_events
    from data.generator.trading_events import generate_trading_events
    p = generate_portal_events(output_path="data/raw/portal_events.csv")
    t = generate_trading_events(output_path="data/raw/trading_events.csv")
    print(f"  Portal: {len(p):,} events | Trading: {len(t):,} events")

    # ── Step 2: Feature engineering ───────────────────────────────
    section("STEP 2 — Feature Engineering")
    from features.feature_store import build_feature_store
    meta, X, feat_cols = build_feature_store()
    print(f"  Feature matrix: {X.shape} | Anomalous: {meta['is_anomalous'].sum()}")

    # ── Step 3: Baseline model ────────────────────────────────────
    section("STEP 3 — Baseline Model (Isolation Forest + LOF)")
    from models.baseline.isolation_forest import train_baseline
    baseline, b_results = train_baseline(save=True)

    # ── Step 4: Advanced model ────────────────────────────────────
    section("STEP 4 — Advanced Model (MLP Autoencoder)")
    from models.advanced.lstm_autoencoder import train_lstm
    ae, a_results = train_lstm(save=True)

    # ── Step 5: Alerts ────────────────────────────────────────────
    section("STEP 5 — Alert Generation")
    from alerts.alert_generator import generate_alerts_from_scores
    alerts = generate_alerts_from_scores(threshold=0.30)

    # ── Step 6: Comparison report ─────────────────────────────────
    section("STEP 6 — Model Comparison Report")
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    import pandas as pd

    store  = pd.read_csv("data/raw/feature_store.csv")
    labels = store["is_anomalous"].astype(int)

    b_scores = pd.read_csv("data/raw/baseline_scores.csv")
    a_scores = pd.read_csv("data/raw/lstm_scores.csv")

    print(f"\n  {'Model':<30} {'AUC':>6}  {'AP':>6}  {'F1':>6}")
    print(f"  {'─'*52}")
    for name, df, score_col, pred_col in [
        ("Isolation Forest",    b_scores, "if_score",       "if_anomaly"),
        ("LOF",                 b_scores, "lof_score",      "lof_anomaly"),
        ("IF+LOF Ensemble",     b_scores, "ensemble_score", "if_anomaly"),
        ("MLP Autoencoder",     a_scores, "lstm_score",     "lstm_anomaly"),
    ]:
        try:
            auc  = roc_auc_score(labels, df[score_col])
            ap   = average_precision_score(labels, df[score_col])
            pred = (df[pred_col] == -1).astype(int)   # -1 = anomalous in sklearn
            f1   = f1_score(labels, pred, zero_division=0)
            print(f"  {name:<30} {auc:>6.3f}  {ap:>6.3f}  {f1:>6.3f}")
        except Exception as e:
            print(f"  {name:<30}  ERROR: {e}")

    elapsed = time.time() - t0
    print(f"\n{'═'*60}")
<<<<<<< HEAD
    print(f"   Pipeline complete in {elapsed:.1f}s")
=======
    print(f"  ✅ Pipeline complete in {elapsed:.1f}s")
>>>>>>> 1de7f2e45308a5d7ca20b46b9a7d37a9f65f2356
    print(f"  Alerts generated: {len(alerts)}")
    print(f"{'═'*60}")
