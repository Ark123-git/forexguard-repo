# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # ForexGuard - EDA and Model Evaluation Notebook
#
# This notebook walks through:
# 1. Dataset exploration
# 2. Feature distributions
# 3. Anomaly pattern visualization
# 4. Model score comparison
# 5. Alert review

# %%
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath("."))))
plt.style.use("seaborn-v0_8-darkgrid")
pd.set_option("display.max_columns", 50)

# %% [markdown]
# ## 1. Load raw event data

# %%
portal  = pd.read_csv("../data/raw/portal_events.csv",  parse_dates=["timestamp"])
trading = pd.read_csv("../data/raw/trading_events.csv",  parse_dates=["open_ts", "close_ts"])

print(f"Portal events  : {len(portal):,}")
print(f"Trading events : {len(trading):,}")
print(f"Portal users   : {portal['user_id'].nunique()}")
print(f"Trading users  : {trading['user_id'].nunique()}")

# %%
print("\nPortal event type distribution:")
print(portal["event_type"].value_counts())

print("\nPortal anomaly type distribution:")
print(portal[portal["anomaly_flag"]]["anomaly_type"].value_counts())

# %%
print("\nTrading anomaly type distribution:")
print(trading[trading["anomaly_flag"]]["anomaly_type"].value_counts())

# %% [markdown]
# ## 2. Feature store exploration

# %%
store = pd.read_csv("../data/raw/feature_store.csv")
print(f"Feature store shape: {store.shape}")
print(f"Anomalous users: {store['is_anomalous'].sum()} / {len(store)}")

portal_feats  = [c for c in store.columns if c.startswith("p_")]
trading_feats = [c for c in store.columns if c.startswith("t_")]
print(f"\nPortal features:  {len(portal_feats)}")
print(f"Trading features: {len(trading_feats)}")

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
features_to_plot = [
    "p_unique_ips", "p_pct_night_logins", "p_deposit_withdraw_ratio",
    "t_pnl_volatility", "t_pct_sub_30s_trades", "t_top_instrument_concentration",
]
for ax, feat in zip(axes.flat, features_to_plot):
    normal   = store[~store["is_anomalous"]][feat]
    anomalous= store[store["is_anomalous"]][feat]
    ax.hist(normal,    bins=30, alpha=0.6, label="Normal",    color="steelblue")
    ax.hist(anomalous, bins=30, alpha=0.6, label="Anomalous", color="tomato")
    ax.set_title(feat, fontsize=9)
    ax.legend(fontsize=7)
plt.suptitle("Feature Distributions: Normal vs Anomalous Users", fontsize=12)
plt.tight_layout()
plt.savefig("../data/raw/feature_distributions.png", dpi=120)
plt.show()

# %% [markdown]
# ## 3. Model score comparison

# %%
b_scores = pd.read_csv("../data/raw/baseline_scores.csv")
a_scores = pd.read_csv("../data/raw/lstm_scores.csv")

merged = b_scores.merge(a_scores[["user_id","lstm_score"]], on="user_id")
merged["is_anomalous"] = store["is_anomalous"].values

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, col, title in [
    (axes[0], "ensemble_score", "Isolation Forest + LOF Ensemble Score"),
    (axes[1], "lstm_score",     "MLP Autoencoder Score"),
]:
    normal   = merged[~merged["is_anomalous"]][col]
    anomalous= merged[merged["is_anomalous"]][col]
    ax.hist(normal,    bins=30, alpha=0.6, label="Normal",    color="steelblue")
    ax.hist(anomalous, bins=30, alpha=0.6, label="Anomalous", color="tomato")
    ax.set_title(title)
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Users")
    ax.legend()
plt.tight_layout()
plt.savefig("../data/raw/score_distributions.png", dpi=120)
plt.show()

# %% [markdown]
# ## 4. ROC curves

# %%
from sklearn.metrics import roc_curve, auc

fig, ax = plt.subplots(figsize=(7, 6))
y = store["is_anomalous"].astype(int)

for name, scores in [
    ("IF+LOF Ensemble", b_scores["ensemble_score"]),
    ("MLP Autoencoder", a_scores["lstm_score"]),
]:
    fpr, tpr, _ = roc_curve(y, scores)
    roc_auc     = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", linewidth=2)

ax.plot([0,1],[0,1], "k--", linewidth=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves - Model Comparison")
ax.legend()
plt.tight_layout()
plt.savefig("../data/raw/roc_curves.png", dpi=120)
plt.show()

# %% [markdown]
# ## 5. Alert review

# %%
alerts = pd.read_csv("../data/raw/alerts.csv")
print(f"Total alerts: {len(alerts)}")
print("\nRisk level distribution:")
print(alerts["risk_level"].value_counts())

print("\nTop 10 alerts by score:")
cols = ["user_id", "risk_level", "final_score", "summary"]
print(alerts.sort_values("final_score", ascending=False)[cols].head(10).to_string(index=False))
