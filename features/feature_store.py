"""
features/feature_store.py
--------------------------
Merges portal + trading feature vectors into one unified user-level
feature matrix ready for model input.

Output columns:
  - user_id
  - all portal features  (prefixed p_)
  - all trading features (prefixed t_) — NaN-filled for non-traders
  - combined anomaly label: is_anomalous (union of both)
  - numeric feature matrix X  (no user_id / label)
"""

import numpy as np
import pandas as pd
from features.portal_features  import build_portal_features
from features.trading_features import build_trading_features


PORTAL_FEATURE_COLS = [
    "total_logins", "failed_login_ratio", "unique_ips", "unique_devices",
    "unique_countries", "ip_entropy", "login_hour_mean", "login_hour_std",
    "pct_night_logins", "max_ips_in_1h", "inter_login_min",
    "total_deposits", "total_withdrawals", "deposit_count", "withdrawal_count",
    "mean_deposit_amt", "mean_withdrawal_amt", "max_deposit_amt", "max_withdrawal_amt",
    "deposit_withdraw_ratio", "structuring_score", "kyc_change_count",
    "kyc_before_withdrawal", "rolling_7d_logins", "rolling_7d_deposits",
    "rolling_7d_withdrawals", "support_ticket_count",
]

TRADING_FEATURE_COLS = [
    "total_trades", "total_lot_size", "mean_lot_size", "std_lot_size",
    "max_lot_size", "lot_zscore_max", "trades_per_day",
    "total_pnl", "mean_pnl", "std_pnl", "pnl_volatility",
    "win_rate", "max_single_win", "max_single_loss", "consecutive_wins_max",
    "mean_duration_min", "std_duration_min", "pct_sub_minute_trades", "pct_sub_30s_trades",
    "unique_instruments", "top_instrument_concentration", "instrument_entropy",
    "mean_margin_used", "max_margin_used",
    "rolling_7d_trades", "rolling_7d_pnl", "rolling_7d_lot_size",
    "trade_hour_std", "weekend_trade_ratio",
]


def build_feature_store(portal_events_path:  str = "data/raw/portal_events.csv",
                        trading_events_path: str = "data/raw/trading_events.csv",
                        output_path:         str = "data/raw/feature_store.csv"
                        ) -> tuple[pd.DataFrame, np.ndarray, list]:
    """
    Returns:
        meta_df  : DataFrame with user_id + is_anomalous
        X        : numpy feature matrix (n_users × n_features)
        feat_cols: list of feature column names matching X
    """
    portal_df  = pd.read_csv(portal_events_path)
    trading_df = pd.read_csv(trading_events_path)

    p_feat = build_portal_features(portal_df,  output_path="data/raw/portal_features.csv")
    t_feat = build_trading_features(trading_df, output_path="data/raw/trading_features.csv")

    # Rename feature cols to avoid collision
    p_rename = {c: f"p_{c}" for c in PORTAL_FEATURE_COLS}
    t_rename = {c: f"t_{c}" for c in TRADING_FEATURE_COLS}

    p_feat = p_feat.rename(columns=p_rename)
    t_feat = t_feat.rename(columns=t_rename)

    p_label_col = "is_anomalous"
    t_label_col = "is_anomalous"

    # Merge on user_id (outer so users without trades still appear)
    merged = pd.merge(
        p_feat[["user_id"] + list(p_rename.values()) + [p_label_col]],
        t_feat[["user_id"] + list(t_rename.values()) + [t_label_col]],
        on="user_id", how="outer", suffixes=("_p", "_t")
    )

    # Combined label: anomalous if either source says so
    merged["is_anomalous"] = (
        merged["is_anomalous_p"].fillna(False) |
        merged["is_anomalous_t"].fillna(False)
    )
    merged = merged.drop(columns=["is_anomalous_p", "is_anomalous_t"])

    # Fill missing trading features with 0 (users who never traded)
    t_cols_prefixed = list(t_rename.values())
    merged[t_cols_prefixed] = merged[t_cols_prefixed].fillna(0)

    # Final feature columns
    feat_cols = list(p_rename.values()) + list(t_rename.values())

    # Impute any remaining NaNs with column median
    X_df = merged[feat_cols].copy()
    X_df = X_df.fillna(X_df.median())

    X = X_df.to_numpy(dtype=np.float32)

    merged.to_csv(output_path, index=False)
    print(f"[feature_store] Saved unified store → {output_path}")
    print(f"[feature_store] Shape: {X.shape}  |  "
          f"Anomalous users: {merged['is_anomalous'].sum()}")

    return merged, X, feat_cols


if __name__ == "__main__":
    build_feature_store()
