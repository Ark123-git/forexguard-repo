"""
features/portal_features.py
----------------------------
Transforms raw portal events into per-user feature vectors.

Features produced (per user):
  Login behavior
    - total_logins, failed_login_ratio
    - unique_ips, unique_devices, unique_countries
    - login_hour_mean, login_hour_std          (circadian rhythm)
    - pct_night_logins                         (00:00–05:59)
    - max_ips_in_1h                            (multi-IP burst)
    - mean_inter_login_minutes                 (session cadence)
    - ip_entropy                               (IP diversity score)

  Financial behavior
    - total_deposits, total_withdrawals
    - deposit_count, withdrawal_count
    - mean_deposit_amt, mean_withdrawal_amt
    - deposit_withdraw_ratio                   (abuse signal)
    - max_deposit_amt, max_withdrawal_amt
    - structuring_score                        (small deposits count)
    - kyc_change_count
    - kyc_before_withdrawal_flag               (burst KYC + withdrawal)

  Rolling 7-day windows (using last 7 days of data)
    - rolling_7d_logins
    - rolling_7d_deposits
    - rolling_7d_withdrawals

  Session
    - support_ticket_count
    - doc_upload_count
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ip_entropy(ips: pd.Series) -> float:
    counts = ips.value_counts(normalize=True)
    return float(scipy_entropy(counts)) if len(counts) > 1 else 0.0


def _max_ips_in_window(group: pd.DataFrame, window_minutes=60) -> int:
    """Max number of distinct IPs used in any rolling window."""
    if group.empty:
        return 0
    logins = group[group["event_type"] == "login"].sort_values("timestamp")
    if logins.empty:
        return 0
    max_ips = 1
    for i, row in logins.iterrows():
        window_end  = row["timestamp"] + pd.Timedelta(minutes=window_minutes)
        in_window   = logins[(logins["timestamp"] >= row["timestamp"]) &
                             (logins["timestamp"] <= window_end)]
        max_ips = max(max_ips, in_window["ip_address"].nunique())
    return max_ips


def _structuring_score(deposits: pd.Series, threshold=500) -> int:
    """Count deposits just below a reporting threshold."""
    return int((deposits < threshold).sum())


def _kyc_before_withdrawal_flag(group: pd.DataFrame, window_hours=4) -> int:
    """1 if user had ≥3 KYC changes within 4h before a withdrawal."""
    kyc_events = group[group["event_type"] == "kyc_change"]["timestamp"]
    withdrawals = group[group["event_type"] == "withdrawal"]["timestamp"]
    if kyc_events.empty or withdrawals.empty:
        return 0
    for wts in withdrawals:
        start = wts - pd.Timedelta(hours=window_hours)
        recent_kyc = kyc_events[(kyc_events >= start) & (kyc_events <= wts)]
        if len(recent_kyc) >= 3:
            return 1
    return 0


# ── Per-user feature builder ──────────────────────────────────────────────────

def _user_features(uid: str, group: pd.DataFrame) -> dict:
    group = group.sort_values("timestamp")

    logins     = group[group["event_type"] == "login"]
    deposits   = group[group["event_type"] == "deposit"]
    withdrawals= group[group["event_type"] == "withdrawal"]
    kyc        = group[group["event_type"] == "kyc_change"]
    tickets    = group[group["event_type"] == "support_ticket"]

    # ── login features ────────────────────────────────────────────────────────
    total_logins       = len(logins)
    failed_logins      = (logins["status"] == "failed").sum()
    failed_login_ratio = failed_logins / max(total_logins, 1)
    unique_ips         = logins["ip_address"].nunique()
    unique_devices     = logins["device_id"].nunique()
    unique_countries   = logins["country"].nunique()
    ip_ent             = _ip_entropy(logins["ip_address"])

    if total_logins > 0:
        hours           = logins["timestamp"].dt.hour
        login_hour_mean = float(hours.mean())
        login_hour_std  = float(hours.std()) if total_logins > 1 else 0.0
        pct_night       = float((hours.between(0, 5)).mean())
        inter_login_min = (logins["timestamp"].diff().dt.total_seconds() / 60
                           ).dropna().mean()
        inter_login_min = float(inter_login_min) if not np.isnan(inter_login_min) else 0.0
    else:
        login_hour_mean = login_hour_std = pct_night = inter_login_min = 0.0

    max_ips_1h = _max_ips_in_window(group)

    # ── financial features ────────────────────────────────────────────────────
    dep_amounts  = deposits["amount"].dropna()
    wd_amounts   = withdrawals["amount"].dropna()

    total_dep    = float(dep_amounts.sum())
    total_wd     = float(wd_amounts.sum())
    dep_count    = len(dep_amounts)
    wd_count     = len(wd_amounts)
    mean_dep     = float(dep_amounts.mean()) if dep_count > 0 else 0.0
    mean_wd      = float(wd_amounts.mean())  if wd_count  > 0 else 0.0
    max_dep      = float(dep_amounts.max())  if dep_count > 0 else 0.0
    max_wd       = float(wd_amounts.max())   if wd_count  > 0 else 0.0
    dep_wd_ratio = total_wd / max(total_dep, 1)
    struct_score = _structuring_score(dep_amounts)
    kyc_count    = len(kyc)
    kyc_wd_flag  = _kyc_before_withdrawal_flag(group)

    # ── rolling 7-day (last 7 days of activity) ───────────────────────────────
    last_ts   = group["timestamp"].max()
    cutoff_7d = last_ts - pd.Timedelta(days=7)
    g7        = group[group["timestamp"] >= cutoff_7d]
    roll_logins = (g7["event_type"] == "login").sum()
    roll_deps   = (g7["event_type"] == "deposit").sum()
    roll_wds    = (g7["event_type"] == "withdrawal").sum()

    # ── misc ──────────────────────────────────────────────────────────────────
    ticket_count = len(tickets)

    return {
        "user_id":              uid,
        "total_logins":         total_logins,
        "failed_login_ratio":   round(failed_login_ratio, 4),
        "unique_ips":           unique_ips,
        "unique_devices":       unique_devices,
        "unique_countries":     unique_countries,
        "ip_entropy":           round(ip_ent, 4),
        "login_hour_mean":      round(login_hour_mean, 2),
        "login_hour_std":       round(login_hour_std, 2),
        "pct_night_logins":     round(pct_night, 4),
        "max_ips_in_1h":        max_ips_1h,
        "inter_login_min":      round(inter_login_min, 2),
        "total_deposits":       round(total_dep, 2),
        "total_withdrawals":    round(total_wd, 2),
        "deposit_count":        dep_count,
        "withdrawal_count":     wd_count,
        "mean_deposit_amt":     round(mean_dep, 2),
        "mean_withdrawal_amt":  round(mean_wd, 2),
        "max_deposit_amt":      round(max_dep, 2),
        "max_withdrawal_amt":   round(max_wd, 2),
        "deposit_withdraw_ratio": round(dep_wd_ratio, 4),
        "structuring_score":    struct_score,
        "kyc_change_count":     kyc_count,
        "kyc_before_withdrawal":kyc_wd_flag,
        "rolling_7d_logins":    int(roll_logins),
        "rolling_7d_deposits":  int(roll_deps),
        "rolling_7d_withdrawals": int(roll_wds),
        "support_ticket_count": ticket_count,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def build_portal_features(portal_df: pd.DataFrame,
                          output_path: str = "data/raw/portal_features.csv") -> pd.DataFrame:
    """
    Args:
        portal_df : raw portal events DataFrame
        output_path: where to save the features CSV
    Returns:
        user-level feature DataFrame
    """
    portal_df = portal_df.copy()
    portal_df["timestamp"] = pd.to_datetime(portal_df["timestamp"])

    print(f"[portal_features] Building features for {portal_df['user_id'].nunique()} users...")

    rows = []
    for uid, group in portal_df.groupby("user_id"):
        rows.append(_user_features(uid, group))

    feat_df = pd.DataFrame(rows)

    # Carry over ground-truth anomaly label (any event flagged → user is anomalous)
    labels = (portal_df.groupby("user_id")["anomaly_flag"]
              .any()
              .reset_index()
              .rename(columns={"anomaly_flag": "is_anomalous"}))
    feat_df = feat_df.merge(labels, on="user_id", how="left")

    feat_df.to_csv(output_path, index=False)
    print(f"[portal_features] Saved {len(feat_df)} user rows → {output_path}")
    return feat_df


if __name__ == "__main__":
    import os
    os.makedirs("data/raw", exist_ok=True)
    df = pd.read_csv("data/raw/portal_events.csv")
    build_portal_features(df)
