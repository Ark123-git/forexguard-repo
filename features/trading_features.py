"""
features/trading_features.py
-----------------------------
Transforms raw trading events into per-user feature vectors.

Features produced:
  Volume & Size
    - total_trades, total_lot_size
    - mean_lot_size, std_lot_size, max_lot_size
    - lot_size_zscore_max          (spike detection)
    - trades_per_day               (activity rate)

  PnL
    - total_pnl, mean_pnl, std_pnl
    - pnl_volatility               (std / abs(mean+1))
    - win_rate                     (% profitable trades)
    - max_single_win, max_single_loss
    - consecutive_wins_max         (latency arb signal)

  Duration / Latency
    - mean_duration_min, std_duration_min
    - pct_sub_minute_trades        (latency arb signal)
    - pct_sub_30s_trades

  Instrument diversity
    - unique_instruments
    - top_instrument_concentration (% trades on single instrument)
    - instrument_entropy

  Margin
    - mean_margin_used, max_margin_used

  Rolling 7-day
    - rolling_7d_trades
    - rolling_7d_pnl
    - rolling_7d_lot_size

  Temporal
    - trade_hour_std               (how scattered across day)
    - weekend_trade_ratio
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy


# ── Helpers ───────────────────────────────────────────────────────────────────

def _instrument_entropy(instruments: pd.Series) -> float:
    counts = instruments.value_counts(normalize=True)
    return float(scipy_entropy(counts)) if len(counts) > 1 else 0.0


def _consecutive_wins(pnl: pd.Series) -> int:
    max_streak = cur = 0
    for v in pnl:
        cur = cur + 1 if v > 0 else 0
        max_streak = max(max_streak, cur)
    return max_streak


def _lot_zscore_max(lots: pd.Series) -> float:
    if len(lots) < 2:
        return 0.0
    mu, sigma = lots.mean(), lots.std()
    if sigma == 0:
        return 0.0
    return float(((lots - mu) / sigma).max())


# ── Per-user feature builder ──────────────────────────────────────────────────

def _user_features(uid: str, group: pd.DataFrame) -> dict:
    group = group.sort_values("open_ts")
    pnl   = group["pnl"]
    lots  = group["lot_size"]
    dur   = group["duration_min"]
    instr = group["instrument"]

    # date range for rate calculation
    date_range_days = max(
        (group["open_ts"].max() - group["open_ts"].min()).days, 1
    )

    # ── volume & size ─────────────────────────────────────────────────────────
    total_trades   = len(group)
    total_lot_size = float(lots.sum())
    mean_lot       = float(lots.mean())
    std_lot        = float(lots.std()) if total_trades > 1 else 0.0
    max_lot        = float(lots.max())
    lot_zscore_max = _lot_zscore_max(lots)
    trades_per_day = total_trades / date_range_days

    # ── PnL ───────────────────────────────────────────────────────────────────
    total_pnl      = float(pnl.sum())
    mean_pnl       = float(pnl.mean())
    std_pnl        = float(pnl.std()) if total_trades > 1 else 0.0
    pnl_volatility = std_pnl / (abs(mean_pnl) + 1.0)
    win_rate       = float((pnl > 0).mean())
    max_win        = float(pnl.max())
    max_loss       = float(pnl.min())
    consec_wins    = _consecutive_wins(pnl)

    # ── duration ──────────────────────────────────────────────────────────────
    mean_dur       = float(dur.mean())
    std_dur        = float(dur.std()) if total_trades > 1 else 0.0
    pct_sub_min    = float((dur < 1.0).mean())
    pct_sub_30s    = float((dur < 0.5).mean())

    # ── instruments ───────────────────────────────────────────────────────────
    unique_instr   = instr.nunique()
    top_conc       = float(instr.value_counts(normalize=True).iloc[0])
    instr_ent      = _instrument_entropy(instr)

    # ── margin ────────────────────────────────────────────────────────────────
    mean_margin    = float(group["margin_used"].mean())
    max_margin     = float(group["margin_used"].max())

    # ── rolling 7-day ─────────────────────────────────────────────────────────
    last_ts   = group["open_ts"].max()
    cutoff_7d = last_ts - pd.Timedelta(days=7)
    g7        = group[group["open_ts"] >= cutoff_7d]
    roll_trades   = len(g7)
    roll_pnl      = float(g7["pnl"].sum())
    roll_lot      = float(g7["lot_size"].sum())

    # ── temporal ──────────────────────────────────────────────────────────────
    hours          = group["open_ts"].dt.hour
    trade_hour_std = float(hours.std()) if total_trades > 1 else 0.0
    weekend_ratio  = float(group["open_ts"].dt.dayofweek.isin([5, 6]).mean())

    return {
        "user_id":                    uid,
        "total_trades":               total_trades,
        "total_lot_size":             round(total_lot_size, 2),
        "mean_lot_size":              round(mean_lot, 4),
        "std_lot_size":               round(std_lot, 4),
        "max_lot_size":               round(max_lot, 2),
        "lot_zscore_max":             round(lot_zscore_max, 4),
        "trades_per_day":             round(trades_per_day, 4),
        "total_pnl":                  round(total_pnl, 2),
        "mean_pnl":                   round(mean_pnl, 4),
        "std_pnl":                    round(std_pnl, 4),
        "pnl_volatility":             round(pnl_volatility, 4),
        "win_rate":                   round(win_rate, 4),
        "max_single_win":             round(max_win, 2),
        "max_single_loss":            round(max_loss, 2),
        "consecutive_wins_max":       consec_wins,
        "mean_duration_min":          round(mean_dur, 4),
        "std_duration_min":           round(std_dur, 4),
        "pct_sub_minute_trades":      round(pct_sub_min, 4),
        "pct_sub_30s_trades":         round(pct_sub_30s, 4),
        "unique_instruments":         unique_instr,
        "top_instrument_concentration": round(top_conc, 4),
        "instrument_entropy":         round(instr_ent, 4),
        "mean_margin_used":           round(mean_margin, 2),
        "max_margin_used":            round(max_margin, 2),
        "rolling_7d_trades":          roll_trades,
        "rolling_7d_pnl":             round(roll_pnl, 2),
        "rolling_7d_lot_size":        round(roll_lot, 2),
        "trade_hour_std":             round(trade_hour_std, 4),
        "weekend_trade_ratio":        round(weekend_ratio, 4),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def build_trading_features(trading_df: pd.DataFrame,
                           output_path: str = "data/raw/trading_features.csv") -> pd.DataFrame:
    trading_df = trading_df.copy()
    trading_df["open_ts"]  = pd.to_datetime(trading_df["open_ts"])
    trading_df["close_ts"] = pd.to_datetime(trading_df["close_ts"])

    print(f"[trading_features] Building features for {trading_df['user_id'].nunique()} users...")

    rows = []
    for uid, group in trading_df.groupby("user_id"):
        rows.append(_user_features(uid, group))

    feat_df = pd.DataFrame(rows)

    labels = (trading_df.groupby("user_id")["anomaly_flag"]
              .any()
              .reset_index()
              .rename(columns={"anomaly_flag": "is_anomalous"}))
    feat_df = feat_df.merge(labels, on="user_id", how="left")

    feat_df.to_csv(output_path, index=False)
    print(f"[trading_features] Saved {len(feat_df)} user rows → {output_path}")
    return feat_df


if __name__ == "__main__":
    import os
    os.makedirs("data/raw", exist_ok=True)
    df = pd.read_csv("data/raw/trading_events.csv")
    build_trading_features(df)
