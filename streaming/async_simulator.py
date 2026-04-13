"""
streaming/async_simulator.py
-----------------------------
Simulates real-time streaming of events through the anomaly detection pipeline.

When Kafka is available, swap in kafka_producer / kafka_consumer.
This module provides a self-contained async simulation that:
  1. Reads raw events sorted by timestamp
  2. Emits them in configurable time-compressed batches
  3. Maintains a rolling per-user feature buffer
  4. Scores each user after each new event (using the trained models)
  5. Emits alerts when anomaly score exceeds threshold

Usage:
    python streaming/async_simulator.py
    python streaming/async_simulator.py --speed 100 --alert-threshold 0.7
"""

import argparse
import asyncio
import json
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ── Load trained models ───────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.baseline.isolation_forest  import BaselineDetector
from models.advanced.lstm_autoencoder  import LSTMAEDetector
from alerts.alert_generator            import generate_alert

# ── Config ────────────────────────────────────────────────────────────────────
PORTAL_PATH  = "data/raw/portal_events.csv"
TRADING_PATH = "data/raw/trading_events.csv"
BATCH_SIZE   = 50           # events per simulated "tick"
ALERT_THRESHOLD = 0.65      # ensemble score to trigger alert


# ── Rolling user state buffer ─────────────────────────────────────────────────

class UserBuffer:
    """
    Maintains a lightweight rolling state for each user seen in the stream.
    Only stores what's needed for fast incremental feature approximation.
    """
    def __init__(self):
        self.portal_events:  dict[str, list] = defaultdict(list)
        self.trading_events: dict[str, list] = defaultdict(list)
        self.last_scored:    dict[str, float] = {}
        self.alert_history:  dict[str, list]  = defaultdict(list)

    def add_portal(self, user_id: str, event: dict):
        self.portal_events[user_id].append(event)
        # Keep last 200 events per user to bound memory
        if len(self.portal_events[user_id]) > 200:
            self.portal_events[user_id] = self.portal_events[user_id][-200:]

    def add_trading(self, user_id: str, trade: dict):
        self.trading_events[user_id].append(trade)
        if len(self.trading_events[user_id]) > 200:
            self.trading_events[user_id] = self.trading_events[user_id][-200:]

    def get_all_users(self) -> list[str]:
        return list(set(list(self.portal_events.keys()) +
                        list(self.trading_events.keys())))


# ── Fast incremental feature extraction ──────────────────────────────────────

def _quick_portal_features(events: list) -> dict:
    """Lightweight feature extraction from a user's event buffer."""
    if not events:
        return {}
    df = pd.DataFrame(events)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    logins    = df[df["event_type"] == "login"]
    deposits  = df[df["event_type"] == "deposit"]
    withdrawals = df[df["event_type"] == "withdrawal"]
    kyc       = df[df["event_type"] == "kyc_change"]

    n_logins  = len(logins)
    n_failed  = (logins["status"] == "failed").sum() if n_logins else 0
    n_ips     = logins["ip_address"].nunique() if n_logins else 0
    n_countries = logins["country"].nunique() if n_logins else 0

    hours     = logins["timestamp"].dt.hour if n_logins else pd.Series([], dtype=float)
    pct_night = float((hours.between(0, 5)).mean()) if n_logins else 0.0

    dep_amt   = deposits["amount"].dropna()
    wd_amt    = withdrawals["amount"].dropna()
    dep_wd_ratio = float(wd_amt.sum()) / max(float(dep_amt.sum()), 1)
    struct    = int((dep_amt < 500).sum()) if len(dep_amt) else 0

    return {
        "p_total_logins":          n_logins,
        "p_failed_login_ratio":    n_failed / max(n_logins, 1),
        "p_unique_ips":            n_ips,
        "p_unique_countries":      n_countries,
        "p_pct_night_logins":      pct_night,
        "p_deposit_count":         len(dep_amt),
        "p_withdrawal_count":      len(wd_amt),
        "p_deposit_withdraw_ratio":dep_wd_ratio,
        "p_structuring_score":     struct,
        "p_kyc_change_count":      len(kyc),
    }


def _quick_trading_features(trades: list) -> dict:
    if not trades:
        return {}
    df = pd.DataFrame(trades)
    df["open_ts"] = pd.to_datetime(df["open_ts"])

    n = len(df)
    lots = df["lot_size"]
    pnl  = df["pnl"]
    dur  = df["duration_min"]

    pnl_vol   = float(pnl.std()) / (abs(float(pnl.mean())) + 1)
    top_conc  = float(df["instrument"].value_counts(normalize=True).iloc[0]) if n else 0
    win_rate  = float((pnl > 0).mean()) if n else 0
    pct_sub30 = float((dur < 0.5).mean()) if n else 0

    return {
        "t_total_trades":                  n,
        "t_mean_lot_size":                 float(lots.mean()) if n else 0,
        "t_max_lot_size":                  float(lots.max())  if n else 0,
        "t_pnl_volatility":                pnl_vol,
        "t_win_rate":                      win_rate,
        "t_top_instrument_concentration":  top_conc,
        "t_pct_sub_30s_trades":            pct_sub30,
    }


def _build_feature_vector(buffer, user_id, feat_cols):
    p_feats  = _quick_portal_features(buffer.portal_events.get(user_id, []))
    t_feats  = _quick_trading_features(buffer.trading_events.get(user_id, []))
    combined = {**p_feats, **t_feats}
    vec = np.array([combined.get(c, 0.0) for c in feat_cols], dtype=np.float32)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)  # ← ADD THIS
    return vec.reshape(1, -1)

# ── Scoring ───────────────────────────────────────────────────────────────────

def score_user(user_id: str, buffer: UserBuffer,
               baseline: BaselineDetector,
               ae: LSTMAEDetector,
               feat_cols: list) -> dict:
    X = _build_feature_vector(buffer, user_id, feat_cols)

    # Baseline score
    b_scores = baseline.predict(X)
    b_score  = float(b_scores["ensemble_score"].iloc[0])
    b_feats  = json.loads(b_scores["top_features"].iloc[0])

    # AE score
    a_scores = ae.predict(X)
    a_score  = float(a_scores["lstm_score"].iloc[0])
    a_feats  = json.loads(a_scores["lstm_top_features"].iloc[0])

    # Ensemble
    final_score = round(0.6 * b_score + 0.4 * a_score, 4)

    # Union top features
    top_feats = list(dict.fromkeys(b_feats + a_feats))[:5]

    return {
        "user_id":       user_id,
        "if_score":      round(b_score, 4),
        "ae_score":      round(a_score, 4),
        "final_score":   final_score,
        "top_features":  top_feats,
        "is_alert":      final_score >= ALERT_THRESHOLD,
        "scored_at":     datetime.utcnow().isoformat(),
    }


# ── Stream simulation ─────────────────────────────────────────────────────────

async def stream_events(portal_df: pd.DataFrame,
                        trading_df: pd.DataFrame,
                        baseline: BaselineDetector,
                        ae: LSTMAEDetector,
                        feat_cols: list,
                        speed: int = 50,
                        alert_threshold: float = ALERT_THRESHOLD,
                        max_alerts: int = 20):
    """
    Simulates real-time streaming.
    speed : how many real-time seconds 1 simulated second represents
    """
    global ALERT_THRESHOLD
    ALERT_THRESHOLD = alert_threshold

    buffer      = UserBuffer()
    alerts_sent = []

    # Merge and sort all events by timestamp
    portal_df  = portal_df.copy()
    trading_df = trading_df.copy()
    portal_df["_source"]    = "portal"
    trading_df["_source"]   = "trading"
    portal_df["_sort_ts"]   = pd.to_datetime(portal_df["timestamp"])
    trading_df["_sort_ts"]  = pd.to_datetime(trading_df["open_ts"])

    all_events = pd.concat([
        portal_df[["user_id","_source","_sort_ts"]].assign(
            _data=portal_df.drop(columns=["_source","_sort_ts"], errors="ignore").to_dict("records")
        ),
        trading_df[["user_id","_source","_sort_ts"]].assign(
            _data=trading_df.drop(columns=["_source","_sort_ts"], errors="ignore").to_dict("records")
        ),
    ]).sort_values("_sort_ts").reset_index(drop=True)

    print(f"\n[stream] Starting simulation | {len(all_events):,} events | speed={speed}x")
    print(f"[stream] Alert threshold: {ALERT_THRESHOLD}")
    print("─" * 60)

    scored_users = set()
    batch        = []
    tick         = 0

    for _, row in all_events.iterrows():
        uid    = row["user_id"]
        source = row["_source"]
        data   = row["_data"]

        if source == "portal":
            buffer.add_portal(uid, data)
        else:
            buffer.add_trading(uid, data)

        batch.append(uid)

        if len(batch) >= BATCH_SIZE:
            tick += 1
            # Score users seen in this batch (deduplicated)
            to_score = list(set(batch))
            batch    = []

            for u in to_score:
                result = score_user(u, buffer, baseline, ae, feat_cols)

                if result["is_alert"] and u not in scored_users:
                    scored_users.add(u)
                    # Generate human-readable alert
                    p_evts  = buffer.portal_events.get(u, [])
                    t_evts  = buffer.trading_events.get(u, [])
                    alert   = generate_alert(result, p_evts, t_evts)
                    alerts_sent.append(alert)

                    print(f"\n🚨 ALERT  tick={tick:>4}  user={u}")
                    print(f"   Score: {result['final_score']:.3f}  "
                          f"(IF={result['if_score']:.3f}, AE={result['ae_score']:.3f})")
                    print(f"   Reason: {alert['summary']}")
                    print(f"   Top features: {', '.join(result['top_features'][:3])}")

                    if len(alerts_sent) >= max_alerts:
                        print(f"\n[stream] Reached max_alerts={max_alerts}. Stopping.")
                        return alerts_sent

            # Simulate time delay (speed-compressed)
            await asyncio.sleep(BATCH_SIZE / (speed * 1000))

    print(f"\n[stream] Done. Total alerts: {len(alerts_sent)}")
    return alerts_sent


# ── Main ──────────────────────────────────────────────────────────────────────

def main(speed: int = 100, alert_threshold: float = 0.65):
    # Load models
    print("[stream] Loading models...")
    baseline  = BaselineDetector.load()
    ae        = LSTMAEDetector.load()
    feat_cols = baseline.feat_cols

    # Load data
    portal_df  = pd.read_csv(PORTAL_PATH)
    trading_df = pd.read_csv(TRADING_PATH)

    alerts = asyncio.run(
        stream_events(portal_df, trading_df, baseline, ae, feat_cols,
                      speed=speed, alert_threshold=alert_threshold)
    )

    # Save alerts
    if alerts:
        pd.DataFrame(alerts).to_csv("data/raw/stream_alerts.csv", index=False)
        print(f"[stream] Alerts saved → data/raw/stream_alerts.csv")
    return alerts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=int, default=100,
                        help="Time compression factor (default 100x)")
    parser.add_argument("--alert-threshold", type=float, default=0.65,
                        help="Score threshold for alert (default 0.65)")
    args = parser.parse_args()
    main(speed=args.speed, alert_threshold=args.alert_threshold)
