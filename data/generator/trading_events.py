"""
trading_events.py
-----------------
Generates synthetic WebTrader / TradingTerminal events.
Columns: trade_id, user_id, instrument, direction, lot_size,
         open_ts, close_ts, duration_min, pnl, margin_used,
         anomaly_flag, anomaly_type

Anomaly types:
  - volume_spike                   : 10x lot size / trade count in one day
  - single_instrument_concentration: only trades one instrument
  - latency_arbitrage              : sub-30s trades, always profitable
  - high_pnl_volatility            : huge swinging PnL
  - synchronized_trading           : colluding accounts trade simultaneously
"""

import random
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

rng = np.random.default_rng(99)

# ── Config ────────────────────────────────────────────────────────────────────
N_NORMAL_USERS    = 450
N_ANOMALOUS_USERS = 50
START_DATE        = datetime(2024, 1, 1)
END_DATE          = datetime(2024, 6, 30)
TOTAL_DAYS        = (END_DATE - START_DATE).days

INSTRUMENTS = ["EURUSD","GBPUSD","USDJPY","XAUUSD","BTCUSD",
               "AUDUSD","USDCAD","NZDUSD","USDCHF","EURJPY"]
DIRECTIONS  = ["buy", "sell"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def random_ts(start=START_DATE, end=END_DATE) -> datetime:
    delta = int((end - start).total_seconds())
    return start + timedelta(seconds=int(rng.integers(0, delta)))


def make_trade(user_id, instrument, direction, lot_size,
               open_ts, close_ts, pnl, margin_used,
               anomaly_flag=False, anomaly_type=""):
    return {
        "trade_id":     str(uuid.uuid4()),
        "user_id":      user_id,
        "instrument":   instrument,
        "direction":    direction,
        "lot_size":     round(float(lot_size), 2),
        "open_ts":      open_ts.isoformat(),
        "close_ts":     close_ts.isoformat(),
        "duration_min": round((close_ts - open_ts).total_seconds() / 60, 2),
        "pnl":          round(float(pnl), 2),
        "margin_used":  round(float(margin_used), 2),
        "anomaly_flag": anomaly_flag,
        "anomaly_type": anomaly_type,
    }


def _normal_trade(user_id: str) -> dict:
    instrument  = random.choice(INSTRUMENTS)
    direction   = random.choice(DIRECTIONS)
    lot_size    = float(rng.choice([0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
                                   p=[0.30, 0.25, 0.20, 0.12, 0.08, 0.05]))
    open_ts     = random_ts()
    hold_min    = int(rng.integers(5, 1440))
    close_ts    = open_ts + timedelta(minutes=hold_min)
    pnl         = float(rng.standard_normal() * lot_size * 50)
    margin_used = lot_size * 1000 * float(rng.uniform(0.01, 0.05))
    return make_trade(user_id, instrument, direction, lot_size,
                      open_ts, close_ts, pnl, margin_used)


# ── Normal users ──────────────────────────────────────────────────────────────

def generate_normal_trading(user_id: str) -> list:
    n = int(rng.integers(10, 120))
    return [_normal_trade(user_id) for _ in range(n)]


# ── Anomaly injectors ─────────────────────────────────────────────────────────

def inject_volume_spike(user_id: str) -> list:
    events = []
    spike_day = START_DATE + timedelta(days=int(rng.integers(0, TOTAL_DAYS)))
    spike_end = spike_day + timedelta(hours=23)
    for _ in range(int(rng.integers(40, 80))):
        instrument  = random.choice(INSTRUMENTS)
        direction   = random.choice(DIRECTIONS)
        lot_size    = float(rng.choice([5.0, 10.0, 20.0]))
        open_ts     = spike_day + timedelta(minutes=int(rng.integers(0, 1380)))
        hold_min    = int(rng.integers(1, 30))
        close_ts    = min(open_ts + timedelta(minutes=hold_min), spike_end)
        pnl         = float(rng.standard_normal() * lot_size * 30)
        margin_used = lot_size * 1000 * 0.02
        events.append(make_trade(user_id, instrument, direction, lot_size,
                                 open_ts, close_ts, pnl, margin_used,
                                 anomaly_flag=True, anomaly_type="volume_spike"))
    return events


def inject_single_instrument(user_id: str) -> list:
    events = []
    instrument = "XAUUSD"
    for _ in range(int(rng.integers(30, 60))):
        direction   = random.choice(DIRECTIONS)
        lot_size    = float(rng.choice([0.5, 1.0, 2.0]))
        open_ts     = random_ts()
        hold_min    = int(rng.integers(5, 120))
        close_ts    = open_ts + timedelta(minutes=hold_min)
        pnl         = float(rng.standard_normal() * lot_size * 40)
        margin_used = lot_size * 1000 * 0.02
        events.append(make_trade(user_id, instrument, direction, lot_size,
                                 open_ts, close_ts, pnl, margin_used,
                                 anomaly_flag=True, anomaly_type="single_instrument_concentration"))
    return events


def inject_latency_arbitrage(user_id: str) -> list:
    events = []
    for _ in range(int(rng.integers(20, 40))):
        instrument  = random.choice(INSTRUMENTS)
        direction   = random.choice(DIRECTIONS)
        lot_size    = float(rng.choice([1.0, 2.0, 5.0]))
        open_ts     = random_ts()
        hold_sec    = int(rng.integers(1, 30))
        close_ts    = open_ts + timedelta(seconds=hold_sec)
        pnl         = float(rng.uniform(10, 200))    # always wins
        margin_used = lot_size * 1000 * 0.02
        events.append(make_trade(user_id, instrument, direction, lot_size,
                                 open_ts, close_ts, pnl, margin_used,
                                 anomaly_flag=True, anomaly_type="latency_arbitrage"))
    return events


def inject_high_pnl_volatility(user_id: str) -> list:
    events = []
    for _ in range(int(rng.integers(15, 30))):
        instrument  = random.choice(INSTRUMENTS)
        direction   = random.choice(DIRECTIONS)
        lot_size    = float(rng.choice([10.0, 20.0, 50.0]))
        open_ts     = random_ts()
        hold_min    = int(rng.integers(5, 60))
        close_ts    = open_ts + timedelta(minutes=hold_min)
        sign        = float(rng.choice([-1.0, 1.0]))
        pnl         = sign * float(rng.uniform(1_000, 10_000))
        margin_used = lot_size * 1000 * 0.02
        events.append(make_trade(user_id, instrument, direction, lot_size,
                                 open_ts, close_ts, pnl, margin_used,
                                 anomaly_flag=True, anomaly_type="high_pnl_volatility"))
    return events


def inject_synchronized_trading(user_ids: list) -> list:
    """Ring of users all placing same trade within seconds."""
    events = []
    base_ts    = random_ts()
    instrument = random.choice(INSTRUMENTS)
    direction  = random.choice(DIRECTIONS)
    lot_size   = 1.0
    for i, uid in enumerate(user_ids):
        open_ts  = base_ts + timedelta(seconds=i * int(rng.integers(1, 5)))
        close_ts = open_ts + timedelta(minutes=int(rng.integers(10, 30)))
        pnl      = float(rng.uniform(50, 300))
        events.append(make_trade(uid, instrument, direction, lot_size,
                                 open_ts, close_ts, pnl, 20.0,
                                 anomaly_flag=True, anomaly_type="synchronized_trading"))
    return events


ANOMALY_INJECTORS = [
    inject_volume_spike,
    inject_single_instrument,
    inject_latency_arbitrage,
    inject_high_pnl_volatility,
]


def generate_anomalous_trading(user_id: str) -> list:
    events = generate_normal_trading(user_id)
    chosen = random.sample(ANOMALY_INJECTORS, k=int(rng.integers(1, 3)))
    for fn in chosen:
        events.extend(fn(user_id))
    return events


# ── Main ──────────────────────────────────────────────────────────────────────

def generate_trading_events(n_normal=N_NORMAL_USERS,
                            n_anomalous=N_ANOMALOUS_USERS,
                            output_path="data/raw/trading_events.csv") -> pd.DataFrame:
    print(f"[trading] Generating {n_normal} normal + {n_anomalous} anomalous users...")
    all_events = []

    normal_ids    = [f"U{i:05d}" for i in range(n_normal)]
    anomalous_ids = [f"U{i:05d}" for i in range(n_normal, n_normal + n_anomalous)]

    for uid in normal_ids:
        all_events.extend(generate_normal_trading(uid))
    for uid in anomalous_ids:
        all_events.extend(generate_anomalous_trading(uid))

    # Synchronized trading ring
    ring = random.sample(anomalous_ids, k=min(5, len(anomalous_ids)))
    all_events.extend(inject_synchronized_trading(ring))

    df = pd.DataFrame(all_events)
    df["open_ts"]  = pd.to_datetime(df["open_ts"])
    df["close_ts"] = pd.to_datetime(df["close_ts"])
    df = df.sort_values("open_ts").reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"[trading] Saved {len(df):,} trades → {output_path}")
    return df


if __name__ == "__main__":
    import os; os.makedirs("data/raw", exist_ok=True)
    generate_trading_events()
