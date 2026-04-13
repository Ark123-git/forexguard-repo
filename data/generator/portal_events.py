"""
portal_events.py
----------------
Generates synthetic Client Portal events for ~N users over a date range.
Events: login, account_modify, kyc_change, deposit, withdrawal, support_ticket, doc_upload

Anomaly types injected (labeled for evaluation):
  - multi_ip_login            : same user from 5+ IPs in minutes
  - unusual_hour_login        : consistent 3 AM logins
  - deposit_withdraw_abuse    : deposit → idle → near-full withdrawal
  - structuring               : many small deposits just under threshold
  - kyc_before_withdrawal     : burst of KYC changes then big withdrawal
  - brute_force_login         : many failed logins then success
"""

import random
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

rng = np.random.default_rng(42)

# ── Config ────────────────────────────────────────────────────────────────────
N_NORMAL_USERS    = 450
N_ANOMALOUS_USERS = 50
START_DATE        = datetime(2024, 1, 1)
END_DATE          = datetime(2024, 6, 30)
TOTAL_DAYS        = (END_DATE - START_DATE).days

# Generate fake IPs deterministically
def _fake_ip(seed):
    r = np.random.default_rng(seed)
    return ".".join(str(r.integers(1, 255)) for _ in range(4))

IP_POOL     = [_fake_ip(i) for i in range(300)]
DEVICE_POOL = [f"dev-{i:04d}" for i in range(200)]
COUNTRY_POOL = ["IN","US","GB","DE","AE","SG","NG","ZA","BR","JP",
                "FR","CA","AU","NL","CH","HK","KR","MX","TR","SA"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def random_ts(start=START_DATE, end=END_DATE) -> datetime:
    delta = int((end - start).total_seconds())
    return start + timedelta(seconds=int(rng.integers(0, delta)))


def make_event(user_id, event_type, ts, ip, device, country,
               amount=None, status="success",
               anomaly_flag=False, anomaly_type=""):
    return {
        "event_id":     str(uuid.uuid4()),
        "user_id":      user_id,
        "event_type":   event_type,
        "timestamp":    ts.isoformat(),
        "ip_address":   ip,
        "device_id":    device,
        "country":      country,
        "amount":       round(float(amount), 2) if amount is not None else None,
        "status":       status,
        "anomaly_flag": anomaly_flag,
        "anomaly_type": anomaly_type,
    }


# ── Normal user ───────────────────────────────────────────────────────────────

def generate_normal_user_events(user_id: str) -> list:
    events = []
    ip      = random.choice(IP_POOL)
    device  = random.choice(DEVICE_POOL)
    country = random.choice(COUNTRY_POOL)
    n_active_days = int(rng.integers(10, TOTAL_DAYS))

    for _ in range(n_active_days):
        day_offset = int(rng.integers(0, TOTAL_DAYS))
        day_start  = START_DATE + timedelta(days=day_offset)
        day_end    = day_start + timedelta(hours=22)

        # 1-3 logins
        for _ in range(int(rng.integers(1, 4))):
            ts = random_ts(day_start, day_end)
            events.append(make_event(user_id, "login", ts, ip, device, country))

        if rng.random() < 0.15:
            ts  = random_ts(day_start, day_end)
            amt = float(rng.integers(100, 5_000))
            events.append(make_event(user_id, "deposit", ts, ip, device, country, amount=amt))

        if rng.random() < 0.07:
            ts  = random_ts(day_start, day_end)
            amt = float(rng.integers(50, 3_000))
            events.append(make_event(user_id, "withdrawal", ts, ip, device, country, amount=amt))

        if rng.random() < 0.03:
            ts = random_ts(day_start, day_end)
            events.append(make_event(user_id, "support_ticket", ts, ip, device, country))

        if rng.random() < 0.01:
            ts = random_ts(day_start, day_end)
            events.append(make_event(user_id, "kyc_change", ts, ip, device, country))

    return events


# ── Anomaly injectors ─────────────────────────────────────────────────────────

def inject_multi_ip_login(user_id: str) -> list:
    events = []
    base_ts = random_ts()
    country = random.choice(COUNTRY_POOL)
    device  = random.choice(DEVICE_POOL)
    ips     = random.sample(IP_POOL, k=int(rng.integers(5, 10)))
    for i, ip in enumerate(ips):
        ts = base_ts + timedelta(minutes=i * int(rng.integers(1, 5)))
        events.append(make_event(user_id, "login", ts, ip, device, country,
                                 anomaly_flag=True, anomaly_type="multi_ip_login"))
    return events


def inject_3am_logins(user_id: str) -> list:
    events = []
    ip      = random.choice(IP_POOL)
    device  = random.choice(DEVICE_POOL)
    country = random.choice(COUNTRY_POOL)
    for _ in range(int(rng.integers(10, 25))):
        day_offset = int(rng.integers(0, TOTAL_DAYS))
        ts = START_DATE + timedelta(days=day_offset, hours=3,
                                    minutes=int(rng.integers(0, 30)))
        events.append(make_event(user_id, "login", ts, ip, device, country,
                                 anomaly_flag=True, anomaly_type="unusual_hour_login"))
    return events


def inject_deposit_withdraw_abuse(user_id: str) -> list:
    events = []
    ip      = random.choice(IP_POOL)
    device  = random.choice(DEVICE_POOL)
    country = random.choice(COUNTRY_POOL)
    base_ts = random_ts(START_DATE, END_DATE - timedelta(days=5))

    deposit_amt = float(rng.integers(5_000, 20_000))
    events.append(make_event(user_id, "deposit", base_ts, ip, device, country,
                             amount=deposit_amt,
                             anomaly_flag=True, anomaly_type="deposit_withdraw_abuse"))

    for i in range(int(rng.integers(1, 3))):
        ts = base_ts + timedelta(hours=i + 1)
        events.append(make_event(user_id, "login", ts, ip, device, country,
                                 anomaly_flag=True, anomaly_type="deposit_withdraw_abuse"))

    withdraw_ts  = base_ts + timedelta(days=int(rng.integers(1, 4)))
    withdraw_amt = deposit_amt * rng.uniform(0.80, 0.99)
    events.append(make_event(user_id, "withdrawal", withdraw_ts, ip, device, country,
                             amount=float(withdraw_amt),
                             anomaly_flag=True, anomaly_type="deposit_withdraw_abuse"))
    return events


def inject_structuring(user_id: str) -> list:
    events = []
    ip      = random.choice(IP_POOL)
    device  = random.choice(DEVICE_POOL)
    country = random.choice(COUNTRY_POOL)
    base_ts = random_ts(START_DATE, END_DATE - timedelta(days=1))
    for i in range(int(rng.integers(15, 30))):
        ts  = base_ts + timedelta(minutes=i * int(rng.integers(2, 10)))
        amt = float(rng.integers(50, 499))
        events.append(make_event(user_id, "deposit", ts, ip, device, country,
                                 amount=amt,
                                 anomaly_flag=True, anomaly_type="structuring"))
    return events


def inject_kyc_before_withdrawal(user_id: str) -> list:
    events = []
    ip      = random.choice(IP_POOL)
    device  = random.choice(DEVICE_POOL)
    country = random.choice(COUNTRY_POOL)
    base_ts = random_ts(START_DATE, END_DATE - timedelta(days=1))
    for i in range(int(rng.integers(3, 7))):
        ts = base_ts + timedelta(minutes=i * 10)
        events.append(make_event(user_id, "kyc_change", ts, ip, device, country,
                                 anomaly_flag=True, anomaly_type="kyc_before_withdrawal"))
    ts  = base_ts + timedelta(hours=2)
    amt = float(rng.integers(10_000, 50_000))
    events.append(make_event(user_id, "withdrawal", ts, ip, device, country,
                             amount=amt,
                             anomaly_flag=True, anomaly_type="kyc_before_withdrawal"))
    return events


def inject_brute_force_login(user_id: str) -> list:
    events = []
    ip      = random.choice(IP_POOL)
    device  = random.choice(DEVICE_POOL)
    country = random.choice(COUNTRY_POOL)
    base_ts = random_ts()
    for i in range(int(rng.integers(5, 15))):
        ts = base_ts + timedelta(seconds=i * 30)
        events.append(make_event(user_id, "login", ts, ip, device, country,
                                 status="failed",
                                 anomaly_flag=True, anomaly_type="brute_force_login"))
    ts = base_ts + timedelta(minutes=10)
    events.append(make_event(user_id, "login", ts, ip, device, country,
                             status="success",
                             anomaly_flag=True, anomaly_type="brute_force_login"))
    return events


ANOMALY_INJECTORS = [
    inject_multi_ip_login,
    inject_3am_logins,
    inject_deposit_withdraw_abuse,
    inject_structuring,
    inject_kyc_before_withdrawal,
    inject_brute_force_login,
]


def generate_anomalous_user_events(user_id: str) -> list:
    events = generate_normal_user_events(user_id)
    chosen = random.sample(ANOMALY_INJECTORS, k=int(rng.integers(1, 3)))
    for fn in chosen:
        events.extend(fn(user_id))
    return events


# ── Main ──────────────────────────────────────────────────────────────────────

def generate_portal_events(n_normal=N_NORMAL_USERS,
                           n_anomalous=N_ANOMALOUS_USERS,
                           output_path="data/raw/portal_events.csv") -> pd.DataFrame:
    print(f"[portal] Generating {n_normal} normal + {n_anomalous} anomalous users...")
    all_events = []

    normal_ids    = [f"U{i:05d}" for i in range(n_normal)]
    anomalous_ids = [f"U{i:05d}" for i in range(n_normal, n_normal + n_anomalous)]

    for uid in normal_ids:
        all_events.extend(generate_normal_user_events(uid))
    for uid in anomalous_ids:
        all_events.extend(generate_anomalous_user_events(uid))

    df = pd.DataFrame(all_events)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"[portal] Saved {len(df):,} events → {output_path}")
    return df


if __name__ == "__main__":
    import os; os.makedirs("data/raw", exist_ok=True)
    generate_portal_events()
