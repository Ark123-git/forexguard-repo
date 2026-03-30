"""
generate_all.py  — run this once to produce all raw data.

    cd forexguard
    python data/generator/generate_all.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from data.generator.portal_events  import generate_portal_events
from data.generator.trading_events import generate_trading_events

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    p = generate_portal_events(output_path="data/raw/portal_events.csv")
    t = generate_trading_events(output_path="data/raw/trading_events.csv")

    print("\n── Summary ──────────────────────────────────────────")
    print(f"  Portal events    : {len(p):>8,}")
    print(f"  Trading events   : {len(t):>8,}")
    print(f"  Total            : {len(p)+len(t):>8,}")
    print(f"\n  Portal anomalies : {p['anomaly_flag'].sum():>6,}  "
          f"({p['anomaly_flag'].mean()*100:.1f}%)")
    print(f"  Trading anomalies: {t['anomaly_flag'].sum():>6,}  "
          f"({t['anomaly_flag'].mean()*100:.1f}%)")
    print("\n  Anomaly types in portal:")
    for k,v in p[p.anomaly_flag].groupby('anomaly_type').size().items():
        print(f"    {k:<35} {v:>5,}")
    print("\n  Anomaly types in trading:")
    for k,v in t[t.anomaly_flag].groupby('anomaly_type').size().items():
        print(f"    {k:<35} {v:>5,}")
