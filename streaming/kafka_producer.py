"""
streaming/kafka_producer.py
----------------------------
Reads raw event CSVs and publishes them to Kafka topics,
simulating a real brokerage event stream.

Topics:
    forex.portal.events   -- login, KYC, deposit, withdrawal events
    forex.trading.events  -- trade open/close events

Usage:
    python streaming/kafka_producer.py
    python streaming/kafka_producer.py --speed 500
"""

import argparse
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

PORTAL_TOPIC  = os.getenv("KAFKA_PORTAL_TOPIC",  "forex.portal.events")
TRADING_TOPIC = os.getenv("KAFKA_TRADING_TOPIC", "forex.trading.events")
BOOTSTRAP     = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

PORTAL_PATH  = "data/raw/portal_events.csv"
TRADING_PATH = "data/raw/trading_events.csv"


def get_producer():
    try:
        from kafka import KafkaProducer
        producer = KafkaProducer(
            bootstrap_servers=BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8"),
        )
        return producer
    except ImportError:
        raise SystemExit("kafka-python not installed. Run: pip install kafka-python")
    except Exception as e:
        raise SystemExit(f"Cannot connect to Kafka at {BOOTSTRAP}: {e}")


def produce(speed: int = 100):
    producer = get_producer()

    portal_df  = pd.read_csv(PORTAL_PATH)
    trading_df = pd.read_csv(TRADING_PATH)

    portal_df["_sort_ts"]  = pd.to_datetime(portal_df["timestamp"])
    trading_df["_sort_ts"] = pd.to_datetime(trading_df["open_ts"])

    portal_records  = portal_df.sort_values("_sort_ts").to_dict("records")
    trading_records = trading_df.sort_values("_sort_ts").to_dict("records")

    # Interleave by timestamp
    all_events = []
    for r in portal_records:
        all_events.append(("portal", r["user_id"], r))
    for r in trading_records:
        all_events.append(("trading", r["user_id"], r))

    all_events.sort(key=lambda x: str(x[2].get("_sort_ts", "")))

    print(f"[producer] Publishing {len(all_events):,} events to Kafka | speed={speed}x")
    sent = 0
    for source, user_id, record in all_events:
        record_clean = {k: str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v
                        for k, v in record.items() if not k.startswith("_")}
        topic = PORTAL_TOPIC if source == "portal" else TRADING_TOPIC
        producer.send(topic, key=user_id, value=record_clean)
        sent += 1
        if sent % 500 == 0:
            print(f"[producer] Sent {sent:,} / {len(all_events):,}")
            producer.flush()
            time.sleep(500 / (speed * 1000))

    producer.flush()
    print(f"[producer] Done. Total sent: {sent:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=int, default=100)
    args = parser.parse_args()
    produce(speed=args.speed)
