"""
streaming/kafka_consumer.py
----------------------------
Consumes events from Kafka, maintains per-user feature buffers,
scores users in real time, and publishes alerts back to Kafka.

Topics consumed:  forex.portal.events, forex.trading.events
Topic produced:   forex.alerts

Usage:
    python streaming/kafka_consumer.py
    python streaming/kafka_consumer.py --threshold 0.65
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.baseline.isolation_forest import BaselineDetector
from models.advanced.lstm_autoencoder import LSTMAEDetector
from alerts.alert_generator import generate_alert
from streaming.async_simulator import (
    UserBuffer, _build_feature_vector,
    _quick_portal_features, _quick_trading_features,
)

BOOTSTRAP      = os.getenv("KAFKA_BOOTSTRAP_SERVERS",  "localhost:9092")
PORTAL_TOPIC   = os.getenv("KAFKA_PORTAL_TOPIC",       "forex.portal.events")
TRADING_TOPIC  = os.getenv("KAFKA_TRADING_TOPIC",      "forex.trading.events")
ALERTS_TOPIC   = os.getenv("KAFKA_ALERTS_TOPIC",       "forex.alerts")
GROUP_ID       = "forexguard-consumer-group"


def get_consumer(topics):
    try:
        from kafka import KafkaConsumer
        return KafkaConsumer(
            *topics,
            bootstrap_servers=BOOTSTRAP,
            group_id=GROUP_ID,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            key_deserializer=lambda k: k.decode("utf-8") if k else None,
            auto_offset_reset="earliest",
            enable_auto_commit=True,
        )
    except ImportError:
        raise SystemExit("kafka-python not installed.")
    except Exception as e:
        raise SystemExit(f"Cannot connect to Kafka at {BOOTSTRAP}: {e}")


def get_producer():
    try:
        from kafka import KafkaProducer
        return KafkaProducer(
            bootstrap_servers=BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8"),
        )
    except Exception as e:
        raise SystemExit(f"Cannot connect to Kafka producer: {e}")


def consume(threshold: float = 0.65):
    print("[consumer] Loading models...")
    baseline  = BaselineDetector.load()
    ae        = LSTMAEDetector.load()
    feat_cols = baseline.feat_cols

    consumer = get_consumer([PORTAL_TOPIC, TRADING_TOPIC])
    producer = get_producer()
    buffer   = UserBuffer()

    alerted_users = set()
    processed     = 0

    print(f"[consumer] Listening on topics: {PORTAL_TOPIC}, {TRADING_TOPIC}")
    print(f"[consumer] Alert threshold: {threshold}")

    for msg in consumer:
        topic   = msg.topic
        user_id = msg.key
        record  = msg.value

        if not user_id:
            continue

        if topic == PORTAL_TOPIC:
            buffer.add_portal(user_id, record)
        else:
            buffer.add_trading(user_id, record)

        processed += 1

        # Score every 10 events per user
        p_count = len(buffer.portal_events.get(user_id, []))
        t_count = len(buffer.trading_events.get(user_id, []))
        if (p_count + t_count) % 10 != 0:
            continue

        # Build feature vector and score
        X = _build_feature_vector(buffer, user_id, feat_cols)

        b_scores  = baseline.predict(X)
        a_scores  = ae.predict(X)
        if_score  = float(b_scores["ensemble_score"].iloc[0])
        ae_score  = float(a_scores["lstm_score"].iloc[0])
        final     = round(0.6 * if_score + 0.4 * ae_score, 4)

        if final >= threshold and user_id not in alerted_users:
            alerted_users.add(user_id)

            import json as _json
            b_feats = _json.loads(b_scores["top_features"].iloc[0])
            a_feats = _json.loads(a_scores["lstm_top_features"].iloc[0])
            top     = list(dict.fromkeys(b_feats + a_feats))[:5]

            score_result = {
                "user_id": user_id, "final_score": final,
                "if_score": if_score, "ae_score": ae_score,
                "top_features": top,
            }
            alert = generate_alert(
                score_result,
                buffer.portal_events.get(user_id, []),
                buffer.trading_events.get(user_id, []),
            )

            producer.send(ALERTS_TOPIC, key=user_id, value=alert)
            print(f"[ALERT] {user_id}  score={final:.3f}  {alert['summary'][:80]}")

        if processed % 1000 == 0:
            print(f"[consumer] Processed {processed:,} messages | "
                  f"Alerts fired: {len(alerted_users)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.65)
    args = parser.parse_args()
    consume(threshold=args.threshold)
