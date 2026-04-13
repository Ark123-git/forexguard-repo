"""
tests/test_pipeline.py
-----------------------
Unit and integration tests for the ForexGuard pipeline.

Run with:
    cd forexguard
    pytest tests/ -v
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ---- Data generation tests --------------------------------------------------

class TestDataGeneration:
    def test_portal_events_shape(self):
        from data.generator.portal_events import generate_portal_events
        df = generate_portal_events(n_normal=10, n_anomalous=2,
                                     output_path="/tmp/test_portal.csv")
        assert len(df) > 0
        required_cols = ["event_id", "user_id", "event_type", "timestamp",
                         "ip_address", "device_id", "country", "anomaly_flag"]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_portal_event_types(self):
        from data.generator.portal_events import generate_portal_events
        df = generate_portal_events(n_normal=20, n_anomalous=5,
                                     output_path="/tmp/test_portal2.csv")
        valid_types = {"login", "deposit", "withdrawal", "kyc_change",
                       "support_ticket", "doc_upload"}
        assert set(df["event_type"].unique()).issubset(valid_types)

    def test_trading_events_shape(self):
        from data.generator.trading_events import generate_trading_events
        df = generate_trading_events(n_normal=10, n_anomalous=2,
                                      output_path="/tmp/test_trading.csv")
        assert len(df) > 0
        required_cols = ["trade_id", "user_id", "instrument", "direction",
                         "lot_size", "open_ts", "close_ts", "pnl", "anomaly_flag"]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_anomaly_flag_present(self):
        from data.generator.portal_events import generate_portal_events
        df = generate_portal_events(n_normal=30, n_anomalous=10,
                                     output_path="/tmp/test_portal3.csv")
        assert df["anomaly_flag"].any(), "No anomalies were injected"

    def test_lot_size_positive(self):
        from data.generator.trading_events import generate_trading_events
        df = generate_trading_events(n_normal=10, n_anomalous=2,
                                      output_path="/tmp/test_trading2.csv")
        assert (df["lot_size"] > 0).all()

    def test_close_after_open(self):
        from data.generator.trading_events import generate_trading_events
        df = generate_trading_events(n_normal=10, n_anomalous=2,
                                      output_path="/tmp/test_trading3.csv")
        df["open_ts"]  = pd.to_datetime(df["open_ts"])
        df["close_ts"] = pd.to_datetime(df["close_ts"])
        assert (df["close_ts"] >= df["open_ts"]).all()


# ---- Feature engineering tests ----------------------------------------------

class TestFeatureEngineering:
    @pytest.fixture(scope="class")
    def small_portal(self):
        from data.generator.portal_events import generate_portal_events
        return generate_portal_events(n_normal=20, n_anomalous=5,
                                       output_path="/tmp/feat_portal.csv")

    @pytest.fixture(scope="class")
    def small_trading(self):
        from data.generator.trading_events import generate_trading_events
        return generate_trading_events(n_normal=20, n_anomalous=5,
                                        output_path="/tmp/feat_trading.csv")

    def test_portal_features_user_count(self, small_portal):
        from features.portal_features import build_portal_features
        feat = build_portal_features(small_portal, output_path="/tmp/pf.csv")
        assert len(feat) == small_portal["user_id"].nunique()

    def test_portal_features_no_nan(self, small_portal):
        from features.portal_features import build_portal_features
        feat = build_portal_features(small_portal, output_path="/tmp/pf2.csv")
        numeric_cols = feat.select_dtypes(include=[np.number]).columns
        assert not feat[numeric_cols].isnull().all(axis=None), \
            "All numeric values are NaN"

    def test_trading_features_user_count(self, small_trading):
        from features.trading_features import build_trading_features
        feat = build_trading_features(small_trading, output_path="/tmp/tf.csv")
        assert len(feat) == small_trading["user_id"].nunique()

    def test_feature_store_shape(self, small_portal, small_trading):
        from features.portal_features  import build_portal_features
        from features.trading_features import build_trading_features
        import pandas as pd
        build_portal_features(small_portal,  output_path="/tmp/pf3.csv")
        build_trading_features(small_trading, output_path="/tmp/tf3.csv")

        from features.feature_store import build_feature_store
        import shutil, os
        os.makedirs("/tmp/raw_test", exist_ok=True)
        small_portal.to_csv("/tmp/raw_test/portal_events.csv", index=False)
        small_trading.to_csv("/tmp/raw_test/trading_events.csv", index=False)

        meta, X, cols = build_feature_store(
            portal_events_path="/tmp/raw_test/portal_events.csv",
            trading_events_path="/tmp/raw_test/trading_events.csv",
            output_path="/tmp/raw_test/feature_store.csv",
        )
        assert X.shape[0] == meta["user_id"].nunique()
        assert X.shape[1] == len(cols)
        assert X.shape[1] > 0


# ---- Model tests ------------------------------------------------------------

class TestModels:
    @pytest.fixture(scope="class")
    def feature_data(self):
        from data.generator.portal_events  import generate_portal_events
        from data.generator.trading_events import generate_trading_events
        from features.feature_store import build_feature_store
        import os
        os.makedirs("/tmp/model_test", exist_ok=True)
        p = generate_portal_events(n_normal=50, n_anomalous=10,
                                    output_path="/tmp/model_test/portal_events.csv")
        t = generate_trading_events(n_normal=50, n_anomalous=10,
                                     output_path="/tmp/model_test/trading_events.csv")
        meta, X, cols = build_feature_store(
            portal_events_path="/tmp/model_test/portal_events.csv",
            trading_events_path="/tmp/model_test/trading_events.csv",
            output_path="/tmp/model_test/feature_store.csv",
        )
        return meta, X, cols

    def test_baseline_fit_predict(self, feature_data):
        from models.baseline.isolation_forest import BaselineDetector
        meta, X, cols = feature_data
        det = BaselineDetector()
        det.fit(X, cols)
        scores = det.predict(X)
        assert len(scores) == len(X)
        assert "ensemble_score" in scores.columns
        assert scores["ensemble_score"].between(0, 1).all()

    def test_baseline_top_features_valid_json(self, feature_data):
        from models.baseline.isolation_forest import BaselineDetector
        meta, X, cols = feature_data
        det = BaselineDetector()
        det.fit(X, cols)
        scores = det.predict(X)
        for val in scores["top_features"]:
            feats = json.loads(val)
            assert isinstance(feats, list)
            assert len(feats) <= 5

    def test_ae_fit_predict(self, feature_data):
        from models.advanced.lstm_autoencoder import LSTMAEDetector
        meta, X, cols = feature_data
        labels = meta["is_anomalous"].astype(int).to_numpy()
        det = LSTMAEDetector(epochs=5)
        det.fit(X, cols, labels=labels)
        scores = det.predict(X)
        assert len(scores) == len(X)
        assert "lstm_score" in scores.columns
        assert scores["lstm_score"].between(0, 1).all()

    def test_ae_top_features_valid_json(self, feature_data):
        from models.advanced.lstm_autoencoder import LSTMAEDetector
        meta, X, cols = feature_data
        labels = meta["is_anomalous"].astype(int).to_numpy()
        det = LSTMAEDetector(epochs=5)
        det.fit(X, cols, labels=labels)
        scores = det.predict(X)
        for val in scores["lstm_top_features"]:
            feats = json.loads(val)
            assert isinstance(feats, list)


# ---- Alert generation tests -------------------------------------------------

class TestAlerts:
    def test_alert_has_required_keys(self):
        from alerts.alert_generator import generate_alert
        score_result = {
            "user_id": "U00001",
            "final_score": 0.8,
            "if_score": 0.75,
            "ae_score": 0.85,
            "top_features": ["p_unique_ips", "p_pct_night_logins"],
        }
        alert = generate_alert(score_result, [], [])
        for key in ["alert_id", "user_id", "risk_level", "final_score",
                    "summary", "findings", "recommended_action"]:
            assert key in alert, f"Missing key: {key}"

    def test_risk_levels(self):
        from alerts.alert_generator import _risk_level
        assert _risk_level(0.90) == "CRITICAL"
        assert _risk_level(0.75) == "HIGH"
        assert _risk_level(0.60) == "MEDIUM"
        assert _risk_level(0.20) == "LOW"

    def test_structuring_detection(self):
        from alerts.alert_generator import _detect_portal_patterns
        from datetime import datetime, timedelta
        base = datetime(2024, 3, 1, 10, 0)
        events = []
        for i in range(20):
            events.append({
                "event_type": "deposit",
                "timestamp": (base + timedelta(minutes=i * 5)).isoformat(),
                "amount": 300.0,
                "ip_address": "1.2.3.4",
                "device_id": "dev-001",
                "country": "IN",
                "status": "success",
            })
        findings = _detect_portal_patterns(events)
        assert any("structuring" in f.lower() for f in findings)

    def test_brute_force_detection(self):
        from alerts.alert_generator import _detect_portal_patterns
        from datetime import datetime, timedelta
        base = datetime(2024, 3, 1, 10, 0)
        events = []
        for i in range(10):
            events.append({
                "event_type": "login",
                "timestamp": (base + timedelta(seconds=i * 30)).isoformat(),
                "status": "failed",
                "ip_address": "5.6.7.8",
                "device_id": "dev-002",
                "country": "US",
                "amount": None,
            })
        findings = _detect_portal_patterns(events)
        assert any("brute force" in f.lower() for f in findings)
