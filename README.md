# ForexGuard

Real-time anomaly detection engine for a forex brokerage platform.
Monitors client portal activity and trading terminal behaviour,
scores each user for suspicious patterns, and generates compliance alerts.

---

## Architecture

```
DATA LAYER
  portal_events.csv    (108k events)    trading_events.csv  (34k events)
  Login, KYC, Deposit, Withdrawal       Trades, PnL, Instruments
          |                                        |
          +------------------+---------------------+
                             |
FEATURE ENGINEERING (features/)
  portal_features.py   27 per-user features
  trading_features.py  29 per-user features
  feature_store.py     unified 500 x 56 matrix
                             |
          +------------------+---------------------+
          |                                        |
MODELS
  Isolation Forest + LOF    MLP Autoencoder (LSTM when PyTorch available)
  AUC = 1.000  F1 = 0.980   AUC = 0.796  Recall = 1.000
          |                                        |
          +------------ Ensemble (0.6/0.4) ---------+
                             |
STREAMING (streaming/)
  async_simulator.py   rolling user buffer, tick-based scoring
  kafka_producer.py    publishes events to Kafka topics
  kafka_consumer.py    consumes events, scores, fires alerts
                             |
ALERTS + API
  alert_generator.py   human-readable findings + recommended actions
  api/main.py          FastAPI: /predict  /predict/batch  /alerts
```

---

## Project Structure

```
forexguard/
├── data/
│   ├── generator/
│   │   ├── portal_events.py        synthetic portal event generator
│   │   ├── trading_events.py       synthetic trading event generator
│   │   └── generate_all.py         standalone data generation runner
│   └── raw/                        generated CSVs (created at runtime)
├── features/
│   ├── portal_features.py          27 portal feature extractors
│   ├── trading_features.py         29 trading feature extractors
│   └── feature_store.py            merges into unified feature matrix
├── models/
│   ├── baseline/
│   │   └── isolation_forest.py     Isolation Forest + LOF ensemble
│   └── advanced/
│       └── lstm_autoencoder.py     MLP Autoencoder (LSTM with PyTorch)
├── streaming/
│   ├── async_simulator.py          async real-time simulation
│   ├── kafka_producer.py           publishes CSV events to Kafka
│   └── kafka_consumer.py           consumes from Kafka, scores, alerts
├── alerts/
│   └── alert_generator.py          compliance alert engine
├── api/
│   ├── main.py                     FastAPI application
│   └── routes/
│       ├── predict.py              /predict route schemas
│       └── alerts.py               /alerts route schemas
├── tracking/
│   └── mlflow_logger.py            MLflow experiment logging
├── tests/
│   └── test_pipeline.py            unit and integration tests
├── notebooks/
│   └── eda_and_model_eval.py       EDA and model evaluation notebook
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── train_all.py                    master pipeline runner
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Setup

### 1. Install dependencies

```bash
git clone <your-repo>
cd forexguard
pip install -r requirements.txt
```

PyTorch is optional. If not installed, the system falls back to a
sklearn MLP Autoencoder with equivalent API.

### 2. Run the full pipeline

```bash
python train_all.py
```

This runs all six steps in order:

1. Generate ~142,000 synthetic events (108k portal + 34k trading)
2. Engineer 56 features per user
3. Train Isolation Forest + LOF baseline
4. Train MLP Autoencoder
5. Generate compliance alerts
6. Print model comparison report

Expected runtime: under 2 minutes on CPU.

### 3. Start the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive docs available at http://localhost:8000/docs

### 4. Run tests

```bash
pytest tests/ -v
```

### 5. Docker with Kafka

```bash
cd docker
docker-compose up --build
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Health check and model status |
| GET | /models/info | Model metadata and feature counts |
| POST | /predict | Score a single user by user_id |
| POST | /predict/batch | Score up to 100 users |
| POST | /score/raw | Score from a raw feature dictionary |
| GET | /alerts | All alerts, filterable by risk_level |
| GET | /alerts/{user_id} | Alerts for a specific user |

### Example: score a user

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"user_id": "U00458"}'
```

Response:

```json
{
  "user_id": "U00458",
  "if_score": 0.9910,
  "ae_score": 0.0170,
  "final_score": 0.6013,
  "risk_level": "MEDIUM",
  "is_anomalous": true,
  "top_features": [
    "p_max_ips_in_1h",
    "p_unique_ips",
    "p_pct_night_logins",
    "t_pct_sub_30s_trades",
    "t_win_rate"
  ],
  "findings": [
    "Multi-IP login burst: 9 distinct IPs within 60 minutes",
    "Brute force pattern: 8 failed login attempts detected"
  ],
  "recommended_action": "IMMEDIATE: Freeze account, initiate security review, notify user via secondary channel"
}
```

---

## Streaming

### Async simulator (no Kafka required)

```bash
python streaming/async_simulator.py --speed 200 --alert-threshold 0.60
```

Replays all events in timestamp order, scoring users after every
50-event batch and printing alerts in real time.

### Kafka (production)

Start Kafka via docker-compose, then in two terminals:

```bash
# Terminal 1
python streaming/kafka_producer.py --speed 500

# Terminal 2
python streaming/kafka_consumer.py --threshold 0.65
```

The producer publishes events to `forex.portal.events` and
`forex.trading.events`. The consumer scores each user after every
10 events and publishes alerts to `forex.alerts`.

---

## Models

### Baseline: Isolation Forest + LOF Ensemble

Isolation Forest partitions the feature space by randomly selecting
a feature and a split value. Anomalous users are isolated in fewer
splits because their feature values are unusual. Runtime is O(n log n)
and the model handles 56 features without dimensionality issues.

Local Outlier Factor catches density-based anomalies: users who
behave differently from their local peer group even when not globally
extreme. The two models are combined as a weighted ensemble (0.6 IF,
0.4 LOF) to improve coverage across different fraud types.

Results: AUC = 1.000, F1 = 0.980.

### Advanced: MLP Autoencoder

Architecture: Input(56) -> Dense(64) -> Dense(16) -> Dense(64) -> Output(56)

The autoencoder is trained only on the 450 normal users. It learns
to compress and reconstruct normal behaviour. When an anomalous user
is passed through, the model cannot reconstruct their unusual patterns,
producing a high reconstruction error that becomes the anomaly score.

Per-feature reconstruction error provides direct explainability: the
features with the highest individual error identify exactly which
behaviour is abnormal for that user.

When PyTorch is installed, the model uses an LSTM Autoencoder with
sequence input, which better captures temporal ordering within sessions.

Results: AUC = 0.796, Recall = 1.000.

---

## Features

### Portal features (27)

| Feature | Description |
|---------|-------------|
| total_logins | Total login count |
| failed_login_ratio | Fraction of logins that failed |
| unique_ips | Number of distinct IPs used |
| unique_devices | Number of distinct devices |
| unique_countries | Number of distinct countries |
| ip_entropy | Shannon entropy of IP distribution |
| login_hour_mean | Mean hour of login activity |
| login_hour_std | Std of login hour (circadian regularity) |
| pct_night_logins | Fraction of logins between midnight and 6am |
| max_ips_in_1h | Max distinct IPs in any 60-minute window |
| inter_login_min | Mean time between logins in minutes |
| deposit_count | Number of deposit events |
| withdrawal_count | Number of withdrawal events |
| deposit_withdraw_ratio | Total withdrawals / total deposits |
| structuring_score | Count of deposits below $500 |
| kyc_change_count | Number of KYC/profile change events |
| kyc_before_withdrawal | 1 if 3+ KYC changes in 4h before withdrawal |
| rolling_7d_logins | Logins in the last 7 days of activity |
| rolling_7d_deposits | Deposits in the last 7 days |
| rolling_7d_withdrawals | Withdrawals in the last 7 days |

### Trading features (29)

| Feature | Description |
|---------|-------------|
| total_trades | Total trade count |
| lot_zscore_max | Max z-score of lot sizes (spike detection) |
| trades_per_day | Average trades per active day |
| pnl_volatility | PnL std / (abs(mean PnL) + 1) |
| win_rate | Fraction of profitable trades |
| consecutive_wins_max | Longest winning streak |
| pct_sub_minute_trades | Fraction of trades closed under 1 minute |
| pct_sub_30s_trades | Fraction of trades closed under 30 seconds |
| top_instrument_concentration | Fraction of trades on the most used instrument |
| instrument_entropy | Shannon entropy of instrument distribution |
| rolling_7d_pnl | PnL in the last 7 days |
| weekend_trade_ratio | Fraction of trades opened on weekends |

---

## Suspicious Patterns Detected

| Pattern | Detection method |
|---------|-----------------|
| Multi-IP login burst | max_ips_in_1h > 4 in any 60-minute window |
| 3am logins | pct_night_logins, login_hour_mean |
| Brute force login | failed_login_ratio, failed count threshold |
| Structuring (small deposits) | structuring_score threshold |
| Deposit then withdrawal cycle | deposit_withdraw_ratio > 0.75 |
| KYC changes before withdrawal | kyc_before_withdrawal flag |
| Trading volume spike | lot_zscore_max, trades_per_day |
| Latency arbitrage | pct_sub_30s_trades with high win_rate |
| Single instrument concentration | top_instrument_concentration > 0.80 |
| High PnL volatility | pnl_volatility threshold |
| Synchronized trading | detected at generation time, scored via clustering features |

---

## Assumptions, Trade-offs and Limitations

**Assumptions**

- Approximately 10% of users are anomalous, which is set as the
  contamination parameter for Isolation Forest and LOF.
- The synthetic anomaly injection covers the main fraud typologies
  listed in the assessment brief.
- Features are computed per user across their full history. A
  session-level model would require a different data structure.

**Trade-offs**

- Isolation Forest AUC of 1.0 on this dataset reflects the clean
  separation between injected anomalies and normal users. Real-world
  data will have more overlap and lower AUC.
- The MLP Autoencoder is used as the advanced model fallback because
  PyTorch is not available in all deployment environments. When
  PyTorch is installed, the LSTM Autoencoder is used instead and
  better handles temporal structure.
- User-level feature aggregation is fast and interpretable but loses
  event-level sequence information. An event-level sequence model
  would catch patterns like session hijacking mid-session.
- The alert threshold is fixed at training time. A production system
  would recalibrate it periodically based on analyst feedback.

**Limitations**

- Graph-level features (collusion ring detection, shared device
  networks) require a graph database such as Neo4j and are not
  included in this prototype. The synchronized_trading anomaly is
  injected at data generation time but the model detects it via
  correlated feature values rather than direct graph analysis.
- No online learning. The model does not retrain as new data arrives.
- The streaming simulator compresses time for demonstration purposes.
  Production Kafka throughput depends on broker configuration.

---

## MLflow Tracking

```bash
python -c "
from tracking.mlflow_logger import log_baseline_run, log_ae_run
import pandas as pd
b = pd.read_csv('data/raw/baseline_scores.csv')
a = pd.read_csv('data/raw/lstm_scores.csv')
s = pd.read_csv('data/raw/feature_store.csv')
log_baseline_run(b, s['is_anomalous'])
log_ae_run(a, s['is_anomalous'])
"
mlflow ui
```

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| ML | scikit-learn (IF, LOF, MLP), PyTorch (LSTM-AE, optional) |
| Data | pandas, numpy, scipy |
| Streaming | asyncio simulator, kafka-python |
| API | FastAPI, uvicorn, pydantic |
| Tracking | MLflow |
| Infra | Docker, docker-compose |
| Testing | pytest |
