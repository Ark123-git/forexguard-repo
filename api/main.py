"""
api/main.py
-----------
FastAPI application exposing the ForexGuard anomaly detection engine.

Endpoints:
  POST /predict          - Score a single user by user_id
  POST /predict/batch    - Score multiple users
  POST /score/raw        - Score from raw feature vector
  GET  /alerts           - Get all generated alerts
  GET  /alerts/{user_id} - Get alerts for a specific user
  GET  /health           - Health check
  GET  /models/info      - Model metadata
"""

import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

from models.baseline.isolation_forest import BaselineDetector
from models.advanced.lstm_autoencoder import LSTMAEDetector
from alerts.alert_generator import generate_alert

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ForexGuard API",
    description="Real-time trader anomaly detection engine",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── Load models and data at startup ──────────────────────────────────────────
print("[API] Loading models...")
try:
    BASELINE  = BaselineDetector.load()
    AE        = LSTMAEDetector.load()
    FEAT_COLS = BASELINE.feat_cols
    STORE     = pd.read_csv("data/raw/feature_store.csv")
    PORTAL    = pd.read_csv("data/raw/portal_events.csv")
    TRADING   = pd.read_csv("data/raw/trading_events.csv")
    ALERTS_DF = pd.read_csv("data/raw/alerts.csv") if os.path.exists("data/raw/alerts.csv") else pd.DataFrame()
    print(f"[API] Ready. {len(STORE)} users loaded.")
except Exception as e:
    print(f"[API] Warning: Could not load models: {e}")
    BASELINE = AE = FEAT_COLS = STORE = PORTAL = TRADING = None
    ALERTS_DF = pd.DataFrame()


# ── Schemas ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    user_id: str

class BatchPredictRequest(BaseModel):
    user_ids: list[str]

class RawScoreRequest(BaseModel):
    features: dict[str, float]

class ScoreResponse(BaseModel):
    user_id: str
    if_score: float
    ae_score: float
    final_score: float
    risk_level: str
    is_anomalous: bool
    top_features: list[str]
    findings: list[str]
    recommended_action: str
    scored_at: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _risk_level(score: float) -> str:
    if score >= 0.85: return "CRITICAL"
    if score >= 0.70: return "HIGH"
    if score >= 0.55: return "MEDIUM"
    if score >= 0.30: return "LOW"
    return "NORMAL"

def _score_user_id(user_id: str) -> dict:
    if STORE is None:
        raise HTTPException(503, "Models not loaded")
    row = STORE[STORE["user_id"] == user_id]
    if row.empty:
        raise HTTPException(404, f"User '{user_id}' not found in feature store")

    X = row[FEAT_COLS].fillna(0).to_numpy(dtype=np.float32)

    b = BASELINE.predict(X)
    a = AE.predict(X)

    if_score = float(b["ensemble_score"].iloc[0])
    ae_score = float(a["lstm_score"].iloc[0])
    final    = round(0.6 * if_score + 0.4 * ae_score, 4)

    b_feats = json.loads(b["top_features"].iloc[0])
    a_feats = json.loads(a["lstm_top_features"].iloc[0])
    top     = list(dict.fromkeys(b_feats + a_feats))[:5]

    p_evts = PORTAL[PORTAL["user_id"] == user_id].to_dict("records")
    t_evts = TRADING[TRADING["user_id"] == user_id].to_dict("records")

    score_result = {
        "user_id": user_id, "final_score": final,
        "if_score": if_score, "ae_score": ae_score,
        "top_features": top,
    }
    alert = generate_alert(score_result, p_evts, t_evts)

    return {
        "user_id":            user_id,
        "if_score":           if_score,
        "ae_score":           ae_score,
        "final_score":        final,
        "risk_level":         _risk_level(final),
        "is_anomalous":       final >= 0.50,
        "top_features":       top,
        "findings":           alert["findings"],
        "recommended_action": alert["recommended_action"],
        "scored_at":          datetime.utcnow().isoformat(),
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": BASELINE is not None,
        "users_in_store": len(STORE) if STORE is not None else 0,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/models/info")
def model_info():
    return {
        "baseline": {
            "type": "Isolation Forest + LOF Ensemble",
            "n_estimators": 200,
            "contamination": 0.10,
            "weight_in_ensemble": 0.6,
        },
        "advanced": {
            "type": "MLP Autoencoder (sklearn fallback, LSTM when torch available)",
            "hidden_layers": [64, 16, 64],
            "anomaly_criterion": "Reconstruction error > 95th percentile of normal users",
            "weight_in_ensemble": 0.4,
        },
        "features": {
            "total": len(FEAT_COLS) if FEAT_COLS else 0,
            "portal": sum(1 for c in (FEAT_COLS or []) if c.startswith("p_")),
            "trading": sum(1 for c in (FEAT_COLS or []) if c.startswith("t_")),
        },
    }


@app.post("/predict", response_model=ScoreResponse)
def predict(req: PredictRequest):
    return _score_user_id(req.user_id)


@app.post("/predict/batch")
def predict_batch(req: BatchPredictRequest):
    if len(req.user_ids) > 100:
        raise HTTPException(400, "Max 100 users per batch request")
    results, errors = [], []
    for uid in req.user_ids:
        try:
            results.append(_score_user_id(uid))
        except HTTPException as e:
            errors.append({"user_id": uid, "error": e.detail})
    return {"results": results, "errors": errors, "total": len(results)}


@app.post("/score/raw")
def score_raw(req: RawScoreRequest):
    """Score from a raw feature dict — useful for real-time streaming integration."""
    if FEAT_COLS is None:
        raise HTTPException(503, "Models not loaded")
    X = np.array([[req.features.get(c, 0.0) for c in FEAT_COLS]], dtype=np.float32)

    b = BASELINE.predict(X)
    a = AE.predict(X)

    if_score = float(b["ensemble_score"].iloc[0])
    ae_score = float(a["lstm_score"].iloc[0])
    final    = round(0.6 * if_score + 0.4 * ae_score, 4)
    b_feats  = json.loads(b["top_features"].iloc[0])
    a_feats  = json.loads(a["lstm_top_features"].iloc[0])
    top      = list(dict.fromkeys(b_feats + a_feats))[:5]

    return {
        "if_score": if_score,
        "ae_score": ae_score,
        "final_score": final,
        "risk_level": _risk_level(final),
        "is_anomalous": final >= 0.50,
        "top_features": top,
    }


@app.get("/alerts")
def get_alerts(risk_level: Optional[str] = None, limit: int = 50):
    if ALERTS_DF.empty:
        return {"alerts": [], "total": 0}
    df = ALERTS_DF.copy()
    if risk_level:
        df = df[df["risk_level"] == risk_level.upper()]
    df = df.sort_values("final_score", ascending=False).head(limit)
    return {"alerts": df.to_dict("records"), "total": len(df)}


@app.get("/alerts/{user_id}")
def get_user_alerts(user_id: str):
    if ALERTS_DF.empty:
        return {"alerts": [], "user_id": user_id}
    df = ALERTS_DF[ALERTS_DF["user_id"] == user_id]
    return {"alerts": df.to_dict("records"), "user_id": user_id, "total": len(df)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
