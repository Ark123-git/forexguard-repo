"""
api/routes/predict.py
---------------------
Prediction route handlers extracted for modularity.
Imported by api/main.py.
"""

import json
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

router = APIRouter(prefix="", tags=["predict"])


class PredictRequest(BaseModel):
    user_id: str


class BatchPredictRequest(BaseModel):
    user_ids: list[str]


class RawScoreRequest(BaseModel):
    features: dict[str, float]


def _risk_level(score: float) -> str:
    if score >= 0.85:
        return "CRITICAL"
    if score >= 0.70:
        return "HIGH"
    if score >= 0.55:
        return "MEDIUM"
    if score >= 0.30:
        return "LOW"
    return "NORMAL"
