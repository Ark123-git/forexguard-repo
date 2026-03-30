"""
api/routes/alerts.py
---------------------
Alert route handlers.
Imported by api/main.py.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/alerts", tags=["alerts"])
