"""
FastAPI serving layer for the Pre-Shipment Return-Risk Scorer.

Endpoints:
    POST /score   - takes an order JSON (schema.json field names), returns risk score + flag
    GET  /metrics - returns the saved precision/recall/PR-AUC/threshold numbers from training
    GET  /health  - liveness check

Run locally:
    uvicorn api.main:app --reload --port 8000
"""

import json
import os
import traceback
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")

app = FastAPI(
    title="Pre-Shipment Return-Risk Scorer",
    description="Scores incoming orders for return/dispute risk before fulfillment.",
    version="1.0.0",
)

# CORS: allow the Lovable frontend (and local dev) to call this API directly.
# Add your deployed Lovable domain here once you have it, e.g. "https://your-app.lovable.app"
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "https://*.lovable.app",
    "https://*.lovableproject.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://.*\.(lovable\.app|lovableproject\.com)|http://localhost:\d+",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TEMP DEBUG: surface the real traceback in the response instead of a bare 500,
# so a production-only failure can be diagnosed without dashboard log access.
# Remove before final submission.
@app.exception_handler(Exception)
async def debug_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "type": type(exc).__name__,
            "traceback": traceback.format_exc(),
        },
    )


_model = None
_metrics = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(status_code=503, detail="Model not trained yet. Run model/train.py.")
        _model = joblib.load(MODEL_PATH)
    return _model


def get_metrics():
    global _metrics
    if _metrics is None:
        if not os.path.exists(METRICS_PATH):
            raise HTTPException(status_code=503, detail="Metrics not available yet. Run model/train.py.")
        with open(METRICS_PATH) as f:
            _metrics = json.load(f)
    return _metrics


class OrderRequest(BaseModel):
    order_id: str = Field(..., example="ORD100234")
    order_value: float = Field(..., gt=0, example=4999)
    category: Literal["electronics", "apparel", "home", "beauty", "grocery"]
    customer_type: Literal["new", "returning"]
    customer_order_count: int = Field(..., ge=0, example=3)
    payment_method: Literal["UPI", "card", "netbanking", "COD", "wallet"]
    delivery_pincode_risk_tier: Literal["low", "medium", "high"]
    discount_percent: float = Field(0, ge=0, le=100, example=20)
    order_hour: int = Field(..., ge=0, le=23)
    day_of_week: Literal["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


class ScoreResponse(BaseModel):
    order_id: str
    risk_score: float
    flagged: bool
    threshold_used: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/debug/versions")
def debug_versions():
    import sys
    import sklearn
    import xgboost
    import numpy
    import pandas as pd_mod

    return {
        "python": sys.version,
        "sklearn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "pandas": pd_mod.__version__,
        "numpy": numpy.__version__,
        "joblib": joblib.__version__,
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
        "model_size_bytes": os.path.getsize(MODEL_PATH) if os.path.exists(MODEL_PATH) else None,
    }


@app.post("/score", response_model=ScoreResponse)
def score_order(order: OrderRequest):
    model = get_model()
    metrics = get_metrics()
    threshold = metrics.get("chosen_threshold", 0.5)

    row = pd.DataFrame(
        [
            {
                "category": order.category,
                "customer_type": order.customer_type,
                "payment_method": order.payment_method,
                "delivery_pincode_risk_tier": order.delivery_pincode_risk_tier,
                "day_of_week": order.day_of_week,
                "order_value": order.order_value,
                "customer_order_count": order.customer_order_count,
                "discount_percent": order.discount_percent,
                "order_hour": order.order_hour,
            }
        ]
    )
    prob = float(model.predict_proba(row)[:, 1][0])

    return ScoreResponse(
        order_id=order.order_id,
        risk_score=round(prob, 4),
        flagged=prob >= threshold,
        threshold_used=threshold,
    )


class BatchScoreRequest(BaseModel):
    orders: list[OrderRequest]


@app.post("/score/batch")
def score_batch(payload: BatchScoreRequest):
    model = get_model()
    metrics = get_metrics()
    threshold = metrics.get("chosen_threshold", 0.5)

    if not payload.orders:
        return {"results": []}

    rows = pd.DataFrame(
        [
            {
                "category": o.category,
                "customer_type": o.customer_type,
                "payment_method": o.payment_method,
                "delivery_pincode_risk_tier": o.delivery_pincode_risk_tier,
                "day_of_week": o.day_of_week,
                "order_value": o.order_value,
                "customer_order_count": o.customer_order_count,
                "discount_percent": o.discount_percent,
                "order_hour": o.order_hour,
            }
            for o in payload.orders
        ]
    )
    probs = model.predict_proba(rows)[:, 1]

    results = [
        {
            "order_id": o.order_id,
            "risk_score": round(float(p), 4),
            "flagged": bool(p >= threshold),
            "threshold_used": threshold,
        }
        for o, p in zip(payload.orders, probs)
    ]
    return {"results": results}


@app.get("/metrics")
def get_saved_metrics():
    return get_metrics()
