"""
FastAPI serving layer for the Pre-Shipment Return-Risk Scorer.

Endpoints:
    GET  /         - serves the interactive dashboard UI
    POST /score   - takes an order JSON (schema.json field names), returns risk score + flag
    GET  /metrics - returns the saved precision/recall/PR-AUC/threshold numbers from training
    GET  /health  - liveness check

Run locally:
    uvicorn api.main:app --reload --port 8000
"""

import json
import os
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(
    title="Pre-Shipment Return-Risk Scorer",
    description="Scores incoming orders for return/dispute risk before fulfillment.",
    version="1.0.0",
)

# Mount static files (CSS, JS)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

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


# ===== Dashboard HTML =====
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AI Risk Manager — Pre-Shipment Return-Risk Scorer</title>
  <meta name="description" content="AI-powered pre-shipment return-risk scoring system. Score incoming e-commerce orders for return/dispute risk before fulfillment using XGBoost ML model." />
  <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🛡️</text></svg>" />
  <link rel="stylesheet" href="/static/style.css" />
</head>
<body>

  <!-- ===== Navigation ===== -->
  <nav class="nav" id="nav">
    <div class="container">
      <a href="#" class="nav-brand">
        <div class="nav-logo">🛡️</div>
        <div>
          <div class="nav-title">AI Risk Manager</div>
          <div class="nav-subtitle">Return-Risk Scorer</div>
        </div>
      </a>
      <div class="nav-links">
        <a href="#metrics" class="nav-link active">Metrics</a>
        <a href="#scorer" class="nav-link">Score Order</a>
        <a href="#features" class="nav-link">Features</a>
        <a href="#api" class="nav-link">API</a>
        <a href="/docs" class="nav-link">Swagger ↗</a>
      </div>
    </div>
  </nav>

  <!-- ===== Hero ===== -->
  <section class="hero">
    <div class="container">
      <div class="hero-badge">
        <span>⚡</span>
        <span>XGBoost · Cost-Optimized Threshold · Real-Time Scoring</span>
      </div>
      <h1>
        Pre-Shipment<br />
        <span class="gradient-text">Return-Risk Scorer</span>
      </h1>
      <p class="hero-description">
        Predict which e-commerce orders are likely to be returned or disputed
        <strong>before</strong> they ship. Reduce losses, optimize fulfillment,
        and protect margins with ML-powered risk scoring.
      </p>
      <div class="hero-actions">
        <a href="#scorer" class="btn btn-primary">⚡ Try the Scorer</a>
        <a href="#metrics" class="btn btn-secondary">📊 View Metrics</a>
        <a href="/docs" class="btn btn-secondary">📖 API Docs</a>
      </div>
    </div>
  </section>

  <!-- ===== Model Metrics ===== -->
  <section class="section" id="metrics">
    <div class="container">
      <div class="section-header">
        <h2 class="section-title">📊 Model Performance</h2>
        <p class="section-subtitle">XGBClassifier trained on 6,400 orders · Evaluated on 1,600 held-out orders</p>
      </div>

      <div class="metrics-grid">
        <div class="metric-card">
          <div class="metric-icon purple">📈</div>
          <div class="metric-label">ROC-AUC</div>
          <div class="metric-value" id="metric-roc-auc">—</div>
          <div class="metric-detail">Area under ROC curve</div>
        </div>
        <div class="metric-card">
          <div class="metric-icon blue">🎯</div>
          <div class="metric-label">PR-AUC</div>
          <div class="metric-value" id="metric-pr-auc">—</div>
          <div class="metric-detail">Precision-recall area</div>
        </div>
        <div class="metric-card">
          <div class="metric-icon green">✅</div>
          <div class="metric-label">Precision</div>
          <div class="metric-value" id="metric-precision">—</div>
          <div class="metric-detail">At chosen threshold</div>
        </div>
        <div class="metric-card">
          <div class="metric-icon orange">🔍</div>
          <div class="metric-label">Recall</div>
          <div class="metric-value" id="metric-recall">—</div>
          <div class="metric-detail">Risky orders caught</div>
        </div>
        <div class="metric-card">
          <div class="metric-icon blue">⚖️</div>
          <div class="metric-label">F1 Score</div>
          <div class="metric-value" id="metric-f1">—</div>
          <div class="metric-detail">Harmonic mean</div>
        </div>
        <div class="metric-card">
          <div class="metric-icon purple">🎚️</div>
          <div class="metric-label">Threshold</div>
          <div class="metric-value" id="metric-threshold">—</div>
          <div class="metric-detail">Cost-optimized cutoff</div>
        </div>
        <div class="metric-card">
          <div class="metric-icon orange">🚩</div>
          <div class="metric-label">Flag Rate</div>
          <div class="metric-value" id="metric-flag-rate">—</div>
          <div class="metric-detail">Orders flagged</div>
        </div>
        <div class="metric-card">
          <div class="metric-icon green">💰</div>
          <div class="metric-label">Cost Savings</div>
          <div class="metric-value" id="metric-cost-savings">—</div>
          <div class="metric-detail">vs. default 0.5 threshold</div>
        </div>
      </div>

      <!-- Confusion Matrix -->
      <div class="section-header" style="margin-top: 40px;">
        <h2 class="section-title">🧮 Confusion Matrix</h2>
        <p class="section-subtitle">At the cost-optimized threshold on held-out test set</p>
      </div>
      <div class="confusion-grid">
        <div class="confusion-cell tn">
          <div class="cell-value" id="cm-tn">—</div>
          <div class="cell-label">True Negative</div>
        </div>
        <div class="confusion-cell fp">
          <div class="cell-value" id="cm-fp">—</div>
          <div class="cell-label">False Positive</div>
        </div>
        <div class="confusion-cell fn">
          <div class="cell-value" id="cm-fn">—</div>
          <div class="cell-label">False Negative</div>
        </div>
        <div class="confusion-cell tp">
          <div class="cell-value" id="cm-tp">—</div>
          <div class="cell-label">True Positive</div>
        </div>
      </div>
    </div>
  </section>

  <!-- ===== Order Scorer ===== -->
  <section class="section" id="scorer">
    <div class="container">
      <div class="section-header">
        <h2 class="section-title">⚡ Score an Order</h2>
        <p class="section-subtitle">Enter order details to get a real-time risk prediction</p>
      </div>

      <div class="scorer-panel">
        <!-- Form -->
        <div class="form-card">
          <form onsubmit="scoreOrder(event)">
            <div class="form-grid">
              <div class="form-group">
                <label class="form-label" for="order_id">Order ID</label>
                <input class="form-input" type="text" name="order_id" id="order_id" placeholder="ORD100234" value="ORD100234" />
              </div>
              <div class="form-group">
                <label class="form-label" for="order_value">Order Value (₹)</label>
                <input class="form-input" type="number" name="order_value" id="order_value" placeholder="4999" value="4999" min="1" step="1" required />
              </div>
              <div class="form-group">
                <label class="form-label" for="category">Category</label>
                <select class="form-select" name="category" id="category">
                  <option value="electronics">Electronics</option>
                  <option value="apparel">Apparel</option>
                  <option value="home">Home</option>
                  <option value="beauty">Beauty</option>
                  <option value="grocery">Grocery</option>
                </select>
              </div>
              <div class="form-group">
                <label class="form-label" for="customer_type">Customer Type</label>
                <select class="form-select" name="customer_type" id="customer_type">
                  <option value="new">New</option>
                  <option value="returning">Returning</option>
                </select>
              </div>
              <div class="form-group">
                <label class="form-label" for="customer_order_count">Order Count</label>
                <input class="form-input" type="number" name="customer_order_count" id="customer_order_count" value="3" min="0" required />
              </div>
              <div class="form-group">
                <label class="form-label" for="payment_method">Payment Method</label>
                <select class="form-select" name="payment_method" id="payment_method">
                  <option value="UPI">UPI</option>
                  <option value="card">Card</option>
                  <option value="netbanking">Netbanking</option>
                  <option value="COD">COD</option>
                  <option value="wallet">Wallet</option>
                </select>
              </div>
              <div class="form-group">
                <label class="form-label" for="delivery_pincode_risk_tier">Pincode Risk</label>
                <select class="form-select" name="delivery_pincode_risk_tier" id="delivery_pincode_risk_tier">
                  <option value="low">Low</option>
                  <option value="medium">Medium</option>
                  <option value="high">High</option>
                </select>
              </div>
              <div class="form-group">
                <label class="form-label" for="discount_percent">Discount %</label>
                <input class="form-input" type="number" name="discount_percent" id="discount_percent" value="20" min="0" max="100" required />
              </div>
              <div class="form-group">
                <label class="form-label" for="order_hour">Order Hour (0-23)</label>
                <input class="form-input" type="number" name="order_hour" id="order_hour" value="14" min="0" max="23" required />
              </div>
              <div class="form-group">
                <label class="form-label" for="day_of_week">Day of Week</label>
                <select class="form-select" name="day_of_week" id="day_of_week">
                  <option value="Mon">Monday</option>
                  <option value="Tue">Tuesday</option>
                  <option value="Wed">Wednesday</option>
                  <option value="Thu">Thursday</option>
                  <option value="Fri">Friday</option>
                  <option value="Sat">Saturday</option>
                  <option value="Sun">Sunday</option>
                </select>
              </div>
              <div class="form-group full-width">
                <button type="submit" class="btn btn-primary form-submit">⚡ Score This Order</button>
              </div>
            </div>
          </form>
        </div>

        <!-- Result -->
        <div class="result-card">
          <div class="result-placeholder" id="result-placeholder">
            <div class="placeholder-icon">🎯</div>
            <p>Fill in the order details and click<br /><strong>"Score This Order"</strong> to see the prediction.</p>
          </div>
          <div class="spinner" id="spinner"></div>
          <div class="result-content" id="result-content">
            <div class="risk-gauge">
              <div class="gauge-circle low" id="gauge-circle" style="--score: 0;">
                <div class="gauge-score" id="gauge-score">0%</div>
                <div class="gauge-label">Risk Score</div>
              </div>
              <div class="risk-badge safe" id="risk-badge">✓ CLEAR</div>
            </div>
            <div class="result-details">
              <div class="detail-row">
                <span class="detail-label">Order ID</span>
                <span class="detail-value" id="res-order-id">—</span>
              </div>
              <div class="detail-row">
                <span class="detail-label">Risk Score</span>
                <span class="detail-value" id="res-risk-score">—</span>
              </div>
              <div class="detail-row">
                <span class="detail-label">Threshold</span>
                <span class="detail-value" id="res-threshold">—</span>
              </div>
              <div class="detail-row">
                <span class="detail-label">Flagged</span>
                <span class="detail-value" id="res-flagged">—</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>

  <!-- ===== Features ===== -->
  <section class="section" id="features">
    <div class="container">
      <div class="section-header">
        <h2 class="section-title">🧬 Model Features</h2>
        <p class="section-subtitle">9 features engineered from order data to predict return/dispute risk</p>
      </div>
      <div class="feature-list">
        <div class="feature-item">
          <span class="feature-name">category</span>
          <div class="feature-bar-wrap"><div class="feature-bar" data-width="95%" style="width:0"></div></div>
          <span class="feature-type">categorical</span>
        </div>
        <div class="feature-item">
          <span class="feature-name">customer_type</span>
          <div class="feature-bar-wrap"><div class="feature-bar" data-width="80%" style="width:0"></div></div>
          <span class="feature-type">categorical</span>
        </div>
        <div class="feature-item">
          <span class="feature-name">payment_method</span>
          <div class="feature-bar-wrap"><div class="feature-bar" data-width="75%" style="width:0"></div></div>
          <span class="feature-type">categorical</span>
        </div>
        <div class="feature-item">
          <span class="feature-name">delivery_pincode_risk_tier</span>
          <div class="feature-bar-wrap"><div class="feature-bar" data-width="70%" style="width:0"></div></div>
          <span class="feature-type">categorical</span>
        </div>
        <div class="feature-item">
          <span class="feature-name">day_of_week</span>
          <div class="feature-bar-wrap"><div class="feature-bar" data-width="55%" style="width:0"></div></div>
          <span class="feature-type">categorical</span>
        </div>
        <div class="feature-item">
          <span class="feature-name">order_value</span>
          <div class="feature-bar-wrap"><div class="feature-bar" data-width="90%" style="width:0"></div></div>
          <span class="feature-type">numeric</span>
        </div>
        <div class="feature-item">
          <span class="feature-name">customer_order_count</span>
          <div class="feature-bar-wrap"><div class="feature-bar" data-width="65%" style="width:0"></div></div>
          <span class="feature-type">numeric</span>
        </div>
        <div class="feature-item">
          <span class="feature-name">discount_percent</span>
          <div class="feature-bar-wrap"><div class="feature-bar" data-width="85%" style="width:0"></div></div>
          <span class="feature-type">numeric</span>
        </div>
        <div class="feature-item">
          <span class="feature-name">order_hour</span>
          <div class="feature-bar-wrap"><div class="feature-bar" data-width="50%" style="width:0"></div></div>
          <span class="feature-type">numeric</span>
        </div>
      </div>
    </div>
  </section>

  <!-- ===== API Reference ===== -->
  <section class="section" id="api">
    <div class="container">
      <div class="section-header">
        <h2 class="section-title">🔌 API Reference</h2>
        <p class="section-subtitle">RESTful endpoints for integration into your fulfillment pipeline</p>
      </div>
      <div class="api-cards">
        <div class="api-card">
          <span class="api-method post">POST</span>
          <div class="api-path">/score</div>
          <p class="api-desc">Score a single order for return/dispute risk. Accepts order JSON, returns risk_score (0–1), flagged boolean, and threshold used.</p>
        </div>
        <div class="api-card">
          <span class="api-method post">POST</span>
          <div class="api-path">/score/batch</div>
          <p class="api-desc">Score multiple orders in a single request. Accepts an array of orders, returns an array of scored results.</p>
        </div>
        <div class="api-card">
          <span class="api-method get">GET</span>
          <div class="api-path">/metrics</div>
          <p class="api-desc">Returns model evaluation metrics — ROC-AUC, PR-AUC, precision, recall, F1, confusion matrix, and cost-curve analysis.</p>
        </div>
        <div class="api-card">
          <span class="api-method get">GET</span>
          <div class="api-path">/health</div>
          <p class="api-desc">Liveness check. Returns {"status": "ok"} when the service is running and ready to accept requests.</p>
        </div>
        <div class="api-card">
          <span class="api-method get">GET</span>
          <div class="api-path">/docs</div>
          <p class="api-desc">Interactive Swagger UI with auto-generated documentation, request schemas, and the ability to test endpoints live.</p>
        </div>
      </div>
    </div>
  </section>

  <!-- ===== Footer ===== -->
  <footer class="footer">
    <div class="container">
      <p>
        🛡️ AI Risk Manager · Pre-Shipment Return-Risk Scorer ·
        Built with <a href="https://fastapi.tiangolo.com" target="_blank">FastAPI</a> &
        <a href="https://xgboost.readthedocs.io" target="_blank">XGBoost</a>
        · <a href="/docs">API Docs</a>
      </p>
    </div>
  </footer>

  <script src="/static/app.js"></script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Serve the interactive dashboard UI."""
    return DASHBOARD_HTML


@app.get("/health")
def health():
    return {"status": "ok"}


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
