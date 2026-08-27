"""

FastAPI serving layer for RazorGuard AI (AI-Powered Transaction Risk Intelligence).

Endpoints:
    GET  /         - serves the interactive dashboard UI
    POST /score   - takes an order JSON (schema.json field names), returns risk score + flag
    GET  /metrics - returns the saved precision/recall/PR-AUC/threshold numbers from training
    GET  /health  - liveness check

Run locally:
    uvicorn api.main:app --reload --port 8000
"""

import csv
import io
import json
import os
from typing import Any, Literal, Optional

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from api.explain import CATEGORICAL, NUMERIC, explain_batch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Optional API key. If API_KEY env var is set, /score, /score/batch and
# /score/csv require an X-API-Key header. If unset, endpoints stay open
# so local dev and unauthenticated demos still work.
API_KEY = os.environ.get("API_KEY", "").strip()

# Cap CSV uploads at ~2 MB / 5k rows to keep a single request bounded on the
# free Render tier (~512 MB RAM, one worker).
MAX_CSV_BYTES = 2 * 1024 * 1024
MAX_CSV_ROWS = 5000

app = FastAPI(
    title="RazorGuard AI",
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


def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


class Reason(BaseModel):
    feature: str
    value: Any = None
    impact: float


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
    top_reasons: list[Reason] = []


# ===== Dashboard HTML =====
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>RazorGuard AI — AI-Powered Transaction Risk Intelligence</title>
  <meta name="description" content="RazorGuard AI: AI-powered transaction risk intelligence. Score incoming e-commerce orders for return/dispute risk before fulfillment using an XGBoost model with a cost-optimized threshold." />
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
          <div class="nav-title">RazorGuard AI</div>
          <div class="nav-subtitle">Transaction Risk Intelligence</div>
        </div>
      </a>
      <div class="nav-links">
        <a href="#scorer" class="nav-link active">Score</a>
        <a href="#batch" class="nav-link">Batch</a>
        <a href="#features" class="nav-link">Features</a>
        <a href="#metrics" class="nav-link">Model</a>
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
        RazorGuard<br />
        <span class="gradient-text">AI Risk Intelligence</span>
      </h1>
      <p class="hero-description">
        Predict which e-commerce orders are likely to be returned or disputed
        <strong>before</strong> they ship. Reduce losses, optimize fulfillment,
        and protect margins with ML-powered risk scoring.
      </p>
      <div class="hero-actions">
        <a href="#scorer" class="btn btn-primary">⚡ Score an Order</a>
        <a href="#batch" class="btn btn-secondary">📂 Batch Upload</a>
        <a href="/docs" class="btn btn-secondary">📖 API Docs</a>
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
            <div class="reasons-panel">
              <div class="reasons-title">Top contributing factors</div>
              <div class="reasons-list" id="reasons-list"></div>
              <div class="reasons-legend">
                <span><i class="dot dot-up"></i>Raises risk</span>
                <span><i class="dot dot-down"></i>Lowers risk</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>

  <!-- ===== Batch Scoring ===== -->
  <section class="section" id="batch">
    <div class="container">
      <div class="section-header">
        <h2 class="section-title">📂 Batch Score from CSV</h2>
        <p class="section-subtitle">Upload a CSV of orders to score them in bulk. Download results with reasons.</p>
      </div>

      <div class="batch-panel">
        <div class="batch-upload">
          <label class="drop-zone" id="drop-zone" for="csv-input">
            <div class="drop-icon">📄</div>
            <div class="drop-text"><strong>Drop CSV here</strong> or click to browse</div>
            <div class="drop-hint">Columns: order_id, category, customer_type, payment_method, delivery_pincode_risk_tier, day_of_week, order_value, customer_order_count, discount_percent, order_hour</div>
            <input type="file" id="csv-input" accept=".csv,text/csv" hidden />
          </label>
          <div class="batch-actions">
            <a href="#" id="download-sample" class="btn btn-secondary">⬇ Sample CSV</a>
            <button id="download-scored" class="btn btn-secondary" disabled>⬇ Download scored CSV</button>
          </div>
          <div class="batch-status" id="batch-status"></div>
        </div>

        <div class="batch-results" id="batch-results" hidden>
          <div class="batch-summary" id="batch-summary"></div>
          <div class="table-wrap">
            <table class="results-table" id="results-table">
              <thead>
                <tr>
                  <th data-sort="order_id">Order ID</th>
                  <th data-sort="risk_score">Risk</th>
                  <th data-sort="flagged">Status</th>
                  <th>Top reasons</th>
                </tr>
              </thead>
              <tbody id="results-tbody"></tbody>
            </table>
          </div>
          <div class="table-pager">
            <button id="page-prev" class="btn btn-secondary">← Prev</button>
            <span id="page-info">—</span>
            <button id="page-next" class="btn btn-secondary">Next →</button>
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
          <p class="api-desc">Score a single order for return/dispute risk. Accepts order JSON, returns risk_score (0–1), flagged boolean, threshold used, and top_reasons (top 3 contributing features with signed impact).</p>
        </div>
        <div class="api-card">
          <span class="api-method post">POST</span>
          <div class="api-path">/score/batch</div>
          <p class="api-desc">Score multiple orders in a single JSON request. Returns per-order score, flagged, and top_reasons.</p>
        </div>
        <div class="api-card">
          <span class="api-method post">POST</span>
          <div class="api-path">/score/csv</div>
          <p class="api-desc">Upload a CSV (multipart file) with the same columns as the JSON schema; returns scored results with reasons. Cap: 2 MB / 5,000 rows per request.</p>
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
        🛡️ RazorGuard AI · AI-Powered Transaction Risk Intelligence ·
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


def _orders_to_frame(orders: list[OrderRequest]) -> pd.DataFrame:
    return pd.DataFrame(
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
            for o in orders
        ]
    )


@app.post("/score", response_model=ScoreResponse, dependencies=[Depends(require_api_key)])
def score_order(order: OrderRequest):
    model = get_model()
    metrics = get_metrics()
    threshold = metrics.get("chosen_threshold", 0.5)

    row = _orders_to_frame([order])
    prob = float(model.predict_proba(row)[:, 1][0])
    reasons = explain_batch(model, row, top_k=3)[0]

    return ScoreResponse(
        order_id=order.order_id,
        risk_score=round(prob, 4),
        flagged=prob >= threshold,
        threshold_used=threshold,
        top_reasons=[Reason(**r) for r in reasons],
    )


class BatchScoreRequest(BaseModel):
    orders: list[OrderRequest]


@app.post("/score/batch", dependencies=[Depends(require_api_key)])
def score_batch(payload: BatchScoreRequest):
    model = get_model()
    metrics = get_metrics()
    threshold = metrics.get("chosen_threshold", 0.5)

    if not payload.orders:
        return {"results": []}

    rows = _orders_to_frame(payload.orders)
    probs = model.predict_proba(rows)[:, 1]
    reasons_all = explain_batch(model, rows, top_k=3)

    results = [
        {
            "order_id": o.order_id,
            "risk_score": round(float(p), 4),
            "flagged": bool(p >= threshold),
            "threshold_used": threshold,
            "top_reasons": reasons,
        }
        for o, p, reasons in zip(payload.orders, probs, reasons_all)
    ]
    return {"results": results}


REQUIRED_CSV_COLUMNS = ["order_id"] + CATEGORICAL + NUMERIC


@app.post("/score/csv", dependencies=[Depends(require_api_key)])
async def score_csv(file: UploadFile = File(...)):
    """
    Score a CSV of orders. Expected columns match schema.json:
    order_id, category, customer_type, payment_method, delivery_pincode_risk_tier,
    day_of_week, order_value, customer_order_count, discount_percent, order_hour.
    Returns JSON with per-row score + flagged + top_reasons.
    """
    raw = await file.read()
    if len(raw) > MAX_CSV_BYTES:
        raise HTTPException(status_code=413, detail=f"CSV exceeds {MAX_CSV_BYTES // 1024} KB limit.")

    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    if len(df) == 0:
        raise HTTPException(status_code=400, detail="CSV has no rows.")
    if len(df) > MAX_CSV_ROWS:
        raise HTTPException(status_code=413, detail=f"CSV exceeds {MAX_CSV_ROWS} row limit.")

    missing = [c for c in REQUIRED_CSV_COLUMNS if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"CSV missing columns: {missing}")

    model = get_model()
    metrics = get_metrics()
    threshold = metrics.get("chosen_threshold", 0.5)

    feature_df = df[CATEGORICAL + NUMERIC].copy()
    try:
        probs = model.predict_proba(feature_df)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Scoring failed — check CSV values: {e}")

    reasons_all = explain_batch(model, feature_df, top_k=3)

    results = [
        {
            "order_id": str(df["order_id"].iloc[i]),
            "risk_score": round(float(probs[i]), 4),
            "flagged": bool(probs[i] >= threshold),
            "threshold_used": threshold,
            "top_reasons": reasons_all[i],
        }
        for i in range(len(df))
    ]
    return {"results": results, "count": len(results)}


@app.get("/metrics")
def get_saved_metrics():
    return get_metrics()
