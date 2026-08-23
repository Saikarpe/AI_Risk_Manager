# Lovable Frontend Prompt

Paste this into Lovable as your initial project prompt. The API is already deployed —
use this URL as `<YOUR_API_URL>`:

```
https://ai-risk-manager-kl0x.onrender.com
```

---

Build a frontend-only React dashboard called **"Return-Risk Scorer"** for an e-commerce
merchant. **Do not scaffold a backend or use Supabase — this app only calls an external
REST API via `fetch`.** The API base URL is `https://ai-risk-manager-kl0x.onrender.com`.

## API contract (use these exact field names — do not rename anything)

`GET {API_URL}/metrics` returns JSON including: `pr_auc`, `roc_auc`, `chosen_threshold`,
`precision_at_threshold`, `recall_at_threshold`, `flag_rate_at_threshold`,
`test_positive_rate`, `n_test`, `threshold_rationale` (a string explaining the threshold
choice), `confusion_matrix_at_threshold` (object with `true_positive`, `false_positive`,
`true_negative`, `false_negative`).

`POST {API_URL}/score/batch` accepts `{"orders": [ ...order objects... ]}` where each
order object has exactly these fields:
```
order_id (string), order_value (number), category (one of: electronics, apparel, home,
beauty, grocery), customer_type (new | returning), customer_order_count (integer),
payment_method (UPI | card | netbanking | COD | wallet), delivery_pincode_risk_tier
(low | medium | high), discount_percent (number 0-100), order_hour (integer 0-23),
day_of_week (Mon | Tue | Wed | Thu | Fri | Sat | Sun)
```
It returns `{"results": [{"order_id", "risk_score" (0-1 float), "flagged" (boolean),
"threshold_used" (float)}, ...]}`.

`POST {API_URL}/score` takes a single order object (same shape) and returns a single
result object (same shape as one item in the batch results).

## Pages / components to build

1. **Metrics panel** (top of page): cards showing PR-AUC, ROC-AUC, precision, recall,
   and flag rate from `GET /metrics`, plus the `threshold_rationale` text displayed as a
   short explanatory note. Show the confusion matrix as a small 2x2 grid.

2. **Order feed table**: a table of ~30-50 sample orders (hardcode a realistic array of
   order objects matching the schema above, spanning a mix of categories, payment
   methods, and risk tiers). On load, POST the full batch to `/score/batch` and merge
   `risk_score` / `flagged` back onto each row.
   - Columns: order_id, category, order_value (formatted as ₹), customer_type,
     payment_method, delivery_pincode_risk_tier, risk_score (as a percentage, 1 decimal),
     and a color-coded risk flag column (green "OK" badge if not flagged, red "REVIEW"
     badge if flagged).
   - Sortable by risk_score descending by default (highest risk first).

3. **Threshold slider**: a slider from 0 to 1 (step 0.01), defaulting to the
   `chosen_threshold` value from `/metrics`. Moving it re-flags every row **client-side**
   (recompute `flagged = risk_score >= sliderValue` locally — do not re-call the API on
   every slider tick) and live-updates: the count of flagged orders, and the flag-rate
   percentage, shown next to the slider.

4. Clean, modern dashboard aesthetic — dark or light, your choice — but make the risk
   badges (green/red) high-contrast and immediately scannable, since this is a
   risk-triage tool a merchant ops team will scan quickly.

Handle API errors gracefully (e.g. if the batch score call fails, show a toast/banner
saying the API is unreachable, don't crash the page).
