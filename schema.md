# Data Schema — RazorGuard AI

**Grain:** one row = one order at the moment of purchase (before fulfillment).

**Target label:** `returned_or_disputed` — binary (1 if the order was returned or disputed within 30 days of delivery; 0 otherwise).

| Field | Type | Example values | Why it signals risk |
|---|---|---|---|
| `order_id` | string | `ORD100234` | Unique key, not a model feature |
| `order_value` | numeric (INR) | 249, 4999, 18500 | High-value orders have more return incentive |
| `category` | categorical | electronics, apparel, home, beauty, grocery | Return rates vary sharply by category |
| `customer_type` | categorical | new, returning | New customers are statistically higher-risk |
| `customer_order_count` | numeric | 0, 3, 27 | Proxy for trust/loyalty |
| `payment_method` | categorical | UPI, card, netbanking, COD, wallet | COD correlates with higher RTO/return risk |
| `delivery_pincode_risk_tier` | categorical | low, medium, high | Proxy for logistics/fraud-prone zones |
| `discount_percent` | numeric | 0, 20, 60 | Heavy discounts sometimes correlate with impulse returns |
| `order_hour` | numeric | 0-23 | Late-night impulse buys skew riskier |
| `day_of_week` | categorical | Mon...Sun | Weekend impulse buys skew riskier |
| `returned_or_disputed` | binary (label) | 0, 1 | Target variable |

## Locked decisions

- **Realistic base rate.** Synthetic data is sampled at ~5-15% return rate, not 50/50. We report PR-AUC and use class weighting, not plain accuracy.
- **Name fields once, use everywhere.** These exact field names are the CSV columns, the FastAPI JSON keys, and the Lovable UI labels — no renaming mid-project.
- Machine-readable copy of this schema lives in [`schema.json`](schema.json); reused verbatim in generation, training, API, and frontend prompts.
