# Honest Failure Cases (from the held-out test set)

The model is not perfect — precision at the chosen threshold is 24%, recall is 54%.
Here are two concrete, real held-out orders it gets wrong, and why. Use these in the
pitch video's "one honest failure case" segment.

## Case 1 — Missed a real return (false negative)

| Field | Value |
|---|---|
| order_id | `ORD100284` |
| category | home |
| order_value | ₹3,232 |
| customer_type | returning |
| customer_order_count | 8 |
| payment_method | UPI |
| delivery_pincode_risk_tier | low |
| discount_percent | 1% |
| order_hour / day | 10:00, Saturday |
| **actual label** | **returned = 1** |
| **model risk score** | **0.075** (well below the 0.53 threshold — not flagged) |

**Why the model gets it wrong:** every order-level signal here looks safe — a loyal
returning customer (8 prior orders), a low-risk delivery pincode, almost no discount,
paid via UPI, ordered on a weekday morning. There is nothing in the order metadata that
distinguishes this from a totally ordinary, low-risk purchase. The return most likely
happened for a *product-specific* reason — wrong size, item didn't match expectations,
changed their mind — that simply isn't observable from order-level features. This is a
real, structural limitation of the feature set: the model can flag statistical risk
patterns, but it cannot see inside the product itself. Fixing this would require
product-level signals (return-rate history per SKU, size/fit variance, review sentiment),
which are out of scope for this order-time synthetic schema.

## Case 2 — False alarm on a legitimate customer (false positive)

| Field | Value |
|---|---|
| order_id | `ORD101436` |
| category | electronics |
| order_value | ₹20,614 |
| customer_type | returning |
| customer_order_count | 5 |
| payment_method | UPI |
| delivery_pincode_risk_tier | high |
| discount_percent | 15% |
| order_hour / day | 00:00, Thursday |
| **actual label** | **returned = 0** (order was fine) |
| **model risk score** | **0.857** (flagged) |

**Why the model gets it wrong:** this customer has a track record (5 prior orders) and
paid electronically (UPI, not COD), which are both trust signals — but the order is
high-value electronics shipping to a `high`-risk pincode tier at midnight, and the model
weighs the pincode-risk and late-night signals heavily enough to override the customer's
loyalty history. In production this is the kind of false positive that costs real
goodwill: a genuine repeat customer gets sent through extra verification because of where
they live, not who they are. It's a concrete argument for adding a customer-trust override
rule downstream (e.g. "never hard-block, only soft-flag, when `customer_order_count` > N"),
which is exactly the kind of human-in-the-loop guardrail this system is designed to support
— the model recommends, it doesn't auto-cancel.
