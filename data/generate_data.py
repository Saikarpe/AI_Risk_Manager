"""
Synthetic order-data generator for the Pre-Shipment Return-Risk Scorer.

Generates order-level rows with realistic return/dispute correlations baked in
(new customer + high value + risky pincode -> higher return probability, COD ->
higher risk, category effects, discount effects, time-of-purchase effects), at a
realistic base rate (~8-12%), not 50/50.

Usage:
    python data/generate_data.py --n 8000 --seed 42 --out-dir data
Produces data/train.csv and data/test.csv (80/20 stratified split).
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from faker import Faker

CATEGORIES = ["electronics", "apparel", "home", "beauty", "grocery"]
PAYMENT_METHODS = ["UPI", "card", "netbanking", "COD", "wallet"]
PINCODE_TIERS = ["low", "medium", "high"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Category base price ranges (INR) so order_value correlates sensibly with category.
CATEGORY_PRICE_RANGE = {
    "electronics": (800, 45000),
    "apparel": (300, 6000),
    "home": (400, 12000),
    "beauty": (150, 4000),
    "grocery": (100, 2500),
}

# Category effect on return-logit: apparel/electronics return more, grocery returns least.
CATEGORY_LOGIT = {
    "electronics": 0.35,
    "apparel": 0.55,
    "home": 0.00,
    "beauty": 0.15,
    "grocery": -0.70,
}

PAYMENT_LOGIT = {
    "UPI": -0.10,
    "card": -0.15,
    "netbanking": -0.05,
    "COD": 0.80,
    "wallet": 0.05,
}

PINCODE_LOGIT = {"low": -0.25, "medium": 0.15, "high": 0.70}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate(n, seed, target_rate=0.10):
    rng = np.random.default_rng(seed)
    fake = Faker()
    Faker.seed(seed)

    category = rng.choice(CATEGORIES, size=n, p=[0.22, 0.28, 0.18, 0.17, 0.15])

    order_value = np.empty(n)
    for cat in CATEGORIES:
        mask = category == cat
        lo, hi = CATEGORY_PRICE_RANGE[cat]
        # log-normal-ish spread within the category's realistic price band
        vals = rng.lognormal(mean=np.log((lo + hi) / 4), sigma=0.6, size=mask.sum())
        order_value[mask] = np.clip(vals, lo, hi)
    order_value = np.round(order_value, 0)

    customer_type = rng.choice(["new", "returning"], size=n, p=[0.38, 0.62])
    customer_order_count = np.where(
        customer_type == "new",
        0,
        rng.poisson(lam=6, size=n) + 1,
    )
    customer_order_count = np.clip(customer_order_count, 0, 80)

    payment_method = rng.choice(
        PAYMENT_METHODS, size=n, p=[0.34, 0.28, 0.10, 0.20, 0.08]
    )

    delivery_pincode_risk_tier = rng.choice(
        PINCODE_TIERS, size=n, p=[0.55, 0.30, 0.15]
    )

    discount_percent = np.clip(rng.exponential(scale=14, size=n), 0, 80).round(0)

    order_hour = rng.integers(0, 24, size=n)
    day_of_week = rng.choice(DAYS, size=n, p=[0.13, 0.13, 0.13, 0.13, 0.14, 0.17, 0.17])

    order_id = np.array([f"ORD{100000 + i}" for i in range(n)])

    # ---- Build return-risk logit from features (the "ground truth" generative process) ----
    logit = np.zeros(n)

    # category effect
    logit += np.array([CATEGORY_LOGIT[c] for c in category])

    # payment method effect
    logit += np.array([PAYMENT_LOGIT[p] for p in payment_method])

    # pincode risk effect
    logit += np.array([PINCODE_LOGIT[t] for t in delivery_pincode_risk_tier])

    # customer trust: new customers riskier; more prior orders -> lower risk (diminishing)
    logit += np.where(customer_type == "new", 0.55, -0.20)
    logit += -0.045 * np.minimum(customer_order_count, 30)

    # order value: higher value -> somewhat higher risk (log-scaled)
    logit += 0.16 * np.log1p(order_value / 1000.0)

    # discount: heavy discounts nudge impulse-return risk up
    logit += 0.010 * discount_percent

    # time-of-purchase: late night (22-4) and weekend nudge risk up
    is_late_night = (order_hour >= 22) | (order_hour <= 4)
    logit += np.where(is_late_night, 0.30, 0.0)
    is_weekend = np.isin(day_of_week, ["Sat", "Sun"])
    logit += np.where(is_weekend, 0.15, 0.0)

    # interaction: new customer + high value + risky pincode is a compounding red flag
    high_value = order_value > np.quantile(order_value, 0.75)
    risky_combo = (
        (customer_type == "new") & high_value & (delivery_pincode_risk_tier == "high")
    )
    logit += np.where(risky_combo, 0.65, 0.0)

    # idiosyncratic noise (real-world orders aren't perfectly explained by these features)
    logit += rng.normal(0, 0.55, size=n)

    # Calibrate intercept via bisection so mean(P(return)) ~= target_rate
    def mean_rate(intercept):
        return sigmoid(logit + intercept).mean()

    lo_b, hi_b = -10.0, 10.0
    for _ in range(60):
        mid = (lo_b + hi_b) / 2
        if mean_rate(mid) < target_rate:
            lo_b = mid
        else:
            hi_b = mid
    intercept = (lo_b + hi_b) / 2

    prob = sigmoid(logit + intercept)
    returned_or_disputed = rng.binomial(1, prob)

    df = pd.DataFrame(
        {
            "order_id": order_id,
            "order_value": order_value.astype(int),
            "category": category,
            "customer_type": customer_type,
            "customer_order_count": customer_order_count.astype(int),
            "payment_method": payment_method,
            "delivery_pincode_risk_tier": delivery_pincode_risk_tier,
            "discount_percent": discount_percent.astype(int),
            "order_hour": order_hour.astype(int),
            "day_of_week": day_of_week,
            "returned_or_disputed": returned_or_disputed.astype(int),
        }
    )
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-rate", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--out-dir", type=str, default="data")
    args = ap.parse_args()

    df = generate(args.n, args.seed, args.target_rate)

    # Stratified train/test split (held-out test is never touched until evaluation)
    rng = np.random.default_rng(args.seed)
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    test_idx = []
    train_idx = []
    for label in [0, 1]:
        idx = df.index[df["returned_or_disputed"] == label].to_numpy()
        rng.shuffle(idx)
        n_test = int(len(idx) * args.test_frac)
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])

    train_df = df.loc[sorted(train_idx)].reset_index(drop=True)
    test_df = df.loc[sorted(test_idx)].reset_index(drop=True)

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "train.csv")
    test_path = os.path.join(args.out_dir, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    summary = {
        "n_total": len(df),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "train_positive_rate": round(train_df["returned_or_disputed"].mean(), 4),
        "test_positive_rate": round(test_df["returned_or_disputed"].mean(), 4),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
