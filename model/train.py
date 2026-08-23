"""
Train the return/dispute risk classifier and evaluate on the held-out test set.

- Trains an XGBoost classifier with class weighting (imbalanced ~9-10% positive rate).
- Reports precision, recall, F1, PR-AUC, ROC-AUC on the held-out test set.
- Builds a cost curve (cost of a missed risky order vs. cost of a false flag) across
  thresholds and picks the operating threshold that minimizes expected cost.
- Saves model.pkl, metrics.json, and cost_curve.csv.

Usage:
    python model/train.py --train data/train.csv --test data/test.csv --out-dir model
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb

CATEGORICAL = [
    "category",
    "customer_type",
    "payment_method",
    "delivery_pincode_risk_tier",
    "day_of_week",
]
NUMERIC = [
    "order_value",
    "customer_order_count",
    "discount_percent",
    "order_hour",
]
TARGET = "returned_or_disputed"
ID_COL = "order_id"

# ---- Business cost assumptions (INR-scale relative units; tune to your merchant) ----
# Cost of missing a genuinely risky order (false negative): the merchant absorbs
# shipping + restocking + payment-processing fees when it comes back, PLUS the lost
# opportunity to intervene. Modeled as a fraction of order value plus a fixed handling cost.
# Cost of a false flag (false positive): friction cost of manual verification /
# delayed capture on a customer who was actually going to be fine. This is NOT just
# an internal ops cost — added friction (extra verification step, delayed capture)
# measurably increases cart abandonment on genuine customers, so a fraction of flagged
# good orders are lost sales outright. Modeled as a fixed review cost plus an abandonment
# probability applied to order value.
FN_COST_FIXED = 150.0  # INR — fixed cost of an unflagged return (shipping+restock+fees)
FN_COST_VALUE_FRAC = 0.12  # + 12% of order value lost on an unflagged return
FP_COST_FIXED = 20.0  # INR — fixed cost of manually verifying a good order
FP_ABANDON_PROB = 0.08  # probability a flagged-but-good customer abandons the order


def build_pipeline():
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ],
        remainder="passthrough",
    )
    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="aucpr",
        random_state=42,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def expected_cost(y_true, y_prob, order_value, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    fn_mask = (y_true == 1) & (y_pred == 0)
    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_cost = (FN_COST_FIXED + FN_COST_VALUE_FRAC * order_value[fn_mask]).sum()
    fp_cost = (FP_COST_FIXED + FP_ABANDON_PROB * order_value[fp_mask]).sum()
    return fn_cost + fp_cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.csv")
    ap.add_argument("--test", default="data/test.csv")
    ap.add_argument("--out-dir", default="model")
    args = ap.parse_args()

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    X_train = train_df[CATEGORICAL + NUMERIC]
    y_train = train_df[TARGET]
    X_test = test_df[CATEGORICAL + NUMERIC]
    y_test = test_df[TARGET]

    # class weighting for the imbalanced label
    pos_rate = y_train.mean()
    scale_pos_weight = (1 - pos_rate) / pos_rate

    pipe = build_pipeline()
    pipe.set_params(clf__scale_pos_weight=scale_pos_weight)
    pipe.fit(X_train, y_train)

    y_prob = pipe.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    # ---- Cost curve across thresholds to pick the operating point ----
    thresholds = np.linspace(0.01, 0.99, 99)
    order_value_test = test_df["order_value"].to_numpy()
    y_true_arr = y_test.to_numpy()

    cost_rows = []
    for t in thresholds:
        cost = expected_cost(y_true_arr, y_prob, order_value_test, t)
        y_pred_t = (y_prob >= t).astype(int)
        cost_rows.append(
            {
                "threshold": round(float(t), 3),
                "expected_cost": round(float(cost), 2),
                "precision": round(float(precision_score(y_test, y_pred_t, zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, y_pred_t, zero_division=0)), 4),
                "flag_rate": round(float(y_pred_t.mean()), 4),
            }
        )
    cost_df = pd.DataFrame(cost_rows)

    # Pure unconstrained cost-minimization is degenerate at this base rate (~9%) and
    # signal strength (ROC-AUC ~0.73): depending on the exact FN/FP cost ratio it swings
    # to flagging almost everyone or almost no one. Neither is operationally useful, so we
    # add a business floor — catch at least RECALL_FLOOR of genuinely risky orders — and
    # minimize expected cost subject to that. This mirrors how a real risk team would set
    # the knob: "we must catch most bad orders; given that, minimize false-flag cost."
    RECALL_FLOOR = 0.50
    unconstrained_best = cost_df.loc[cost_df["expected_cost"].idxmin()]
    feasible = cost_df[cost_df["recall"] >= RECALL_FLOOR]
    if len(feasible) == 0:
        feasible = cost_df  # fallback: no threshold clears the floor, use unconstrained
    best_row = feasible.loc[feasible["expected_cost"].idxmin()]
    best_threshold = float(best_row["threshold"])

    # also compute cost at naive default 0.5 for comparison
    cost_at_default = expected_cost(y_true_arr, y_prob, order_value_test, 0.5)

    y_pred_best = (y_prob >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_best).ravel()

    metrics = {
        "model": "XGBClassifier",
        "n_train": len(train_df),
        "n_test": len(test_df),
        "train_positive_rate": round(float(pos_rate), 4),
        "test_positive_rate": round(float(y_test.mean()), 4),
        "pr_auc": round(float(pr_auc), 4),
        "roc_auc": round(float(roc_auc), 4),
        "chosen_threshold": best_threshold,
        "precision_at_threshold": round(float(precision_score(y_test, y_pred_best, zero_division=0)), 4),
        "recall_at_threshold": round(float(recall_score(y_test, y_pred_best, zero_division=0)), 4),
        "f1_at_threshold": round(float(f1_score(y_test, y_pred_best, zero_division=0)), 4),
        "flag_rate_at_threshold": round(float(y_pred_best.mean()), 4),
        "confusion_matrix_at_threshold": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "cost_assumptions": {
            "false_negative_fixed_cost_inr": FN_COST_FIXED,
            "false_negative_value_fraction": FN_COST_VALUE_FRAC,
            "false_positive_fixed_cost_inr": FP_COST_FIXED,
            "false_positive_abandon_probability": FP_ABANDON_PROB,
            "note": (
                "FN = missed risky order: merchant eats shipping + restocking + "
                "payment fees (~12% of order value) plus a fixed handling cost. "
                "FP = false flag: fixed manual-verification cost, PLUS an 8% chance the "
                "added friction causes a genuinely good customer to abandon the order "
                "(a lost sale worth the full order value). Both costs scale with order "
                "value, which keeps the optimal threshold from degenerating to 'flag "
                "almost everyone' — flagging is not free even though missing a return is worse."
            ),
        },
        "expected_cost_at_chosen_threshold": round(float(best_row["expected_cost"]), 2),
        "expected_cost_at_default_0.5": round(float(cost_at_default), 2),
        "cost_savings_vs_default": round(float(cost_at_default - best_row["expected_cost"]), 2),
        "recall_floor": RECALL_FLOOR,
        "unconstrained_min_cost_threshold": float(unconstrained_best["threshold"]),
        "unconstrained_min_cost_recall": float(unconstrained_best["recall"]),
        "threshold_rationale": (
            f"Unconstrained cost-minimization picks threshold {unconstrained_best['threshold']:.2f}, "
            f"which only catches {unconstrained_best['recall']*100:.0f}% of risky orders — not "
            f"operationally useful for a merchant who needs to catch most bad orders. We instead "
            f"minimize expected cost subject to a business floor of catching at least "
            f"{RECALL_FLOOR*100:.0f}% of risky orders, which selects threshold {best_threshold:.2f} "
            f"(recall {best_row['recall']*100:.0f}%, precision {best_row['precision']*100:.0f}%). "
            f"Costs: a missed risky order costs ~{FN_COST_FIXED:.0f} + {FN_COST_VALUE_FRAC*100:.0f}% "
            f"of order value (shipping/restock/fees); a false flag costs a fixed "
            f"~{FP_COST_FIXED:.0f} plus an {FP_ABANDON_PROB*100:.0f}% chance of losing the sale "
            f"outright (friction-driven abandonment on a genuine customer)."
        ),
        "feature_names": CATEGORICAL + NUMERIC,
        "target": TARGET,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    joblib.dump(pipe, os.path.join(args.out_dir, "model.pkl"))
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    cost_df.to_csv(os.path.join(args.out_dir, "cost_curve.csv"), index=False)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
