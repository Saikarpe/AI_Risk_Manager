"""
Derived features used by the training pipeline.

Kept in its own module (not train.py or api/) so joblib.load can resolve
``add_derived_features`` at inference time — pickling a function defined in
``__main__`` breaks when the model is loaded from a different process.
"""

import numpy as np
import pandas as pd

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
DERIVED_NUMERIC = [
    "is_late_night",
    "is_weekend",
    "order_value_log",
    "new_cod",
    "high_value",
    "new_high_pincode",
]


def add_derived_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["is_late_night"] = ((X["order_hour"] >= 22) | (X["order_hour"] <= 4)).astype(int)
    X["is_weekend"] = X["day_of_week"].isin(["Sat", "Sun"]).astype(int)
    X["order_value_log"] = np.log1p(X["order_value"].astype(float))
    X["new_cod"] = (
        (X["customer_type"] == "new") & (X["payment_method"] == "COD")
    ).astype(int)
    X["high_value"] = (X["order_value"] > 5000).astype(int)
    X["new_high_pincode"] = (
        (X["customer_type"] == "new") & (X["delivery_pincode_risk_tier"] == "high")
    ).astype(int)
    return X
