"""
Per-prediction explanations for the return-risk model.

Uses XGBoost's pred_contribs (SHAP values) on the transformed feature matrix,
then aggregates one-hot columns back to their original schema fields so the
UI can display "which fields pushed this order's score up/down".
"""

from typing import Any

import numpy as np
import pandas as pd
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


def _map_transformed_to_original(transformed_names: list[str]) -> list[str]:
    """
    Map each ColumnTransformer output name back to the original schema field.

    OneHotEncoder columns look like ``cat__category_electronics``; passthrough
    numerics look like ``remainder__order_value``. We strip the transformer
    prefix and match against the known categorical / numeric lists.
    """
    out = []
    for name in transformed_names:
        tail = name.split("__", 1)[1] if "__" in name else name
        matched = None
        for cat in CATEGORICAL:
            if tail == cat or tail.startswith(cat + "_"):
                matched = cat
                break
        if matched:
            out.append(matched)
        elif tail in NUMERIC:
            out.append(tail)
        else:
            out.append(tail)
    return out


def _stringify(v: Any) -> Any:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return round(float(v), 2)
    return str(v)


def explain_batch(pipe, X_df: pd.DataFrame, top_k: int = 3) -> list[list[dict[str, Any]]]:
    """
    Return top-k reasons per row.

    Each reason: ``{"feature": str, "value": <schema value>, "impact": float}``.
    ``impact > 0`` pushes the risk score up, ``impact < 0`` pushes it down.
    Values are in log-odds units (XGBoost's raw margin), so the sign is what
    matters for UI display, not the absolute magnitude.
    """
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]

    X_transformed = pre.transform(X_df)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    dmat = xgb.DMatrix(np.asarray(X_transformed))
    contribs = clf.get_booster().predict(dmat, pred_contribs=True)
    contribs = contribs[:, :-1]  # drop bias column

    transformed_names = pre.get_feature_names_out().tolist()
    original_cols = _map_transformed_to_original(transformed_names)

    results: list[list[dict[str, Any]]] = []
    for row_idx in range(contribs.shape[0]):
        agg: dict[str, float] = {}
        for col_idx, orig in enumerate(original_cols):
            agg[orig] = agg.get(orig, 0.0) + float(contribs[row_idx, col_idx])
        top = sorted(agg.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
        row = X_df.iloc[row_idx]
        results.append(
            [
                {
                    "feature": feat,
                    "value": _stringify(row[feat]) if feat in row.index else None,
                    "impact": round(imp, 4),
                }
                for feat, imp in top
            ]
        )
    return results
