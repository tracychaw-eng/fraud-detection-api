"""
routers/ab_testing.py — A/B comparison between xgb-shallow-fast and xgb-medium-balanced.

POST /predict/ab runs both models on the same transaction and returns side-by-side results.
Requires fraud_model_xgb-shallow-fast.joblib and fraud_model_xgb-medium-balanced.joblib.
Run train.py first to generate these files.
"""
import os
import time
import shap
import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from pydantic import BaseModel
from state import FEATURE_NAMES, THRESHOLD


class Transaction(BaseModel):
    Time: float;  V1: float;  V2: float;  V3: float;  V4: float;  V5: float
    V6: float;    V7: float;  V8: float;  V9: float;  V10: float; V11: float
    V12: float;   V13: float; V14: float; V15: float; V16: float; V17: float
    V18: float;   V19: float; V20: float; V21: float; V22: float; V23: float
    V24: float;   V25: float; V26: float; V27: float; V28: float; Amount: float

router = APIRouter(prefix="/predict", tags=["A/B Testing"])

MODEL_A_NAME = "xgb-shallow-fast"
MODEL_B_NAME = "xgb-medium-balanced"

def _try_load(path: str):
    return joblib.load(path) if os.path.exists(path) else None

_model_a = _try_load(f"fraud_model_{MODEL_A_NAME}.joblib")
_model_b = _try_load(f"fraud_model_{MODEL_B_NAME}.joblib")
_explainer_a = shap.TreeExplainer(_model_a) if _model_a else None
_explainer_b = shap.TreeExplainer(_model_b) if _model_b else None


def _run(model, explainer, input_df: pd.DataFrame, name: str) -> dict:
    t0    = time.perf_counter()
    proba = float(model.predict_proba(input_df)[0, 1])
    shap_vals = explainer.shap_values(input_df)
    top_shap  = dict(
        sorted(
            zip(FEATURE_NAMES, shap_vals[0]),
            key=lambda x: abs(x[1]), reverse=True
        )[:5]
    )
    return {
        "model":             name,
        "prediction":        int(proba >= THRESHOLD),
        "label":             "fraud" if proba >= THRESHOLD else "legitimate",
        "fraud_probability": round(proba, 4),
        "threshold_used":    round(THRESHOLD, 4),
        "top_shap_features": {k: round(float(v), 4) for k, v in top_shap.items()},
        "latency_ms":        round((time.perf_counter() - t0) * 1000, 2),
    }


@router.post("/ab")
def predict_ab(transaction: Transaction):
    """
    Runs xgb-shallow-fast and xgb-medium-balanced on the same transaction.
    Returns both predictions side by side with agreement analysis and latency comparison.
    Requires both variant model files on disk — run train.py first.
    """
    missing = []
    if _model_a is None: missing.append(f"fraud_model_{MODEL_A_NAME}.joblib")
    if _model_b is None: missing.append(f"fraud_model_{MODEL_B_NAME}.joblib")
    if missing:
        raise HTTPException(
            status_code=503,
            detail={"error": "Variant model files missing. Re-run train.py.", "missing": missing}
        )

    input_df = pd.DataFrame([transaction.model_dump()])[FEATURE_NAMES]
    try:
        result_a = _run(_model_a, _explainer_a, input_df, MODEL_A_NAME)
        result_b = _run(_model_b, _explainer_b, input_df, MODEL_B_NAME)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "model_a":    result_a,
        "model_b":    result_b,
        "comparison": {
            "agree":                       result_a["prediction"] == result_b["prediction"],
            "probability_delta_b_minus_a": round(result_b["fraud_probability"] - result_a["fraud_probability"], 4),
            "faster_model":                MODEL_A_NAME if result_a["latency_ms"] <= result_b["latency_ms"] else MODEL_B_NAME,
            "latency_delta_ms":            round(result_b["latency_ms"] - result_a["latency_ms"], 2),
        },
    }
