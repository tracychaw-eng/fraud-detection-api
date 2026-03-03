import joblib
import shap
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
import uvicorn

# ── Load model and threshold at startup — NOT per request ─────────
print("Loading model...")
model     = joblib.load("fraud_model.joblib")
THRESHOLD = joblib.load("threshold.joblib")
explainer = shap.TreeExplainer(model)
print(f"Model loaded. Threshold: {THRESHOLD:.3f}")

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud detection with SHAP explainability",
    version="1.0.0"
)

# ── Feature names — must match training data exactly ──────────────
FEATURE_NAMES = (
    ["Time"] +
    [f"V{i}" for i in range(1, 29)] +
    ["Amount"]
)

# ── Request / Response schemas ────────────────────────────────────
class Transaction(BaseModel):
    Time:   float = Field(..., example=0.0)
    V1:     float = Field(..., example=-1.3598)
    V2:     float = Field(..., example=-0.0728)
    V3:     float = Field(..., example=2.5363)
    V4:     float = Field(..., example=1.3782)
    V5:     float = Field(..., example=-0.3383)
    V6:     float = Field(..., example=0.4624)
    V7:     float = Field(..., example=0.2396)
    V8:     float = Field(..., example=0.0987)
    V9:     float = Field(..., example=0.3638)
    V10:    float = Field(..., example=0.0908)
    V11:    float = Field(..., example=-0.5516)
    V12:    float = Field(..., example=-0.6178)
    V13:    float = Field(..., example=-0.9913)
    V14:    float = Field(..., example=-0.3112)
    V15:    float = Field(..., example=1.4682)
    V16:    float = Field(..., example=-0.4704)
    V17:    float = Field(..., example=0.2079)
    V18:    float = Field(..., example=0.0258)
    V19:    float = Field(..., example=0.4040)
    V20:    float = Field(..., example=0.2514)
    V21:    float = Field(..., example=-0.0183)
    V22:    float = Field(..., example=0.2778)
    V23:    float = Field(..., example=-0.1105)
    V24:    float = Field(..., example=0.0669)
    V25:    float = Field(..., example=0.1285)
    V26:    float = Field(..., example=-0.1891)
    V27:    float = Field(..., example=0.1336)
    V28:    float = Field(..., example=-0.0211)
    Amount: float = Field(..., example=149.62)

class BatchRequest(BaseModel):
    transactions: List[Transaction] = Field(..., max_items=100)


def predict_single(transaction: Transaction):
    """Core prediction logic — used by both /predict and /invocations."""
    input_dict = transaction.model_dump()
    input_df   = pd.DataFrame([input_dict])[FEATURE_NAMES]

    # Prediction
    proba      = float(model.predict_proba(input_df)[0, 1])
    prediction = int(proba >= THRESHOLD)
    label      = "fraud" if prediction == 1 else "legitimate"

    # SHAP values — explains WHY this prediction was made
    shap_values = explainer.shap_values(input_df)
    shap_dict   = dict(zip(FEATURE_NAMES, shap_values[0]))

    # Return top 5 most influential features (by absolute SHAP value)
    top_shap = dict(
        sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    )
    # Round for readability
    top_shap = {k: round(float(v), 4) for k, v in top_shap.items()}

    return {
        "prediction":       prediction,
        "label":            label,
        "fraud_probability": round(proba, 4),
        "threshold_used":   round(THRESHOLD, 4),
        "top_shap_features": top_shap,
        "shap_interpretation": (
            "Positive SHAP = pushed toward fraud. "
            "Negative SHAP = pushed toward legitimate. "
            "Larger absolute value = stronger influence."
        )
    }


# ── Endpoints ─────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model":  "fraud-detector-xgb-medium-balanced",
        "threshold": round(THRESHOLD, 4)
    }

@app.post("/predict")
def predict(transaction: Transaction):
    """
    Single transaction fraud prediction with SHAP explainability.
    Returns prediction, fraud probability, and top 5 SHAP features.
    """
    try:
        return predict_single(transaction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
def predict_batch(request: BatchRequest):
    """
    Batch prediction — up to 100 transactions at once.
    Returns summary counts plus per-transaction predictions.
    Real fraud systems process transactions in bulk, not one at a time.
    """
    try:
        results = []
        for i, txn in enumerate(request.transactions):
            result = predict_single(txn)
            results.append({
                "transaction_id":   i,
                "prediction":       result["prediction"],
                "label":            result["label"],
                "fraud_probability": result["fraud_probability"],
            })

        flagged = [r for r in results if r["prediction"] == 1]

        return {
            "total_transactions": len(results),
            "flagged_count":      len(flagged),
            "legitimate_count":   len(results) - len(flagged),
            "results":            results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── SageMaker BYOC endpoints — identical to Project 1 ────────────
@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.post("/invocations")
def invocations(transaction: Transaction):
    try:
        return predict_single(transaction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)