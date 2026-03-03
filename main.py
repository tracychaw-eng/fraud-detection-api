import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import uvicorn

from state import model, THRESHOLD, explainer, FEATURE_NAMES
from routers.monitor import record_prediction
from routers import shap_plots, monitor, ab_testing

# ── App ───────────────────────────────────────────────────────────
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud detection with SHAP explainability",
    version="2.0.0"
)

app.include_router(shap_plots.router)
app.include_router(monitor.router)
app.include_router(ab_testing.router)

# ── Request / Response schemas ────────────────────────────────────
class Transaction(BaseModel):
    model_config = {"json_schema_extra": {"example": {
        "Time": 0.0, "V1": -1.3598, "V2": -0.0728, "V3": 2.5363,
        "V4": 1.3782, "V5": -0.3383, "V6": 0.4624, "V7": 0.2396,
        "V8": 0.0987, "V9": 0.3638, "V10": 0.0908, "V11": -0.5516,
        "V12": -0.6178, "V13": -0.9913, "V14": -0.3112, "V15": 1.4682,
        "V16": -0.4704, "V17": 0.2079, "V18": 0.0258, "V19": 0.4040,
        "V20": 0.2514, "V21": -0.0183, "V22": 0.2778, "V23": -0.1105,
        "V24": 0.0669, "V25": 0.1285, "V26": -0.1891, "V27": 0.1336,
        "V28": -0.0211, "Amount": 149.62
    }}}

    Time:   float
    V1:     float
    V2:     float
    V3:     float
    V4:     float
    V5:     float
    V6:     float
    V7:     float
    V8:     float
    V9:     float
    V10:    float
    V11:    float
    V12:    float
    V13:    float
    V14:    float
    V15:    float
    V16:    float
    V17:    float
    V18:    float
    V19:    float
    V20:    float
    V21:    float
    V22:    float
    V23:    float
    V24:    float
    V25:    float
    V26:    float
    V27:    float
    V28:    float
    Amount: float

class BatchRequest(BaseModel):
    transactions: List[Transaction] = Field(..., max_length=100)


def predict_single(transaction: Transaction):
    """Core prediction logic — used by both /predict and /invocations."""
    input_dict = transaction.model_dump()
    input_df   = pd.DataFrame([input_dict])[FEATURE_NAMES]

    proba      = float(model.predict_proba(input_df)[0, 1])
    prediction = int(proba >= THRESHOLD)
    label      = "fraud" if prediction == 1 else "legitimate"

    record_prediction(input_dict, proba)  # feed monitoring ring buffer

    shap_values = explainer.shap_values(input_df)
    shap_dict   = dict(zip(FEATURE_NAMES, shap_values[0]))
    top_shap    = dict(sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5])
    top_shap    = {k: round(float(v), 4) for k, v in top_shap.items()}

    return {
        "prediction":        prediction,
        "label":             label,
        "fraud_probability": round(proba, 4),
        "threshold_used":    round(THRESHOLD, 4),
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
        "status":    "ok",
        "model":     "fraud-detector-xgb-medium-balanced",
        "threshold": round(THRESHOLD, 4)
    }

@app.post("/predict")
def predict(transaction: Transaction):
    """Single transaction fraud prediction with SHAP explainability."""
    try:
        return predict_single(transaction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
def predict_batch(request: BatchRequest):
    """Batch prediction — up to 100 transactions at once."""
    try:
        results = []
        for i, txn in enumerate(request.transactions):
            result = predict_single(txn)
            results.append({
                "transaction_id":    i,
                "prediction":        result["prediction"],
                "label":             result["label"],
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
