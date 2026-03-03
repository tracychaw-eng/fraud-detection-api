"""
routers/shap_plots.py — SHAP visualisation endpoints.

GET  /shap/summary/beeswarm  — global feature impact across 200 background samples
GET  /shap/summary/bar       — mean |SHAP| per feature
POST /shap/summary/waterfall — per-transaction waterfall explanation
"""
import io
import matplotlib
matplotlib.use("Agg")   # must be before pyplot import — no display needed
import matplotlib.pyplot as plt
import shap
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from state import FEATURE_NAMES, explainer

router = APIRouter(prefix="/shap", tags=["SHAP Plots"])

# ── 200-row background sample — computed once at startup ───────────
_background_df = (
    pd.read_csv("data/creditcard.csv")
      .drop(columns=["Class"])
      .sample(n=200, random_state=42)
      [FEATURE_NAMES]
)
_background_shap: shap.Explanation = explainer(_background_df)


def _fig_to_png(fig: plt.Figure) -> StreamingResponse:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return StreamingResponse(buf, media_type="image/png")


@router.get("/summary/beeswarm")
def shap_beeswarm(max_display: int = Query(20, ge=5, le=30)):
    """
    Beeswarm plot — global feature importance across 200 background samples.
    Each dot is one sample; color = feature value; x-axis = SHAP impact on fraud probability.
    Returns a PNG image.
    """
    try:
        plt.figure(figsize=(10, 8))
        shap.plots.beeswarm(_background_shap, max_display=max_display, show=False)
        fig = plt.gcf()
        fig.suptitle("SHAP Beeswarm — Feature Impact on Fraud Probability", y=1.01, fontsize=13)
        return _fig_to_png(fig)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary/bar")
def shap_bar(max_display: int = Query(20, ge=5, le=30)):
    """
    Bar plot — mean absolute SHAP value per feature (global importance ranking).
    Returns a PNG image.
    """
    try:
        plt.figure(figsize=(10, 6))
        shap.plots.bar(_background_shap, max_display=max_display, show=False)
        fig = plt.gcf()
        fig.suptitle("SHAP Bar — Mean |SHAP| per Feature", y=1.01, fontsize=13)
        return _fig_to_png(fig)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/summary/waterfall")
def shap_waterfall(transaction: dict, max_display: int = Query(15, ge=5, le=30)):
    """
    Waterfall plot — per-transaction explanation showing how each feature
    pushed the prediction from the base value to the final fraud probability.
    POST body: same JSON as /predict. Returns a PNG image.
    """
    try:
        input_df = pd.DataFrame([transaction])[FEATURE_NAMES]
        shap_exp = explainer(input_df)
        plt.figure(figsize=(10, 7))
        shap.plots.waterfall(shap_exp[0], max_display=max_display, show=False)
        fig = plt.gcf()
        return _fig_to_png(fig)
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Missing feature: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
