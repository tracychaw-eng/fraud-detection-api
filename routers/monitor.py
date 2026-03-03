"""
routers/monitor.py — Prediction monitoring and drift detection.

record_prediction() is called by main.py on every /predict request.
GET /monitor/stats computes KS and PSI drift stats on demand.
DELETE /monitor/reset clears all state (useful for testing).
"""
import threading
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime, timezone
from typing import Dict
from fastapi import APIRouter, HTTPException
from scipy import stats

from state import FEATURE_NAMES

router = APIRouter(prefix="/monitor", tags=["Monitoring"])

# ── Reference distributions (5000-row training sample) ────────────
_ref_df: pd.DataFrame = (
    pd.read_csv("data/creditcard.csv")
      .drop(columns=["Class"])
      .sample(n=5000, random_state=42)
      [FEATURE_NAMES]
)
_ref_arrays: Dict[str, np.ndarray] = {
    col: _ref_df[col].values for col in FEATURE_NAMES
}

# ── Live prediction store ──────────────────────────────────────────
_lock = threading.Lock()
_predictions: deque = deque(maxlen=10_000)
_feature_buffers: Dict[str, deque] = {col: deque(maxlen=10_000) for col in FEATURE_NAMES}
_request_count: int = 0
_fraud_count:   int = 0


def record_prediction(features: dict, probability: float) -> None:
    """Called by main.predict_single() for every prediction. Thread-safe."""
    global _request_count, _fraud_count
    with _lock:
        _request_count += 1
        if probability >= 0.5:
            _fraud_count += 1
        _predictions.append({"prob": probability, "ts": datetime.now(timezone.utc).isoformat()})
        for col in FEATURE_NAMES:
            if col in features:
                _feature_buffers[col].append(features[col])


def _compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """PSI < 0.1: stable. 0.1-0.25: moderate drift. >0.25: significant drift."""
    _, bin_edges = np.histogram(reference, bins=n_bins)
    bin_edges[0]  = -np.inf
    bin_edges[-1] =  np.inf
    ref_pct = (np.histogram(reference, bins=bin_edges)[0] / len(reference)).clip(1e-8)
    cur_pct = (np.histogram(current,   bins=bin_edges)[0] / len(current)).clip(1e-8)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


@router.get("/stats")
def monitor_stats(min_samples: int = 100):
    """
    Drift detection stats comparing incoming features to training distribution.
    Requires at least min_samples predictions before drift tests are meaningful.
    """
    with _lock:
        n_live     = len(_predictions)
        req_count  = _request_count
        fraud_rate = _fraud_count / max(req_count, 1)
        if n_live < min_samples:
            return {
                "status":        "insufficient_data",
                "message":       f"Need {min_samples} predictions, have {n_live}.",
                "request_count": req_count,
            }
        feature_snapshots = {
            col: np.array(list(_feature_buffers[col]))
            for col in FEATURE_NAMES
            if len(_feature_buffers[col]) >= min_samples
        }
        prob_array = np.array([p["prob"] for p in _predictions])

    feature_stats  = {}
    drifted_features = []
    for col, live_vals in feature_snapshots.items():
        ks_stat, ks_pval = stats.ks_2samp(_ref_arrays[col], live_vals)
        psi_val          = _compute_psi(_ref_arrays[col], live_vals)
        is_drifted       = (ks_pval < 0.05) or (psi_val > 0.1)
        if is_drifted:
            drifted_features.append(col)
        feature_stats[col] = {
            "ks_statistic": round(float(ks_stat), 4),
            "ks_pvalue":    round(float(ks_pval), 4),
            "psi":          round(psi_val, 4),
            "drift_flag":   is_drifted,
            "n_live":       len(live_vals),
        }

    return {
        "status":            "ok",
        "request_count":     req_count,
        "live_samples":      n_live,
        "fraud_rate_live":   round(fraud_rate, 4),
        "drifted_features":  drifted_features,
        "n_drifted":         len(drifted_features),
        "probability_stats": {
            "mean": round(float(prob_array.mean()), 4),
            "std":  round(float(prob_array.std()),  4),
            "p5":   round(float(np.percentile(prob_array, 5)),  4),
            "p95":  round(float(np.percentile(prob_array, 95)), 4),
            "min":  round(float(prob_array.min()), 4),
            "max":  round(float(prob_array.max()), 4),
        },
        "feature_drift": feature_stats,
    }


@router.delete("/reset")
def monitor_reset():
    """Reset all monitoring state."""
    global _request_count, _fraud_count
    with _lock:
        _predictions.clear()
        for buf in _feature_buffers.values():
            buf.clear()
        _request_count = 0
        _fraud_count   = 0
    return {"status": "reset"}
