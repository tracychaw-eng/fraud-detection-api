import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# ── Reusable test data ─────────────────────────────────────────────

# A legitimate-looking transaction (low fraud probability expected)
LEGIT_TRANSACTION = {
    "Time": 0.0, "V1": -1.3598, "V2": -0.0728, "V3": 2.5363,
    "V4": 1.3782, "V5": -0.3383, "V6": 0.4624, "V7": 0.2396,
    "V8": 0.0987, "V9": 0.3638, "V10": 0.0908, "V11": -0.5516,
    "V12": -0.6178, "V13": -0.9913, "V14": -0.3112, "V15": 1.4682,
    "V16": -0.4704, "V17": 0.2079, "V18": 0.0258, "V19": 0.4040,
    "V20": 0.2514, "V21": -0.0183, "V22": 0.2778, "V23": -0.1105,
    "V24": 0.0669, "V25": 0.1285, "V26": -0.1891, "V27": 0.1336,
    "V28": -0.0211, "Amount": 149.62
}

# A fraud-pattern transaction (high fraud probability expected)
FRAUD_TRANSACTION = {
    "Time": 406.0, "V1": -2.3122, "V2": 1.9519, "V3": -1.6096,
    "V4": 3.9979, "V5": -0.5222, "V6": -1.4265, "V7": -2.5374,
    "V8": 1.3918, "V9": -2.7706, "V10": -2.7722, "V11": 3.2020,
    "V12": -2.8992, "V13": -0.5950, "V14": -4.2895, "V15": 0.3898,
    "V16": -1.1405, "V17": -2.8300, "V18": -0.0168, "V19": 0.4165,
    "V20": 0.1260, "V21": 0.5173, "V22": -0.0355, "V23": -0.4658,
    "V24": 0.3200, "V25": 0.0445, "V26": 0.1774, "V27": 0.2613,
    "V28": 0.1436, "Amount": 239.93
}


# ── Test 1: Health check ───────────────────────────────────────────
def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "threshold" in data
    assert 0.0 < data["threshold"] < 1.0


# ── Test 2: Valid predict returns correct structure + SHAP ─────────
def test_predict_returns_valid_structure():
    response = client.post("/predict", json=LEGIT_TRANSACTION)
    assert response.status_code == 200
    data = response.json()

    # Prediction fields
    assert data["prediction"] in [0, 1]
    assert data["label"] in ["fraud", "legitimate"]
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert 0.0 < data["threshold_used"] < 1.0

    # SHAP fields — this is new vs Project 1
    assert "top_shap_features" in data
    assert isinstance(data["top_shap_features"], dict)
    assert len(data["top_shap_features"]) == 5   # exactly 5 features returned
    assert "shap_interpretation" in data

    # All SHAP values should be floats
    for feature, value in data["top_shap_features"].items():
        assert isinstance(value, float), f"SHAP value for {feature} is not float"


# ── Test 3: Fraud pattern returns fraud label ──────────────────────
def test_fraud_transaction_flagged():
    response = client.post("/predict", json=FRAUD_TRANSACTION)
    assert response.status_code == 200
    data = response.json()

    assert data["prediction"] == 1
    assert data["label"] == "fraud"
    assert data["fraud_probability"] > 0.5

    # V14 should be in top SHAP features for this fraud pattern
    assert "V14" in data["top_shap_features"], (
        f"Expected V14 in top SHAP features, got: {list(data['top_shap_features'].keys())}"
    )


# ── Test 4: Batch endpoint returns correct structure ───────────────
def test_batch_predict_returns_correct_structure():
    payload = {
        "transactions": [LEGIT_TRANSACTION, FRAUD_TRANSACTION, LEGIT_TRANSACTION]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Summary counts
    assert data["total_transactions"] == 3
    assert data["flagged_count"] + data["legitimate_count"] == 3

    # Per-transaction results
    assert len(data["results"]) == 3
    for i, result in enumerate(data["results"]):
        assert result["transaction_id"] == i
        assert result["prediction"] in [0, 1]
        assert result["label"] in ["fraud", "legitimate"]
        assert 0.0 <= result["fraud_probability"] <= 1.0

    # The fraud transaction (index 1) should be flagged
    assert data["results"][1]["prediction"] == 1


# ── Test 5: Missing field returns 422 ─────────────────────────────
def test_missing_field_returns_422():
    incomplete = LEGIT_TRANSACTION.copy()
    del incomplete["V14"]   # remove a required field
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


# ── Test 6: Wrong data type returns 422 ───────────────────────────
def test_wrong_type_returns_422():
    bad_input = LEGIT_TRANSACTION.copy()
    bad_input["Amount"] = "not_a_number"   # Amount must be float
    response = client.post("/predict", json=bad_input)
    assert response.status_code == 422