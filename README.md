# Fraud Detection API

A machine learning pipeline and REST API that trains an XGBoost classifier to detect fraudulent credit card transactions in real time. Features SHAP explainability, real-time model monitoring with drift detection, A/B testing across model variants, and experiment tracking via Weights & Biases.

---

## Problem Statement

Credit card fraud detection is a highly imbalanced classification problem. In this dataset only **0.173%** of transactions are fraudulent. A naive model that predicts every transaction as legitimate achieves ~99.83% accuracy — but catches **zero fraud**. This project addresses that with SMOTE oversampling, threshold tuning, and recall-optimized evaluation.

---

## Dataset

**Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

| Property | Value |
| --- | --- |
| Total rows | 284,807 |
| Features | 30 (`Time`, `Amount`, `V1`–`V28`) |
| Fraud cases | 492 (0.173%) |
| Legitimate cases | 284,315 (99.827%) |

`V1`–`V28` are PCA-transformed features (anonymized). `Time` and `Amount` are the only original features.

> The dataset is not committed to this repo. Download it manually — see Setup below.

---

## Project Structure

```text
fraud-detection-api/
├── eda.py                          # Exploratory data analysis + visualizations
├── train.py                        # Model training, threshold tuning, W&B logging
├── main.py                         # FastAPI app — /predict, /batch, /invocations
├── state.py                        # Shared model/explainer/threshold singletons
├── generate_shap_plots.py          # Offline script to generate SHAP PNGs
├── test_main.py                    # Pytest test suite (6 tests)
├── routers/
│   ├── shap_plots.py               # GET /shap/summary/* — beeswarm, bar, waterfall
│   ├── monitor.py                  # GET /monitor/stats — KS + PSI drift detection
│   └── ab_testing.py               # POST /predict/ab — side-by-side model comparison
├── fraud_model.joblib              # Best model (xgb-medium-balanced)
├── threshold.joblib                # Tuned threshold (0.959)
├── fraud_model_xgb-shallow-fast.joblib    # Variant A for A/B testing
├── fraud_model_xgb-medium-balanced.joblib # Variant B for A/B testing
├── class_imbalance.png             # Output chart: class distribution
├── precision_recall_curve.png      # Output chart: threshold analysis
├── data/
│   └── creditcard.csv              # Dataset (not committed — download separately)
├── venv/                           # Virtual environment (not committed)
└── wandb/                          # W&B local run logs (not committed)
```

---

## Setup

### 1. Clone and create virtual environment

```bash
git clone https://github.com/tracychaw-eng/fraud-detection-api.git
cd fraud-detection-api

python -m venv venv
```

### 2. Activate the virtual environment

```bash
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

Always use `python -m pip` to ensure packages install into the venv:

```bash
python -m pip install xgboost shap imbalanced-learn scikit-learn pandas matplotlib seaborn wandb fastapi uvicorn pydantic httpx pytest joblib
```

### 4. Download the dataset

```bash
python -m pip install kaggle

# Place your Kaggle API credentials at:
# Windows: C:\Users\<you>\.kaggle\kaggle.json
# Mac/Linux: ~/.kaggle/kaggle.json
# Format: {"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}

kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip
```

### 5. Log in to Weights & Biases

```bash
wandb login
# Paste your API key from https://wandb.ai/authorize
# Input is hidden — just paste and hit Enter
```

---

## Usage

### Exploratory Data Analysis

```bash
venv\Scripts\python.exe eda.py
```

Outputs:

- Prints dataset shape, fraud rate, and class imbalance stats
- Saves `class_imbalance.png` — amount distribution and class bar chart
- Prints top 10 features most different between fraud and legitimate transactions

### Train Models

```bash
venv\Scripts\python.exe train.py
```

What it does:

1. Loads `data/creditcard.csv`
2. Splits 80/20 train/test (stratified)
3. Applies SMOTE to training set only (grows to ~454,902 rows after balancing)
4. Trains a baseline model and tunes the classification threshold (targets Recall ≥ 0.80)
5. Trains 3 XGBoost variants and logs all metrics + confusion matrices to W&B
6. Saves the best model by F1 score to `fraud_model.joblib` and `threshold.joblib`

### Run the API

```bash
venv\Scripts\python.exe main.py
```

The server starts at `http://localhost:8000`. Open `http://localhost:8000/docs` for the interactive Swagger UI.

---

## API Endpoints

Open `http://localhost:8000/docs` for the interactive Swagger UI where you can test every endpoint directly in the browser.

### Core Prediction

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/health` | Returns model name and threshold |
| POST | `/predict` | Single transaction prediction + SHAP |
| POST | `/predict/batch` | Batch prediction (up to 100 transactions) |
| GET | `/ping` | SageMaker health check |
| POST | `/invocations` | SageMaker inference endpoint |

### SHAP Explainability Visualizations

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/shap/summary/beeswarm` | Global feature importance — each dot is one sample |
| GET | `/shap/summary/bar` | Mean \|SHAP\| per feature — ranked bar chart |
| POST | `/shap/summary/waterfall` | Per-transaction explanation — how each feature moved the score |

> All SHAP endpoints return a PNG image. Open the URL directly in a browser tab or use Swagger UI's "Download" button.

### Model Monitoring & Drift Detection

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/monitor/stats` | KS test + PSI drift scores per feature vs training distribution |
| DELETE | `/monitor/reset` | Clear all monitoring state |

### A/B Testing

| Method | Endpoint | Description |
| --- | --- | --- |
| POST | `/predict/ab` | Run the same transaction through two model variants side by side |

### `GET /health`

```json
{
  "status": "ok",
  "model": "fraud-detector-xgb-medium-balanced",
  "threshold": 0.959
}
```

### `POST /predict`

Input: a single transaction with fields `Time`, `V1`–`V28`, `Amount`.

**Legitimate transaction response:**

```json
{
  "prediction": 0,
  "label": "legitimate",
  "fraud_probability": 0.0222,
  "threshold_used": 0.9591,
  "top_shap_features": {
    "V3":  -1.3787,
    "V14": -1.2787,
    "V10": -0.7214,
    "V4":   0.6741,
    "V15": -0.4291
  },
  "shap_interpretation": "Positive SHAP = pushed toward fraud. Negative SHAP = pushed toward legitimate. Larger absolute value = stronger influence."
}
```

**Fraud transaction response:**

```json
{
  "prediction": 1,
  "label": "fraud",
  "fraud_probability": 0.9996,
  "threshold_used": 0.9591,
  "top_shap_features": {
    "V14": 2.3513,
    "V17": 1.5670,
    "V10": 1.4338,
    "V4":  0.9410,
    "V7":  0.5464
  },
  "shap_interpretation": "Positive SHAP = pushed toward fraud. Negative SHAP = pushed toward legitimate. Larger absolute value = stronger influence."
}
```

Notice how `V14` appears in both responses but with opposite signs — negative (`-1.2787`) in the legitimate case pushing away from fraud, and strongly positive (`2.3513`) in the fraud case pushing toward it. This is the most important feature in the model.

### `POST /predict/batch`

Input: `{"transactions": [...]}` — up to 100 transaction objects.

```json
{
  "total_transactions": 3,
  "flagged_count": 1,
  "legitimate_count": 2,
  "results": [...]
}
```

### `GET /shap/summary/beeswarm` and `GET /shap/summary/bar`

Returns a PNG image. Open in browser:

```text
http://localhost:8000/shap/summary/beeswarm
http://localhost:8000/shap/summary/bar
```

- **Beeswarm**: every dot is one of the 200 background samples. The x-axis shows how much that feature pushed the model toward fraud (positive) or legitimate (negative). Color = feature value (red = high, blue = low).
- **Bar**: the mean absolute SHAP value per feature — a simple global importance ranking.

Both accept an optional `?max_display=20` query parameter (5–30).

### `POST /shap/summary/waterfall`

Explains a single transaction — how each feature moved the prediction from the base rate to the final fraud probability.

```bash
curl -X POST "http://localhost:8000/shap/summary/waterfall" \
  -H "Content-Type: application/json" \
  -d '{"Time": 0.0, "V1": -1.3598, ..., "Amount": 149.62}' \
  --output waterfall.png
```

### `GET /monitor/stats` — Where to check drift

Call this endpoint to see whether the live traffic distribution has shifted from the training data:

```text
http://localhost:8000/monitor/stats
```

**Example response (after 100+ predictions):**

```json
{
  "status": "ok",
  "request_count": 250,
  "live_samples": 250,
  "fraud_rate_live": 0.032,
  "drifted_features": ["V4", "Amount"],
  "n_drifted": 2,
  "probability_stats": {
    "mean": 0.0841,
    "std":  0.2103,
    "p5":   0.0012,
    "p95":  0.9871,
    "min":  0.0001,
    "max":  0.9998
  },
  "feature_drift": {
    "V4": {
      "ks_statistic": 0.1823,
      "ks_pvalue":    0.0031,
      "psi":          0.1342,
      "drift_flag":   true,
      "n_live":       250
    },
    "Amount": { "...": "..." }
  }
}
```

**How to read the drift metrics:**

| Metric | Meaning | Threshold |
| --- | --- | --- |
| `ks_pvalue` | Kolmogorov-Smirnov test p-value. Low = distributions differ. | Flag if < 0.05 |
| `psi` | Population Stability Index. Measures distribution shift magnitude. | Flag if > 0.1 |
| `drift_flag` | `true` if either KS or PSI triggers | — |

**Possible `status` values:**

| Status | Meaning |
| --- | --- |
| `ok` | Enough data, drift stats computed |
| `insufficient_data` | Fewer than 100 predictions recorded so far |
| `no_reference_data` | API started without `data/creditcard.csv` (CI/production without dataset) |

### `POST /predict/ab` — Where to check A/B results

Runs the same transaction through two trained model variants and returns a side-by-side comparison:

```text
http://localhost:8000/docs → POST /predict/ab
```

**Example response:**

```json
{
  "model_a": {
    "model": "xgb-shallow-fast",
    "prediction": 0,
    "label": "legitimate",
    "fraud_probability": 0.0198,
    "latency_ms": 4.2
  },
  "model_b": {
    "model": "xgb-medium-balanced",
    "prediction": 0,
    "label": "legitimate",
    "fraud_probability": 0.0222,
    "latency_ms": 6.8
  },
  "comparison": {
    "agree": true,
    "probability_delta_b_minus_a": 0.0024,
    "faster_model": "model_a",
    "latency_delta_ms": 2.6
  }
}
```

`agree: false` means the two models disagree on the label — useful for finding transactions where the models are uncertain. `probability_delta_b_minus_a` shows which model assigns higher fraud risk.

> **Prerequisite:** A/B testing requires re-running `train.py` to generate `fraud_model_xgb-shallow-fast.joblib` and `fraud_model_xgb-medium-balanced.joblib`. Without these files the endpoint returns HTTP 503.

---

### Run Tests

```bash
venv\Scripts\python.exe -m pytest test_main.py -v
```

The test suite covers 6 cases:

| Test | What it checks |
| --- | --- |
| `test_health_returns_200` | `/health` returns 200 with `status`, `model`, and a valid threshold |
| `test_predict_returns_valid_structure` | `/predict` returns all required fields including 5 SHAP values as plain floats |
| `test_fraud_transaction_flagged` | Known fraud pattern (high negative V14) is correctly classified as fraud |
| `test_batch_predict_returns_correct_structure` | `/predict/batch` returns correct counts and per-transaction results |
| `test_missing_field_returns_422` | Missing a required field (e.g. `V14`) returns HTTP 422 |
| `test_wrong_type_returns_422` | Wrong data type (e.g. `Amount: "text"`) returns HTTP 422 |

Test fixtures used:

- **`LEGIT_TRANSACTION`** — real transaction from row 0 of the dataset, expected label: `legitimate`
- **`FRAUD_TRANSACTION`** — known fraud pattern with strongly negative `V14` (`-4.2895`), expected label: `fraud`

---

## Model Variants & Results

All three variants used the same tuned threshold: **0.959**

| Model | n_estimators | max_depth | learning_rate | Precision | Recall | F1 | ROC-AUC | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xgb-shallow-fast | 100 | 3 | 0.10 | 0.769 | 0.847 | 0.806 | 0.981 | 83 | 25 | 15 |
| **xgb-medium-balanced** | **200** | **4** | **0.05** | **0.783** | **0.847** | **0.814** | **0.980** | **83** | **23** | **15** |
| xgb-deep-slow | 300 | 5 | 0.02 | 0.755 | 0.816 | 0.784 | 0.974 | 80 | 26 | 18 |

**Best model: `xgb-medium-balanced`** (highest F1: 0.814)

Test set: 56,962 transactions (98 actual fraud cases)

### Reading the confusion matrix

| | Predicted Legit | Predicted Fraud |
| --- | --- | --- |
| **Actually Legit** | 56,841 (TN) | 23 (FP — false alarms) |
| **Actually Fraud** | 15 (FN — missed fraud) | 83 (TP — caught fraud) |

---

## Key Design Decisions

### Why not use accuracy?

A model predicting "all legitimate" scores 99.83% accuracy but catches 0 fraud. We optimize for **Recall** (catching fraud) and **F1** (balancing precision vs recall).

### Why SMOTE?

The dataset has 492 fraud vs 284,315 legitimate transactions. SMOTE (Synthetic Minority Oversampling Technique) generates synthetic fraud examples in the training set so the model sees a balanced distribution. It is applied **only to training data** — never to the test set, which would contaminate evaluation.

### Why threshold tuning?

XGBoost outputs a probability. The default threshold of 0.5 is rarely optimal for imbalanced problems. We scan the Precision-Recall curve and select the threshold that maximizes F1 while keeping Recall ≥ 0.80. The selected threshold was **0.959** — much higher than default, meaning we only flag a transaction as fraud when the model is highly confident.

### Why SHAP?

Raw fraud predictions ("fraud" / "legitimate") aren't enough for a real system. SHAP (SHapley Additive exPlanations) shows *which features* drove each decision and by how much. This makes the model auditable and helps analysts investigate flagged transactions.

---

## Experiment Tracking

Runs are logged to [Weights & Biases](https://wandb.ai) under project `fraud-detection-api`. Each run logs:

- Hyperparameters (n_estimators, max_depth, learning_rate, subsample, threshold)
- Metrics (precision, recall, F1, ROC-AUC, average precision)
- Confusion matrix (TP, FP, FN, TN)

---

## Environment

| Property | Value |
| --- | --- |
| Python | 3.13.9 |
| OS | Windows 11 |
| GPU | NVIDIA RTX A1000 6GB (Ampere) |
| CUDA | 12.4 |
| RAM | 32 GB |
