import joblib
import shap

FEATURE_NAMES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

print("Loading model...")
model     = joblib.load("fraud_model.joblib")
THRESHOLD = joblib.load("threshold.joblib")
explainer = shap.TreeExplainer(model)
print(f"Model loaded. Threshold: {THRESHOLD:.3f}")
