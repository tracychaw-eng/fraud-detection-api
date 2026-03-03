"""
generate_shap_plots.py — Offline SHAP plot generation for reports and README.

Usage:
    python generate_shap_plots.py

Outputs:
    shap_beeswarm.png   — global feature importance (500 background samples)
    shap_bar.png        — mean |SHAP| per feature
    shap_waterfall.png  — per-transaction waterfall for one known fraud case
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import joblib
import pandas as pd

FEATURE_NAMES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

print("Loading model...")
model     = joblib.load("fraud_model.joblib")
explainer = shap.TreeExplainer(model)

print("Sampling background data...")
df = pd.read_csv("data/creditcard.csv")
bg_df    = df.drop(columns=["Class"]).sample(n=500, random_state=42)[FEATURE_NAMES]
shap_exp = explainer(bg_df)

# ── 1. Beeswarm ───────────────────────────────────────────────────
print("Generating shap_beeswarm.png...")
shap.plots.beeswarm(shap_exp, max_display=20, show=False)
plt.savefig("shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: shap_beeswarm.png")

# ── 2. Bar ────────────────────────────────────────────────────────
print("Generating shap_bar.png...")
shap.plots.bar(shap_exp, max_display=20, show=False)
plt.savefig("shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: shap_bar.png")

# ── 3. Waterfall (first known fraud case) ─────────────────────────
print("Generating shap_waterfall.png...")
fraud_row = df[df["Class"] == 1][FEATURE_NAMES].iloc[[0]]
fraud_exp = explainer(fraud_row)
shap.plots.waterfall(fraud_exp[0], max_display=15, show=False)
plt.savefig("shap_waterfall.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: shap_waterfall.png")

print("\nDone. All 3 plots saved.")
