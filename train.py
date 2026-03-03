import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import wandb

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    precision_recall_curve, confusion_matrix
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# ── 1. Load data ──────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv('data/creditcard.csv')
X = df.drop('Class', axis=1)
y = df['Class']

print(f"Loaded: {X.shape[0]:,} rows, fraud rate: {y.mean():.3%}")

# ── 2. Train/test split — stratify keeps fraud rate equal in both ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
print(f"Train fraud rate: {y_train.mean():.3%} | Test fraud rate: {y_test.mean():.3%}")

# ── 3. Apply SMOTE to training set ONLY ───────────────────────────
# NEVER apply SMOTE to test set — that would contaminate your evaluation
print("\nApplying SMOTE to training set...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f"After SMOTE — Train: {len(X_train_resampled):,} rows")
print(f"Fraud in resampled train: {y_train_resampled.mean():.1%} (was {y_train.mean():.3%})")

# ── 4. Find optimal threshold using Precision-Recall curve ────────
# We do this ONCE on the test set with a baseline model
# to understand the tradeoff before training all 3 variants
print("\nFinding optimal threshold...")
baseline = xgb.XGBClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.1,
    random_state=42, eval_metric='logloss', verbosity=0
)
baseline.fit(X_train_resampled, y_train_resampled)
y_proba_baseline = baseline.predict_proba(X_test)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba_baseline)

# Plot so you can SEE the tradeoff
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(thresholds, precisions[:-1], label='Precision', color='steelblue')
plt.plot(thresholds, recalls[:-1], label='Recall', color='crimson')
plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Default threshold')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision vs Recall by Threshold')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(recalls[:-1], precisions[:-1], color='purple')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('precision_recall_curve.png', dpi=150, bbox_inches='tight')
print("✅ Saved: precision_recall_curve.png")

# Choose threshold: find where Recall >= 0.80 with best F1
# For fraud detection we prioritize Recall over Precision
f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
high_recall_mask = recalls[:-1] >= 0.80
if high_recall_mask.any():
    best_idx = np.argmax(f1_scores * high_recall_mask)
    THRESHOLD = float(thresholds[best_idx])
else:
    THRESHOLD = 0.3  # fallback

print(f"Selected threshold: {THRESHOLD:.3f}")
print(f"At this threshold — Precision: {precisions[best_idx]:.3f} | Recall: {recalls[best_idx]:.3f}")

# ── 5. Train 3 variants and log to W&B ───────────────────────────
variants = [
    {
        'name': 'xgb-shallow-fast',
        'n_estimators': 100,
        'max_depth': 3,
        'learning_rate': 0.1,
        'subsample': 0.8,
    },
    {
        'name': 'xgb-medium-balanced',
        'n_estimators': 200,
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.8,
    },
    {
        'name': 'xgb-deep-slow',
        'n_estimators': 300,
        'max_depth': 5,
        'learning_rate': 0.02,
        'subsample': 0.9,
    },
]

best_f1 = -1
best_model = None
best_run_name = None

for v in variants:
    run_name = v.pop('name')
    print(f"\nTraining: {run_name}...")

    # Initialize W&B run
    run = wandb.init(
        project="fraud-detection-api",
        name=run_name,
        config={
            **v,
            'threshold': THRESHOLD,
            'smote': True,
            'train_size': len(X_train_resampled),
            'test_size': len(X_test),
        }
    )

    # Train
    model = xgb.XGBClassifier(
        **v,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )
    model.fit(X_train_resampled, y_train_resampled)

    # Save each variant individually for A/B testing
    joblib.dump(model, f'fraud_model_{run_name}.joblib')
    print(f"  Saved: fraud_model_{run_name}.joblib")

    # Evaluate at tuned threshold
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= THRESHOLD).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_test, y_proba)
    avg_prec  = average_precision_score(y_test, y_proba)
    cm        = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    # Log all metrics to W&B
    wandb.log({
        'precision':          precision,
        'recall':             recall,
        'f1':                 f1,
        'roc_auc':            roc_auc,
        'average_precision':  avg_prec,
        'true_positives':     int(tp),
        'false_positives':    int(fp),
        'false_negatives':    int(fn),
        'true_negatives':     int(tn),
        'threshold':          THRESHOLD,
    })

    # Log confusion matrix as W&B table
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_test.tolist(),
            preds=y_pred.tolist(),
            class_names=["Legitimate", "Fraud"]
        )
    })

    print(f"  Precision: {precision:.3f} | Recall: {recall:.3f} | "
          f"F1: {f1:.3f} | ROC-AUC: {roc_auc:.4f}")
    print(f"  TP: {tp} (fraud caught) | FN: {fn} (fraud missed) | FP: {fp} (false alarms)")

    # Track best model by F1
    if f1 > best_f1:
        best_f1       = f1
        best_model    = model
        best_run_name = run_name

    wandb.finish()

# ── 6. Save best model to disk ────────────────────────────────────
print(f"\nBest model: {best_run_name} (F1: {best_f1:.3f})")
joblib.dump(best_model, 'fraud_model.joblib')
joblib.dump(THRESHOLD, 'threshold.joblib')
print("✅ Saved: fraud_model.joblib")
print("✅ Saved: threshold.joblib")
print(f"\nView your W&B dashboard at:")
print(f"https://wandb.ai/[your-username]/fraud-detection-api")