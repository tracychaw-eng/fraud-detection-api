import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml

# ── 1. Load dataset ──────────────────────────────────────────────
df = pd.read_csv('data/creditcard.csv')
X = df.drop('Class', axis=1)
y = df['Class']


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Amount distribution
axes[0].hist(df[df['Class']==0]['Amount'], bins=50, alpha=0.6,
             label='Legitimate', color='steelblue', density=True)
axes[0].hist(df[df['Class']==1]['Amount'], bins=50, alpha=0.6,
             label='Fraud', color='crimson', density=True)
axes[0].set_title('Transaction Amount: Fraud vs Legitimate')
axes[0].set_xlabel('Amount ($)')
axes[0].set_ylabel('Density')
axes[0].legend()
axes[0].set_xlim(0, 500)  # zoom in — most transactions are under $500

# Class imbalance bar
class_counts = y.value_counts()
bars = axes[1].bar(['Legitimate', 'Fraud'],
                   [class_counts[0], class_counts[1]],
                   color=['steelblue', 'crimson'], alpha=0.8)
axes[1].set_title('Class Imbalance (log scale)')
axes[1].set_ylabel('Number of transactions')
axes[1].set_yscale('log')

for bar, count in zip(bars, [class_counts[0], class_counts[1]]):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() * 1.1,
                 f'{count:,}', ha='center', fontsize=11)

plt.tight_layout()
plt.savefig('class_imbalance.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved: class_imbalance.png")
plt.show()

# ── 4. Feature stats ─────────────────────────────────────────────
stat_cols = [c for c in ['Amount', 'Time'] if c in df.columns]
print("\nFraud transaction stats:")
print(df[df['Class']==1][stat_cols].describe().round(2))

print("\nLegit transaction stats:")
print(df[df['Class']==0][stat_cols].describe().round(2))

# ── 5. Correlation: which V features differ most between classes ──
fraud_means    = df[df['Class']==1].mean()
legit_means    = df[df['Class']==0].mean()
feature_diff   = (fraud_means - legit_means).abs()
top_features   = feature_diff.drop('Class').nlargest(10)

print(f"\nTop 10 features most different between fraud and legit:")
print(top_features.round(4))

print("\n✅ EDA complete. Key insight:")
print(f"   Only {y.mean():.3%} of transactions are fraud.")
print("   A naive 'predict all legit' model gets 99.83% accuracy but catches NOTHING.")
print("   We need SMOTE + threshold tuning + Recall optimization.")