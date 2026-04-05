"""
train_model.py
==============
Run this ONCE before starting app.py

Steps:
  1. Loads ClassSurvey.csv
  2. Preprocesses data
  3. Trains Logistic Regression
  4. Evaluates with accuracy, confusion matrix, ROC-AUC
  5. Saves model.pkl for Flask to use
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, f1_score, precision_score, recall_score
)

np.random.seed(123)
print("=" * 60)
print("  SOCIAL MEDIA ADDICTION - MODEL TRAINING")
print("=" * 60)


print("\n[1] Loading dataset...")
df = pd.read_csv("ClassSurvey.csv")
print(f"    Rows: {len(df)}  |  Columns: {len(df.columns)}")
print(f"    Columns: {list(df.columns)}")

print("\n[2] Preparing features...")

DROP_COLS = [
    'Name', 'Age',
    'TotalSocialMediaScreenTime',
    'Number.of.times.opened..hourly.intervals.',
    'SocialMediaAddiction'
]

FEATURE_COLS = [c for c in df.columns if c not in DROP_COLS]
TARGET_COL   = 'SocialMediaAddiction'

print(f"    Feature columns: {FEATURE_COLS}")

X = df[FEATURE_COLS].copy()
y = df[TARGET_COL].copy()


y_encoded = (y == 'Addicted').astype(int)

X = X.fillna(X.mean())

print(f"\n    Target distribution:")
for label, cnt in y.value_counts().items():
    print(f"      {label}: {cnt} students ({cnt/len(y)*100:.1f}%)")

#3. Split and scale data
print("\n[3] Splitting & scaling data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=123, stratify=y_encoded
)

scaler  = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"    Train: {len(X_train)}  |  Test: {len(X_test)}")

# 4. Train model 
print("\n[4] Training Logistic Regression...")
model = LogisticRegression(max_iter=1000, random_state=123, C=1.0)
model.fit(X_train_s, y_train)
print("    ✓ Model trained")

# 5. Evaluate model 
print("\n[5] Evaluating model...")

y_pred       = model.predict(X_test_s)
y_pred_proba = model.predict_proba(X_test_s)[:, 1]

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_pred_proba)

# Cross-validation
cv_scores = cross_val_score(model, scaler.transform(X), y_encoded, cv=5, scoring='accuracy')

print("\n" + "─" * 40)
print("  EVALUATION METRICS")
print("─" * 40)
print(f"  Accuracy       : {accuracy*100:.2f}%")
print(f"  Precision      : {precision*100:.2f}%")
print(f"  Recall         : {recall*100:.2f}%")
print(f"  F1 Score       : {f1:.4f}")
print(f"  ROC-AUC        : {roc_auc:.4f}")
print(f"  CV Accuracy    : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
print("─" * 40)

print("\n  Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"  {'':20} Predicted")
print(f"  {'':20} Not Add.  Addicted")
labels_cm = ['Not Addicted', 'Addicted']
for i, row_label in enumerate(labels_cm):
    print(f"  Actual {row_label:14} {cm[i][0]:6}    {cm[i][1]:6}")

print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Addicted', 'Addicted']))

# ── 6. Feature importance ──────────────────────────────────────
print("\n  Feature Importance (Coefficients):")
coef_df = pd.DataFrame({
    'Feature': FEATURE_COLS,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient', ascending=False)

for _, row in coef_df.iterrows():
    bar = "█" * int(abs(row['Coefficient']) * 5)
    sign = "+" if row['Coefficient'] > 0 else "-"
    print(f"  {row['Feature']:15} {sign}{abs(row['Coefficient']):.3f}  {bar}")

# ── 7. Save plots ─────────────────────────────────────────────
print("\n[6] Saving evaluation plots...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Logistic Regression - Evaluation', fontsize=14, fontweight='bold')

# Confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Not Addicted', 'Addicted'],
            yticklabels=['Not Addicted', 'Addicted'])
axes[0].set_title('Confusion Matrix')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[1].plot(fpr, tpr, color='blue', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
axes[1].plot([0,1],[0,1],'--', color='gray', label='Random')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('evaluation.png', dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ evaluation.png saved")

# ── 8. Save model ─────────────────────────────────────────────
print("\n[7] Saving model...")
payload = {
    "model":    model,
    "scaler":   scaler,
    "features": FEATURE_COLS,
    "accuracy": round(accuracy * 100, 2),
    "roc_auc":  round(roc_auc, 4),
    "f1":       round(f1, 4)
}
with open("model.pkl", "wb") as f:
    pickle.dump(payload, f)

print("    ✓ model.pkl saved")
print("\n" + "=" * 60)
print(f"  MODEL READY  |  Accuracy: {accuracy*100:.2f}%  |  AUC: {roc_auc:.4f}")
print("=" * 60)
print("\n  Now run:  python app.py")
print("  Then open: http://localhost:5000\n")