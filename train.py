"""
Churn Model Training Script
────────────────────────────
Called by train_op in the KFP pipeline.
Usage: python train.py <input_csv_path> <model_output_path> <metrics_output_path>
"""

import sys
import json
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# ─────────────────────────────────────────────
# Args — injected by KFP component (artifact paths)
# ─────────────────────────────────────────────
input_path  = sys.argv[1]   # KFP Dataset artifact path
model_path  = sys.argv[2]   # KFP Model artifact path
metrics_path = sys.argv[3]  # KFP Metrics artifact path

print(f"📥 Input  : {input_path}")
print(f"📤 Model  : {model_path}")
print(f"📊 Metrics: {metrics_path}")

# ─────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────
df = pd.read_csv(input_path)
print(f"Loaded {len(df)} rows | Churn rate: {df['churn'].mean():.2%}")

FEATURES = ['age', 'tenure_months', 'monthly_charges',
            'total_charges', 'num_support_calls']
X = df[FEATURES]
y = df['churn']

# ─────────────────────────────────────────────
# Split
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,           # Prevents overfitting on small data
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

# ─────────────────────────────────────────────
# Evaluate
# ─────────────────────────────────────────────
y_pred  = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc      = roc_auc_score(y_test, y_proba)

print(f"✅ Accuracy : {accuracy:.4f}")
print(f"✅ AUC      : {auc:.4f}")

# ─────────────────────────────────────────────
# Save model — KFP artifact path
# ─────────────────────────────────────────────
with open(model_path, "wb") as f:
    pickle.dump(clf, f)
print(f"💾 Model saved to {model_path}")

# ─────────────────────────────────────────────
# Save metrics — KFP Metrics format
# ─────────────────────────────────────────────
metrics = {
    "metrics": [
        {"name": "accuracy", "numberValue": float(accuracy), "format": "RAW"},
        {"name": "auc",      "numberValue": float(auc),      "format": "RAW"},
    ]
}
with open(metrics_path, "w") as f:
    json.dump(metrics, f)
print(f"📊 Metrics saved to {metrics_path}")