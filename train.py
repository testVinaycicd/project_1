import boto3
import pandas as pd
import pickle
import sys
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

input_path = sys.argv[1]
model_path = sys.argv[2]
metrics_path = sys.argv[3]

# Handle S3 (optional)
if input_path.startswith("s3://"):
    s3 = boto3.client("s3")
    bucket = input_path.split("/")[2]
    key = "/".join(input_path.split("/")[3:])

    local_file = "/tmp/data.csv"
    s3.download_file(bucket, key, local_file)
    input_path = local_file

# Load data
df = pd.read_csv(input_path)

features = ['age', 'tenure_months', 'monthly_charges', 'total_charges', 'num_support_calls']
X = df[features]
y = df['churn']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

# Save model
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

# ✅ Correct KFP metrics format
metrics = {
    "metrics": [
        {
            "name": "accuracy",
            "numberValue": float(accuracy),
            "format": "RAW"
        },
        {
            "name": "auc",
            "numberValue": float(auc),
            "format": "RAW"
        }
    ]
}

with open(metrics_path, "w") as f:
    json.dump(metrics, f)