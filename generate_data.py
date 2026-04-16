"""
Generate synthetic churn dataset and upload to MinIO.
Run this ONCE locally to seed your MinIO bucket before running the pipeline.

Usage:
  python generate_data.py

Requires MinIO running in minikube:
  kubectl port-forward -n kubeflow svc/minio-service 9000:9000
"""

import pandas as pd
import numpy as np
import boto3
import os

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
MINIO_ENDPOINT  = "http://localhost:9000"   # port-forwarded
MINIO_ACCESS    = "minio"
MINIO_SECRET    = "minio123"
BUCKET          = "mlpipeline"
S3_KEY          = "churn/data/churn_data.csv"
LOCAL_PATH      = "/tmp/churn_data.csv"
N_SAMPLES       = 1000

# ─────────────────────────────────────────────
# Generate
# ─────────────────────────────────────────────
np.random.seed(42)

data = {
    "customer_id":       range(1, N_SAMPLES + 1),
    "age":               np.random.randint(18, 70, N_SAMPLES),
    "tenure_months":     np.random.randint(1, 72, N_SAMPLES),
    "monthly_charges":   np.random.uniform(20, 120, N_SAMPLES),
    "total_charges":     np.random.uniform(100, 8000, N_SAMPLES),
    "num_support_calls": np.random.randint(0, 10, N_SAMPLES),
}

churn_prob = (
        (data["monthly_charges"] / 120) * 0.3 +
        (data["num_support_calls"] / 10) * 0.4 +
        (1 - data["tenure_months"] / 72) * 0.3
)
data["churn"] = (np.random.random(N_SAMPLES) < churn_prob).astype(int)

df = pd.DataFrame(data)
df.to_csv(LOCAL_PATH, index=False)
print(f"Generated {len(df)} rows | Churn rate: {df['churn'].mean():.2%}")

# ─────────────────────────────────────────────
# Upload to MinIO
# ─────────────────────────────────────────────
s3 = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS,
    aws_secret_access_key=MINIO_SECRET,
)

# Create bucket if it doesn't exist
try:
    s3.head_bucket(Bucket=BUCKET)
except Exception:
    s3.create_bucket(Bucket=BUCKET)
    print(f"Created bucket: {BUCKET}")

s3.upload_file(LOCAL_PATH, BUCKET, S3_KEY)
print(f"✅ Uploaded to s3://{BUCKET}/{S3_KEY}")
print("You can now run the Kubeflow pipeline.")