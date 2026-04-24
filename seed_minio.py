"""
seed_minio.py
──────────────
Run this ONCE from your laptop before submitting the pipeline.
It generates the CSV and uploads it to MinIO so the pipeline
can pull it in generate_data_op.

Prerequisite:
  kubectl port-forward -n kubeflow svc/minio-service 9000:9000

Usage:
  python seed_minio.py
"""

import io
import numpy as np
import pandas as pd
import boto3
from botocore.exceptions import ClientError

# ── Config ────────────────────────────────────────────────────────
MINIO_ENDPOINT = "http://localhost:9000"   # port-forwarded
MINIO_ACCESS   = "minio"
MINIO_SECRET   = "minio123"
BUCKET         = "mlpipeline"
S3_KEY         = "churn/data/churn_data.csv"
N              = 1000

# ── Generate ──────────────────────────────────────────────────────
print("Generating synthetic churn dataset...")
np.random.seed(42)

data = {
    "customer_id":       range(1, N + 1),
    "age":               np.random.randint(18, 70, N),
    "tenure_months":     np.random.randint(1, 72, N),
    "monthly_charges":   np.random.uniform(20, 120, N),
    "total_charges":     np.random.uniform(100, 8000, N),
    "num_support_calls": np.random.randint(0, 10, N),
}
churn_prob = (
    (data["monthly_charges"] / 120) * 0.3 +
    (data["num_support_calls"] / 10) * 0.4 +
    (1 - data["tenure_months"] / 72)  * 0.3
)
data["churn"] = (np.random.random(N) < churn_prob).astype(int)
df = pd.DataFrame(data)
print(f"  Rows: {len(df)}  |  Churn rate: {df['churn'].mean():.2%}")

# ── Connect to MinIO ──────────────────────────────────────────────
print(f"\nConnecting to MinIO at {MINIO_ENDPOINT}...")
s3 = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS,
    aws_secret_access_key=MINIO_SECRET,
)

# Create bucket if missing
try:
    s3.head_bucket(Bucket=BUCKET)
    print(f"  Bucket '{BUCKET}' exists")
except ClientError:
    s3.create_bucket(Bucket=BUCKET)
    print(f"  Created bucket '{BUCKET}'")

# ── Upload ────────────────────────────────────────────────────────
buf = io.BytesIO()
df.to_csv(buf, index=False)
buf.seek(0)
s3.put_object(Bucket=BUCKET, Key=S3_KEY, Body=buf.getvalue())

print(f"\nUploaded → s3://{BUCKET}/{S3_KEY}")
print("You can now run the Kubeflow pipeline.")