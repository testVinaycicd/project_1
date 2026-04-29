"""
simulate_drift.py
──────────────────
Generates synthetic production prediction requests with configurable drift.
Uploads them to MinIO so drift_detector.py has data to compare against.

WHY THIS EXISTS:
  In production, your inference API logs every request to MinIO.
  In your local minikube, you have no real production traffic.
  This script simulates that traffic — including drift — so you can
  test the full drift detection → alerting → retraining loop.

WHAT IT SIMULATES:
  Three scenarios you can choose from:

  1. "clean"   → Same distribution as training data. Detector should say NO DRIFT.
  2. "drift"   → monthly_charges shifted up 40%, age skewed younger.
                 Detector should say DATA DRIFT DETECTED.
  3. "concept" → Features normal but prediction distribution shifted.
                 Detector should say CONCEPT DRIFT DETECTED.

Usage:
  kubectl port-forward -n kubeflow svc/minio-service 9000:9000 &
  python simulate_drift.py --scenario clean    # no drift
  python simulate_drift.py --scenario drift    # data drift
  python simulate_drift.py --scenario concept  # concept drift
  python simulate_drift.py --scenario drift --n 300  # 300 samples
"""

import argparse
import io
import pickle
import time

import boto3
import numpy as np
import pandas as pd
import yaml
from botocore.exceptions import ClientError


def load_config(path: str = "drift_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_minio_client(cfg: dict):
    storage = cfg["storage"]
    return boto3.client(
        "s3",
        endpoint_url=storage["minio_endpoint"],
        aws_access_key_id=storage["minio_access_key"],
        aws_secret_access_key=storage["minio_secret_key"],
    )


# ─────────────────────────────────────────────────────────────────
# SCENARIO GENERATORS
# ─────────────────────────────────────────────────────────────────
def generate_clean(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Same distribution as training data.
    Drift detector should return: NO DRIFT.
    """
    age            = rng.integers(18, 70, n)
    tenure_months  = rng.integers(1, 72, n)
    monthly_charges = rng.uniform(20, 120, n)
    total_charges  = rng.uniform(100, 8000, n)
    support_calls  = rng.integers(0, 10, n)

    churn_prob = (
            (monthly_charges / 120) * 0.3 +
            (support_calls / 10) * 0.4 +
            (1 - tenure_months / 72) * 0.3
    )
    # Model outputs (simulate RandomForest predictions)
    churn_probability = churn_prob + rng.normal(0, 0.05, n)
    churn_probability = np.clip(churn_probability, 0.01, 0.99)

    return pd.DataFrame({
        "age": age,
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "num_support_calls": support_calls,
        "churn_probability": churn_probability,
        "timestamp": [time.time() + i for i in range(n)],
        "scenario": "clean",
    })


def generate_data_drift(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Shifted input distributions.
    - monthly_charges shifted UP by 40% (price increase scenario)
    - age skewed YOUNGER (new demographic acquiring the product)
    - num_support_calls DOWN (product improved)

    Drift detector should return: DATA DRIFT DETECTED
    KS test should flag: monthly_charges, age, num_support_calls
    """
    print("\n  Injecting data drift:")
    print("    monthly_charges: shifted +40% (price increase)")
    print("    age:             skewed younger (new demographic)")
    print("    num_support_calls: reduced (product improvement)")

    age            = rng.integers(18, 35, n)                   # ← younger
    tenure_months  = rng.integers(1, 72, n)                    # ← same
    monthly_charges = rng.uniform(60, 168, n)                  # ← +40% shift
    total_charges  = rng.uniform(100, 8000, n)                 # ← same
    support_calls  = rng.integers(0, 3, n)                     # ← reduced

    churn_prob = (
            (monthly_charges / 168) * 0.3 +
            (support_calls / 3) * 0.4 +
            (1 - tenure_months / 72) * 0.3
    )
    churn_probability = churn_prob + rng.normal(0, 0.05, n)
    churn_probability = np.clip(churn_probability, 0.01, 0.99)

    return pd.DataFrame({
        "age": age,
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "num_support_calls": support_calls,
        "churn_probability": churn_probability,
        "timestamp": [time.time() + i for i in range(n)],
        "scenario": "data_drift",
    })


def generate_concept_drift(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Features look similar to training, but the model's predictions
    have shifted — it's now predicting much higher churn for everyone.

    This simulates: the model's learned relationship is no longer valid.
    E.g., it learned "high monthly_charges = churn" but now high charges
    just mean premium customers who are very loyal.

    Drift detector should return: CONCEPT DRIFT DETECTED
    (prediction_shift above threshold, but feature distributions ok)
    """
    print("\n  Injecting concept drift:")
    print("    Features: same distribution as training")
    print("    Predictions: mean shifted from ~0.35 to ~0.72")
    print("    (model is now predicting high churn for everyone)")

    age            = rng.integers(18, 70, n)
    tenure_months  = rng.integers(1, 72, n)
    monthly_charges = rng.uniform(20, 120, n)
    total_charges  = rng.uniform(100, 8000, n)
    support_calls  = rng.integers(0, 10, n)

    # Features are normal — but predictions are shifted high
    # This simulates model degradation / wrong world model
    churn_probability = rng.beta(a=5, b=2, size=n)  # ← mean ~0.71

    return pd.DataFrame({
        "age": age,
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "num_support_calls": support_calls,
        "churn_probability": churn_probability,
        "timestamp": [time.time() + i for i in range(n)],
        "scenario": "concept_drift",
    })


# ─────────────────────────────────────────────────────────────────
# UPLOAD TO MINIO
# ─────────────────────────────────────────────────────────────────
def upload_production_log(df: pd.DataFrame, cfg: dict, s3, append: bool = True):
    """
    Upload (or append) the production log to MinIO.
    drift_detector.py reads this file on each check.
    """
    storage = cfg["storage"]
    bucket = storage["bucket"]
    key = storage["production_log_key"]

    if append:
        # Try to load existing log and append
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            existing = pd.read_parquet(io.BytesIO(obj["Body"].read()))
            combined = pd.concat([existing, df], ignore_index=True)
            # Keep last 1000 rows
            combined = combined.tail(1000)
            print(f"  Appended {len(df)} rows → {len(combined)} total in log")
            df = combined
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                print(f"  No existing log — creating new one with {len(df)} rows")
            else:
                raise

    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    print(f"  Uploaded → s3://{bucket}/{key}")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Simulate production drift")
    parser.add_argument(
        "--scenario",
        choices=["clean", "drift", "concept"],
        default="drift",
        help=(
            "clean   → no drift (detector should say NORMAL)\n"
            "drift   → data drift (features shifted)\n"
            "concept → concept drift (predictions shifted)"
        ),
    )
    parser.add_argument("--n", type=int, default=200,
                        help="Number of production samples to generate")
    parser.add_argument("--no-append", action="store_true",
                        help="Replace existing log instead of appending")
    parser.add_argument("--config", default="drift_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    s3  = get_minio_client(cfg)
    rng = np.random.default_rng(seed=42)

    print(f"\n{'─' * 55}")
    print(f"  DRIFT SIMULATOR")
    print(f"  Scenario : {args.scenario}")
    print(f"  Samples  : {args.n}")
    print(f"{'─' * 55}")

    # Generate scenario data
    generators = {
        "clean":   generate_clean,
        "drift":   generate_data_drift,
        "concept": generate_concept_drift,
    }
    df = generators[args.scenario](args.n, rng)

    print(f"\n  Generated {len(df)} rows")
    print(f"  churn_probability: mean={df['churn_probability'].mean():.3f}  "
          f"std={df['churn_probability'].std():.3f}")
    for col in ["age", "monthly_charges", "tenure_months",
                "num_support_calls", "total_charges"]:
        print(f"    {col:<22}: mean={df[col].mean():.2f}")

    upload_production_log(df, cfg, s3, append=not args.no_append)

    print(f"\n  Done. Run drift_detector.py to check for drift.")
    print(f"{'─' * 55}\n")


if __name__ == "__main__":
    main()