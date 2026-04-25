"""
Churn Prediction — Kubeflow Pipeline (KFP v2)
Updated for Day 4: adds save_reference_data_op after training.

NEW COMPONENT — save_reference_data_op:
  After training, this component:
    1. Loads the training data from the KFP artifact
    2. Runs the trained model on the test split to get predictions
    3. Saves training data + predictions to MinIO as a parquet file

  This parquet file is the REFERENCE DATASET for drift detection.
  drift_detector.py compares production requests against this.

  Why parquet not CSV?
    Parquet preserves dtypes (int vs float vs bool).
    Much faster to read/write for large datasets.
    Standard format in MLops pipelines.
"""

import kfp
from kfp import dsl
from kfp.dsl import component, Output, Input, Dataset, Model, Metrics

BASE_IMAGE = "crysis307/churn-train:v14"


@component(base_image=BASE_IMAGE)
def generate_data_op(data: Output[Dataset]):
    import boto3, shutil
    LOCAL = "/tmp/churn_data.csv"
    s3 = boto3.client(
        "s3",
        endpoint_url="http://minio-service.kubeflow:9000",
        aws_access_key_id="minio",
        aws_secret_access_key="minio123",
    )
    s3.download_file("mlpipeline", "churn/data/churn_data.csv", LOCAL)
    shutil.copy(LOCAL, data.path)
    print(f"Data written to artifact: {data.path}")


@component(base_image=BASE_IMAGE)
def validate_data_op(data: Input[Dataset]) -> bool:
    import pandas as pd
    df = pd.read_csv(data.path)
    required = ['age', 'tenure_months', 'monthly_charges',
                'total_charges', 'num_support_calls', 'churn']
    if len(df) < 100: return False
    if df.isnull().sum().sum() > 0: return False
    for col in required:
        if col not in df.columns: return False
    print(f"Validated: {df.shape}")
    return True


@component(base_image=BASE_IMAGE)
def train_op(
        data: Input[Dataset],
        model: Output[Model],
        metrics: Output[Metrics],
):
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "/app/training/train.py",
         data.path, model.path, metrics.path],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"Training failed:\n{result.stderr}")


@component
def evaluate_op(metrics: Input[Metrics]) -> float:
    import json
    with open(metrics.path) as f:
        m = json.load(f)
    for metric in m["metrics"]:
        if metric["name"] == "auc":
            print(f"AUC: {metric['numberValue']:.4f}")
            return float(metric["numberValue"])
    raise ValueError("AUC not found")


@component(base_image=BASE_IMAGE)
def get_production_metric_op() -> float:
    import boto3, json
    from botocore.exceptions import ClientError
    FALLBACK = 0.60
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url="http://minio-service.kubeflow:9000",
            aws_access_key_id="minio",
            aws_secret_access_key="minio123",
        )
        obj = s3.get_object(Bucket="mlpipeline", Key="churn/production/metrics.json")
        m = json.loads(obj["Body"].read())
        for metric in m["metrics"]:
            if metric["name"] == "auc":
                return float(metric["numberValue"])
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return FALLBACK
        raise
    return FALLBACK


@component(base_image=BASE_IMAGE)
def promote_model_op(model: Input[Model], metrics: Input[Metrics]):
    import boto3
    s3 = boto3.client(
        "s3",
        endpoint_url="http://minio-service.kubeflow:9000",
        aws_access_key_id="minio",
        aws_secret_access_key="minio123",
    )
    s3.upload_file(model.path, "mlpipeline", "churn/production/model.pkl")
    s3.upload_file(metrics.path, "mlpipeline", "churn/production/metrics.json")
    print("Model and metrics promoted to production path")


# ─────────────────────────────────────────────────────────────────
# NEW — Day 4: save reference dataset for drift detection
# ─────────────────────────────────────────────────────────────────
@component(base_image=BASE_IMAGE)
def save_reference_data_op(data: Input[Dataset], model: Input[Model]):
    """
    Save the training data + model predictions to MinIO as parquet.
    This becomes the REFERENCE DATASET for drift_detector.py.

    Why we save predictions alongside features:
      Drift detection needs both to detect concept drift.
      Without predictions in the reference, we can only detect data drift.
      With predictions, we can also detect when the model's output
      distribution shifts (concept drift proxy).

    What gets saved:
      All input features + churn_probability (model's prediction on
      the training set). The test split is used, not the train split,
      because the model hasn't seen it — more representative of
      what production predictions will look like.
    """
    import io
    import pickle
    import boto3
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # Load training data
    df = pd.read_csv(data.path)
    FEATURES = ['age', 'tenure_months', 'monthly_charges',
                'total_charges', 'num_support_calls']

    X = df[FEATURES]
    y = df['churn']

    # Same split as train.py — same random_state → same test set
    _, X_test, _, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Load model and predict on test set
    with open(model.path, 'rb') as f:
        clf = pickle.load(f)

    test_proba = clf.predict_proba(X_test)[:, 1]

    # Build reference DataFrame: features + predictions
    reference_df = X_test.copy().reset_index(drop=True)
    reference_df['churn_probability'] = test_proba
    reference_df['split'] = 'test'

    print(f"Reference dataset: {reference_df.shape}")
    print(f"  churn_probability mean: {test_proba.mean():.4f}")
    print(f"  churn_probability std:  {test_proba.std():.4f}")

    # Upload to MinIO as parquet
    s3 = boto3.client(
        "s3",
        endpoint_url="http://minio-service.kubeflow:9000",
        aws_access_key_id="minio",
        aws_secret_access_key="minio123",
    )

    buf = io.BytesIO()
    reference_df.to_parquet(buf, index=False)
    buf.seek(0)

    s3.put_object(
        Bucket="mlpipeline",
        Key="churn/drift/reference_data.parquet",
        Body=buf.getvalue(),
    )
    print("Reference data saved → s3://mlpipeline/churn/drift/reference_data.parquet")
    print("drift_detector.py will use this as the baseline for future drift checks.")


# ─────────────────────────────────────────────────────────────────
# PIPELINE DEFINITION
# ─────────────────────────────────────────────────────────────────
@dsl.pipeline(
    name="churn-train-pipeline",
    description="Generate → Validate → Train → Evaluate → Promote → Save Reference"
)
def churn_pipeline():

    data_task  = generate_data_op()
    data_task.set_caching_options(False)

    valid_task = validate_data_op(data=data_task.outputs["data"])

    with dsl.If(valid_task.output == True, name="data-is-valid"):

        train_task = train_op(data=data_task.outputs["data"])
        auc_task   = evaluate_op(metrics=train_task.outputs["metrics"])
        prod_task  = get_production_metric_op()

        with dsl.If(auc_task.output > prod_task.output, name="new-model-is-better"):
            promote_model_op(
                model=train_task.outputs["model"],
                metrics=train_task.outputs["metrics"],
            )

        # Save reference data ALWAYS after training (not gated by AUC comparison)
        # Even if the model isn't promoted, we want the reference updated
        # so drift detector has the latest baseline.
        save_reference_data_op(
            data=data_task.outputs["data"],
            model=train_task.outputs["model"],
        )


def run_pipeline():
    client = kfp.Client(host="http://127.0.0.1:8080")
    run = client.create_run_from_pipeline_func(
        churn_pipeline,
        arguments={},
        run_name="churn-pipeline-with-drift-reference",
        enable_caching=False,
    )
    print(f"Pipeline submitted — Run ID: {run.run_id}")
    print("Watch at: http://127.0.0.1:8080")


if __name__ == "__main__":
    run_pipeline()