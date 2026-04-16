# import kfp
# from kfp import dsl
# from kfp.dsl import component, Output, Dataset, Model, Metrics , Input
#
#
#
# @component(base_image="crysis307/churn-train:v9")
# def generate_data_op(data: Output[Dataset]):
#     import subprocess
#     import shutil
#     import boto3
#     s3 = boto3.client("s3")
#
#     print("ENDPOINT:", s3.meta.endpoint_url)
#
#     response = s3.list_objects_v2(Bucket="mlops-demo-dvc-mikey")
#
#     print("RESPONSE:################################################", response)
#     # Pull data from S3 via DVC
#     subprocess.run(["dvc", "pull"], check=True)
#     # Copy dataset to KFP artifact path
#     # shutil.copy("data/churn_data.csv", data.path)
#
#
#
#
# @component(base_image="crysis307/churn-train:v9")
# def validate_data_op(data: Input[Dataset]) -> bool:
#     import pandas as pd
#
#     df = pd.read_csv(data.path)
#
#     # simple checks
#     print(df.head())
#     print(df.shape)
#     print(df.isnull().sum())
#     if len(df) < 100:
#         return False
#
#     if df.isnull().sum().sum() > 0:
#         return False
#
#     return True
#
#
#
#
#
# @component(base_image="crysis307/churn-train:v9")
# def train_op(
#         data: Input[Dataset],
#         model: Output[Model],
#         metrics: Output[Metrics],
# ):
#     import subprocess
#
#     subprocess.run([
#         "python", "train.py",
#         data.path,
#         model.path,
#         metrics.path
#     ], check=True)
#
#
# @component
# def evaluate_op(metrics: Input[Metrics]) -> float:
#     import json
#
#     with open(metrics.path) as f:
#         m = json.load(f)
#
#     for metric in m["metrics"]:
#         if metric["name"] == "auc":
#             print("AUC====================================",metric["numberValue"])
#             return metric["numberValue"]
#
#
#
#
# @component
# def get_production_metric_op() -> float:
#     # load stored metric from previous production model
#     return 0.60  # placeholder
#     # need to write code to get previous model metrics from s3 before that implement s3 in the current project
#
#
# @component(base_image="crysis307/churn-train:v9")
# def promote_model_op(model: Input[Model]):
#
#     import boto3
#
#     s3 = boto3.client(
#         "s3",
#         endpoint_url="http://minio-service.kubeflow:9000",
#         aws_access_key_id="minio",
#         aws_secret_access_key="minio123",
#     )
#
#     bucket = "mlpipeline"  # Kubeflow default bucket
#
#     key = "churn/production/model.pkl"
#
#     s3.upload_file(model.path, bucket, key)
#
#     print(f"Uploaded to MinIO: s3://{bucket}/{key}")
#
#
#
#
# @dsl.pipeline(name="churn-train-pipeline")
# def churn_pipeline():
#
#
#
#     data_task = generate_data_op()
#     data_task.set_caching_options(True)
#     data_task.set_env_variable("AWS_ACCESS_KEY_ID", "")
#     data_task.set_env_variable("AWS_SECRET_ACCESS_KEY", "")
#     # data_task.set_service_account("dvc-access-sa")
#     valid = validate_data_op(data=data_task.outputs["data"])
#
#     # with dsl.If(valid.output == True):
#     #
#     #     train_task = train_op(data=data_task.outputs["data"])
#     #     auc = evaluate_op(metrics=train_task.outputs["metrics"])
#     #     prod_score = get_production_metric_op()
#     #
#     #     with dsl.If(auc.output > prod_score.output):
#     #         promote_model_op(model=train_task.outputs["model"])
#     #
#     #         print("Model Ready to Deploy")
#
#
#
# def run_pipeline():
#     client = kfp.Client(host="http://127.0.0.1:8080")
#
#     run = client.create_run_from_pipeline_func(
#         churn_pipeline,
#         arguments={}
#     )
#
#     print("Run submitted:", run)
#
#
# if __name__ == "__main__":
#     run_pipeline()
#
#
#
#
#
# #     1) Produces output
# # def generate_data_op(data: Output[Dataset])
# #
# # ✔ Produces:
# #
# # data
# # def train_op(data: Input[Dataset], model: Output[Model], metrics: Output[Metrics])
# #
# # ✔ Produces:
# #
# # model
# # metrics
# # 2) Produces return value (not artifact)
# # def validate_data_op(data: Input[Dataset]) -> bool
# #
# # ✔ Produces:
# #
# # a parameter output (boolean)
# #
# # NOT stored as file/artifact
# #
# # def evaluate_op(metrics: Input[Metrics]) -> float
# #
# # ✔ Produces:
# #
# # parameter output (float)
# # def get_production_metric_op() -> float
# #
# # ✔ Produces:
# #
# # parameter output
# # 3) Produces nothing
# # def promote_model_op(model: Input[Model])
# #
# # ❌ No outputs

"""
Churn Prediction - Kubeflow Pipeline (KFP v2)

Flow:
  generate_data → validate_data → train → evaluate
                                        ↓ (if AUC > prod threshold)
                                    promote_model
"""

import kfp
from kfp import dsl
from kfp.dsl import component, Output, Input, Dataset, Model, Metrics

# ─────────────────────────────────────────────
# IMPORTANT: Never hardcode secrets here.
# Inject AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
# via a Kubernetes Secret and reference it with
# set_env_variable_from_secret() in production.
# For local minikube dev, we use MinIO (no AWS needed).
# ─────────────────────────────────────────────

BASE_IMAGE = "crysis307/churn-train:v9"
MINIO_ENDPOINT = "http://minio-service.kubeflow:9000"
MINIO_ACCESS_KEY = "minio"
MINIO_SECRET_KEY = "minio123"
MINIO_BUCKET = "mlpipeline"
PROD_MODEL_KEY = "churn/production/model.pkl"
PROD_METRICS_KEY = "churn/production/metrics.json"


# ─────────────────────────────────────────────
# COMPONENT 1 — Generate / Pull Data
# Bug fixed: shutil.copy() was commented out so
# data.path was never written → crash in step 2.
# ─────────────────────────────────────────────
@component(base_image=BASE_IMAGE)
def generate_data_op(data: Output[Dataset]):
    """
    Pull churn_data.csv from MinIO and write it to the
    KFP artifact path so downstream components can read it.
    """
    import boto3
    import shutil
    import os

    LOCAL_FILE = "/tmp/churn_data.csv"

    s3 = boto3.client(
        "s3",
        endpoint_url="http://minio-service.kubeflow:9000",
        aws_access_key_id="minio",
        aws_secret_access_key="minio123",
    )

    print("⬇️  Downloading churn_data.csv from MinIO...")
    s3.download_file("mlpipeline", "churn/data/churn_data.csv", LOCAL_FILE)
    print(f"✅ Downloaded to {LOCAL_FILE}")

    # THIS is what was missing/commented out — write to KFP artifact path
    shutil.copy(LOCAL_FILE, data.path)
    print(f"✅ Written to artifact path: {data.path}")


# ─────────────────────────────────────────────
# COMPONENT 2 — Validate Data
# ─────────────────────────────────────────────
@component(base_image=BASE_IMAGE)
def validate_data_op(data: Input[Dataset]) -> bool:
    """
    Basic data quality checks.
    Returns True only if data is safe to train on.
    """
    import pandas as pd

    df = pd.read_csv(data.path)

    print(f"Shape: {df.shape}")
    print(df.head())
    print("Nulls:\n", df.isnull().sum())

    if len(df) < 100:
        print("❌ FAIL: Too few rows")
        return False

    if df.isnull().sum().sum() > 0:
        print("❌ FAIL: Null values found")
        return False

    required_cols = ['age', 'tenure_months', 'monthly_charges',
                     'total_charges', 'num_support_calls', 'churn']
    for col in required_cols:
        if col not in df.columns:
            print(f"❌ FAIL: Missing column '{col}'")
            return False

    print("✅ Validation passed")
    return True


# ─────────────────────────────────────────────
# COMPONENT 3 — Train
# ─────────────────────────────────────────────
@component(base_image=BASE_IMAGE)
def train_op(
        data: Input[Dataset],
        model: Output[Model],
        metrics: Output[Metrics],
):
    """
    Train RandomForestClassifier.
    Writes model.pkl and metrics.json to KFP artifact paths.
    """
    import subprocess
    import os

    # train.py lives in the Docker image at /app/training/train.py
    result = subprocess.run(
        ["python", "/app/training/train.py",
         data.path, model.path, metrics.path],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Training failed:\n{result.stderr}")

    print("✅ Training complete")


# ─────────────────────────────────────────────
# COMPONENT 4 — Evaluate (returns AUC as float)
# ─────────────────────────────────────────────
@component
def evaluate_op(metrics: Input[Metrics]) -> float:
    """Read metrics.json and return the AUC score."""
    import json

    with open(metrics.path) as f:
        m = json.load(f)

    auc = None
    for metric in m["metrics"]:
        if metric["name"] == "auc":
            auc = metric["numberValue"]
            break

    if auc is None:
        raise ValueError("AUC not found in metrics file")

    print(f"📊 New model AUC: {auc:.4f}")
    return float(auc)


# ─────────────────────────────────────────────
# COMPONENT 5 — Get Production Baseline Metric
# Reads the last promoted model's metrics from MinIO.
# Falls back to 0.60 if no production model exists yet.
# ─────────────────────────────────────────────
@component(base_image=BASE_IMAGE)
def get_production_metric_op() -> float:
    """
    Fetch the AUC of the currently deployed production model.
    This is what the new model must BEAT to get promoted.
    """
    import boto3
    import json
    from botocore.exceptions import ClientError

    FALLBACK_AUC = 0.60  # First ever run — nothing to compare against

    try:
        s3 = boto3.client(
            "s3",
            endpoint_url="http://minio-service.kubeflow:9000",
            aws_access_key_id="minio",
            aws_secret_access_key="minio123",
        )
        obj = s3.get_object(
            Bucket="mlpipeline",
            Key="churn/production/metrics.json"
        )
        m = json.loads(obj["Body"].read())
        for metric in m["metrics"]:
            if metric["name"] == "auc":
                prod_auc = float(metric["numberValue"])
                print(f"📦 Production model AUC: {prod_auc:.4f}")
                return prod_auc

    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            print(f"⚠️  No production metrics found. Using fallback: {FALLBACK_AUC}")
            return FALLBACK_AUC
        raise

    return FALLBACK_AUC


# ─────────────────────────────────────────────
# COMPONENT 6 — Promote Model
# Only runs when new AUC > production AUC
# ─────────────────────────────────────────────
@component(base_image=BASE_IMAGE)
def promote_model_op(model: Input[Model], metrics: Input[Metrics]):
    """
    Upload winning model + its metrics to MinIO production path.
    KServe InferenceService watches this path.
    """
    import boto3
    import shutil

    s3 = boto3.client(
        "s3",
        endpoint_url="http://minio-service.kubeflow:9000",
        aws_access_key_id="minio",
        aws_secret_access_key="minio123",
    )

    bucket = "mlpipeline"

    # Upload model
    s3.upload_file(model.path, bucket, "churn/production/model.pkl")
    print("✅ Promoted model → s3://mlpipeline/churn/production/model.pkl")

    # Upload metrics so next run can compare against it
    s3.upload_file(metrics.path, bucket, "churn/production/metrics.json")
    print("✅ Promoted metrics → s3://mlpipeline/churn/production/metrics.json")


# ─────────────────────────────────────────────
# PIPELINE DEFINITION
# ─────────────────────────────────────────────
@dsl.pipeline(
    name="churn-train-pipeline",
    description="End-to-end churn prediction: ingest → validate → train → evaluate → promote"
)
def churn_pipeline():

    # Step 1: Pull data from MinIO
    data_task = generate_data_op()
    data_task.set_caching_options(False)  # Always get fresh data

    # Step 2: Validate — gate everything behind this
    valid_task = validate_data_op(data=data_task.outputs["data"])

    # Step 3–6: Only run if validation passed
    # dsl.If compares the *output parameter* of validate_data_op to True
    with dsl.If(valid_task.output == True, name="data-is-valid"):

        # Step 3: Train
        train_task = train_op(
            data=data_task.outputs["data"]
        )

        # Step 4: Get new model's AUC
        auc_task = evaluate_op(
            metrics=train_task.outputs["metrics"]
        )

        # Step 5: Get current production AUC
        prod_task = get_production_metric_op()

        # Step 6: Only promote if the new model is better
        with dsl.If(
                auc_task.output > prod_task.output,
                name="new-model-is-better"
        ):
            promote_model_op(
                model=train_task.outputs["model"],
                metrics=train_task.outputs["metrics"],
            )


# ─────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────
def run_pipeline():
    """Submit pipeline to Kubeflow. Run after: kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80"""
    client = kfp.Client(host="http://127.0.0.1:8080")

    run = client.create_run_from_pipeline_func(
        churn_pipeline,
        arguments={},
        run_name="churn-pipeline-run",
        enable_caching=False,
    )
    print(f"✅ Pipeline submitted. Run ID: {run.run_id}")
    print("   Open http://127.0.0.1:8080 to track progress")


if __name__ == "__main__":
    run_pipeline()