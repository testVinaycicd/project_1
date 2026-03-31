import kfp
from kfp import dsl
from kfp.dsl import component, Output, Dataset, Model, Metrics , Input



@component(base_image="crysis307/churn-train:v9")
def generate_data_op(data: Output[Dataset]):
    import subprocess
    import shutil

    # Pull data from S3 via DVC
    subprocess.run(["dvc", "pull"], check=True)

    # Copy dataset to KFP artifact path
    shutil.copy("data/churn_data.csv", data.path)




@component(base_image="crysis307/churn-train:v9")
def validate_data_op(data: Input[Dataset]) -> bool:
    import pandas as pd

    df = pd.read_csv(data.path)

    # simple checks
    print(df.head())
    print(df.shape)
    print(df.isnull().sum())
    if len(df) < 100:
        return False

    if df.isnull().sum().sum() > 0:
        return False

    return True





@component(base_image="crysis307/churn-train:v9")
def train_op(
        data: Input[Dataset],
        model: Output[Model],
        metrics: Output[Metrics],
):
    import subprocess

    subprocess.run([
        "python", "train.py",
        data.path,
        model.path,
        metrics.path
    ], check=True)


@component
def evaluate_op(metrics: Input[Metrics]) -> float:
    import json

    with open(metrics.path) as f:
        m = json.load(f)

    for metric in m["metrics"]:
        if metric["name"] == "auc":
            print("AUC====================================",metric["numberValue"])
            return metric["numberValue"]




@component
def get_production_metric_op() -> float:
    # load stored metric from previous production model
    return 0.60  # placeholder
    # need to write code to get previous model metrics from s3 before that implement s3 in the current project


@component(base_image="crysis307/churn-train:v9")
def promote_model_op(model: Input[Model]):

    import boto3

    s3 = boto3.client(
        "s3",
        endpoint_url="http://minio-service.kubeflow:9000",
        aws_access_key_id="minio",
        aws_secret_access_key="minio123",
    )

    bucket = "mlpipeline"  # Kubeflow default bucket

    key = "churn/production/model.pkl"

    s3.upload_file(model.path, bucket, key)

    print(f"Uploaded to MinIO: s3://{bucket}/{key}")




@dsl.pipeline(name="churn-train-pipeline")
def churn_pipeline():



    data_task = generate_data_op()
    data_task.set_caching_options(True)
    data_task.set_service_account("dvc-access-sa")
    valid = validate_data_op(data=data_task.outputs["data"])

    # with dsl.If(valid.output == True):
    #
    #     train_task = train_op(data=data_task.outputs["data"])
    #     auc = evaluate_op(metrics=train_task.outputs["metrics"])
    #     prod_score = get_production_metric_op()
    #
    #     with dsl.If(auc.output > prod_score.output):
    #         promote_model_op(model=train_task.outputs["model"])
    #
    #         print("Model Ready to Deploy")



def run_pipeline():
    client = kfp.Client(host="http://127.0.0.1:8080")

    run = client.create_run_from_pipeline_func(
        churn_pipeline,
        arguments={}
    )

    print("Run submitted:", run)


if __name__ == "__main__":
    run_pipeline()