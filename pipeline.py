import kfp
from kfp import dsl
from kfp.dsl import component, Output, Model, Metrics

@component(base_image="crysis307/churn-train:v2")
def train_op(
        input_path: str,
        model: Output[Model],
        metrics: Output[Metrics],
):
    import subprocess

    subprocess.run([
        "python", "/app/train.py",
        input_path,
        model.path,
        metrics.path
    ], check=True)


@dsl.pipeline(name="churn-train-pipeline")
def churn_pipeline():
    train_op(
        input_path="/app/churn_data.csv",

    )


def run_pipeline():
    client = kfp.Client(host="http://127.0.0.1:8080")

    run = client.create_run_from_pipeline_func(
        churn_pipeline,
        arguments={}
    )

    print("Run submitted:", run)


if __name__ == "__main__":
    run_pipeline()