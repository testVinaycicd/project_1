from kfp import dsl
from kfp import Client
from train_component import train_op

@dsl.pipeline(name="churn-train-pipeline")
def pipeline():
    train_op(
        input_path="data/churn_data.csv",
        model_path="/tmp/model.pkl",
        metrics_path="/tmp/metrics.txt"
    )

if __name__ == "__main__":
    client = Client(host="http://localhost:8080")
    client.create_run_from_pipeline_func(pipeline)