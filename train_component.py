from kfp import dsl
from kfp.dsl import component

@component
def train_op(input_path: str, model_path: str, metrics_path: str):
    import subprocess

    subprocess.run([
        "python", "train.py",
        input_path,
        model_path,
        metrics_path
    ], check=True)

