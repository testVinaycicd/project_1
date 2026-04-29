"""
drift_detector.py
──────────────────
Core drift detection logic using Evidently AI.

WHAT EVIDENTLY AI IS:
  An open-source ML observability library. It compares two DataFrames
  (reference = training data, current = recent production requests)
  and runs statistical tests to answer: "have the distributions changed?"

  It returns structured results you can threshold and act on.

TWO THINGS THIS FILE DETECTS:

  1. DATA DRIFT (statistical comparison of feature distributions)
     ─────────────────────────────────────────────────────────
     Uses Kolmogorov-Smirnov test (numerical features):
       H0: The two samples come from the same distribution.
       If p_value < 0.05 → reject H0 → distributions differ → DRIFT.

     Returns per-feature drift scores + dataset-level flag.
     Example output:
       age:              drift_score=0.12  drifted=False
       monthly_charges:  drift_score=0.67  drifted=True   ← problem
       tenure_months:    drift_score=0.08  drifted=False
       dataset_drift: True (2/5 features drifted = 40% > 30% threshold)

  2. CONCEPT DRIFT (prediction distribution monitoring)
     ─────────────────────────────────────────────────
     We don't have ground truth labels in production (we don't know
     if a predicted churner actually churned until months later).

     So we monitor the MODEL'S OWN OUTPUT distribution instead:
       - If mean(churn_probability) shifts from 0.32 → 0.71, the model
         is predicting churn for most customers — unusual.
       - If std(churn_probability) drops near 0, the model lost confidence
         and is returning nearly identical scores for everyone.

     These are strong signals that the model's world has changed.

HOW IT CONNECTS TO THE PIPELINE:
  The KFP pipeline now has a new step: save_reference_data_op.
  After training, it uploads the training DataFrame to MinIO.
  This file reads that reference, reads the production log,
  and runs the comparison.
"""

from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import boto3
import numpy as np
import pandas as pd
import yaml
from botocore.exceptions import ClientError

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import (
    DatasetDriftMetric,
    ColumnDriftMetric,
)
from evidently import ColumnMapping


# ─────────────────────────────────────────────────────────────────
# CONFIG LOADER
# ─────────────────────────────────────────────────────────────────
def load_config(path: str = "drift_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────
# RESULT DATACLASSES — structured outputs
# ─────────────────────────────────────────────────────────────────
@dataclass
class FeatureDriftResult:
    name: str
    drift_score: float      # 0.0 (no drift) → 1.0 (maximum drift)
    p_value: float          # KS test p-value
    drifted: bool           # p_value < threshold
    current_mean: float
    reference_mean: float
    mean_shift_pct: float   # % change in mean


@dataclass
class DataDriftResult:
    timestamp: float
    dataset_drifted: bool
    dataset_drift_score: float        # fraction of features drifted
    drifted_features_count: int
    total_features: int
    feature_results: list[FeatureDriftResult] = field(default_factory=list)
    production_samples: int = 0
    error: Optional[str] = None


@dataclass
class ConceptDriftResult:
    timestamp: float
    concept_drifted: bool
    prediction_shift: float           # |current_mean - reference_mean|
    reference_mean_prediction: float
    current_mean_prediction: float
    current_std_prediction: float
    low_confidence_detected: bool     # std too low
    error: Optional[str] = None


@dataclass
class DriftCheckResult:
    timestamp: float
    data_drift: DataDriftResult
    concept_drift: ConceptDriftResult
    overall_drift_score: float        # 0.0 → 1.0 combined score
    retraining_recommended: bool


# ─────────────────────────────────────────────────────────────────
# MINIO CLIENT HELPER
# ─────────────────────────────────────────────────────────────────
class MinIOClient:
    """Thin wrapper around boto3 for MinIO access."""

    def __init__(self, cfg: dict):
        storage = cfg["storage"]
        self.client = boto3.client(
            "s3",
            endpoint_url=storage["minio_endpoint"],
            aws_access_key_id=storage["minio_access_key"],
            aws_secret_access_key=storage["minio_secret_key"],
        )
        self.bucket = storage["bucket"]
        self.reference_key = storage["reference_data_key"]
        self.production_key = storage["production_log_key"]

    def load_parquet(self, key: str) -> Optional[pd.DataFrame]:
        """Download a parquet file from MinIO into a DataFrame."""
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=key)
            return pd.read_parquet(io.BytesIO(obj["Body"].read()))
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                return None
            raise

    def save_parquet(self, df: pd.DataFrame, key: str):
        """Upload a DataFrame as parquet to MinIO."""
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        buf.seek(0)
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=buf.getvalue(),
        )

    def append_production_log(self, new_rows: pd.DataFrame):
        """
        Append new prediction requests to the production log.
        Implements the sliding window: keeps only last N rows.
        """
        existing = self.load_parquet(self.production_key)
        if existing is not None:
            combined = pd.concat([existing, new_rows], ignore_index=True)
        else:
            combined = new_rows

        # Keep only the last 1000 rows (configurable)
        if len(combined) > 1000:
            combined = combined.tail(1000)

        self.save_parquet(combined, self.production_key)
        return combined


# ─────────────────────────────────────────────────────────────────
# DRIFT DETECTOR — main class
# ─────────────────────────────────────────────────────────────────
class DriftDetector:
    """
    Loads reference data from MinIO, loads production log from MinIO,
    runs Evidently AI, returns structured results.

    Usage:
        detector = DriftDetector("drift_config.yaml")
        result = detector.run_check()
        print(result.overall_drift_score)
        print(result.retraining_recommended)
    """

    def __init__(self, config_path: str = "drift_config.yaml"):
        self.cfg = load_config(config_path)
        self.minio = MinIOClient(self.cfg)

        # Feature names from config (in order)
        self.feature_names = [
                                 f["name"] for f in self.cfg["detection"]["features"]
                                 if "features" in self.cfg.get("data_drift", {})
                             ] or [
                                 f["name"] for f in self.cfg["data_drift"]["features"]
                             ]

        self.p_threshold = self.cfg["data_drift"]["p_value_threshold"]
        self.dataset_threshold = self.cfg["data_drift"]["dataset_drift_threshold"]
        self.feature_threshold = self.cfg["data_drift"]["feature_drift_threshold"]
        self.prediction_col = self.cfg["concept_drift"]["prediction_column"]
        self.pred_shift_threshold = self.cfg["concept_drift"]["prediction_shift_threshold"]
        self.min_pred_std = self.cfg["concept_drift"]["min_prediction_std"]
        self.min_samples = self.cfg["detection"]["min_production_samples"]
        self.window_size = self.cfg["detection"]["production_window_size"]

    # ─────────────────────────────────────────────────────────────
    def run_check(self) -> DriftCheckResult:
        """
        Full drift check. Entry point for the detection loop.
        Returns a complete DriftCheckResult.
        """
        ts = time.time()
        print(f"\n[{time.strftime('%H:%M:%S')}] Running drift check...")

        # Load data
        reference = self.minio.load_parquet(self.minio.reference_key)
        production = self.minio.load_parquet(self.minio.production_key)

        if reference is None:
            print("  ⚠  No reference data found in MinIO.")
            print("     Run the Kubeflow pipeline first to generate it.")
            err_data = DataDriftResult(
                timestamp=ts, dataset_drifted=False,
                dataset_drift_score=0.0, drifted_features_count=0,
                total_features=len(self.feature_names), error="no_reference_data"
            )
            err_concept = ConceptDriftResult(
                timestamp=ts, concept_drifted=False,
                prediction_shift=0.0, reference_mean_prediction=0.0,
                current_mean_prediction=0.0, current_std_prediction=0.0,
                low_confidence_detected=False, error="no_reference_data"
            )
            return DriftCheckResult(ts, err_data, err_concept, 0.0, False)

        if production is None or len(production) < self.min_samples:
            n = len(production) if production is not None else 0
            print(f"  ⚠  Only {n} production samples (need {self.min_samples}).")
            print("     Run simulate_drift.py to generate production traffic.")
            err_data = DataDriftResult(
                timestamp=ts, dataset_drifted=False,
                dataset_drift_score=0.0, drifted_features_count=0,
                total_features=len(self.feature_names),
                production_samples=n, error="insufficient_production_samples"
            )
            err_concept = ConceptDriftResult(
                timestamp=ts, concept_drifted=False,
                prediction_shift=0.0, reference_mean_prediction=0.0,
                current_mean_prediction=0.0, current_std_prediction=0.0,
                low_confidence_detected=False,
                error="insufficient_production_samples"
            )
            return DriftCheckResult(ts, err_data, err_concept, 0.0, False)

        # Use sliding window of recent production data
        current = production.tail(self.window_size)

        print(f"  Reference: {len(reference)} rows")
        print(f"  Current:   {len(current)} rows (window={self.window_size})")

        data_result   = self._check_data_drift(reference, current, ts)
        concept_result = self._check_concept_drift(reference, current, ts)

        # Combined score: weighted average
        # Data drift is the primary signal, concept drift is secondary
        overall = (
                data_result.dataset_drift_score * 0.6 +
                (1.0 if concept_result.concept_drifted else 0.0) * 0.4
        )

        min_score = self.cfg["alerting"]["min_drift_score_for_retraining"]
        retrain = overall >= min_score

        result = DriftCheckResult(
            timestamp=ts,
            data_drift=data_result,
            concept_drift=concept_result,
            overall_drift_score=round(overall, 4),
            retraining_recommended=retrain,
        )

        self._print_summary(result)
        return result

    # ─────────────────────────────────────────────────────────────
    def _check_data_drift(
            self,
            reference: pd.DataFrame,
            current: pd.DataFrame,
            ts: float,
    ) -> DataDriftResult:
        """
        Run Evidently DataDriftPreset.

        Evidently builds a Report with DataDriftPreset, which:
          1. Detects column types automatically
          2. Selects appropriate statistical test per type
             (KS test for numerical, chi-squared for categorical)
          3. Returns per-column drift score + dataset-level flag

        We extract the structured JSON result and convert to our
        DataDriftResult dataclass.
        """
        print("\n  [DATA DRIFT]")

        try:
            ref_features = reference[self.feature_names]
            cur_features = current[self.feature_names]

            # ColumnMapping tells Evidently which columns are features
            # vs targets vs predictions
            column_mapping = ColumnMapping(
                numerical_features=self.feature_names,
            )

            report = Report(metrics=[
                DataDriftPreset(
                    # stattest: 'ks' = Kolmogorov-Smirnov
                    # This is the standard test for numerical distributions
                    stattest="ks",
                    # drift_share: fraction of drifted features to call
                    # dataset drifted
                    drift_share=self.dataset_threshold,
                ),
            ])

            report.run(
                reference_data=ref_features,
                current_data=cur_features,
                column_mapping=column_mapping,
            )

            # Extract structured result as dict
            result_dict = report.as_dict()
            drift_metric = result_dict["metrics"][0]["result"]

            # ── Dataset-level result ──────────────────────────────
            dataset_drifted = drift_metric["dataset_drift"]
            n_drifted = drift_metric["number_of_drifted_columns"]
            n_total = drift_metric["number_of_columns"]
            drift_share = n_drifted / n_total if n_total > 0 else 0.0

            # ── Per-feature results ───────────────────────────────
            feature_results = []
            drift_by_col = drift_metric.get("drift_by_columns", {})

            for feat_name in self.feature_names:
                col_data = drift_by_col.get(feat_name, {})
                score = col_data.get("drift_score", 0.0)
                p_val = col_data.get("drift_score", 1.0)  # KS score ≈ p-value
                drifted = col_data.get("drift_detected", False)

                ref_mean = float(reference[feat_name].mean())
                cur_mean = float(current[feat_name].mean())
                shift_pct = abs(cur_mean - ref_mean) / (abs(ref_mean) + 1e-10) * 100

                feature_results.append(FeatureDriftResult(
                    name=feat_name,
                    drift_score=round(float(score), 4),
                    p_value=round(float(p_val), 4),
                    drifted=bool(drifted),
                    current_mean=round(cur_mean, 4),
                    reference_mean=round(ref_mean, 4),
                    mean_shift_pct=round(float(shift_pct), 2),
                ))

                status = "DRIFT" if drifted else "ok"
                print(f"    {feat_name:<22} score={score:.3f}  "
                      f"ref_mean={ref_mean:.2f}  cur_mean={cur_mean:.2f}  "
                      f"shift={shift_pct:.1f}%  [{status}]")

            print(f"\n    Dataset drifted: {dataset_drifted}  "
                  f"({n_drifted}/{n_total} features)")

            return DataDriftResult(
                timestamp=ts,
                dataset_drifted=bool(dataset_drifted),
                dataset_drift_score=round(drift_share, 4),
                drifted_features_count=int(n_drifted),
                total_features=int(n_total),
                feature_results=feature_results,
                production_samples=len(current),
            )

        except Exception as e:
            print(f"    ERROR: {e}")
            return DataDriftResult(
                timestamp=ts, dataset_drifted=False,
                dataset_drift_score=0.0, drifted_features_count=0,
                total_features=len(self.feature_names), error=str(e)
            )

    # ─────────────────────────────────────────────────────────────
    def _check_concept_drift(
            self,
            reference: pd.DataFrame,
            current: pd.DataFrame,
            ts: float,
    ) -> ConceptDriftResult:
        """
        Monitor prediction distribution as a concept drift proxy.

        We don't have ground truth labels in production
        (we don't know IF customers actually churned yet).

        So we monitor the model's OWN predictions:
          - Reference: predictions on the test set during training
          - Current:   predictions on recent production requests

        If the distribution shifts significantly → concept drift.

        This is called "proxy concept drift detection" and is the
        standard approach when labels are delayed or unavailable.
        """
        print("\n  [CONCEPT DRIFT]")

        try:
            if self.prediction_col not in reference.columns:
                print(f"    ⚠  '{self.prediction_col}' not in reference data.")
                print(f"       Save predictions alongside features in save_reference_data_op.")
                return ConceptDriftResult(
                    timestamp=ts, concept_drifted=False,
                    prediction_shift=0.0, reference_mean_prediction=0.0,
                    current_mean_prediction=0.0, current_std_prediction=0.0,
                    low_confidence_detected=False,
                    error="prediction_column_missing_from_reference"
                )

            if self.prediction_col not in current.columns:
                return ConceptDriftResult(
                    timestamp=ts, concept_drifted=False,
                    prediction_shift=0.0, reference_mean_prediction=0.0,
                    current_mean_prediction=0.0, current_std_prediction=0.0,
                    low_confidence_detected=False,
                    error="prediction_column_missing_from_production"
                )

            ref_preds = reference[self.prediction_col].dropna()
            cur_preds = current[self.prediction_col].dropna()

            ref_mean = float(ref_preds.mean())
            cur_mean = float(cur_preds.mean())
            cur_std  = float(cur_preds.std())

            shift = abs(cur_mean - ref_mean)
            concept_drifted = shift > self.pred_shift_threshold
            low_confidence  = cur_std < self.min_pred_std

            print(f"    Reference mean prediction: {ref_mean:.4f}")
            print(f"    Current mean prediction:   {cur_mean:.4f}")
            print(f"    Shift:                     {shift:.4f}  "
                  f"(threshold={self.pred_shift_threshold})")
            print(f"    Current std:               {cur_std:.4f}")
            print(f"    Concept drifted: {concept_drifted}")
            if low_confidence:
                print(f"    ⚠  Low prediction variance — model may be stuck!")

            return ConceptDriftResult(
                timestamp=ts,
                concept_drifted=concept_drifted,
                prediction_shift=round(shift, 4),
                reference_mean_prediction=round(ref_mean, 4),
                current_mean_prediction=round(cur_mean, 4),
                current_std_prediction=round(cur_std, 4),
                low_confidence_detected=low_confidence,
            )

        except Exception as e:
            print(f"    ERROR: {e}")
            return ConceptDriftResult(
                timestamp=ts, concept_drifted=False,
                prediction_shift=0.0, reference_mean_prediction=0.0,
                current_mean_prediction=0.0, current_std_prediction=0.0,
                low_confidence_detected=False, error=str(e)
            )

    # ─────────────────────────────────────────────────────────────
    def _print_summary(self, result: DriftCheckResult):
        print("\n" + "─" * 55)
        print("  DRIFT CHECK SUMMARY")
        print("─" * 55)
        print(f"  Data drift score  : {result.data_drift.dataset_drift_score:.4f}")
        print(f"  Dataset drifted   : {result.data_drift.dataset_drifted}")
        print(f"  Drifted features  : {result.data_drift.drifted_features_count}"
              f"/{result.data_drift.total_features}")
        print(f"  Concept drifted   : {result.concept_drift.concept_drifted}")
        print(f"  Prediction shift  : {result.concept_drift.prediction_shift:.4f}")
        print(f"  Overall score     : {result.overall_drift_score:.4f}")
        print(f"  Retrain needed    : {'YES' if result.retraining_recommended else 'no'}")
        print("─" * 55 + "\n")


# ─────────────────────────────────────────────────────────────────
# Standalone run — test without the full server
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    detector = DriftDetector("drift_config.yaml")
    result = detector.run_check()
    print(f"\nFinal: overall_drift_score={result.overall_drift_score}")
    print(f"       retraining_recommended={result.retraining_recommended}")