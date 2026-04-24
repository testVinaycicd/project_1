"""
iforest_detector.py
────────────────────
Isolation Forest anomaly detector. Driven by REGISTRY.

WHAT ISOLATION FOREST ACTUALLY DOES:
  Imagine you have 1000 normal data points and 5 anomalies on a 2D scatter plot.
  Normal points cluster together. Anomalies sit far from the cluster.

  The algorithm builds many random binary trees:
    Pick a random feature → pick a random split value → split the data → repeat

  To isolate a NORMAL point: many splits needed (it's deep in the cluster)
  To isolate an ANOMALY   : few splits needed  (it's alone at the edge)

  The anomaly score = average depth across all trees.
  Short path = anomalous. Long path = normal.

  score_samples() returns a float in roughly [-0.5, 0.0]:
    0.0   → perfectly normal (very long path to isolate)
   -0.5   → extreme anomaly (isolated in 1-2 splits)
   -0.1   → typical threshold (below this = ANOMALY)

WHAT'S NEW vs your Day 1 iforest.py:
  1. Reads feature list from REGISTRY — no hardcoded column names
  2. Feature contributions use registry metadata (higher_is_bad, unit)
  3. severity is continuous [0,1] not just binary
  4. save/load with joblib — model survives process restart
  5. Layer-grouped contribution report (node/container/workload/network/pipeline)
"""

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from metric_registry import REGISTRY


class IsolationForestDetector:

    def __init__(
            self,
            contamination: float = 0.05,
            n_estimators: int = 200,
            random_state: int = 42,
    ):
        """
        contamination:
            The fraction of TRAINING data you expect was already anomalous.
            0.05 = "5% of my 24h baseline had brief incidents — that's fine."
            Lower = more conservative (fewer alerts).
            Higher = more sensitive (more alerts, more false positives).

        n_estimators:
            Number of isolation trees to build.
            More trees = more stable, consistent scores. Slower to train.
            200 is standard for production. 100 is fine for development.
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )

        self.threshold_score: Optional[float] = None
        self.training_stats: Optional[dict] = None
        self.is_fitted = False

    # ─────────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame) -> "IsolationForestDetector":
        """
        Train on baseline data.
        df must contain all columns in REGISTRY.feature_names.
        """
        features = REGISTRY.feature_names
        print(f"\n🌲 Training Isolation Forest")
        print(f"   Samples      : {len(df)}")
        print(f"   Features     : {len(features)}")
        print(f"   Estimators   : {self.n_estimators}")
        print(f"   Contamination: {self.contamination}")

        X = self.scaler.fit_transform(df[features])
        self.model.fit(X)

        # Compute score distribution on training data
        # threshold = the contamination-th percentile of training scores
        # i.e., 5% of training data is considered anomalous — this sets the cutoff
        scores = self.model.score_samples(X)
        self.threshold_score = float(np.percentile(scores, self.contamination * 100))

        self.training_stats = {
            "n_samples":     len(df),
            "score_mean":    float(np.mean(scores)),
            "score_std":     float(np.std(scores)),
            "score_min":     float(np.min(scores)),
            "threshold":     self.threshold_score,
            # Per-feature stats for contribution calculation
            "feat_means":    df[features].mean().to_dict(),
            "feat_stds":     df[features].std().to_dict(),
        }
        self.is_fitted = True

        print(f"   Score range  : [{self.training_stats['score_min']:.4f}, "
              f"{self.training_stats['score_mean']:.4f}]")
        print(f"   Threshold    : {self.threshold_score:.4f}  (below = ANOMALY)")
        print("✅ Isolation Forest ready")
        return self

    # ─────────────────────────────────────────────────────────────
    def predict(self, snapshot_df: pd.DataFrame) -> dict:
        """
        Detect anomaly in a single snapshot (1-row DataFrame).

        Returns a rich dict:
        {
            "detector"    : "IsolationForest",
            "status"      : "ANOMALY" | "NORMAL",
            "score"       : -0.14,     ← raw IF score
            "threshold"   : -0.09,     ← learned threshold
            "severity"    : 0.73,      ← 0.0 normal → 1.0 extreme
            "top_culprit" : "container_mem_bytes",
            "contributions": {          ← z-score per metric
                "node_cpu_pct":          0.3,   (σ from training mean)
                "container_mem_bytes":   4.2,   ← this is the problem
                ...
            },
            "values_formatted": {       ← human-readable current values
                "container_mem_bytes": "1.82 GB",
                ...
            }
        }
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict()")

        features = REGISTRY.feature_names
        X = self.scaler.transform(snapshot_df[features])

        raw_pred = self.model.predict(X)[0]           # -1 (anomaly) or 1 (normal)
        score    = float(self.model.score_samples(X)[0])  # continuous score

        status = "ANOMALY" if raw_pred == -1 else "NORMAL"

        # ── Severity [0, 1] ───────────────────────────────────────
        # At the threshold: severity = 0.5
        # score much more negative than threshold: severity → 1.0
        score_range = (
                self.training_stats["score_mean"] - self.training_stats["score_min"]
        )
        if score_range > 0 and status == "ANOMALY":
            severity = float(np.clip(
                (self.threshold_score - score) / score_range,
                0.0, 1.0
            ))
        else:
            severity = 0.0

        # ── Feature contributions (z-scores) ──────────────────────
        # How many standard deviations is each current value from
        # its training mean? Large z-score = that feature is unusual.
        row = snapshot_df[features].iloc[0]
        contributions = {}
        for feat in features:
            mean = self.training_stats["feat_means"][feat]
            std  = self.training_stats["feat_stds"][feat]
            z    = abs((row[feat] - mean) / (std + 1e-10))
            contributions[feat] = round(float(z), 3)

        top_culprit = max(contributions, key=contributions.get)

        # ── Human-readable values ─────────────────────────────────
        values_formatted = {
            feat: REGISTRY.get_spec(feat).format_value(row[feat])
            for feat in features
        }

        return {
            "detector":          "IsolationForest",
            "status":            status,
            "score":             round(score, 4),
            "threshold":         round(self.threshold_score, 4),
            "severity":          round(severity, 3),
            "top_culprit":       top_culprit,
            "contributions":     contributions,
            "values_formatted":  values_formatted,
        }

    def contributions_by_layer(self, result: dict) -> dict:
        """
        Group feature contributions by infrastructure layer.
        Returns: { "node": 0.45_avg_z, "container": 2.1_avg_z, ... }
        Used in the comparison report to show WHICH LAYER is most anomalous.
        """
        layer_scores = {}
        for layer in REGISTRY.layers:
            specs = REGISTRY.specs_by_layer(layer)
            z_scores = [
                result["contributions"].get(s.name, 0.0)
                for s in specs
            ]
            layer_scores[layer] = round(float(np.mean(z_scores)), 3)
        return layer_scores

    # ─────────────────────────────────────────────────────────────
    def save(self, path: str = "models/iforest_detector.joblib"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "scaler":          self.scaler,
            "model":           self.model,
            "threshold_score": self.threshold_score,
            "training_stats":  self.training_stats,
            "contamination":   self.contamination,
            "n_estimators":    self.n_estimators,
        }, path)
        print(f"💾 IsolationForest saved → {path}")

    @classmethod
    def load(cls, path: str = "models/iforest_detector.joblib") -> "IsolationForestDetector":
        p = joblib.load(path)
        d = cls(contamination=p["contamination"], n_estimators=p["n_estimators"])
        d.scaler          = p["scaler"]
        d.model           = p["model"]
        d.threshold_score = p["threshold_score"]
        d.training_stats  = p["training_stats"]
        d.is_fitted       = True
        print(f"📂 IsolationForest loaded ← {path}")
        return d