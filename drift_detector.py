"""
drift_detector.py
──────────────────
Model drift monitoring. Sits alongside the ensemble — does not replace it.

TWO TYPES OF DRIFT WE DETECT:
  1. Data drift   — the INPUT distribution has shifted from what the model trained on.
                    Example: CPU values that were normally 20-40% are now always 80%.
                    The model still runs, but it's scoring on data it never saw in training.
                    KS-test catches this.

  2. Feature drift (PSI) — a softer, ranked measure of how much each feature has moved.
                    PSI < 0.1  → no drift       (safe)
                    PSI 0.1-0.2 → moderate drift  (monitor)
                    PSI > 0.2  → significant drift (consider retraining)
                    Used in banking/finance for model stability monitoring.

WHY BOTH:
  KS-test is binary and statistically rigorous (p-value based).
  PSI is continuous and industry-standard — easier to threshold and alert on.
  Together they give you "IS there drift?" (KS) + "HOW BAD is it?" (PSI).

ARCHITECTURE — fits into existing code with no changes to ensemble_engine.py:
  drift = DriftDetector.from_training_stats(ensemble.iforest.training_stats)
  drift.observe(snapshot)          ← call every poll, same as ensemble.predict()
  result = drift.report()          ← get full drift report
  drift.set_gauges()               ← push to Prometheus
"""

import json
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from metric_registry import REGISTRY


# ─────────────────────────────────────────────────────────────────
# PSI HELPERS
# ─────────────────────────────────────────────────────────────────
def _psi_score(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index between a reference and current distribution.

    PSI = Σ (current_pct - reference_pct) * ln(current_pct / reference_pct)

    Bins are computed on the reference distribution.
    Current values are bucketed into the same bins.
    This makes the comparison apples-to-apples.
    """
    # Compute bin edges from reference
    breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
    # Deduplicate edges (happens when many identical values)
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 3:
        # Not enough distinct values to bin — return 0 (no drift detectable)
        return 0.0

    # Count reference and current in each bin
    ref_counts, _  = np.histogram(reference, bins=breakpoints)
    cur_counts, _  = np.histogram(current,   bins=breakpoints)

    # Convert to proportions — add tiny epsilon to avoid log(0)
    eps = 1e-6
    ref_pct = (ref_counts + eps) / (len(reference) + eps * len(ref_counts))
    cur_pct = (cur_counts + eps) / (len(current)   + eps * len(cur_counts))

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(np.clip(psi, 0.0, 10.0))   # clip pathological cases


def _ks_test(reference_sample: np.ndarray, current_window: np.ndarray) -> tuple[float, float]:
    """
    Kolmogorov-Smirnov two-sample test.
    Returns (statistic, p_value).
    p_value < 0.05 means distributions are significantly different → drift.
    """
    if len(current_window) < 5:
        return 0.0, 1.0   # not enough data yet
    stat, pval = stats.ks_2samp(reference_sample, current_window)
    return float(stat), float(pval)


# ─────────────────────────────────────────────────────────────────
# DRIFT DETECTOR
# ─────────────────────────────────────────────────────────────────
class DriftDetector:
    """
    Monitors each feature for distribution drift vs. training baseline.

    Usage:
        # At startup (after loading ensemble):
        drift = DriftDetector.from_training_stats(ensemble.iforest.training_stats)

        # Every poll (inside live_loop):
        drift.observe(snapshot)
        report = drift.report()
        drift.set_gauges(ml_drift_score, ml_drift_psi, ml_drift_feature_label)
    """

    # PSI thresholds (industry standard)
    PSI_OK       = 0.1
    PSI_WARN     = 0.2

    # KS p-value threshold
    KS_ALPHA     = 0.05

    def __init__(
            self,
            reference_means: dict,
            reference_stds:  dict,
            reference_samples: Optional[dict] = None,
            window_size: int = 30,
    ):
        """
        reference_means   — per-feature mean from training   (from iforest.training_stats)
        reference_stds    — per-feature std  from training   (from iforest.training_stats)
        reference_samples — optional dict of per-feature np.arrays for KS-test
                            If None, synthetic reference is generated from means/stds
        window_size       — rolling window of snapshots to compare against reference
                            30 polls × 30s = 15 minutes of live data
        """
        self.features       = REGISTRY.feature_names
        self.ref_means      = reference_means
        self.ref_stds       = reference_stds
        self.window_size    = window_size

        # Rolling buffer of recent snapshots per feature
        self._buffers: dict[str, deque] = {
            f: deque(maxlen=window_size) for f in self.features
        }

        # Generate synthetic reference samples from training stats
        # (1000 samples per feature, normally distributed around training mean/std)
        # This is used for KS-test when real baseline samples aren't saved
        rng = np.random.default_rng(42)
        self._ref_samples: dict[str, np.ndarray] = {}
        for feat in self.features:
            if reference_samples and feat in reference_samples:
                self._ref_samples[feat] = np.asarray(reference_samples[feat])
            else:
                mu  = reference_means.get(feat, 0.0)
                sig = max(reference_stds.get(feat, 1.0), 1e-6)
                self._ref_samples[feat] = rng.normal(mu, sig, 1000)

        # Last report cache
        self._last_report: Optional[dict] = None

    # ── Factory ───────────────────────────────────────────────────
    @classmethod
    def from_training_stats(
            cls,
            training_stats: dict,
            window_size: int = 30,
    ) -> "DriftDetector":
        """
        Build from iforest.training_stats — no extra args needed.

        training_stats is the dict stored inside the saved IsolationForest:
          {
            "n_samples":  1000,
            "score_mean": -0.12,
            ...
            "feat_means": { "aiops:node_cpu_pct": 22.3, ... },
            "feat_stds":  { "aiops:node_cpu_pct":  4.1, ... },
          }
        """
        return cls(
            reference_means   = training_stats["feat_means"],
            reference_stds    = training_stats["feat_stds"],
            window_size       = window_size,
        )

    @classmethod
    def load(cls, path: str = "models/drift_reference.json") -> "DriftDetector":
        """Load reference from saved JSON (includes actual baseline samples)."""
        with open(path) as f:
            data = json.load(f)
        ref_samples = {k: np.asarray(v) for k, v in data.get("samples", {}).items()}
        return cls(
            reference_means   = data["means"],
            reference_stds    = data["stds"],
            reference_samples = ref_samples,
            window_size       = data.get("window_size", 30),
        )

    def save_reference(
            self,
            baseline_df: pd.DataFrame,
            path: str = "models/drift_reference.json",
            n_samples: int = 500,
    ):
        """
        Save actual baseline samples for KS-test.
        Call this once after training — baseline_df is the same df used for fit().

        Saves a random sample (not all rows) to keep the file small.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(42)
        idx = rng.choice(len(baseline_df), min(n_samples, len(baseline_df)), replace=False)
        sampled = baseline_df.iloc[idx]

        data = {
            "means":       self.ref_means,
            "stds":        self.ref_stds,
            "window_size": self.window_size,
            "samples": {
                feat: sampled[feat].tolist()
                for feat in self.features
                if feat in sampled.columns
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"💾 Drift reference saved → {path}")

    # ── Core ──────────────────────────────────────────────────────
    def observe(self, snapshot: pd.DataFrame):
        """
        Feed one snapshot into the rolling buffer.
        Call every poll — same cadence as ensemble.predict().
        """
        row = snapshot[self.features].iloc[0]
        for feat in self.features:
            self._buffers[feat].append(float(row[feat]))

    @property
    def buffer_fill(self) -> int:
        """How many snapshots we've collected so far."""
        return len(self._buffers[self.features[0]])

    @property
    def is_ready(self) -> bool:
        """Need at least 10 snapshots before drift is meaningful."""
        return self.buffer_fill >= 10

    def report(self) -> dict:
        """
        Compute drift report for all features.

        Returns:
        {
            "status":       "DRIFTING" | "NORMAL" | "WARMING_UP",
            "overall_psi":  0.23,      ← max PSI across all features
            "n_drifting":   3,         ← features with PSI > threshold
            "worst_feature": "aiops:container_mem_bytes",
            "worst_psi":    0.41,
            "features": {
                "aiops:node_cpu_pct": {
                    "psi":      0.03,
                    "ks_stat":  0.08,
                    "ks_pval":  0.61,
                    "drifting": False,
                    "current_mean": 21.4,
                    "ref_mean":     22.3,
                    "z_shift":      0.22,   ← how many σ the mean has shifted
                },
                ...
            },
            "buffer_fill": 15,
        }
        """
        if not self.is_ready:
            return {
                "status":        "WARMING_UP",
                "buffer_fill":   self.buffer_fill,
                "window_size":   self.window_size,
                "overall_psi":   0.0,
                "n_drifting":    0,
                "worst_feature": None,
                "worst_psi":     0.0,
                "features":      {},
            }

        feature_results = {}
        for feat in self.features:
            current  = np.asarray(list(self._buffers[feat]))
            ref      = self._ref_samples[feat]

            psi              = _psi_score(ref, current)
            ks_stat, ks_pval = _ks_test(ref, current)
            cur_mean         = float(np.mean(current))
            ref_mean         = self.ref_means.get(feat, 0.0)
            ref_std          = max(self.ref_stds.get(feat, 1.0), 1e-6)
            z_shift          = abs(cur_mean - ref_mean) / ref_std

            drifting = (psi > self.PSI_WARN) or (ks_pval < self.KS_ALPHA and ks_stat > 0.2)

            feature_results[feat] = {
                "psi":          round(psi, 4),
                "ks_stat":      round(ks_stat, 4),
                "ks_pval":      round(ks_pval, 4),
                "drifting":     drifting,
                "current_mean": round(cur_mean, 4),
                "ref_mean":     round(ref_mean, 4),
                "z_shift":      round(z_shift, 3),
            }

        # Overall summary
        psi_values   = [v["psi"] for v in feature_results.values()]
        n_drifting   = sum(1 for v in feature_results.values() if v["drifting"])
        worst_feat   = max(feature_results, key=lambda f: feature_results[f]["psi"])
        worst_psi    = feature_results[worst_feat]["psi"]
        overall_psi  = round(float(np.mean(psi_values)), 4)

        status = "NORMAL"
        if worst_psi > self.PSI_WARN:
            status = "DRIFTING"
        elif worst_psi > self.PSI_OK:
            status = "WARN"

        self._last_report = {
            "status":        status,
            "buffer_fill":   self.buffer_fill,
            "window_size":   self.window_size,
            "overall_psi":   overall_psi,
            "worst_psi":     worst_psi,
            "worst_feature": worst_feat,
            "n_drifting":    n_drifting,
            "features":      feature_results,
        }
        return self._last_report

    def set_gauges(self, gauge_psi, gauge_active, gauge_feature_psi: dict):
        """
        Push drift metrics to Prometheus gauges.

        gauge_psi          — ml_drift_psi          (overall mean PSI)
        gauge_active       — ml_drift_active        (1 if DRIFTING)
        gauge_feature_psi  — dict of {feat: Gauge}  (per-feature PSI)

        Define gauges in ensemble_engine.py:
            from prometheus_client import Gauge
            ml_drift_psi    = Gauge("ml_drift_psi",    "Mean PSI across all features")
            ml_drift_active = Gauge("ml_drift_active",  "1 if drift detected")
        """
        if self._last_report is None:
            return
        r = self._last_report
        gauge_psi.set(r["overall_psi"])
        gauge_active.set(1 if r["status"] == "DRIFTING" else 0)
        for feat, g in gauge_feature_psi.items():
            if feat in r["features"]:
                g.set(r["features"][feat]["psi"])

    def print_report(self, report: Optional[dict] = None):
        """Human-readable drift report for the terminal."""
        r = report or self._last_report
        if r is None:
            print("  No drift report yet — call report() first")
            return

        status_icon = {
            "WARMING_UP": "⏳",
            "NORMAL":     "✅",
            "WARN":       "⚠️ ",
            "DRIFTING":   "🚨",
        }.get(r["status"], "❓")

        print(f"\n{'─'*65}")
        print(f"  DRIFT MONITOR  {status_icon} {r['status']}")
        print(f"  Buffer: {r['buffer_fill']}/{r['window_size']} snapshots")
        if r["status"] == "WARMING_UP":
            print(f"{'─'*65}\n")
            return

        print(f"  Overall PSI : {r['overall_psi']:.4f}  "
              f"(warn>{self.PSI_OK:.1f}  drift>{self.PSI_WARN:.1f})")
        print(f"  Worst feat  : {r['worst_feature']}  PSI={r['worst_psi']:.4f}")
        print(f"  Drifting    : {r['n_drifting']}/{len(r['features'])} features")

        if r["n_drifting"] > 0:
            print(f"\n  {'FEATURE':<42} {'PSI':>6}  {'KS-p':>6}  {'z-shift':>7}  STATUS")
            print(f"  {'─'*42} {'─'*6}  {'─'*6}  {'─'*7}  {'─'*8}")
            # Sort by PSI descending — worst first
            sorted_feats = sorted(
                r["features"].items(),
                key=lambda x: x[1]["psi"],
                reverse=True,
            )
            for feat, fd in sorted_feats:
                flag = "🚨" if fd["drifting"] else "  "
                print(
                    f"  {flag}{feat:<40} "
                    f"{fd['psi']:>6.4f}  "
                    f"{fd['ks_pval']:>6.3f}  "
                    f"{fd['z_shift']:>7.2f}σ  "
                    f"{'DRIFT' if fd['drifting'] else 'ok'}"
                )
        print(f"{'─'*65}\n")