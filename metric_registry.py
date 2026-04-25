"""
metric_registry.py
───────────────────
Loads metrics_config.yaml into typed dataclasses.

WHAT CHANGED (and why it's better):

  BEFORE:
    metrics_config.yaml stored complex PromQL per metric.
    Python code had to maintain query strings.
    Any PromQL change = edit Python config = redeploy code.

  AFTER:
    Recording rules in prometheusrule.yaml (Kubernetes CRD).
    Prometheus pre-computes every expression on a 30s interval.
    Result is stored as a simple named gauge: aiops:node_cpu_pct
    metrics_config.yaml only stores ML metadata.
    promql property just returns self.name — the gauge IS the query.

  Benefits:
    1. PromQL maintained in k8s — version controlled, kubectl apply
    2. Recording rules are faster — Prometheus pre-computes, no
       on-demand evaluation for every scrape
    3. Python is simpler — no embedded PromQL strings
    4. Separation of concerns: k8s owns queries, Python owns ML logic
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yaml



@dataclass
class MetricSpec:
    name: str            # e.g. "aiops:node_cpu_pct"
    # This IS the PromQL query — just reference the recording rule
    unit: str            # bytes | percent | ratio | count | bytes_per_sec
    layer: str           # node | container | workload | network | pipeline
    normal_min: float
    normal_max: float
    normal_std_pct: float
    higher_is_bad: bool

    @property
    def promql(self) -> str:
        """
        The PromQL query for this metric.

        Because we use Prometheus recording rules, the pre-computed
        metric name IS a valid instant PromQL query.

        Prometheus evaluates 'aiops:node_cpu_pct' the same way it
        evaluates '100 - avg(rate(node_cpu_seconds_total{...}[5m]))'.
        The recording rule runs the complex expression on a schedule
        and stores the result under this name.
        """
        return self.name

    @property
    def normal_mean(self) -> float:
        return (self.normal_min + self.normal_max) / 2

    @property
    def normal_std(self) -> float:
        return (self.normal_max - self.normal_min) * (self.normal_std_pct / 100)

    def synthetic_sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Generate n synthetic normal samples for fallback baseline."""
        return rng.normal(
            loc=self.normal_mean,
            scale=self.normal_std,
            size=n,
        ).clip(0, self.normal_max * 2)

    def format_value(self, value: float) -> str:
        if self.unit == "bytes":
            if value > 1e9: return f"{value/1e9:.2f} GB"
            if value > 1e6: return f"{value/1e6:.2f} MB"
            return f"{value/1e3:.2f} KB"
        if self.unit == "bytes_per_sec":
            if value > 1e6: return f"{value/1e6:.2f} MB/s"
            return f"{value/1e3:.2f} KB/s"
        if self.unit == "percent": return f"{value:.1f}%"
        if self.unit == "ratio":   return f"{value:.4f}"
        return f"{value:.4f}"

    @staticmethod
    def prepare_for_lstm(df, metric_name):
        if df is None or df.empty:
            print(f"DEBUG: {metric_name} returned NO data.")
            return None

        # Change: Check if all values in the entire snapshot are 0
        # or check a specific column if you are passing just one series
        if (df == 0).all().all():
            print(f"DEBUG: Data is currently all 0.0 (Healthy/Idle). Skipping AI.")
            return None

        return df
class MetricRegistry:
    """
    Central registry. Load once, use everywhere.

        from metric_registry import REGISTRY

        REGISTRY.feature_names          # ['aiops:node_cpu_pct', ...]
        REGISTRY.promql_map             # {'aiops:node_cpu_pct': 'aiops:node_cpu_pct', ...}
        REGISTRY.synthetic_baseline(500) # pd.DataFrame
    """

    def __init__(self, config_path: str | Path):
        self._specs: List[MetricSpec] = []
        self._load(Path(config_path))

    def _load(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"metrics_config.yaml not found at {path}")
        with open(path) as f:
            raw = yaml.safe_load(f)
        for entry in raw["metrics"]:
            self._specs.append(MetricSpec(
                name=entry["name"],
                unit=entry["unit"],
                layer=entry["layer"],
                normal_min=float(entry["normal_min"]),
                normal_max=float(entry["normal_max"]),
                normal_std_pct=float(entry["normal_std_pct"]),
                higher_is_bad=bool(entry["higher_is_bad"]),
            ))
        layers = set(s.layer for s in self._specs)
        print(f"MetricRegistry: {len(self._specs)} metrics, {len(layers)} layers")

    @property
    def all_specs(self) -> List[MetricSpec]:
        return self._specs

    @property
    def feature_names(self) -> List[str]:
        return [s.name for s in self._specs]

    @property
    def promql_map(self) -> dict[str, str]:
        """
        name -> promql mapping.
        Because promql == name for recording rules, this is
        {'aiops:node_cpu_pct': 'aiops:node_cpu_pct', ...}
        PrometheusClient iterates this — zero change needed there.
        """
        return {s.name: s.promql for s in self._specs}

    @property
    def layers(self) -> List[str]:
        seen = []
        for s in self._specs:
            if s.layer not in seen:
                seen.append(s.layer)
        return seen

    def get_spec(self, name: str) -> MetricSpec:
        for s in self._specs:
            if s.name == name:
                return s
        raise KeyError(f"No metric: {name}")

    def specs_by_layer(self, layer: str) -> List[MetricSpec]:
        return [s for s in self._specs if s.layer == layer]

    def __len__(self) -> int:
        return len(self._specs)

    def synthetic_baseline(self, n: int = 500) -> pd.DataFrame:
        """
        Synthetic normal data — used when Prometheus has < 24h of history.
        Column names are recording rule names: 'aiops:node_cpu_pct' etc.
        """
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            spec.name: spec.synthetic_sample(n, rng)
            for spec in self._specs
        })
        print(f"  Synthetic baseline: {df.shape[0]} rows x {df.shape[1]} metrics")
        return df





    def print_summary(self):
        print(f"\n{'─'*65}")
        print(f"  METRIC REGISTRY  ({len(self._specs)} recording rules)")
        print(f"{'─'*65}")
        current_layer = None
        for s in self._specs:
            if s.layer != current_layer:
                current_layer = s.layer
                print(f"\n  [{s.layer.upper()}]")
            direction = "spike=bad" if s.higher_is_bad else "drop=bad"
            print(f"    {s.name:<40}  {s.unit:<14}  {direction}")
        print(f"{'─'*65}\n")


_CONFIG_PATH = Path(__file__).parent / "metrics_config.yaml"
REGISTRY = MetricRegistry(_CONFIG_PATH)


if __name__ == "__main__":
    REGISTRY.print_summary()
    df = REGISTRY.synthetic_baseline(100)
    print(df.describe().round(4))