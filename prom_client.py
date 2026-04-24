"""
prometheus_client.py
─────────────────────
Talks to Prometheus. Pure pull architecture.

WHAT SIMPLIFIED:
  Before: each metric had a complex PromQL embedded in Python.
  After:  each metric is a recording rule name — a simple gauge.

  Querying 'aiops:node_cpu_pct' is identical to querying
  '100 - avg(rate(node_cpu_seconds_total{mode="idle"}[5m]))'.
  Prometheus evaluated the complex expression when the recording
  rule fired (every 30s) and stored the result.
  We just read the stored value.

  The instant query and range query APIs are unchanged.
  The only difference: the PromQL strings are now trivially short.

PULL MODEL (unchanged from Day 3):
  Recording rules run in Prometheus → stored as time series.
  We call /api/v1/query or /api/v1/query_range to pull values.
  Nothing pushes to us. We pull everything.
"""

import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests

from metric_registry import REGISTRY


class PrometheusClient:

    def __init__(self, base_url: str = "http://localhost:9090", timeout: int = 10):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["Accept"] = "application/json"

    def is_healthy(self) -> bool:
        try:
            return self.session.get(
                f"{self.base_url}/-/ready", timeout=3
            ).status_code == 200
        except Exception:
            return False

    # ── Low level ─────────────────────────────────────────────────
    def _instant(self, promql: str, default: float = 0.0) -> float:
        """
        Instant query — current value of a PromQL expression.
        For recording rules, promql is just the metric name.
        e.g. promql = 'aiops:node_cpu_pct'
        """
        try:
            resp = self.session.get(
                f"{self.base_url}/api/v1/query",
                params={"query": promql},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            results = resp.json()["data"]["result"]
            if not results:
                print(f"  No data: {promql}")
                return default
            return float(results[0]["value"][1])
        except requests.exceptions.ConnectionError:
            print(f"  Cannot reach Prometheus at {self.base_url}")
            print(f"  Run: kubectl port-forward -n monitoring "
                  f"svc/monitoring-kube-prometheus-prometheus 9090:9090")
            return default
        except Exception as e:
            print(f"  Query error [{promql}]: {e}")
            return default

    def _range(
            self,
            promql: str,
            hours_back: float,
            step_seconds: int,
    ) -> pd.Series:
        """
        Range query — time series of values over last N hours.
        Recording rules have been running since the PrometheusRule
        was applied, so history depth = time since you ran:
          kubectl apply -f prometheusrule.yaml
        """
        now = time.time()
        start = now - (hours_back * 3600)
        try:
            resp = self.session.get(
                f"{self.base_url}/api/v1/query_range",
                params={
                    "query": promql,
                    "start": start,
                    "end":   now,
                    "step":  step_seconds,
                },
                timeout=30,
            )
            resp.raise_for_status()
            results = resp.json()["data"]["result"]
            if not results:
                return pd.Series(dtype=float)
            pairs = results[0]["values"]
            timestamps = [
                datetime.fromtimestamp(float(p[0]), tz=timezone.utc)
                for p in pairs
            ]
            return pd.Series([float(p[1]) for p in pairs], index=timestamps)
        except Exception as e:
            print(f"  Range error [{promql}]: {e}")
            return pd.Series(dtype=float)

    # ── High level ────────────────────────────────────────────────
    def fetch_snapshot(self) -> pd.DataFrame:
        row = {}
        for spec in REGISTRY.all_specs:
            # Use normal_mean as default, not 0.0
            # Keeps snapshot consistent with how baseline was filled
            value = self._instant(spec.promql, default=spec.normal_mean)
            row[spec.name] = value
        return pd.DataFrame([row])[REGISTRY.feature_names]

    def fetch_baseline(
            self,
            hours_back: float = 24.0,
            step_seconds: int = 30,
    ) -> pd.DataFrame:
        """
        Historical data for ML training.
        Recording rules must have been running for at least hours_back hours.

        On a fresh minikube where you just applied prometheusrule.yaml:
          Prometheus will have only minutes of history.
          The fallback to synthetic data triggers automatically.

        Once the cluster has been running with recording rules for 24h:
          This returns real historical data and synthetic is not needed.
        """
        expected = int(hours_back * 3600 / step_seconds)
        print(f"\nFetching {hours_back}h baseline ({expected} rows expected)...")
        print(f"Note: history depth = time since prometheusrule.yaml was applied")

        series = {}
        for spec in REGISTRY.all_specs:
            print(f"  {spec.name:<40} ", end="", flush=True)
            s = self._range(spec.promql, hours_back, step_seconds)
            series[spec.name] = s
            print(f"{len(s):>5} pts")

        df = pd.DataFrame(series)
        df = df.ffill(limit=3)
        for spec in REGISTRY.all_specs:
            if spec.name in df.columns:
                df[spec.name] = df[spec.name].fillna(spec.normal_mean)

        if len(df) < 100:
            print(f"\n  Only {len(df)} clean rows — recording rules need more time.")
            print(f"  Using synthetic baseline.\n")
            return REGISTRY.synthetic_baseline(n=500)

        before = len(df)



        df = df.reset_index(drop=True)[REGISTRY.feature_names]
        print(f"\n  Baseline ready: {df.shape[0]} rows x {df.shape[1]} features")
        return df

    def print_live_readings(self):
        """Verify recording rules are working — prints current values."""
        print(f"\n{'─'*65}")
        print(f"  LIVE READINGS  (recording rules via {self.base_url})")
        print(f"{'─'*65}")
        for layer in REGISTRY.layers:
            print(f"\n  [{layer.upper()}]")
            for spec in REGISTRY.specs_by_layer(layer):
                value = self._instant(spec.promql)
                status = "no data" if value == 0.0 else spec.format_value(value)
                direction = "spike=bad" if spec.higher_is_bad else "drop=bad"
                print(f"    {spec.name:<40}  {status:>15}  ({direction})")
        print(f"{'─'*65}\n")


if __name__ == "__main__":
    client = PrometheusClient()
    if not client.is_healthy():
        print("Prometheus not reachable.")
        print("kubectl port-forward -n monitoring "
              "svc/monitoring-kube-prometheus-prometheus 9090:9090")
    else:
        print("Prometheus healthy\n")
        client.print_live_readings()