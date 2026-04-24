"""
check_metrics.py
─────────────────
Run this BEFORE training the anomaly detectors.
Tells you which of the 11 declared metrics Prometheus can actually return.

Why this matters:
  Some metrics (e.g. argo_workflows_gauge) only appear after a pipeline
  has actually run. Others (e.g. container_oom_kills) may return 0 or
  no data on a fresh cluster. This script tells you exactly what you have.

Usage:
  cd monitoring/aiops
  python check_metrics.py

Prerequisite:
  kubectl port-forward -n monitoring \
    svc/monitoring-kube-prometheus-prometheus 9090:9090
"""

import requests
from metric_registry import REGISTRY

PROM = "http://localhost:9090/api/v1/query"


def query(promql: str) -> tuple[bool, float]:
    """Returns (has_data, value)"""
    try:
        resp = requests.get(PROM, params={"query": promql}, timeout=5).json()
        results = resp.get("data", {}).get("result", [])
        if not results:
            return False, 0.0
        return True, float(results[0]["value"][1])
    except Exception as e:
        return False, 0.0


def main():
    print("\n" + "═" * 72)
    print("  PROMETHEUS METRIC AVAILABILITY CHECK")
    print("═" * 72)

    available   = []
    unavailable = []

    for layer in REGISTRY.layers:
        print(f"\n  [{layer.upper()} LAYER]")
        for spec in REGISTRY.specs_by_layer(layer):
            has_data, value = query(spec.promql)
            formatted = spec.format_value(value) if has_data else "—"
            status = "✅" if has_data else "⚠️ "
            print(f"    {status}  {spec.name:<35}  {formatted}")
            if has_data:
                available.append(spec.name)
            else:
                unavailable.append(spec.name)

    print(f"\n{'─' * 72}")
    print(f"  Available   : {len(available)}/{len(REGISTRY)}  metrics returning data")
    print(f"  Unavailable : {len(unavailable)}/{len(REGISTRY)}  metrics returning no data")

    if unavailable:
        print(f"\n  Unavailable metrics:")
        for name in unavailable:
            spec = REGISTRY.get_spec(name)
            print(f"    • {name}  ({spec.layer} layer)")
            if spec.layer == "pipeline":
                print(f"        → Run the Kubeflow pipeline first — "
                      f"argo_workflows_gauge appears after at least one run")
            elif "oom" in name:
                print(f"        → Normal on healthy cluster (no OOM kills yet)")
            elif "throttle" in name:
                print(f"        → May be 0 on idle cluster — that is valid data")

    if len(available) < 4:
        print(f"\n  ❌ Too few metrics available ({len(available)}).")
        print(f"     Check that Prometheus is scraping node-exporter and kube-state-metrics:")
        print(f"       kubectl get servicemonitor -n monitoring")
    elif len(available) < len(REGISTRY):
        print(f"\n  ⚠️  Training will proceed on {len(available)} metrics.")
        print(f"     Synthetic fallback will be used for missing ones.")
        print(f"     This is fine for local minikube.")
    else:
        print(f"\n  ✅ All metrics available. Safe to run training.")

    print("═" * 72 + "\n")


if __name__ == "__main__":
    main()