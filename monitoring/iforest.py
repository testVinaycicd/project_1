"""
AIops Anomaly Detection Engine
───────────────────────────────
Pulls live metrics from Prometheus, runs Isolation Forest,
alerts with cooldown so you don't get paged 200 times.

Bugs fixed vs original:
  1. COOLDOWN_PERIOD was undefined → now uses ALERT_COOLDOWN everywhere
  2. process_anomaly() was defined but never called → now called from main loop
  3. IsolationForest was being re-fit every single loop iteration → now fit once on baseline
  4. score_samples() not used → now we log the anomaly score for visibility
  5. Hardcoded baseline of 2 metrics → expanded to 4 real k8s metrics
"""

import time
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
PROM_URL = "http://localhost:9090/api/v1/query"
POLL_INTERVAL_SEC = 10       # How often we check metrics
ALERT_COOLDOWN_SEC = 120     # Don't re-alert within this window (seconds)
BASELINE_SAMPLES = 100       # How many "normal" samples to train on

# ─────────────────────────────────────────────
# STATE  (module-level, persists across loop iterations)
# ─────────────────────────────────────────────
last_alert_time: float = 0.0
scaler: StandardScaler = None
model: IsolationForest = None


# ─────────────────────────────────────────────
# METRIC FETCHER
# ─────────────────────────────────────────────
def fetch_metric(query: str, default: float = 0.0) -> float:
    """
    Query Prometheus instant API.
    Returns default if metric is missing (safe fallback).
    """
    try:
        resp = requests.get(
            PROM_URL,
            params={"query": query},
            timeout=5
        ).json()
        results = resp.get("data", {}).get("result", [])
        if not results:
            print(f"  ⚠️  No data for: {query}")
            return default
        return float(results[0]["value"][1])
    except Exception as e:
        print(f"  ❌ Fetch error [{query}]: {e}")
        return default


def get_current_metrics() -> pd.DataFrame:
    """
    Collect the 4 metrics we monitor.
    Add more queries here as your stack grows.
    """
    metrics = {
        # Pipeline latency (seconds) — from your Pushgateway
        "duration_sec": fetch_metric("pipeline_processing_seconds"),

        # Memory used by kubeflow namespace containers (bytes)
        "mem_bytes": fetch_metric(
            'sum(container_memory_usage_bytes{namespace="kubeflow"})'
        ),

        # CPU throttle ratio across kubeflow — leading indicator of overload
        "cpu_throttle_ratio": fetch_metric(
            'sum(rate(container_cpu_throttled_seconds_total{namespace="kubeflow"}[5m]))'
        ),

        # Pod restart count — detects crashlooping
        "pod_restarts": fetch_metric(
            'sum(kube_pod_container_status_restarts_total{namespace="kubeflow"})'
        ),
    }
    return pd.DataFrame([metrics])


# ─────────────────────────────────────────────
# MODEL INITIALISATION (runs once at startup)
# ─────────────────────────────────────────────
def build_baseline() -> tuple:
    """
    Build a synthetic 'normal' baseline.
    In production you would replace this with 24h of real Prometheus
    historical data pulled via the /api/v1/query_range endpoint.
    """
    print("🏗️  Building baseline model from synthetic normal data...")

    rng = np.random.default_rng(42)
    baseline = pd.DataFrame({
        "duration_sec":      rng.normal(loc=0.3,   scale=0.05,  size=BASELINE_SAMPLES).clip(0),
        "mem_bytes":         rng.normal(loc=5e7,   scale=5e6,   size=BASELINE_SAMPLES).clip(0),
        "cpu_throttle_ratio":rng.normal(loc=0.01,  scale=0.005, size=BASELINE_SAMPLES).clip(0),
        "pod_restarts":      rng.normal(loc=2.0,   scale=0.5,   size=BASELINE_SAMPLES).clip(0),
    })

    # Fit scaler and model ONCE on baseline — never re-fit in the loop
    sc = StandardScaler()
    sc.fit(baseline)

    # contamination='auto' → Isolation Forest picks its own threshold
    # n_estimators=200 → more trees = more stable anomaly scores
    mdl = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42
    )
    mdl.fit(sc.transform(baseline))

    print("✅ Baseline model ready")
    return sc, mdl


# ─────────────────────────────────────────────
# ANOMALY DETECTION
# ─────────────────────────────────────────────
def detect(current_df: pd.DataFrame) -> tuple:
    """
    Returns (status, score)
    score < 0  → more anomalous
    score >= 0 → more normal
    """
    global scaler, model
    scaled = scaler.transform(current_df)

    prediction = model.predict(scaled)[0]        # -1 = anomaly, 1 = normal
    score = model.score_samples(scaled)[0]       # raw anomaly score

    status = "ANOMALY" if prediction == -1 else "NORMAL"
    return status, score


# ─────────────────────────────────────────────
# ALERT HANDLER (with cooldown)
# ─────────────────────────────────────────────
def handle_anomaly(score: float, current_df: pd.DataFrame):
    """
    Send alert only if we're outside the cooldown window.
    Bug fix: original used undefined COOLDOWN_PERIOD variable.
    """
    global last_alert_time
    now = time.time()
    elapsed = now - last_alert_time
    remaining = int(ALERT_COOLDOWN_SEC - elapsed)

    if elapsed < ALERT_COOLDOWN_SEC:
        print(f"  🤫 MUTED — cooldown active for another {remaining}s")
        return

    # Outside cooldown — fire the alert
    row = current_df.iloc[0]
    print("\n" + "!" * 55)
    print("🚨 CRITICAL ALERT — MULTIVARIATE ANOMALY DETECTED")
    print(f"   Score       : {score:.4f}  (lower = more anomalous)")
    print(f"   Duration    : {row['duration_sec']:.3f}s")
    print(f"   Memory      : {row['mem_bytes']/1e6:.1f} MB")
    print(f"   CPU Throttle: {row['cpu_throttle_ratio']:.4f}")
    print(f"   Pod Restarts: {row['pod_restarts']:.0f}")
    print("   Action      : → Notifying SRE Team (Slack / PagerDuty)")
    print("!" * 55 + "\n")

    # TODO Day 5: replace this print with a real webhook call
    # requests.post(SLACK_WEBHOOK_URL, json={"text": "..."})

    last_alert_time = now  # reset cooldown


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Initialise once
    scaler, model = build_baseline()
    print(f"\n🚀 AIOps Engine running — polling every {POLL_INTERVAL_SEC}s\n")

    while True:
        try:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            current = get_current_metrics()
            status, score = detect(current)

            if status == "ANOMALY":
                print(f"[{ts}]  🚨 ANOMALY  (score={score:.4f})")
                handle_anomaly(score, current)   # Bug fix: was never called before
            else:
                print(f"[{ts}]  ✅ NORMAL   (score={score:.4f})")

        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}]  💥 Loop error: {e}")

        time.sleep(POLL_INTERVAL_SEC)