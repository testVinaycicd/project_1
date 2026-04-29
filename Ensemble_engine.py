"""
ensemble_engine.py
───────────────────
Orchestrates everything. The only file you run directly.

Three modes:
  --mode train    → pull 24h from Prometheus, train both models, save
  --mode run      → load saved models, start live loop
  --mode compare  → one comparison report (good for demos/interviews)

WHAT WE BUILT — the full picture:

  metrics_config.yaml    ← DECLARE what to observe (11 metrics, 5 layers)
         ↓
  metric_registry.py     ← LOAD declarations into typed Python objects
         ↓
  prometheus_client.py   ← PULL data (instant + range) from Prometheus
         ↓
  iforest_detector.py    ← POINT anomaly detection (one snapshot)
  lstm_detector.py       ← SEQUENCE anomaly detection (rolling window)
         ↓
  ensemble_engine.py     ← COMBINE both, alert with cooldown, loop

  Adding a new metric = one YAML block. Zero code changes elsewhere.
  That is the declarative pattern in production.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from metric_registry import REGISTRY
from prom_client import PrometheusClient
from iforest_detector import IsolationForestDetector
from lstm_detecter import LSTMDetector

import requests
from datetime import datetime, timezone

from prometheus_client import Gauge, start_http_server

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
PROM_URL           = "http://localhost:9090"
POLL_INTERVAL_SEC  = 30
ALERT_COOLDOWN_SEC = 120
BASELINE_HOURS     = 20
BASELINE_STEP_SEC  = 30
VOTING             = "all"     # "any" | "all" | "majority"

MODEL_DIR          = Path("models")
IF_PATH            = MODEL_DIR / "iforest_detector.joblib"
LSTM_PATH          = MODEL_DIR / "lstm_detector.pt"
ALERTMANAGER_URL = "http://localhost:9093/api/v2/alerts"
METRICS_PORT = 8000



# ─────────────────────────────────────────────────────────────────
# PROMETHEUS METRICS
# ─────────────────────────────────────────────────────────────────
ml_severity      = Gauge("ml_anomaly_severity",  "Ensemble anomaly severity 0-1")
ml_active        = Gauge("ml_anomaly_active",     "1 if ANOMALY 0 if NORMAL")
ml_if_anomaly    = Gauge("ml_iforest_anomaly",    "1 if IsolationForest says ANOMALY")
ml_lstm_anomaly  = Gauge("ml_lstm_anomaly",       "1 if LSTM says ANOMALY")
ml_votes         = Gauge("ml_anomaly_votes",      "Detectors voting ANOMALY")

# ─────────────────────────────────────────────────────────────────
# ENSEMBLE
# ─────────────────────────────────────────────────────────────────
class EnsembleDetector:
    """
    Holds both detectors. Combines their predictions.

    Voting strategies:
      "any"      → ANOMALY if either detector says ANOMALY  (more sensitive)
      "all"      → ANOMALY only if both say ANOMALY         (fewer alerts)
      "majority" → ANOMALY if more than half say ANOMALY
    """

    def __init__(self, voting: str = VOTING):
        self.voting  = voting
        self.iforest = IsolationForestDetector(contamination=0.03, n_estimators=200)
        self.lstm    = LSTMDetector(seq_len=10, hidden_dim=64, epochs=50, threshold_sigma=5.0)
        self.is_fitted = False

    def fit(self, baseline: pd.DataFrame) -> "EnsembleDetector":
        print("\n" + "═" * 60)
        print("  TRAINING ENSEMBLE")
        print(f"  {len(REGISTRY)} metrics · {len(baseline)} baseline rows")
        print("═" * 60)
        self.iforest.fit(baseline)
        self.lstm.fit(baseline)
        self.is_fitted = True
        return self

    def predict(self, snapshot: pd.DataFrame) -> dict:
        if_result   = self.iforest.predict(snapshot)
        lstm_result = self.lstm.update_and_predict(snapshot)

        # ── NEW: suppress weak IF signals ──────────────────────────
        IF_MIN_SEVERITY = 0.15   # tune this — ignore IF if sev < 0.15
        if if_result["status"] == "ANOMALY" and if_result.get("severity", 0) < IF_MIN_SEVERITY:
            if_result = dict(if_result, status="NORMAL")   # downgrade, don't mutate original
            # ───────────────────────────────────────────────────────────
        # valid_snapshot = REGISTRY.get_spec("aiops:container_cpu_throttle_ratio").prepare_for_lstm(
        #     snapshot, "Ensemble-Precheck"
        # )
        #
        # if valid_snapshot is not None:
        #     # Data is real/active — run LSTM
        #     lstm_result = self.lstm.update_and_predict(snapshot)
        # else:
        #     # Data is 0.0 or empty — Force LSTM to stay 'NORMAL' to avoid false alerts
        #     lstm_result = {
        #         "status": "NORMAL",
        #         "reconstruction_error": 0.0,
        #         "threshold": 0.1, # Use your model's default
        #         "severity": 0.0
        #     }

        # Collect votes (skip WARMING_UP)
        active = [
            r["status"] for r in [if_result, lstm_result]
            if r["status"] in ("ANOMALY", "NORMAL")
        ]
        n_anomaly = active.count("ANOMALY")
        n_total   = len(active)

        # ADD this — don't fire if LSTM hasn't warmed up yet and voting is "all"
        lstm_warming = lstm_result.get("status") == "WARMING_UP"

        if self.voting == "any":
            verdict = "ANOMALY" if n_anomaly >= 1 else "NORMAL"
        elif self.voting == "all":
            # Require ALL detectors ready AND all saying ANOMALY
            if lstm_warming:
                verdict = "NORMAL"   # ← wait for LSTM before firing
            else:
                verdict = "ANOMALY" if n_anomaly == n_total and n_total > 0 else "NORMAL"
        else:  # majority
            verdict = "ANOMALY" if n_anomaly > n_total / 2 else "NORMAL"

        # ── FIX: severity = 0 when verdict is NORMAL ──────────────
        if verdict == "NORMAL":
            severity = 0.0
        else:
            severities = [r.get("severity", 0.0) for r in [if_result, lstm_result]]
            severity = round(max(severities), 3)
            # ──────────────────────────────────────────────────────────

        return {
            "verdict":          verdict,
            "voting":           self.voting,
            "n_anomaly_votes":  n_anomaly,
            "n_votes":          n_total,
            "severity":         severity,
            "if_result":        if_result,
            "lstm_result":      lstm_result,
        }

    def save(self):
        MODEL_DIR.mkdir(exist_ok=True)
        self.iforest.save(str(IF_PATH))
        self.lstm.save(str(LSTM_PATH))

    @classmethod
    def load(cls) -> "EnsembleDetector":
        e = cls()
        e.iforest = IsolationForestDetector.load(str(IF_PATH))
        e.lstm    = LSTMDetector.load(str(LSTM_PATH))
        e.is_fitted = True
        return e


# ─────────────────────────────────────────────────────────────────
# COMPARISON REPORT
# ─────────────────────────────────────────────────────────────────
def run_comparison(ensemble: EnsembleDetector, client: PrometheusClient):
    """
    Run 4 scenarios showing how IF and LSTM behave differently.
    This is your interview demo — it proves you understand the difference.
    """
    snap = client.fetch_snapshot()

    print("\n" + "═" * 70)
    print("  ANOMALY DETECTOR COMPARISON")
    print(f"  {len(REGISTRY)} metrics across layers: "
          + " · ".join(REGISTRY.layers))
    print("═" * 70)

    print("\n📊 CURRENT LIVE READINGS (from Prometheus):")
    for layer in REGISTRY.layers:
        print(f"  [{layer.upper()}]")
        for spec in REGISTRY.specs_by_layer(layer):
            val = snap[spec.name].iloc[0]
            print(f"    {spec.name:<35}  {spec.format_value(val)}")

    # Build scenarios — normal + 3 injected anomalies
    scenarios = [
        {
            "label": "✅ Normal — live readings right now",
            "desc":  "Real values from Prometheus. Both detectors should say NORMAL.",
            "snap":  snap.copy(),
        },
        {
            "label": "🔴 Point anomaly — container memory 15× normal",
            "desc":  "One metric spikes suddenly. IF catches this easily. LSTM may not yet.",
            "snap":  snap.assign(**{"aiops:container_mem_bytes": snap["aiops:container_mem_bytes"] * 15}),
        },
        {
            "label": "🟡 Workload anomaly — pods failing + restarts",
            "desc":  "Multiple workload-layer metrics deviate together.",
            "snap":  snap.assign(**{
                "aiops:pods_not_running": 8,
                "aiops:pod_restart_rate": snap["aiops:pod_restart_rate"] * 20 + 0.5,
                "aiops:deployment_unavailable_ratio": 0.8,
            }),
        },
        {
            "label": "🔵 Node pressure — CPU + memory + disk all stressed",
            "desc":  "Node-layer wide stress. Cascade failure pattern.",
            "snap":  snap.assign(**{
                "aiops:node_cpu_pct": 92.0,
                "aiops:node_mem_available_ratio": 0.04,
                "aiops:node_disk_available_ratio": 0.06,
                "aiops:node_load1": 8.5,
            }),
        },
    ]

    for sc in scenarios:
        print(f"\n{'─' * 70}")
        print(f"  {sc['label']}")
        print(f"  {sc['desc']}")

        result = ensemble.predict(sc["snap"])
        if_r   = result["if_result"]
        lstm_r = result["lstm_result"]

        # IF output
        print(f"\n  🌲 ISOLATION FOREST  →  {_badge(if_r['status'])}")
        print(f"     Score     : {if_r['score']:.4f}  (threshold: {if_r['threshold']:.4f})")
        print(f"     Severity  : {_bar(if_r.get('severity', 0))}")
        print(f"     Culprit   : {if_r.get('top_culprit', 'N/A')}")

        # Layer breakdown from IF contributions
        layer_scores = ensemble.iforest.contributions_by_layer(if_r)
        print(f"     By layer  : ", end="")
        for layer, z in sorted(layer_scores.items(), key=lambda x: -x[1]):
            print(f"{layer}={z:.1f}σ", end="  ")
        print()

        # LSTM output
        print(f"\n  🧠 LSTM AUTOENCODER  →  ", end="")
        if lstm_r["status"] == "WARMING_UP":
            eta = lstm_r.get("eta_seconds", "?")
            print(f"⏳ WARMING UP  ({lstm_r['buffer_fill']}/{lstm_r['seq_len']} snapshots, ~{eta}s to go)")
        else:
            print(_badge(lstm_r["status"]))
            print(f"     Recon err : {lstm_r['reconstruction_error']:.6f}  "
                  f"(threshold: {lstm_r['threshold']:.6f})")
            print(f"     Severity  : {_bar(lstm_r.get('severity', 0))}")

        # Ensemble verdict
        print(f"\n  🏆 ENSEMBLE  →  {_badge(result['verdict'])}")
        print(f"     Votes     : {result['n_anomaly_votes']}/{result['n_votes']} say ANOMALY")
        print(f"     Strategy  : {result['voting']}")
        print(f"     Severity  : {_bar(result['severity'])}")

    print(f"\n{'═' * 70}")
    print("  KEY TAKEAWAYS:")
    print("  • IF: fast, no warmup, catches sudden point anomalies")
    print("  • LSTM: needs seq_len snapshots to warm up, catches slow drifts")
    print("  • Voting 'any': sensitive (good for alerting)")
    print("  • Voting 'all': conservative (good for automated remediation)")
    print("  • Both firing = critical. Only one = investigate.")
    print(f"{'═' * 70}\n")


# ─────────────────────────────────────────────────────────────────
# LIVE LOOP
# ─────────────────────────────────────────────────────────────────
_last_alert: float = 0.0

def live_loop(ensemble: EnsembleDetector, client: PrometheusClient):
    warmup_steps = ensemble.lstm.seq_len
    print(f"\n🚀 Ensemble AIops Engine — LIVE")
    print(f"   Metrics   : {len(REGISTRY)}")
    print(f"   Poll      : {POLL_INTERVAL_SEC}s")
    print(f"   Cooldown  : {ALERT_COOLDOWN_SEC}s")
    print(f"   LSTM warmup: {warmup_steps} polls = {warmup_steps * POLL_INTERVAL_SEC}s\n")

    # ADD: skip first 2 polls to let metrics stabilize
    print("  Stabilizing — skipping first 2 polls...")
    for _ in range(2):
        client.fetch_snapshot()
        time.sleep(POLL_INTERVAL_SEC)

    while True:
        try:
            ts       = time.strftime("%Y-%m-%d %H:%M:%S")
            snap     = client.fetch_snapshot()
            result   = ensemble.predict(snap)
            verdict  = result["verdict"]
            sev      = result["severity"]
            if_st    = result["if_result"]["status"]
            lstm_st  = result["lstm_result"]["status"]


            ml_severity.set(sev)
            ml_active.set(1 if verdict == "ANOMALY" else 0)
            ml_if_anomaly.set(1 if if_st == "ANOMALY" else 0)
            ml_lstm_anomaly.set(1 if lstm_st == "ANOMALY" else 0)
            ml_votes.set(result["n_anomaly_votes"])


            print(
                f"[{ts}]  Ensemble: {_badge(verdict):14}  "
                f"IF: {_badge(if_st):10}  "
                f"LSTM: {_badge(lstm_st):14}  "
                f"Sev: {sev:.2f}"
            )

            if verdict == "ANOMALY":
                _fire_alert(result)

        except KeyboardInterrupt:
            print("\n🛑 Stopped")
            break
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] 💥 {e}")

        time.sleep(POLL_INTERVAL_SEC)


# def _fire_alert(result: dict):
#     global _last_alert
#     now = time.time()
#     if (now - _last_alert) < ALERT_COOLDOWN_SEC:
#         remaining = int(ALERT_COOLDOWN_SEC - (now - _last_alert))
#         print(f"  🤫 Alert muted — {remaining}s cooldown remaining")
#         return
#
#     if_r = result["if_result"]
#     print("\n" + "!" * 60)
#     print("🚨 ENSEMBLE ANOMALY ALERT")
#     print(f"   Verdict  : {result['verdict']}")
#     print(f"   Severity : {_bar(result['severity'])}")
#     print(f"   IF culprit: {if_r.get('top_culprit', 'N/A')}")
#     print("   → SRE notified (webhook goes here on Day 5)")
#     print("!" * 60 + "\n")
#     _last_alert = now

def _fire_alert(result: dict):
    global _last_alert
    now = time.time()
    if (now - _last_alert) < ALERT_COOLDOWN_SEC:
        remaining = int(ALERT_COOLDOWN_SEC - (now - _last_alert))
        print(f"  🤫 Alert muted — {remaining}s cooldown remaining")
        return

    if_r   = result["if_result"]
    sev    = result["severity"]
    culprit = if_r.get("top_culprit", "N/A")

    print("\n" + "!" * 60)
    print("🚨 ENSEMBLE ANOMALY ALERT")
    print(f"   Severity  : {_bar(sev)}")
    print(f"   IF culprit: {culprit}")

    # ── Real webhook to AlertManager ──────────────────────────────
    payload = [{
        "labels": {
            "alertname": "MLAnomalyDetected",
            "severity":  "critical" if sev >= 0.7 else "warning",
            "source":    "ensemble_engine",
            "culprit":   str(culprit),
        },
        "annotations": {
            "summary":     f"ML anomaly — severity {sev:.0%}",
            "description": (
                f"IF={if_r['status']} | "
                f"LSTM={result['lstm_result']['status']} | "
                f"votes={result['n_anomaly_votes']}/{result['n_votes']} | "
                f"top culprit={culprit}"
            ),
        },
        "startsAt": datetime.now(timezone.utc).isoformat(),
    }]

    try:
        resp = requests.post(ALERTMANAGER_URL, json=payload, timeout=5)
        resp.raise_for_status()
        print(f"   🔔 AlertManager → {resp.status_code} OK")
    except requests.exceptions.ConnectionError:
        print("   ⚠️  AlertManager unreachable — is port-forward running?")
    except Exception as e:
        print(f"   ⚠️  Alert failed: {e}")

    print("!" * 60 + "\n")
    _last_alert = now

# ─────────────────────────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────────
def _badge(status: str) -> str:
    return {"ANOMALY": "🚨 ANOMALY", "NORMAL": "✅ NORMAL",
            "WARMING_UP": "⏳ WARMING"}.get(status, status)

def _bar(severity: float, w: int = 20) -> str:
    f = int(severity * w)
    return f"[{'█'*f}{'░'*(w-f)}] {severity:.0%}"


# ─────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["train", "run", "compare"],
        default="compare",
    )
    args = parser.parse_args()

    client = PrometheusClient(base_url=PROM_URL)

    if not client.is_healthy():
        print(f"❌ Prometheus unreachable at {PROM_URL}")
        print("   kubectl port-forward -n monitoring "
              "svc/monitoring-kube-prometheus-prometheus 9090:9090")
        sys.exit(1)

    REGISTRY.print_summary()

    if args.mode == "train":
        baseline = client.fetch_baseline(BASELINE_HOURS, BASELINE_STEP_SEC)
        ensemble = EnsembleDetector(voting=VOTING)
        ensemble.fit(baseline)
        ensemble.save()
        print("\n✅ Done. Run with --mode run")

    elif args.mode == "run":
        if not IF_PATH.exists():
            print("❌ No saved models. Run --mode train first.")
            sys.exit(1)
        ensemble = EnsembleDetector.load()
        start_http_server(METRICS_PORT)
        print(f"📡 Metrics → http://localhost:{METRICS_PORT}/metrics")
        live_loop(ensemble, client)

    elif args.mode == "compare":
        if IF_PATH.exists():
            ensemble = EnsembleDetector.load()
        else:
            print("⚠️  No saved models. Training on fresh baseline first...")
            baseline = client.fetch_baseline(BASELINE_HOURS, BASELINE_STEP_SEC)
            ensemble = EnsembleDetector(voting=VOTING)
            ensemble.fit(baseline)
            ensemble.save()
        run_comparison(ensemble, client)


if __name__ == "__main__":
    main()