"""
drift_server.py
────────────────
Two servers in one process:

  PORT 8765 → Prometheus /metrics endpoint  (PULL — Prometheus scrapes us)
  PORT 8766 → AlertManager webhook receiver (drift_trigger receives here)

WHY PROMETHEUS SCRAPES US (not the other way around):
  We are an EXPORTER. We expose /metrics in Prometheus text format.
  Prometheus pulls from us every 30 seconds via a ServiceMonitor.
  We never push to Prometheus. This is the correct pull architecture.

  Our metrics look like this in Prometheus format:
    # HELP churn_model_data_drift_score Fraction of features that drifted
    # TYPE churn_model_data_drift_score gauge
    churn_model_data_drift_score 0.4

  Grafana reads these from Prometheus and shows them on a dashboard.

HOW THE DETECTION LOOP WORKS:
  A background thread runs every check_interval_seconds:
    1. Load reference data from MinIO
    2. Load production log from MinIO
    3. Run Evidently drift detection
    4. Update Prometheus Gauges with results
    5. If overall_drift_score > threshold → POST to drift_trigger webhook

  Prometheus keeps scraping /metrics and sees the updated values.

Run this file to start both servers:
  cd monitoring/drift
  python drift_server.py
"""

import threading
import time
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler

import requests
import yaml

from drift_detector import DriftDetector, DriftCheckResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# PROMETHEUS METRICS STATE
# We use simple module-level variables instead of prometheus_client
# library to avoid port conflicts and keep dependencies minimal.
# Format: Prometheus text exposition format
# ─────────────────────────────────────────────────────────────────
_metrics_state: dict = {
    "data_drift_score":          0.0,
    "drifted_features_count":    0.0,
    "dataset_drift_detected":    0.0,
    "concept_drift_score":       0.0,
    "concept_drift_detected":    0.0,
    "production_samples_count":  0.0,
    "last_check_timestamp":      0.0,
    "overall_drift_score":       0.0,
    "retraining_triggered_total": 0.0,
}
_metrics_lock = threading.Lock()


def _update_metrics(result: DriftCheckResult):
    """Update module-level metrics state from a drift check result."""
    with _metrics_lock:
        _metrics_state["data_drift_score"] = result.data_drift.dataset_drift_score
        _metrics_state["drifted_features_count"] = result.data_drift.drifted_features_count
        _metrics_state["dataset_drift_detected"] = 1.0 if result.data_drift.dataset_drifted else 0.0
        _metrics_state["concept_drift_score"] = result.concept_drift.prediction_shift
        _metrics_state["concept_drift_detected"] = 1.0 if result.concept_drift.concept_drifted else 0.0
        _metrics_state["production_samples_count"] = result.data_drift.production_samples
        _metrics_state["last_check_timestamp"] = result.timestamp
        _metrics_state["overall_drift_score"] = result.overall_drift_score


def _render_metrics(prefix: str) -> str:
    """
    Render current metric state as Prometheus text format.

    Prometheus text format:
      # HELP metric_name Description
      # TYPE metric_name gauge
      metric_name value
    """
    lines = []
    descriptions = {
        "data_drift_score":           "Fraction of features with statistical drift (0=none, 1=all)",
        "drifted_features_count":     "Number of features with detected drift",
        "dataset_drift_detected":     "1 if dataset-level drift detected, 0 otherwise",
        "concept_drift_score":        "Absolute shift in mean prediction probability",
        "concept_drift_detected":     "1 if concept drift detected, 0 otherwise",
        "production_samples_count":   "Number of production prediction requests logged",
        "last_check_timestamp":       "Unix timestamp of last drift check",
        "overall_drift_score":        "Combined weighted drift score (0=clean, 1=maximum drift)",
        "retraining_triggered_total": "Total number of retraining runs triggered by drift",
    }

    with _metrics_lock:
        for name, value in _metrics_state.items():
            full_name = f"{prefix}_{name}"
            desc = descriptions.get(name, name)
            lines.append(f"# HELP {full_name} {desc}")
            lines.append(f"# TYPE {full_name} gauge")
            lines.append(f"{full_name} {value}")

    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────
# HTTP SERVER — exposes /metrics for Prometheus to scrape
# ─────────────────────────────────────────────────────────────────
class MetricsHandler(BaseHTTPRequestHandler):
    """
    Minimal HTTP handler.
    GET /metrics → Prometheus text format
    GET /health  → 200 OK (for readiness probes)
    """
    metric_prefix: str = "churn_model"

    def do_GET(self):
        if self.path == "/metrics":
            body = _render_metrics(self.metric_prefix).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == "/health":
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(body)

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        # Suppress default request logging — too noisy
        pass


# ─────────────────────────────────────────────────────────────────
# TRIGGER — send webhook to drift_trigger.py
# ─────────────────────────────────────────────────────────────────
_last_retrain_time: float = 0.0


def _maybe_trigger_retraining(result: DriftCheckResult, cfg: dict):
    """
    If drift is above threshold and cooldown has passed,
    POST to drift_trigger.py to start a KFP retraining run.
    """
    global _last_retrain_time

    if not result.retraining_recommended:
        return

    cooldown_hours = cfg["alerting"]["retraining_cooldown_hours"]
    cooldown_sec = cooldown_hours * 3600
    now = time.time()

    if (now - _last_retrain_time) < cooldown_sec:
        remaining_h = (cooldown_sec - (now - _last_retrain_time)) / 3600
        log.info(f"Retraining needed but in cooldown ({remaining_h:.1f}h remaining)")
        return

    webhook_url = cfg["alerting"]["trigger_webhook_url"]
    payload = {
        "reason": "drift_detected",
        "overall_drift_score": result.overall_drift_score,
        "data_drift_score": result.data_drift.dataset_drift_score,
        "concept_drift_score": result.concept_drift.prediction_shift,
        "drifted_features": [
            f.name for f in result.data_drift.feature_results if f.drifted
        ],
        "timestamp": result.timestamp,
    }

    try:
        log.info(f"Triggering retraining — drift_score={result.overall_drift_score:.3f}")
        resp = requests.post(webhook_url, json=payload, timeout=10)
        if resp.status_code == 200:
            log.info("Retraining triggered successfully")
            _last_retrain_time = now
            with _metrics_lock:
                _metrics_state["retraining_triggered_total"] += 1
        else:
            log.warning(f"Trigger webhook returned {resp.status_code}: {resp.text}")
    except requests.exceptions.ConnectionError:
        log.warning(f"drift_trigger.py not reachable at {webhook_url}")
        log.warning("Start drift_trigger.py to enable automatic retraining")


# ─────────────────────────────────────────────────────────────────
# DETECTION LOOP — background thread
# ─────────────────────────────────────────────────────────────────
def _detection_loop(detector: DriftDetector, cfg: dict, interval: int):
    """
    Runs in a background thread.
    Wakes up every `interval` seconds, runs drift check,
    updates metrics, optionally triggers retraining.
    """
    log.info(f"Detection loop started — checking every {interval}s")

    while True:
        try:
            result = detector.run_check()
            _update_metrics(result)
            _maybe_trigger_retraining(result, cfg)

        except Exception as e:
            log.error(f"Detection loop error: {e}", exc_info=True)

        time.sleep(interval)


# ─────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────
def main():
    cfg = yaml.safe_load(open("drift_config.yaml"))
    prom_cfg = cfg["prometheus"]
    det_cfg = cfg["detection"]

    metrics_port = prom_cfg["metrics_port"]
    prefix = prom_cfg["metric_prefix"]
    interval = det_cfg["check_interval_seconds"]

    MetricsHandler.metric_prefix = prefix

    detector = DriftDetector("drift_config.yaml")

    # Start detection loop in background thread
    thread = threading.Thread(
        target=_detection_loop,
        args=(detector, cfg, interval),
        daemon=True,
    )
    thread.start()

    # Run one immediate check so metrics are populated at startup
    log.info("Running initial drift check...")
    try:
        result = detector.run_check()
        _update_metrics(result)
    except Exception as e:
        log.warning(f"Initial check failed: {e}")

    # Start Prometheus metrics server (blocking)
    log.info(f"Drift metrics server → http://localhost:{metrics_port}/metrics")
    log.info(f"Health check         → http://localhost:{metrics_port}/health")
    log.info(f"Detection interval   → every {interval}s")

    server = HTTPServer(("0.0.0.0", metrics_port), MetricsHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Stopped")


if __name__ == "__main__":
    main()