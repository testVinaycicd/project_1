"""
drift_trigger.py
─────────────────
FastAPI server that receives drift alerts and triggers KFP retraining.

TWO WAYS THIS GETS CALLED:

  1. FROM drift_server.py (Python webhook):
     When the detection loop decides drift is severe enough,
     it POSTs directly to http://localhost:8766/retrain.

  2. FROM AlertManager (Prometheus alert webhook):
     When Prometheus fires an alert (e.g. churn_model_data_drift_score > 0.5),
     AlertManager sends a webhook to http://localhost:8766/alert.
     We receive it, parse it, and trigger retraining.

  Both paths call the same _trigger_kfp_retraining() function.

WHAT TRIGGERS RETRAINING:
  This creates a new KFP pipeline run — the same pipeline you run manually,
  but started programmatically. The pipeline will:
    1. Pull fresh data from MinIO
    2. Validate it
    3. Train a new model
    4. Compare AUC vs production
    5. Promote if better

  After a successful retrain, the reference data is overwritten,
  so the drift baseline resets.

Run:
  cd monitoring/drift
  uvicorn drift_trigger:app --port 8766
"""

import logging
import time
from datetime import datetime
from typing import Optional

import kfp
import yaml
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="Drift Trigger Service")

# Load config
with open("drift_config.yaml") as f:
    CFG = yaml.safe_load(f)

KFP_HOST = CFG["kubeflow"]["host"]
PIPELINE_NAME = CFG["kubeflow"]["pipeline_name"]
RUN_PREFIX = CFG["kubeflow"]["run_name_prefix"]
COOLDOWN_HOURS = CFG["alerting"]["retraining_cooldown_hours"]

_last_trigger_time: float = 0.0
_trigger_history: list = []


# ─────────────────────────────────────────────────────────────────
def _trigger_kfp_retraining(reason: str, drift_score: float) -> dict:
    """
    Create a new Kubeflow pipeline run.

    KFP Client connects to the KFP API server (port-forwarded to 8080).
    It finds the pipeline by name and creates a new run.

    Returns: {"status": "triggered", "run_id": "...", ...}
    """
    global _last_trigger_time

    # Cooldown check
    now = time.time()
    elapsed_h = (now - _last_trigger_time) / 3600
    if _last_trigger_time > 0 and elapsed_h < COOLDOWN_HOURS:
        remaining = COOLDOWN_HOURS - elapsed_h
        log.info(f"In cooldown. {remaining:.1f}h until next allowed retrain.")
        return {
            "status": "cooldown",
            "message": f"Retraining blocked — {remaining:.1f}h cooldown remaining",
            "last_trigger": datetime.fromtimestamp(_last_trigger_time).isoformat(),
        }

    try:
        log.info(f"Connecting to KFP at {KFP_HOST}...")
        client = kfp.Client(host=KFP_HOST)

        # List pipelines and find by name
        pipelines = client.list_pipelines(page_size=50)
        pipeline_id = None
        for p in pipelines.pipelines or []:
            if p.display_name == PIPELINE_NAME or p.name == PIPELINE_NAME:
                pipeline_id = p.pipeline_id
                break

        if not pipeline_id:
            log.error(f"Pipeline '{PIPELINE_NAME}' not found in KFP.")
            log.error("Run pipeline/pipeline.py first to register it.")
            return {
                "status": "error",
                "message": f"Pipeline '{PIPELINE_NAME}' not found",
            }

        run_name = f"{RUN_PREFIX}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        log.info(f"Creating run: {run_name}")

        run = client.create_run_from_pipeline_id(
            pipeline_id=pipeline_id,
            run_name=run_name,
            params={},
        )

        _last_trigger_time = now
        record = {
            "run_id": run.run_id,
            "run_name": run_name,
            "reason": reason,
            "drift_score": drift_score,
            "triggered_at": datetime.fromtimestamp(now).isoformat(),
        }
        _trigger_history.append(record)

        log.info(f"Retraining triggered — Run ID: {run.run_id}")
        return {"status": "triggered", **record}

    except Exception as e:
        log.error(f"Failed to trigger KFP retraining: {e}")
        return {"status": "error", "message": str(e)}


# ─────────────────────────────────────────────────────────────────
# ROUTE 1: Direct webhook from drift_server.py
# ─────────────────────────────────────────────────────────────────
@app.post("/retrain")
async def retrain_endpoint(request: Request):
    """
    Called by drift_server.py when drift score exceeds threshold.

    Expected payload:
    {
        "reason": "drift_detected",
        "overall_drift_score": 0.72,
        "data_drift_score": 0.6,
        "concept_drift_score": 0.15,
        "drifted_features": ["monthly_charges", "age"],
        "timestamp": 1700000000.0
    }
    """
    payload = await request.json()
    drift_score = payload.get("overall_drift_score", 0.0)
    reason = payload.get("reason", "drift_detected")

    log.info(f"Received retrain request — drift_score={drift_score:.3f}")
    log.info(f"Drifted features: {payload.get('drifted_features', [])}")

    result = _trigger_kfp_retraining(reason, drift_score)
    return JSONResponse(content=result)


# ─────────────────────────────────────────────────────────────────
# ROUTE 2: AlertManager webhook
# ─────────────────────────────────────────────────────────────────
@app.post("/alert")
async def alertmanager_webhook(request: Request):
    """
    Receives AlertManager webhook when Prometheus fires a drift alert.

    AlertManager sends a JSON payload with this structure:
    {
        "receiver": "drift-trigger",
        "status": "firing",
        "alerts": [{
            "status": "firing",
            "labels": {
                "alertname": "ModelDriftDetected",
                "severity": "warning"
            },
            "annotations": {
                "summary": "Churn model drift score above threshold",
                "drift_score": "0.72"
            }
        }]
    }
    """
    payload = await request.json()
    status = payload.get("status", "unknown")

    if status != "firing":
        log.info(f"AlertManager notification (status={status}) — no action needed")
        return JSONResponse({"status": "ignored", "reason": f"alert_status={status}"})

    alerts = payload.get("alerts", [])
    if not alerts:
        return JSONResponse({"status": "ignored", "reason": "no_alerts"})

    # Extract drift score from annotations if present
    annotations = alerts[0].get("annotations", {})
    labels = alerts[0].get("labels", {})
    alert_name = labels.get("alertname", "unknown")

    try:
        drift_score = float(annotations.get("drift_score", 0.5))
    except (ValueError, TypeError):
        drift_score = 0.5

    log.info(f"AlertManager fired: {alert_name}  drift_score={drift_score:.3f}")
    result = _trigger_kfp_retraining(f"alertmanager:{alert_name}", drift_score)
    return JSONResponse(content=result)


# ─────────────────────────────────────────────────────────────────
# ROUTE 3: Health + history
# ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "healthy", "kfp_host": KFP_HOST}


@app.get("/history")
def trigger_history():
    """Returns history of all retraining triggers."""
    return {
        "total_triggers": len(_trigger_history),
        "cooldown_hours": COOLDOWN_HOURS,
        "triggers": _trigger_history[-10:],  # Last 10
    }


@app.post("/retrain/manual")
async def manual_retrain():
    """Trigger retraining manually — useful for testing."""
    log.info("Manual retraining triggered via API")
    result = _trigger_kfp_retraining("manual_trigger", 1.0)
    return JSONResponse(content=result)