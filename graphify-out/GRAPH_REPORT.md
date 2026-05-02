# Graph Report - project_1  (2026-05-02)

## Corpus Check
- 24 files · ~19,054 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 262 nodes · 366 edges · 27 communities detected
- Extraction: 78% EXTRACTED · 22% INFERRED · 0% AMBIGUOUS · INFERRED: 80 edges (avg confidence: 0.68)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]

## God Nodes (most connected - your core abstractions)
1. `DriftDetector` - 19 edges
2. `PrometheusClient` - 15 edges
3. `LSTMDetector` - 15 edges
4. `IsolationForestDetector` - 13 edges
5. `EnsembleDetector` - 11 edges
6. `live_loop()` - 11 edges
7. `main()` - 11 edges
8. `run_comparison()` - 10 edges
9. `DevopsAiProject` - 10 edges
10. `MetricRegistry` - 9 edges

## Surprising Connections (you probably didn't know these)
- `live_loop()` --calls--> `from_training_stats()`  [INFERRED]
  Ensemble_engine.py → drift_detector.py
- `train_op()` --calls--> `run()`  [INFERRED]
  train_component.py → devops_ai_project/src/devops_ai_project/main.py
- `PrometheusClient` --calls--> `main()`  [INFERRED]
  prom_client.py → Ensemble_engine.py
- `train_op()` --calls--> `run()`  [INFERRED]
  pipeline.py → devops_ai_project/src/devops_ai_project/main.py
- `evaluate_op()` --calls--> `load()`  [INFERRED]
  pipeline.py → Ensemble_engine.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.1
Nodes (16): EnsembleDetector, ensemble_engine.py ─────────────────── Orchestrates everything. The only file, Run 4 scenarios showing how IF and LSTM behave differently.     This is your in, Run 4 scenarios showing how IF and LSTM behave differently.     This is your in, Holds both detectors. Combines their predictions.      Voting strategies:, Holds both detectors. Combines their predictions.      Voting strategies:, IsolationForestDetector, iforest_detector.py ──────────────────── Isolation Forest anomaly detector. Dr (+8 more)

### Community 1 - "Community 1"
Cohesion: 0.12
Nodes (15): main(), query(), check_metrics.py ───────────────── Run this BEFORE training the anomaly detect, Returns (has_data, value), _badge(), _bar(), _fire_alert(), live_loop() (+7 more)

### Community 2 - "Community 2"
Cohesion: 0.1
Nodes (20): BaseHTTPRequestHandler, DriftDetector, reference_means   — per-feature mean from training   (from iforest.training_stat, Save actual baseline samples for KS-test.         Call this once after training, Feed one snapshot into the rolling buffer.         Call every poll — same caden, Push drift metrics to Prometheus gauges.          gauge_psi          — ml_drif, Human-readable drift report for the terminal., Monitors each feature for distribution drift vs. training baseline.      Usage (+12 more)

### Community 3 - "Community 3"
Cohesion: 0.12
Nodes (16): ConceptDriftResult, DataDriftResult, DriftCheckResult, DriftDetector, FeatureDriftResult, load_config(), MinIOClient, drift_detector.py ────────────────── Core drift detection logic using Evidentl (+8 more)

### Community 4 - "Community 4"
Cohesion: 0.08
Nodes (19): CustomerData, predict(), FastAPI inference server for churn prediction, BaseModel, BaseTool, MyCustomTool, MyCustomToolInput, Input schema for MyCustomTool. (+11 more)

### Community 5 - "Community 5"
Cohesion: 0.11
Nodes (6): MetricRegistry, MetricSpec, metric_registry.py ─────────────────── Loads metrics_config.yaml into typed da, Central registry. Load once, use everywhere.          from metric_registry imp, Synthetic normal data — used when Prometheus has < 24h of history.         Colu, Generate n synthetic normal samples for fallback baseline.

### Community 6 - "Community 6"
Cohesion: 0.18
Nodes (12): crew(), DevopsAiProject, Train the crew for a given number of iterations., Replay the crew execution from a specific task., Test the crew execution and returns the results., Run the crew with trigger payload., replay(), run() (+4 more)

### Community 7 - "Community 7"
Cohesion: 0.15
Nodes (8): load(), LSTMAutoencoder, lstm_detector.py ───────────────── LSTM Autoencoder for sequential anomaly det, x: (batch, seq_len, n_features) → reconstruction: same shape, seq_len:             Number of timesteps in one training/inference window., Sliding window over 2D array.         (N, features) → (N - seq_len + 1, seq_len, Train on baseline DataFrame., Encoder-Decoder LSTM.      Encoder:       (batch, seq_len, n_features)  →  LS

### Community 8 - "Community 8"
Cohesion: 0.19
Nodes (12): generate_clean(), generate_concept_drift(), generate_data_drift(), get_minio_client(), load_config(), main(), simulate_drift.py ────────────────── Generates synthetic production prediction, Features look similar to training, but the model's predictions     have shifted (+4 more)

### Community 9 - "Community 9"
Cohesion: 0.19
Nodes (11): alertmanager_webhook(), manual_retrain(), drift_trigger.py ───────────────── FastAPI server that receives drift alerts a, Called by drift_server.py when drift score exceeds threshold.      Expected pa, Receives AlertManager webhook when Prometheus fires a drift alert.      AlertM, Returns history of all retraining triggers., Trigger retraining manually — useful for testing., Create a new Kubeflow pipeline run.      KFP Client connects to the KFP API se (+3 more)

### Community 10 - "Community 10"
Cohesion: 0.27
Nodes (10): churn_pipeline(), evaluate_op(), generate_data_op(), get_production_metric_op(), promote_model_op(), Churn Prediction — Kubeflow Pipeline (KFP v2) Updated for Day 4: adds save_refe, Save the training data + model predictions to MinIO as parquet.     This become, save_reference_data_op() (+2 more)

### Community 11 - "Community 11"
Cohesion: 0.18
Nodes (7): from_training_stats(), _ks_test(), _psi_score(), drift_detector.py ────────────────── Model drift monitoring. Sits alongside th, Compute drift report for all features.          Returns:         {, Population Stability Index between a reference and current distribution., Kolmogorov-Smirnov two-sample test.     Returns (statistic, p_value).     p_va

### Community 12 - "Community 12"
Cohesion: 1.0
Nodes (1): seed_minio.py ────────────── Run this ONCE from your laptop before submitting

### Community 13 - "Community 13"
Cohesion: 1.0
Nodes (1): Churn Model Training Script ──────────────────────────── Called by train_op in t

### Community 14 - "Community 14"
Cohesion: 1.0
Nodes (1): Generate synthetic churn dataset and upload to MinIO. Run this ONCE locally to s

### Community 16 - "Community 16"
Cohesion: 1.0
Nodes (1): The PromQL query for this metric.          Because we use Prometheus recording

### Community 17 - "Community 17"
Cohesion: 1.0
Nodes (1): name -> promql mapping.         Because promql == name for recording rules, thi

### Community 18 - "Community 18"
Cohesion: 1.0
Nodes (1): Build from iforest.training_stats — no extra args needed.          training_st

### Community 19 - "Community 19"
Cohesion: 1.0
Nodes (1): Load reference from saved JSON (includes actual baseline samples).

### Community 20 - "Community 20"
Cohesion: 1.0
Nodes (1): How many snapshots we've collected so far.

### Community 21 - "Community 21"
Cohesion: 1.0
Nodes (1): Need at least 10 snapshots before drift is meaningful.

### Community 23 - "Community 23"
Cohesion: 1.0
Nodes (1): Creates the DevopsAiProject crew

### Community 25 - "Community 25"
Cohesion: 1.0
Nodes (1): Central registry. Load once, use everywhere.          from metric_registry imp

### Community 26 - "Community 26"
Cohesion: 1.0
Nodes (1): name -> promql mapping.         Because promql == name for recording rules, thi

### Community 27 - "Community 27"
Cohesion: 1.0
Nodes (1): Synthetic normal data — used when Prometheus has < 24h of history.         Colu

### Community 28 - "Community 28"
Cohesion: 1.0
Nodes (1): Range query — time series of values over last N hours.         Recording rules

### Community 29 - "Community 29"
Cohesion: 1.0
Nodes (1): Verify recording rules are working — prints current values.

## Knowledge Gaps
- **78 isolated node(s):** `metric_registry.py ─────────────────── Loads metrics_config.yaml into typed da`, `The PromQL query for this metric.          Because we use Prometheus recording`, `Generate n synthetic normal samples for fallback baseline.`, `Central registry. Load once, use everywhere.          from metric_registry imp`, `name -> promql mapping.         Because promql == name for recording rules, thi` (+73 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 12`** (2 nodes): `seed_minio.py`, `seed_minio.py ────────────── Run this ONCE from your laptop before submitting`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 13`** (2 nodes): `Churn Model Training Script ──────────────────────────── Called by train_op in t`, `train.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 14`** (2 nodes): `Generate synthetic churn dataset and upload to MinIO. Run this ONCE locally to s`, `generate_data.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 16`** (1 nodes): `The PromQL query for this metric.          Because we use Prometheus recording`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 17`** (1 nodes): `name -> promql mapping.         Because promql == name for recording rules, thi`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 18`** (1 nodes): `Build from iforest.training_stats — no extra args needed.          training_st`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 19`** (1 nodes): `Load reference from saved JSON (includes actual baseline samples).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 20`** (1 nodes): `How many snapshots we've collected so far.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 21`** (1 nodes): `Need at least 10 snapshots before drift is meaningful.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 23`** (1 nodes): `Creates the DevopsAiProject crew`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 25`** (1 nodes): `Central registry. Load once, use everywhere.          from metric_registry imp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 26`** (1 nodes): `name -> promql mapping.         Because promql == name for recording rules, thi`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 27`** (1 nodes): `Synthetic normal data — used when Prometheus has < 24h of history.         Colu`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 28`** (1 nodes): `Range query — time series of values over last N hours.         Recording rules`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 29`** (1 nodes): `Verify recording rules are working — prints current values.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `DriftDetector` connect `Community 2` to `Community 0`, `Community 11`?**
  _High betweenness centrality (0.203) - this node is a cross-community bridge._
- **Why does `EnsembleDetector` connect `Community 0` to `Community 1`, `Community 2`, `Community 4`?**
  _High betweenness centrality (0.146) - this node is a cross-community bridge._
- **Why does `LSTMDetector` connect `Community 0` to `Community 7`?**
  _High betweenness centrality (0.139) - this node is a cross-community bridge._
- **Are the 11 inferred relationships involving `DriftDetector` (e.g. with `EnsembleDetector` and `ensemble_engine.py ─────────────────── Orchestrates everything. The only file`) actually correct?**
  _`DriftDetector` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `PrometheusClient` (e.g. with `EnsembleDetector` and `ensemble_engine.py ─────────────────── Orchestrates everything. The only file`) actually correct?**
  _`PrometheusClient` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `LSTMDetector` (e.g. with `EnsembleDetector` and `ensemble_engine.py ─────────────────── Orchestrates everything. The only file`) actually correct?**
  _`LSTMDetector` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `IsolationForestDetector` (e.g. with `EnsembleDetector` and `ensemble_engine.py ─────────────────── Orchestrates everything. The only file`) actually correct?**
  _`IsolationForestDetector` has 7 INFERRED edges - model-reasoned connections that need verification._