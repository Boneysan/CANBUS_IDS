# CAN-IDS Configuration Reference

**Last Updated**: March 1, 2026  
**Config files**: `config/can_ids.yaml` (desktop/laptop), `config/can_ids_rpi4.yaml` (Raspberry Pi 4)

This document describes every configuration parameter. Parameters are grouped by section.

---

## Quick Start

```yaml
# Minimal working config — rules-only detection
log_level: INFO
interface: can0
bustype: socketcan

detection_modes:
  - rule_based

rules_file: config/rules.yaml

alerts:
  log_file: logs/alerts.json
  console_output: true
```

---

## System Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_level` | string | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `interface` | string | `can0` | CAN interface name (overridden by `-i` CLI flag) |
| `bustype` | string | `socketcan` | python-can bus type. Use `socketcan` on Linux |

```yaml
log_level: INFO
interface: can0
bustype: socketcan
```

---

## Detection Modes

Controls which detection engines are active. This is the most important section.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `detection_modes` | list | `[rule_based]` | Active detection engines |
| `rules_file` | string | `config/rules.yaml` | Path to YAML rules file |
| `ml_threshold` | float | `0.75` | ML anomaly score threshold (0.0–1.0) |

### Available Modes

| Mode | Engine | Throughput | Status |
|------|--------|-----------|--------|
| `rule_based` | Rule engine (18 rule types) | ~759 msg/s | **Active** — always include this |
| `ml_based` | IsolationForest ensemble | ~15 msg/s | **Deprecated** for real-time; use for offline analysis only |

```yaml
detection_modes:
  - rule_based
  # - ml_based  # Deprecated for real-time — use decision_tree instead

rules_file: config/rules_adaptive.yaml
ml_threshold: 0.75
```

> **Note:** The Decision Tree (Stage 3) and Pre-Filter (Stage 1) have their own
> config sections and are **not** controlled by `detection_modes`.

---

## Decision Tree (Stage 3 ML)

The Decision Tree classifier replaced IsolationForest for real-time ML detection. It runs as Stage 3 in the hierarchical pipeline, processing only messages that survive Stages 1+2.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `decision_tree.enabled` | bool | `true` | Enable Stage 3 ML detection |
| `decision_tree.model_path` | string | `data/models/decision_tree.pkl` | Path to trained model |
| `decision_tree.visualization_path` | string | `data/models/decision_tree_rules.txt` | Human-readable tree rules |

```yaml
decision_tree:
  enabled: true
  model_path: data/models/decision_tree.pkl
  visualization_path: data/models/decision_tree_rules.txt
```

> **Important:** The `decision_tree.pkl` file is **not shipped** in the repository.
> You must train it before Stage 3 will activate:
> ```bash
> python scripts/train_decision_tree.py --synthetic
> ```
> Without the model file, Stage 3 silently falls back to Stages 1+2 only.

---

## Pre-Filter (Stage 1)

The fast statistical pre-filter eliminates 80–95% of known-benign traffic before rule evaluation, dramatically reducing CPU load.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prefilter.enabled` | bool | `true` | Enable Stage 1 pre-filtering |
| `prefilter.timing_tolerance` | float | `0.3` | Timing tolerance (0.3 = ±30%) |
| `prefilter.known_good_ids` | list | `[]` | Known-good CAN IDs (hex). If empty, extracted from rules at init |

```yaml
prefilter:
  enabled: true
  timing_tolerance: 0.3
  known_good_ids:
    - 0x0C1
    - 0x0C5
    - 0x120
    # ... add your vehicle's legitimate CAN IDs
```

**Tuning tips:**
- If you see false negatives (missed attacks), reduce `timing_tolerance` (e.g., 0.2)
- If throughput is too low, add more `known_good_ids` so the pre-filter skips more traffic
- Set `enabled: false` to disable and send all traffic through the full pipeline

---

## IsolationForest ML Model (Legacy)

These settings control the **original** IsolationForest detection path. Only active when `ml_based` is in `detection_modes` (currently commented out).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ml_model.path` | string | `data/models/aggressive_load_shedding.joblib` | Path to IsolationForest model |
| `ml_model.contamination` | float | `0.20` | Expected anomaly proportion (0.01–0.5) |
| `ml_model.retrain_interval` | int | `86400` | Auto-retrain interval in seconds (24h) |
| `ml_model.auto_update` | bool | `false` | Enable automatic model retraining |
| `ml_model.feature_cache_size` | int | `100` | Feature extraction cache size (Pi 4 only) |

```yaml
ml_model:
  path: data/models/aggressive_load_shedding.joblib
  contamination: 0.20
  retrain_interval: 86400
  auto_update: false
```

### Available Pre-Trained Models

All shipped in `data/models/` (IsolationForest-based):

| Model | Size | Best For |
|-------|------|----------|
| `aggressive_load_shedding.joblib` | 1.3 MB | Default — multi-stage pipeline, load shedding |
| `adaptive_load_shedding.joblib` | 1.3 MB | Alternative load shedding configuration |
| `full_pipeline.joblib` | 1.3 MB | Complete feature pipeline |
| `enhanced_detector.joblib` | 356 KB | RandomForest-based, 13 CAN features |
| `can_feature_engineer.joblib` | 21 KB | Feature engineering only |
| `adaptive_weighted_detector.joblib` | 618 B | Best lab accuracy (97.20% recall) |

---

## Enhanced ML Detection (Multi-Stage Pipeline)

Advanced ML pipeline with adaptive gating and load shedding. Requires `ml_based` in `detection_modes`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ml_detection.enable_multistage` | bool | `true` | Enable multi-stage ML pipeline |
| `ml_detection.multistage.models_dir` | string | `models` | Directory for stage models |
| `ml_detection.multistage.enable_adaptive_gating` | bool | `true` | Dynamic threshold adjustment |
| `ml_detection.multistage.enable_load_shedding` | bool | `true` | Drop low-priority analysis under load |
| `ml_detection.multistage.max_stage3_load` | float | `0.15` | Max CPU fraction for deep analysis |
| `ml_detection.multistage.enable_vehicle_calibration` | bool | `false` | Per-vehicle threshold calibration |
| `ml_detection.multistage.stage1_threshold` | float | `0.0` | Stage 1 anomaly threshold |
| `ml_detection.multistage.stage2_threshold` | float | `0.5` | Stage 2 anomaly threshold |
| `ml_detection.multistage.stage3_threshold` | float | `0.7` | Stage 3 anomaly threshold |
| `ml_detection.multistage.enable_performance_monitoring` | bool | `true` | Track stage metrics |
| `ml_detection.multistage.stats_window_size` | int | `1000` | Rolling stats window size |
| `ml_detection.fallback.model_type` | string | `isolation_forest` | Fallback model type |
| `ml_detection.fallback.contamination` | float | `0.02` | Fallback contamination rate |
| `ml_detection.fallback.feature_window` | int | `100` | Fallback feature window |

```yaml
ml_detection:
  enable_multistage: true
  multistage:
    models_dir: "models"
    enable_adaptive_gating: true
    enable_load_shedding: true
    max_stage3_load: 0.15
    stage1_threshold: 0.0
    stage2_threshold: 0.5
    stage3_threshold: 0.7
  fallback:
    model_type: "isolation_forest"
    contamination: 0.02
    feature_window: 100
```

---

## Alert Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alerts.log_file` | string | `logs/alerts.json` | JSON alert log path |
| `alerts.console_output` | bool | `true` | Print alerts to stdout |
| `alerts.email_alerts` | bool | `false` | Enable email notifications |
| `alerts.email_recipients` | list | `[]` | Email addresses for notifications |
| `alerts.rate_limit` | int | `10` | Maximum alerts per second |

```yaml
alerts:
  log_file: logs/alerts.json
  console_output: true
  email_alerts: false
  email_recipients:
    - security@example.com
  rate_limit: 10
```

---

## Capture Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `capture.buffer_size` | int | `1000` | Message ring buffer size |
| `capture.pcap_enabled` | bool | `true` | Enable packet capture to disk |
| `capture.pcap_directory` | string | `data/raw/` | Capture output directory |
| `capture.pcap_rotation_size` | string | `100MB` | Rotate capture files at this size |

```yaml
capture:
  buffer_size: 1000
  pcap_enabled: true
  pcap_directory: data/raw/
  pcap_rotation_size: 100MB
```

**Pi 4 adjustments:** Reduce `buffer_size` to 500 and `pcap_rotation_size` to 50MB to save memory and reduce SD card wear.

---

## Performance Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `performance.max_cpu_percent` | int | `80` | CPU usage ceiling (%) |
| `performance.max_memory_mb` | int | `500` | Memory usage ceiling (MB) |
| `performance.processing_threads` | int | `2` | Worker threads for message processing |

```yaml
performance:
  max_cpu_percent: 80
  max_memory_mb: 500
  processing_threads: 2
```

**Pi 4 adjustments:**
```yaml
performance:
  max_cpu_percent: 70     # Conservative for thermal management
  max_memory_mb: 300      # For 2GB Pi 4 models
  processing_threads: 1   # Single thread to reduce heat
  enable_turbo_boost: false
```

---

## Raspberry Pi 4 Settings

Only in `config/can_ids_rpi4.yaml`. Ignored on desktop/laptop.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `raspberry_pi.enable_hardware_watchdog` | bool | `true` | Hardware watchdog timer |
| `raspberry_pi.thermal_throttling_temp` | int | `70` | Throttle above this °C |
| `raspberry_pi.log_rotation_size` | string | `10MB` | Log rotation threshold |
| `raspberry_pi.use_tmpfs_logs` | bool | `true` | Write logs to RAM (saves SD card) |

```yaml
raspberry_pi:
  enable_hardware_watchdog: true
  thermal_throttling_temp: 70
  log_rotation_size: 10MB
  use_tmpfs_logs: true
```

---

## CLI Arguments

Command-line flags override config file values.

```
python main.py [OPTIONS]

Required (pick one):
  -i, --interface IFACE     Live CAN monitoring (e.g., can0, vcan0)
  --mode replay             Replay a capture file (requires --file)
  --test-interface IFACE    Test CAN interface connectivity and exit
  --monitor-traffic IFACE   Monitor traffic without detection (with --duration)

Optional:
  --file PATH               PCAP or candump log file (for --mode replay)
  --config PATH             Config file (default: config/can_ids.yaml)
  --log-level LEVEL         DEBUG, INFO, WARNING, ERROR (default: INFO)
  --duration SECONDS        Monitoring duration (default: 30, for --monitor-traffic)
  --version                 Show version and exit
```

### Examples

```bash
# Live monitoring
python main.py -i can0

# Pi 4 with dedicated config
python main.py -i can0 --config config/can_ids_rpi4.yaml

# Replay a candump log file
python main.py --mode replay --file data/raw/traffic.log

# Test interface connectivity
python main.py --test-interface can0

# Monitor traffic for 60 seconds (no detection)
python main.py --monitor-traffic can0 --duration 60

# Debug logging
python main.py -i vcan0 --log-level DEBUG
```

---

## Config File Comparison

| Parameter | `can_ids.yaml` | `can_ids_rpi4.yaml` | Why Different |
|-----------|----------------|---------------------|---------------|
| `capture.buffer_size` | 1000 | 500 | Less RAM on Pi |
| `capture.pcap_rotation_size` | 100MB | 50MB | SD card wear |
| `performance.max_cpu_percent` | 80 | 70 | Thermal management |
| `performance.max_memory_mb` | 500 | 300 | 2GB Pi models |
| `performance.processing_threads` | 2 | 1 | Reduce heat |
| `rules_file` | `rules_adaptive.yaml` | `rules.yaml` | Adaptive rules tuned for desktop testing |

---

## Rule Files

Multiple rule files are available. Point `rules_file` at the one you want:

| File | Description |
|------|-------------|
| `config/rules.yaml` | Hand-written production rules |
| `config/rules_adaptive.yaml` | Auto-generated from baseline data (recommended for testing) |
| `config/example_rules.yaml` | Templates and examples |
| `config/rules_fuzzing_only.yaml` | Fuzzing-focused rules only |
| `config/rules_timing_only.yaml` | Timing rules only |
| `config/rules_timing_1sigma.yaml` | Tight timing thresholds (more alerts) |
| `config/rules_timing_2sigma.yaml` | Loose timing thresholds (fewer alerts) |

To generate vehicle-specific rules from your own baseline data:

```bash
python scripts/generate_rules_from_baseline.py \
  --input test_data/attack-free-1.csv test_data/attack-free-2.csv \
  --output config/rules_my_vehicle.yaml
```

See [rules_guide.md](rules_guide.md) for how to write custom rules covering all 18 rule types.

---

## Model Inventory

### Files Shipped in Repository (`data/models/`)

| File | Size | Type | Notes |
|------|------|------|-------|
| `aggressive_load_shedding.joblib` | 1.3 MB | IsolationForest | Config default for legacy `ml_based` path |
| `adaptive_load_shedding.joblib` | 1.3 MB | IsolationForest | Alternative multi-stage |
| `full_pipeline.joblib` | 1.3 MB | IsolationForest | Complete pipeline |
| `enhanced_detector.joblib` | 356 KB | RandomForest | 13 CAN-specific features |
| `can_feature_engineer.joblib` | 21 KB | Feature engineering | CAN feature transformer |
| `adaptive_weighted_detector.joblib` | 618 B | Weighted ensemble | Best lab accuracy |

### Files NOT Shipped (Must Be Trained Locally)

| File | How to Create | Used By |
|------|---------------|---------|
| `decision_tree.pkl` | `python scripts/train_decision_tree.py --synthetic` | Stage 3 Decision Tree |
| `decision_tree_rules.txt` | Generated alongside `decision_tree.pkl` | Human-readable tree visualization |

### Files Referenced in Older Docs (Do Not Exist)

These models were validated in the Vehicle_Models research project and are mentioned in some development logs and test scripts, but were never copied to this repository:

- `ensemble_detector.joblib`, `improved_isolation_forest.joblib`, `improved_svm.joblib`
- `adaptive_only.joblib`, `ensemble_impala.joblib`, `ensemble_traverse.joblib`

---

## Further Reading

- [Rules Guide](rules_guide.md) — Write custom detection rules (all 18 types)
- [Implementation Status](../implementation/IMPLEMENTATION_STATUS.md) — Current feature status
- [Raspberry Pi Deployment](../deployment/RASPBERRY_PI_DEPLOYMENT_GUIDE.md) — Pi 4 setup
- [ML Optimization](../ml/ML_OPTIMIZATION_GUIDE.md) — ML performance strategies
