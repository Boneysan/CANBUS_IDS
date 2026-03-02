# CAN-IDS Project Context & Status

**Project:** Controller Area Network Intrusion Detection System (CAN-IDS)  
**Platform:** Raspberry Pi 4 Model B  
**Language:** Python 3.11.2  
**Last Updated:** March 1, 2026  
**Status:** ✅ **FUNCTIONAL - PRODUCTION CANDIDATE**

---

## Quick Status Summary

### ✅ What's Working

- **3-stage hierarchical detection pipeline** — fully implemented (Dec 14, 2025)
- **Stage 1 Pre-Filter** (`prefilter.py`) — eliminates 80–95% of benign traffic before rule evaluation
- **Stage 2 Rule Engine** (`rule_engine.py`) — 18/18 rule types, CAN ID hash indexing, priority-based early exit; ~7,002 msg/s peak / ~759 msg/s measured
- **Stage 3 Decision Tree ML** (`decision_tree_detector.py`) — 8,000+ msg/s, replaces IsolationForest for real-time detection
- **False positive rate reduced** from 81.7% → **8.43%** via Tier 3 payload repetition analysis (Dec 14)
- **Recall:** 94.76% across all attack types
- System processes offline datasets and live SocketCAN interfaces
- Comprehensive testing framework: 61/61 rule engine tests passing

### ⚠️ Remaining Gaps

- `decision_tree.pkl` not committed — must be trained locally before Stage 3 activates (`python scripts/train_decision_tree.py --synthetic`)
- Rule thresholds are still generic — vehicle-specific baseline extraction would reduce 8.43% FPR further
- End-to-end Pi 4 benchmark of the full 3-stage pipeline not yet measured (Stages 1+2+3 combined)
- IsolationForest (`ml_based`) commented out — intentional; 15 msg/s is too slow for real-time

### 💡 Rule Tuning Strategy

Rules use generic thresholds. For your specific vehicle, extract baselines from normal traffic:

```bash
python scripts/generate_rules_from_baseline.py \
  --input test_data/attack-free-1.csv test_data/attack-free-2.csv \
  --output config/rules_my_vehicle.yaml
```

The script calculates per-CAN-ID `interval_mean`, `interval_std`, and `freq_last_1s` statistics and generates thresholds at mean ± 3σ (covers 99.7% of normal traffic). This is expected to push FPR below 5%.

---

## Project Structure & Key Files

### Documentation Files

#### Guides (this directory)

| File | Purpose |
|------|---------|
| **GETTING_STARTED.md** | End-to-end setup guide — install, vCAN test, Pi 4 deployment |
| **configuration.md** | Complete parameter reference for all config sections and CLI args |
| **rules_guide.md** | All 18 rule types with YAML examples |
| **PROJECT_CONTEXT.md** | This file — overall status and context |
| **traffic_monitoring_guide.md** | `--test-interface` and `--monitor-traffic` usage |
| **can_packet_format.md** | CAN 2.0A/2.0B frame structure reference |

#### Implementation Docs (`docs/implementation/`)

| File | Purpose |
|------|---------|
| **IMPLEMENTATION_STATUS.md** | Authoritative status: 3-stage pipeline, model inventory, performance table |
| **UNIMPLEMENTED_FEATURES.md** | All 18 rule types with method names and line numbers |

#### Development Logs (`docs/development_logs/`)

| File | Purpose |
|------|---------|
| **CHANGELOG.md** | Consolidated timeline of all development sessions |
| **DEVELOPMENT_SUMMARY_DEC14_2025.md** | Dec 14 session — Tier 3, Decision Tree, hash indexing |

#### Performance Plans (`docs/performance/`)

These were future plans as of Dec 7. Most have been implemented:

| File | Status |
|------|--------|
| **BUILD_PLAN_7000_MSG_SEC.md** | ✅ Implemented Dec 11–14 |
| **IMPROVEMENT_ROADMAP.md** | ✅ Mostly implemented (hash indexing, DT ML, pre-filter) |
| **ACHIEVING_7000_MSG_PER_SEC.md** | ✅ Achieved — DT 8,000+ msg/s, hash indexing 7,002 msg/s |

### Source Code Structure

```
src/
├── capture/
│   ├── can_sniffer.py              # Real-time SocketCAN monitoring
│   └── pcap_reader.py              # Offline PCAP/CSV analysis
├── detection/
│   ├── rule_engine.py              # Stage 2 — 18 rule types, hash indexing, priority exit
│   ├── prefilter.py                # Stage 1 — fast statistical pre-filter (NEW Dec 14)
│   ├── decision_tree_detector.py   # Stage 3 — Decision Tree ML, 8,000+ msg/s (NEW Dec 14)
│   ├── ml_detector.py              # Legacy IsolationForest — code present, not in real-time path
│   └── multistage_detector.py      # Enhanced multi-stage pipeline framework
├── preprocessing/
│   ├── feature_extractor.py        # 50 basic + 8 enhanced research features
│   └── normalizer.py               # Data normalization
└── alerts/
    ├── alert_manager.py            # Alert coordination and deduplication
    └── notifiers.py                # Notification channels

scripts/
├── train_decision_tree.py          # Train Stage 3 model (--synthetic or --vehicle-models)
├── generate_rules_from_baseline.py # Auto-generate vehicle-specific rule thresholds
├── convert_candump.py              # Convert candump logs to CSV
├── comprehensive_test.py           # Main testing framework
├── benchmark.py                    # Performance benchmarking
└── batch_test_set01.sh             # Batch testing script

config/
├── can_ids.yaml                    # Main config (Stage 1+2+3 all enabled)
├── can_ids_rpi4.yaml               # Pi 4 optimized (reduced buffers, 1 thread)
├── rules_adaptive.yaml             # Auto-generated adaptive rules (recommended)
└── rules.yaml                      # Hand-written production rules

data/
└── models/                         # ML models
    ├── aggressive_load_shedding.joblib  # IsolationForest (legacy, 1.3 MB)
    ├── adaptive_load_shedding.joblib    # IsolationForest (legacy, 1.3 MB)
    ├── full_pipeline.joblib             # IsolationForest (legacy, 1.3 MB)
    ├── enhanced_detector.joblib         # RandomForest (356 KB)
    ├── can_feature_engineer.joblib      # Feature engineering (21 KB)
    ├── adaptive_weighted_detector.joblib # Best lab accuracy (618 B)
    └── decision_tree.pkl                # Stage 3 model — NOT COMMITTED, train locally

test_data/                          # 16 labeled CSV files (DoS, fuzzing, rpm, interval, etc.)
```

---

## Current Performance Metrics

These are **measured numbers** from batch test set 01 (Dec 3–14, 2025).

### 3-Stage Pipeline

| Stage | Component | Throughput | Notes |
|-------|-----------|-----------|-------|
| Stage 1 | Pre-Filter | — | Filters 80–95% benign before rule eval |
| Stage 2 | Rule Engine | ~759 msg/s measured / ~7,002 msg/s with hash indexing | 18 rule types, O(1) lookup |
| Stage 3 | Decision Tree ML | 8,000+ msg/s | 14.1 KB model, 12 features |
| Legacy | IsolationForest | ~15 msg/s | Deprecated for real-time use |

### Detection Quality

| Metric | Dec 7 (Pre-Optimization) | Dec 14 (Current) |
|--------|--------------------------|------------------|
| Recall | 100% (DoS only) | 94.76% (all attack types) |
| False Positive Rate | 81.7% | **8.43%** |
| Precision | 18.28% | ~91.57% |

### Resource Usage (Pi 4, Rules-Only, DoS-1 dataset)

```
Throughput:     759 msg/s
Mean Latency:   1.284 ms
P95 Latency:    2.038 ms
CPU:            25.3% avg, 28.7% peak
Memory:         173.3 MB avg
Temperature:    52.8°C avg
```

End-to-end 3-stage Pi 4 benchmark has not been measured yet (open item).

---

## Architecture Overview

```
Every CAN Message
        │
        ▼
  Stage 1: Fast Pre-Filter (prefilter.py)
        │  Known-good CAN IDs + timing tolerance (±30%)
        │  Filters 80–95% benign traffic → passes ~5–20%
        │
  Suspicious messages only
        │
        ▼
  Stage 2: Rule Engine (rule_engine.py)
        │  18 rule types, 30+ parameters
        │  CAN ID hash indexing — O(1) rule lookup
        │  Priority-based early exit (priority 0–10)
        │  ~759 msg/s measured; ~7,002 msg/s peak (indexed)
        │
  Messages still suspicious
        │
        ▼
  Stage 3: Decision Tree ML (decision_tree_detector.py)
        │  sklearn DecisionTreeClassifier, depth 10
        │  12 features: 8 byte values + DLC + interval + freq + entropy
        │  Feature importance: frequency 52%, entropy 41%
        │  8,000+ msg/s (278× faster than IsolationForest)
        │  Requires: data/models/decision_tree.pkl (train locally)
        │
        ▼
  Alert Manager → logs/alerts.json + console
```

### What Happened to IsolationForest?

The original `ml_based` mode used an IsolationForest ensemble (300 estimators). At ~15 msg/s it was impractical for real-time CAN monitoring. The project replaced it with a Decision Tree classifier and kept the code and models for offline analysis. The `ml_based` config flag is commented out — not deleted.

---

## How to Run the System

### Quick Validation

```bash
source venv/bin/activate

# Replay a labeled dataset
python main.py --mode replay --file test_data/DoS-1.csv

# Live monitoring (vCAN)
sudo ip link add dev vcan0 type vcan && sudo ip link set up vcan0
python main.py -i vcan0

# Pi 4 with dedicated config
python main.py -i can0 --config config/can_ids_rpi4.yaml
```

### Train Stage 3 (Decision Tree)

```bash
# Quick — synthetic data
python scripts/train_decision_tree.py --synthetic

# Better — from bundled test data
python scripts/train_decision_tree.py --vehicle-models . --output data/models/decision_tree.pkl

# Verify Stage 3 activates
python main.py -i vcan0 --log-level DEBUG 2>&1 | grep -i "stage 3"
```

### Generate Vehicle-Specific Rules

```bash
python scripts/generate_rules_from_baseline.py \
  --input test_data/attack-free-1.csv test_data/attack-free-2.csv \
  --output config/rules_my_vehicle.yaml
```

### Run Test Suite

```bash
python -m pytest tests/test_rule_engine_phase1.py \
                 tests/test_rule_engine_phase2.py \
                 tests/test_rule_engine_phase3.py -v
# Expected: 61/61 passing
```

---

## Known Issues & Open Items

### Must Do Before Production

- [ ] Train `decision_tree.pkl` — Stage 3 ML silently skipped without it
- [ ] Extract vehicle-specific timing/frequency baselines and regenerate rules
- [ ] Measure end-to-end 3-stage pipeline throughput on Pi 4 hardware

### Lower Priority

- [ ] Rule engine measured throughput (759 msg/s) to be reconciled with hash-indexed peak (7,002 msg/s) — run benchmark on target hardware
- [ ] Config `ml_based` comment says "temporarily disabled" — should say "deprecated for real-time; use decision_tree instead" (cosmetic)
- [ ] Config comment references `docs/ML_MODEL_ISSUE.md` — actual path is `docs/ml/ML_MODEL_ISSUE.md`

### Not Issues (Common Questions)

- *"Why is IsolationForest commented out?"* — Deliberate: 15 msg/s is too slow. Decision Tree replaced it.
- *"Why is recall 94.76% instead of 100%?"* — Tier 3 payload repetition check correctly suppresses some edge-case normal-jitter events that Tier 1/2 would have flagged. Lower FPR, slightly lower recall — better overall.
- *"Why isn't `decision_tree.pkl` in the repo?"* — It should be trained on data representative of your target vehicle environment.

---

## Configuration Quick Reference

### Active Pipeline (default `can_ids.yaml`)

```yaml
detection_modes:
  - rule_based            # Stage 2: always active

rules_file: config/rules_adaptive.yaml

decision_tree:
  enabled: true           # Stage 3: active if model file exists
  model_path: data/models/decision_tree.pkl

prefilter:
  enabled: true           # Stage 1: always active
  timing_tolerance: 0.3

# ml_based is commented out — legacy IsolationForest, deprecated for real-time
```

See [configuration.md](configuration.md) for every parameter explained.

---

## Success Metrics

### Current State (March 2026)

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| **Recall** | >95% | 94.76% | ✅ Near target |
| **FPR** | <10% | 8.43% | ✅ Met |
| **Rule throughput** | 1,500+ msg/s | 759 msg/s measured | ⚠️ Marginal (hash indexing peaks at 7,002) |
| **ML throughput** | 1,500+ msg/s | 8,000+ msg/s (DT) | ✅ Exceeded |
| **Stage 3 model** | Trained & loaded | Not committed | ⚠️ Must train locally |
| **CPU (Pi 4)** | <70% | 25.3% | ✅ Excellent |
| **Memory** | <400 MB | 173.3 MB | ✅ Excellent |
| **Production ready** | Yes | Near-ready | ⚠️ Train DT + tune rules |

---

## Development History Summary

| Date | Milestone |
|------|-----------|
| Jan 2025 | Project scaffolding, 7 original rule types |
| Nov 30, 2025 | First batch testing: 759 msg/s, 100% recall, 81.7% FPR |
| Dec 2, 2025 | Phases 1–3 rule engine: all 18 rule types, 61/61 tests |
| Dec 3, 2025 | IsolationForest bug fixes; profiled at 15 msg/s → motivated replacement |
| Dec 11, 2025 | Dual-sigma timing (Tier 1+2); FPR → 23% |
| Dec 14, 2025 | Hash indexing (7,002 msg/s), priority early exit, Decision Tree (8,000+ msg/s), Tier 3 payload check → FPR **8.43%** |
| Mar 1, 2026 | Documentation overhaul: GETTING_STARTED, configuration.md, all guides updated |

Full details: [docs/development_logs/CHANGELOG.md](../development_logs/CHANGELOG.md)

---

## Troubleshooting

### Stage 3 ML Not Activating

```bash
# Check if model exists
ls -lh data/models/decision_tree.pkl

# Train it
python scripts/train_decision_tree.py --synthetic

# Confirm activation
python main.py -i vcan0 --log-level DEBUG 2>&1 | grep -i "stage 3"
```

### High False Positive Rate

1. Generate vehicle-specific rules: `scripts/generate_rules_from_baseline.py`
2. Use adaptive rules: set `rules_file: config/rules_adaptive.yaml`
3. See [rules_guide.md](rules_guide.md) > Troubleshooting

### System Too Slow

1. Confirm pre-filter is enabled (`prefilter.enabled: true`)
2. Ensure `ml_based` is not in `detection_modes` (IsolationForest at 15 msg/s)
3. Profile: `python -m cProfile scripts/comprehensive_test.py test_data/DoS-1.csv`

### Import Errors

```bash
source venv/bin/activate
pip install -e .
python -c "import sklearn; import pandas; import numpy; print('OK')"
```

---

**Last Updated:** March 1, 2026  
**Status:** Production candidate — train Decision Tree and tune rules for your vehicle before deploying
