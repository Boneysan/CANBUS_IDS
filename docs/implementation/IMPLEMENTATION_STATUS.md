# CAN-IDS Implementation Status

**Last Updated**: March 1, 2026  
**Project**: CANBUS_IDS - Controller Area Network Intrusion Detection System  
**Architecture**: Hierarchical 3-Stage Detection Pipeline

> This is the single authoritative status document. It replaces the previous
> `IMPLEMENTATION_STATUS.md` (Dec 2, 2025 — showed rules as 5/15 partial) and
> `IMPLEMENTATION_STATUS_UPDATED.md` (Dec 2, 2025 — overclaimed production readiness).
> Both have been merged here with corrections verified against the actual source code
> and config files.

---

## Executive Summary

| Component | Code Status | Runtime Status | Notes |
|-----------|-------------|----------------|-------|
| **Stage 1: Pre-Filter** | ✅ `prefilter.py` | ✅ Active (default-enabled) | Filters 80–95% benign traffic before rule evaluation |
| **Stage 2: Rule Engine** | ✅ 18/18 rule types implemented | ✅ Active (`rule_based` enabled) | ~759 msg/s; thresholds generic without vehicle-specific tuning |
| **Stage 3: Decision Tree ML** | ✅ `decision_tree_detector.py` | ⚠️ Enabled in config, but `.pkl` model not committed | 8,000+ msg/s; must train locally via `scripts/train_decision_tree.py` |
| **Legacy: IsolationForest** | ✅ Code + 6 `.joblib` models present | ❌ Deprecated for real-time (`ml_based` commented out) | ~15 msg/s — replaced by Decision Tree for real-time use |

---

## Architecture

The detection pipeline evolved from a dual (rule + IsolationForest) design to a hierarchical 3-stage system, driven by the IsolationForest’s 15 msg/s throughput being impractical for real-time CAN monitoring.

```
Every CAN Message
        │
        ▼
  Stage 1: Fast Pre-Filter (prefilter.py)
        │  • Statistical filtering, enabled by default
        │  • Filters 80–95% of benign traffic
        │  • Known-good CAN IDs + timing tolerance
        │
    Suspicious messages only
        │
        ▼
  Stage 2: Rule Engine (rule_engine.py)
        │  • 18 rule types (Phases 1–3)
        │  • Priority-sorted, early exit on critical match
        │  • CAN ID hash indexing (O(1) lookup)
        │  • ~759 msg/s measured throughput
        │
    Messages still suspicious
        │
        ▼
  Stage 3: Decision Tree ML (decision_tree_detector.py)
        │  • sklearn DecisionTreeClassifier (depth 10)
        │  • 12 features: bytes + DLC + timing + entropy
        │  • 8,000+ msg/s (278× faster than IsolationForest)
        │  • 14.1 KB model
        │
        ▼
  Alert Manager (correlates all stages)
        │
        ▼
  Notifications
```

### What happened to IsolationForest?

The original `ml_based` detection mode used an IsolationForest ensemble (300 estimators). At ~15 msg/s, it was 50–270× too slow for real-time CAN traffic (1,000–4,000 msg/s typical). Rather than optimizing it, the project:

1. **Replaced it** with a Decision Tree classifier (8,000+ msg/s) as Stage 3
2. **Added a pre-filter** (Stage 1) that eliminates most benign traffic before any detection runs
3. **Kept the code and models** for offline analysis — `ml_based` is commented out in config, not deleted

The `ml_based` config flag controls only the legacy IsolationForest path. The Decision Tree and pre-filter have their own config sections (`decision_tree.enabled`, `prefilter.enabled`).

---

## Rule Engine — 18/18 Rule Types Implemented

All rule types are implemented in `src/detection/rule_engine.py`. The original 7 shipped pre-November 2025; the remaining 11 were added in Phases 1–3 (completed December 14, 2025). Test suite: 61/61 tests passing.

### Original Rule Types (7)

| Rule Type | Method | Purpose |
|-----------|--------|---------|
| `data_pattern` | `_check_data_pattern()` | Pattern matching in message data |
| `max_frequency` | `_check_frequency_violation()` | Per-CAN-ID message frequency monitoring |
| `check_timing` | `_check_timing_violation()` | Inter-arrival time validation |
| `allowed_sources` | `_validate_source()` | Source validation for specific ECUs |
| `check_checksum` | (inline) | Checksum validation |
| `check_counter` | (inline) | Counter sequence validation |
| `entropy_threshold` | (inline) | Data entropy analysis |

### Phase 1 — Critical Parameters (3)

| Rule Type | Method | Purpose |
|-----------|--------|---------|
| `validate_dlc` | `_validate_dlc_strict()` | Strict DLC validation against CAN 2.0 spec |
| `check_frame_format` | `_check_frame_format()` | Frame format and structure validation |
| `global_message_rate` | `_check_global_message_rate()` | Bus-wide flooding detection |

### Phase 2 — Important Parameters (4)

| Rule Type | Method | Purpose |
|-----------|--------|---------|
| `check_source` | `_validate_source_enhanced()` | Enhanced diagnostic source validation (OBD-II/UDS) |
| `check_replay` | `_check_replay_attack()` | Replay attack detection with time windows |
| `data_byte_0`–`7` | `_check_byte_values()` | Individual byte-level validation (8 parameters) |
| `whitelist_mode` | (inline) | CAN ID whitelist enforcement |

### Phase 3 — Specialized Parameters (4)

| Rule Type | Method | Purpose |
|-----------|--------|---------|
| `check_data_integrity` | `_check_data_integrity()` | XOR checksum validation for safety systems |
| `check_steering_range` | `_check_steering_range()` | Steering angle physical limits validation |
| `check_repetition` | `_check_repetition_pattern()` | Repetitive pattern detection |
| `frame_type` | `_validate_frame_type()` | Standard vs extended frame validation |

### Test Coverage

| Suite | Tests | Status |
|-------|-------|--------|
| Phase 1 (`tests/test_rule_engine_phase1.py`) | 19/19 | ✅ |
| Phase 2 (`tests/test_rule_engine_phase2.py`) | 21/21 | ✅ |
| Phase 3 (`tests/test_rule_engine_phase3.py`) | 21/21 | ✅ |
| **Total** | **61/61** | **✅** |

---

## ML Detection — Evolved to Hierarchical Pipeline

### Stage 3: Decision Tree Classifier (Active)

Implemented in `src/detection/decision_tree_detector.py`. This is the current real-time ML approach.

| Attribute | Value |
|-----------|-------|
| Algorithm | `sklearn.tree.DecisionTreeClassifier` (max depth 10) |
| Features | 12: 8 byte values + DLC + interval + frequency + entropy |
| Throughput | 8,000+ msg/s (278× faster than IsolationForest) |
| Model size | 14.1 KB |
| Config | `decision_tree.enabled: true` in `config/can_ids.yaml` |
| Model file | `data/models/decision_tree.pkl` (**not committed** — train via `scripts/train_decision_tree.py`) |

### Stage 1: Fast Pre-Filter (Active)

Implemented in `src/detection/prefilter.py`. Enabled by default.

- Filters 80–95% of known-benign traffic before rule evaluation
- Uses known-good CAN ID set + timing tolerance (±30%)
- CAN IDs extracted from rules at init, or configured in `prefilter.known_good_ids`

### Legacy: IsolationForest (Deprecated for Real-Time)

Implemented in `src/detection/ml_detector.py`. The `ml_based` detection mode is **commented out** in both config files—not because ML failed, but because the IsolationForest algorithm is inherently too slow (15 msg/s) for real-time CAN bus monitoring. It was replaced by the Decision Tree for that purpose.

**IsolationForest remains useful for**:
- Offline batch analysis of captured traffic
- Research and model comparison
- Novel attack discovery where latency is not a constraint

### Feature Extraction

Implemented in `src/preprocessing/feature_extractor.py`.

**Basic features** (50): CAN ID, DLC, data bytes, statistical, temporal, pattern, behavioral, communication features.

**Enhanced features** (8, opt-in via `enable_enhanced_features=True`):

| Feature | Source Paper |
|---------|-------------|
| `payload_entropy` | TCE-IDS |
| `hamming_distance` | Novel Architecture |
| `iat_zscore` | SAIDuCANT |
| `unknown_bigram` / `unknown_trigram` | Novel Architecture |
| `bit_time_mean` / `bit_time_rms` / `bit_time_energy` | BTMonitor |

Tests: `tests/test_enhanced_features.py` — 9/9 passing.

### IsolationForest Models in Repository

6 model files exist in `data/models/` (all IsolationForest-based, from Vehicle_Models research):

| Model | Size | Notes |
|-------|------|-------|
| `aggressive_load_shedding.joblib` | 1.3 MB | Config default for legacy `ml_based` mode |
| `adaptive_load_shedding.joblib` | 1.3 MB | Alternative multi-stage config |
| `full_pipeline.joblib` | 1.3 MB | Complete IF pipeline |
| `enhanced_detector.joblib` | 356 KB | RandomForest-based |
| `can_feature_engineer.joblib` | 21 KB | CAN-specific feature engineering |
| `adaptive_weighted_detector.joblib` | 618 B | Best lab accuracy (97.20% recall, 100% precision) |

> **Correction**: Earlier docs claimed 12 models. The other 6 were validated in the
> Vehicle_Models research project but were never copied to this repository.

### All ML Detector Modules

| Module | Purpose | Runtime Status |
|--------|---------|----------------|
| `decision_tree_detector.py` | Stage 3 Decision Tree classifier | ✅ Config-enabled |
| `prefilter.py` | Stage 1 fast statistical filter | ✅ Default-enabled |
| `ml_detector.py` | IsolationForest detector | Deprecated for real-time |
| `enhanced_ml_detector.py` | Multi-stage ML pipeline with gating | Configured |
| `multistage_detector.py` | Multi-stage detector framework | Available |
| `weighted_ensemble_detector.py` | Weighted ensemble voting | Available |
| `ensemble_crosscheck_detector.py` | Cross-validation ensemble | Available |
| `advanced_detectors.py` | Additional detector implementations | Available |
| `improved_detectors.py` | Improved detector variants | Available |
| `vehicle_models_compat.py` | Vehicle_Models pickle compatibility | Support module |
| `vehicle_calibration.py` | Per-vehicle calibration | Available |

---

## Measured Performance

These are **actual measured numbers**, not projections.

| Metric | Stage 2: Rule Engine | Stage 3: Decision Tree | Legacy: IsolationForest | Source |
|--------|---------------------|----------------------|------------------------|--------|
| **Throughput** | ~759 msg/s | 8,000+ msg/s | ~15 msg/s | Dec 3–14, 2025 |
| **Recall** | 100% (DoS), 94.76% (overall) | 85–88% (pre-filtered) | 97.20% (lab) | Batch test set 01 |
| **Precision** | 18% (generic) → 91%+ (tuned) | TBD on production data | 100% (lab, best model) | Dec 7–14, 2025 |
| **FPR** | 81.7% → **8.43%** (Tier 3) | TBD | <2% (lab) | Dec 14, 2025 |
| **Model size** | N/A (rules) | 14.1 KB | 280 KB–1.3 MB | — |

**Key takeaway**: The rule engine works and catches attacks, but generic thresholds cause high false positives on real traffic. Vehicle-specific baseline extraction would fix this (see `docs/guides/PROJECT_CONTEXT.md`).

---

## Known Gaps & Remaining Work

### Performance

- [ ] Rule engine throughput: 759 msg/s → target 2,000–4,000 msg/s
- [ ] End-to-end 3-stage pipeline benchmark on Raspberry Pi 4 (Stages 1+2+3 combined)
- [x] ~~ML throughput: 15 msg/s~~ → Replaced by Decision Tree at 8,000+ msg/s

### Rule Tuning

- [ ] Extract vehicle-specific timing/frequency baselines from training data
- [ ] Auto-generate tuned `rules.yaml` using mean ± 3σ thresholds
- [ ] Reduce FPR from ~8.43% (Tier 3) further toward <5%

### Config Issues

- [ ] `decision_tree.pkl` not committed — must be trained locally (Stage 3 falls back to Stages 1+2 without it)
- [ ] Config comment for `ml_based` says "temporarily disabled" — should say "deprecated for real-time; use decision_tree instead"
- [ ] Config comment paths reference `docs/ML_MODEL_ISSUE.md` — actual file is `docs/ml/ML_MODEL_ISSUE.md`

### Missing Documentation

- [x] ~~`docs/guides/rules_guide.md` only covers 7 rule types~~ → Updated to cover all 18 (Mar 1, 2026)
- [ ] No consolidated configuration parameter reference
- [ ] No end-to-end getting-started guide that works out of the box

---

## Configuration Reference

### Current Config (`config/can_ids.yaml`)

```yaml
# Stage 2: Rule engine (always active)
detection_modes:
  - rule_based
  # - ml_based  # Deprecated for real-time — IsolationForest at 15 msg/s

rules_file: config/rules_adaptive.yaml

# Stage 3: Decision Tree ML (replaces IsolationForest for real-time)
decision_tree:
  enabled: true
  model_path: data/models/decision_tree.pkl

# Stage 1: Pre-filter (enabled by default)
prefilter:
  enabled: true
  timing_tolerance: 0.3
```

### To Re-Enable IsolationForest (Offline Analysis Only)

```yaml
# Only for batch/offline analysis — not viable for real-time
detection_modes:
  - rule_based
  - ml_based

ml_model:
  path: data/models/adaptive_weighted_detector.joblib
  contamination: 0.20
```

---

## Documentation References

| Topic | File |
|-------|------|
| Architecture design | `docs/planning/current_architecture_design.md` |
| Dual detection design | `docs/implementation/multistage_rule_integration.md` |
| Enhanced ML features | `docs/implementation/enhanced_features_integration.md` |
| ML optimization strategies | `docs/ml/ML_OPTIMIZATION_GUIDE.md` |
| Performance optimization plans | `docs/performance/IMPROVEMENT_ROADMAP.md`, `docs/performance/BUILD_PLAN_7000_MSG_SEC.md` |
| Rule tuning strategy | `docs/guides/PROJECT_CONTEXT.md` |
| Pi4 deployment | `docs/deployment/RASPBERRY_PI_DEPLOYMENT_GUIDE.md` |
| Feature status (detailed) | `docs/implementation/UNIMPLEMENTED_FEATURES.md` |
| Phase completion logs | `docs/development_logs/PHASE_1_COMPLETE.md` through `PHASE_3_COMPLETE.md` |

---

## Change History

| Date | Change |
|------|--------|
| Oct 2025 | Original 7 rule types implemented |
| Nov 30, 2025 | 6 pre-trained models integrated from Vehicle_Models |
| Dec 2, 2025 | 8 enhanced ML features integrated; Phase 1–3 rule implementation completed (18/18 types) |
| Dec 3, 2025 | ML bugs fixed; throughput benchmarks run (759 msg/s rules, 15 msg/s ML) |
| Dec 7, 2025 | FPR analysis: 81.7% with generic thresholds; tuning strategy defined |
| Dec 14, 2025 | Phase 3 finalized; 61/61 rule tests passing; Tier 3 FPR improved to 8.43% |
| Dec 16–17, 2025 | PCA testing, additional performance testing |
| Mar 1, 2026 | Merged IMPLEMENTATION_STATUS.md + IMPLEMENTATION_STATUS_UPDATED.md into this document |
