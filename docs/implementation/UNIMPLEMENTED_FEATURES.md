# CAN-IDS Feature Implementation Status

**Last Updated**: March 1, 2026  
**Source of truth**: `src/detection/rule_engine.py`, `src/detection/decision_tree_detector.py`, `src/preprocessing/feature_extractor.py`

---

## Rule Engine — All 18 Rule Types Implemented

All rule types are implemented in `src/detection/rule_engine.py`. The original 7 types shipped pre-November 2025; the remaining 11 were added in Phases 1–3 (completed December 14, 2025).

### Original Rule Types (7)

| # | Rule Type | Method | Purpose |
|---|-----------|--------|---------|
| 1 | `data_pattern` | `_check_data_pattern()` | Pattern matching in message data |
| 2 | `max_frequency` | `_check_frequency_violation()` | Per-CAN-ID message frequency monitoring |
| 3 | `check_timing` | `_check_timing_violation()` | Inter-arrival time validation |
| 4 | `allowed_sources` | `_validate_source()` | Source validation for specific ECUs |
| 5 | `check_checksum` | (inline) | Checksum validation |
| 6 | `check_counter` | (inline) | Counter sequence validation |
| 7 | `entropy_threshold` | (inline) | Data entropy analysis |

### Phase 1 — Critical Parameters (3)

| # | Rule Type | Method | Line | Purpose |
|---|-----------|--------|------|---------|
| 8 | `validate_dlc` | `_validate_dlc_strict()` | L808 | Strict DLC validation against CAN 2.0 spec |
| 9 | `check_frame_format` | `_check_frame_format()` | L853 | Frame format and structure validation |
| 10 | `global_message_rate` | `_check_global_message_rate()` | L912 | Bus-wide flooding detection |

### Phase 2 — Important Parameters (4)

| # | Rule Type | Method | Line | Purpose |
|---|-----------|--------|------|---------|
| 11 | `check_source` | `_validate_source_enhanced()` | L955 | Enhanced diagnostic source validation (OBD-II/UDS) |
| 12 | `check_replay` | `_check_replay_attack()` | L1019 | Replay attack detection with time windows |
| 13 | `data_byte_0`–`data_byte_7` | `_check_byte_values()` | L465 | Individual byte-level validation (8 parameters) |
| 14 | `whitelist_mode` | (inline) | — | CAN ID whitelist enforcement |

### Phase 3 — Specialized Parameters (4)

| # | Rule Type | Method | Line | Purpose |
|---|-----------|--------|------|---------|
| 15 | `check_data_integrity` | `_check_data_integrity()` | L1140 | XOR checksum validation for safety systems |
| 16 | `check_steering_range` | `_check_steering_range()` | L1196 | Steering angle physical limits validation |
| 17 | `check_repetition` | `_check_repetition_pattern()` | L1252 | Repetitive pattern detection |
| 18 | `frame_type` | `_validate_frame_type()` | L1298 | Standard vs extended frame validation |

---

## ML Feature Extraction — Implemented

Implemented in `src/preprocessing/feature_extractor.py`.

**Basic Features** (50 features, always active):
- Message-level: CAN ID, DLC, data bytes, frame flags
- Statistical: mean, median, std, min, max, range, sum, entropy
- Temporal: frequency tracking, IAT statistics, jitter
- Pattern: repetition, sequential, alternating patterns
- Behavioral: DLC consistency, data change rate, payload variance
- Communication: message rate, ID diversity, priority estimation

**Enhanced Features** (8 features, opt-in via `enable_enhanced_features=True`):

| Feature | Source Paper | Description |
|---------|-------------|-------------|
| `payload_entropy` | TCE-IDS | Shannon entropy of payload bytes |
| `hamming_distance` | Novel Architecture | Bit flips between consecutive payloads |
| `iat_zscore` | SAIDuCANT | Normalized timing deviation (IAT − μ)/σ |
| `unknown_bigram` | Novel Architecture | Novel 2-ID sequences |
| `unknown_trigram` | Novel Architecture | Novel 3-ID sequences |
| `bit_time_mean` | BTMonitor | Physical layer bit timing mean |
| `bit_time_rms` | BTMonitor | RMS of bit timing |
| `bit_time_energy` | BTMonitor | Bit timing energy metric |

Tests: `tests/test_enhanced_features.py` (9/9 passing)  
Docs: `docs/implementation/enhanced_features_integration.md`

---

## ML Detection — Evolved from IsolationForest to Hierarchical Pipeline

ML was **not disabled** — it was **reimplemented**. The original IsolationForest approach (15 msg/s) proved too slow for real-time CAN monitoring on Raspberry Pi 4. Rather than abandoning ML, the architecture evolved into a hierarchical 3-stage detection pipeline that separates concerns by cost:

### Architecture Evolution

```
Original (Nov 2025)                    Current (Dec 14, 2025)
─────────────────                      ─────────────────────
CAN Message                            CAN Message
    │                                      │
    ├─► Rule Engine                        ├─► Stage 1: Fast Pre-Filter
    │                                      │   (prefilter.py, enabled by default)
    └─► IsolationForest ← TOO SLOW        │   Filters 80-95% benign traffic
        (~15 msg/s)                        │
                                           ├─► Stage 2: Rule Engine
                                           │   (rule_engine.py, 18 rule types)
                                           │   ~759 msg/s, deterministic
                                           │
                                           └─► Stage 3: Decision Tree ML
                                               (decision_tree_detector.py)
                                               ~8,000+ msg/s, 14.1 KB model
                                               Feature importance: freq 52%, entropy 41%
```

### What Controls What

| Config Flag | Controls | Status |
|---|---|---|
| `detection_modes: [rule_based]` | Stage 2 rule engine | ✅ Active |
| `# - ml_based` | **Original** IsolationForest path | ❌ Commented out (deprecated — 15 msg/s) |
| `decision_tree.enabled: true` | Stage 3 Decision Tree ML | ✅ Enabled in config |
| `prefilter.enabled: true` | Stage 1 fast pre-filter | ✅ Enabled by default |
| `ml_detection.enable_multistage: true` | Enhanced multi-stage pipeline | ✅ Configured |

The `ml_based` flag only controls the **original IsolationForest** detection mode. The Decision Tree detector and pre-filter have their own independent config sections and are both enabled.

### Why IsolationForest Was Deprecated (Not "Disabled")

The IsolationForest model runs at ~15 msg/s — far below the 1,000–4,000 msg/s needed for real-time CAN monitoring. This is inherent to the algorithm (300 estimators × feature extraction overhead). Rather than optimizing it, the project replaced it with a Decision Tree classifier that achieves 8,000+ msg/s with comparable accuracy on pre-filtered traffic.

The IsolationForest code and models remain in the codebase for offline analysis and research, but the real-time pipeline uses the Decision Tree.

### Decision Tree Detector (`src/detection/decision_tree_detector.py`)

- **Algorithm**: `sklearn.tree.DecisionTreeClassifier` (max depth 10)
- **12 features**: 8 byte values + DLC + interval + frequency + entropy
- **Throughput**: 8,000+ msg/s (278× faster than IsolationForest)
- **Model size**: 14.1 KB (vs. 280 KB for IsolationForest)
- **Integration**: Stage 3 in `main.py`, processes messages that pass Stages 1+2
- **Config**: `decision_tree.enabled: true` in `config/can_ids.yaml`

### Pre-Trained Models — Available in `data/models/`

6 IsolationForest model files are present (from Vehicle_Models research):

| Model File | Size | Notes |
|------------|------|-------|
| `aggressive_load_shedding.joblib` | 1.3 MB | Multi-stage IF pipeline (config default for `ml_based`) |
| `adaptive_load_shedding.joblib` | 1.3 MB | Alternative multi-stage config |
| `full_pipeline.joblib` | 1.3 MB | Complete IF pipeline |
| `enhanced_detector.joblib` | 356 KB | RandomForest-based detector |
| `can_feature_engineer.joblib` | 21 KB | 13 CAN-specific features |
| `adaptive_weighted_detector.joblib` | 618 B | Best lab accuracy (97.20% recall, 100% precision) |

> **Note:** The Decision Tree model (`decision_tree.pkl`) is referenced in config with
> `decision_tree.enabled: true`, but the `.pkl` file is not committed to the repository.
> It must be trained locally via `scripts/train_decision_tree.py`.

### ML Detector Modules in `src/detection/`

| Module | Purpose | Status |
|--------|---------|--------|
| `decision_tree_detector.py` | Stage 3 Decision Tree classifier | ✅ Active (config-enabled) |
| `prefilter.py` | Stage 1 fast statistical filter | ✅ Active (default-enabled) |
| `ml_detector.py` | Original IsolationForest detector | ⚠️ Code present, config commented out |
| `enhanced_ml_detector.py` | Multi-stage ML pipeline with gating | ✅ Configured via `ml_detection` |
| `multistage_detector.py` | Multi-stage detector framework | Available |
| `weighted_ensemble_detector.py` | Weighted ensemble voting | Available |
| `ensemble_crosscheck_detector.py` | Cross-validation ensemble | Available |
| `advanced_detectors.py` | Additional detector implementations | Available |
| `improved_detectors.py` | Improved detector variants | Available |
| `vehicle_models_compat.py` | Vehicle_Models pickle compatibility | Support module |
| `vehicle_calibration.py` | Per-vehicle calibration | Available |

---

## Known Gaps & Remaining Work

These are the actual open items as of March 2026:

### Decision Tree Model Not Committed
- `config/can_ids.yaml` has `decision_tree.enabled: true` referencing `data/models/decision_tree.pkl`
- This `.pkl` file is **not in the repository** — must be trained locally via `scripts/train_decision_tree.py`
- Without it, Stage 3 ML detection silently falls back to Stages 1+2 only

### IsolationForest Deprecated for Real-Time
- The `ml_based` detection mode (IsolationForest) is commented out in both config files
- This was a deliberate architectural decision: 15 msg/s throughput is impractical for real-time CAN monitoring
- The IsolationForest code and models remain for offline batch analysis
- **This is not a gap** — the Decision Tree replaced it for real-time use

### Performance Targets
- Rule engine: 759 msg/s (target: 2,000–4,000 msg/s for peak CAN loads)
- Decision Tree: 8,000+ msg/s (exceeds target, but only on pre-filtered traffic)
- Combined 3-stage pipeline throughput needs end-to-end benchmarking on Pi 4
- False positive rate: ~8.43% with Tier 3 tuning; vehicle-specific baselines would reduce further

### Rule Tuning Needed
- All 18 rule types work, but thresholds are generic, not vehicle-specific
- Extracting timing/frequency baselines from real training data would dramatically reduce false positives
- See `docs/guides/PROJECT_CONTEXT.md` for the tuning strategy

### Config Inconsistencies
- Config comment references `docs/ML_MODEL_ISSUE.md` but actual file is `docs/ml/ML_MODEL_ISSUE.md`
- `ml_based` comment says "temporarily disabled" — should say "deprecated for real-time; use decision_tree instead"

---

## Change History

| Date | Change |
|------|--------|
| Oct 2025 | Original 7 rule types implemented |
| Nov 30, 2025 | 6 pre-trained models integrated from Vehicle_Models |
| Dec 2, 2025 | 8 enhanced ML features integrated; Phases 1–3 rule implementation completed (18/18 rule types) |
| Dec 14, 2025 | Phase 3 finalized; 61/61 rule tests passing |
| Dec 16–17, 2025 | PCA testing, performance testing |
| Mar 1, 2026 | This document rewritten to reflect actual implementation state |
