# CAN-IDS Development Changelog

**Consolidated timeline of all development sessions and milestones.**

This file distills the 12 individual development logs into a single chronological reference.
The original session logs are preserved in this directory for detailed context.

---

## Timeline

### January 2025 — Project Scaffolding

**Log**: [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)

- Created core module structure: sniffer, rule engine, ML detector, alert system, feature extractor
- Added Raspberry Pi 4 deployment support (systemd units, boot config)
- Initial configuration files (`can_ids.yaml`, `can_ids_rpi4.yaml`)
- Packaging with `setup.py`, requirements files, and project skeleton
- 7 original rule types: data_pattern, max_frequency, check_timing, allowed_sources, check_checksum, check_counter, entropy_threshold

---

### November 30, 2025 — First Batch Testing Session

**Logs**: [SESSION_LOG_20251130.md](SESSION_LOG_20251130.md), [TONIGHT_SUMMARY.md](TONIGHT_SUMMARY.md) (second half)

Key accomplishments:
- **Entropy bug fixed**: `probability.bit_length()` was called on a float — replaced with proper Shannon entropy calculation
- **Detection metrics added**: TP/FP/TN/FN counts, precision, recall, F1 score
- **CPU reporting normalized**: Per-core % replaced with system-wide average
- **Batch testing**: 12 datasets (9.6M messages) tested, found 100% recall but 0–10% precision due to aggressive timing rules
- **ML contamination parameter experiment**: Proved IsolationForest detection mode was not active (0% contamination = no model effect)

Performance baseline: ~759 msg/s (rules only), ~15 msg/s (with ML)

---

### December 2, 2025 — Rule Engine Phase 1–3 (18/18 Rule Types)

**Logs**: [PHASE_1_COMPLETE.md](PHASE_1_COMPLETE.md), [PHASE_2_COMPLETE.md](PHASE_2_COMPLETE.md), [PHASE_3_COMPLETE.md](PHASE_3_COMPLETE.md), [100_PERCENT_COMPLETE.md](100_PERCENT_COMPLETE.md)

#### Phase 1 — Critical Parameters (19/19 tests)
- `validate_dlc`: Strict DLC validation (0–8 range, data length match)
- `check_frame_format`: CAN frame structure validation (ID range, DLC, error/remote frames)
- `global_message_rate`: Bus-wide flooding detection

Impact: Precision improved from ~10% to 40–60%; DoS coverage 100%, fuzzing coverage 99%

#### Phase 2 — Important Parameters (21/21 tests)
- `check_source`: Enhanced OBD-II/UDS diagnostic source validation
- `check_replay`: Replay attack detection (signature + time window tracking)
- `data_byte_0`–`data_byte_7`: Byte-level payload validation
- `whitelist_mode` + `allowed_can_ids`: CAN ID whitelisting

Impact: Precision improved to 70–85%; FPR reduced to ~15%

#### Phase 3 — Specialized Parameters (21/21 tests, 2 skipped for hardware)
- `check_data_integrity`: XOR checksum validation at configurable offset
- `check_steering_range`: Steering angle range validation (16-bit signed LE, 0.1° resolution)
- `check_repetition`: Consecutive identical message detection
- `frame_type`: Standard vs. extended frame enforcement

Total: 61/61 tests passing across all phases. All 18 rule types implemented.

---

### December 3, 2025 — Real-World Testing & Profiling

**Log**: [DECEMBER_3_SESSION_SUMMARY.md](DECEMBER_3_SESSION_SUMMARY.md)

- **ML model loading bug fixed**: joblib unpickling error on Vehicle_Models dataset
- **Timing tracker optimization**: `list` → `deque` for bounded-size timing history
- **IsolationForest profiled as bottleneck**: 15 msg/s with IsolationForest vs 759 msg/s rules-only — motivated replacement with Decision Tree
- **Documented ML optimization roadmap**: batch inference, model distillation, PCA feature reduction
- Tested against Vehicle_Models real-world data (attack-free + attack scenarios)

---

### December 11, 2025 — Dual-Sigma Adaptive Timing

**Log**: [TONIGHT_SUMMARY.md](TONIGHT_SUMMARY.md) (first half)

- **Dual-sigma timing architecture**: Split timing detection into Tier 1 (`sigma_extreme`) and Tier 2 (`sigma_moderate` + `consecutive_required`)
- **Critical bug fix**: Tier 2 previously used hardcoded 1-sigma threshold instead of configurable value
- **Results**: 94.81% recall with 23% FPR
- Auto-generated adaptive rules via `scripts/generate_rules_from_baseline.py`

---

### December 14, 2025 — Hierarchical Detection & Tier 3

**Logs**: [STAGE2_RULE_INDEXING_COMPLETE.md](STAGE2_RULE_INDEXING_COMPLETE.md), [PHASE2_EARLY_EXIT_COMPLETE.md](PHASE2_EARLY_EXIT_COMPLETE.md), [PHASE3_IMPLEMENTATION_COMPLETE.md](PHASE3_IMPLEMENTATION_COMPLETE.md), [DEVELOPMENT_SUMMARY_DEC14_2025.md](DEVELOPMENT_SUMMARY_DEC14_2025.md)

#### Stage 2 — Rule Engine Optimizations
- **CAN ID hash indexing**: O(1) lookup replacing O(n×m) sequential search → 7,002 msg/s (341× evaluation reduction)
- **Priority-based early exit**: Rules sorted by priority (0=critical), critical rules (priority ≤ 2) cause immediate exit → 10–20% CPU reduction during attacks

#### Stage 3 — Decision Tree ML Detector
- **Trained Decision Tree**: 92.6% accuracy, 14.1 KB model (vs. IsolationForest at 280 KB)
- **Feature importance**: Frequency 52%, entropy 41%
- **Throughput**: 4,171 msg/s (278× faster than IsolationForest)
- **Integration**: Added to `main.py` as Stage 3 of hierarchical detection pipeline

#### Tier 3 — Payload Repetition Analysis
- **`payload_repetition_threshold`** (0.55): Discriminates attack traffic (identical payloads) from normal jitter
- **FPR reduced from 23% to 8.43%** — biggest single improvement
- **Final metrics**: 94.76% recall, 8.43% FPR

---

## Naming Clarification

Two files have confusingly similar names but cover different things:

| File | Date | Content |
|------|------|---------|
| `PHASE_3_COMPLETE.md` | Dec 2 | Phase 3 **rule engine parameters** (integrity, steering, repetition, frame type) |
| `PHASE3_IMPLEMENTATION_COMPLETE.md` | Dec 14 | Phase 3 **ML integration** (Decision Tree detector, hierarchical pipeline) |

---

## Current State (as of March 2026)

- **Rule engine** (Stage 2): 18/18 rule types, 61/61 tests passing, 759 msg/s throughput
- **Decision Tree ML** (Stage 3): 8,000+ msg/s, `decision_tree.enabled: true` (model must be trained locally)
- **Pre-filter** (Stage 1): Filters 80–95% benign traffic, enabled by default
- **IsolationForest** (Legacy): Code + 6 `.joblib` models remain for offline analysis; `ml_based` commented out in config
- **Detection quality**: 94.76% recall, 8.43% FPR (rules-only, Tier 3 tuning)
- **Models on disk**: 6 `.joblib` files in `data/models/` (IsolationForest); Decision Tree `.pkl` must be trained locally

---

## Individual Log Files (Preserved)

| File | Date | Topic |
|------|------|-------|
| [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) | Jan 2025 | Initial project scaffolding |
| [SESSION_LOG_20251130.md](SESSION_LOG_20251130.md) | Nov 30, 2025 | First batch testing, entropy bug fix |
| [TONIGHT_SUMMARY.md](TONIGHT_SUMMARY.md) | Dec 11, 2025 | Dual-sigma timing implementation |
| [PHASE_1_COMPLETE.md](PHASE_1_COMPLETE.md) | Dec 2, 2025 | Phase 1 rule params (DLC, frame format, global rate) |
| [PHASE_2_COMPLETE.md](PHASE_2_COMPLETE.md) | Dec 2, 2025 | Phase 2 rule params (source, replay, byte-level, whitelist) |
| [PHASE_3_COMPLETE.md](PHASE_3_COMPLETE.md) | Dec 2, 2025 | Phase 3 rule params (integrity, steering, repetition, frame type) |
| [100_PERCENT_COMPLETE.md](100_PERCENT_COMPLETE.md) | Dec 2, 2025 | Phase 1–3 summary, 18/18 rule types |
| [DECEMBER_3_SESSION_SUMMARY.md](DECEMBER_3_SESSION_SUMMARY.md) | Dec 3, 2025 | Real-world testing, ML profiling |
| [STAGE2_RULE_INDEXING_COMPLETE.md](STAGE2_RULE_INDEXING_COMPLETE.md) | Dec 14, 2025 | CAN ID hash indexing optimization |
| [PHASE2_EARLY_EXIT_COMPLETE.md](PHASE2_EARLY_EXIT_COMPLETE.md) | Dec 14, 2025 | Priority-based early exit |
| [PHASE3_IMPLEMENTATION_COMPLETE.md](PHASE3_IMPLEMENTATION_COMPLETE.md) | Dec 14, 2025 | Decision Tree ML integration |
| [DEVELOPMENT_SUMMARY_DEC14_2025.md](DEVELOPMENT_SUMMARY_DEC14_2025.md) | Dec 14, 2025 | Tier 3 payload repetition analysis |
