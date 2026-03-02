# CAN-IDS Detection Rules Writing Guide

**Last Updated**: March 1, 2026  
**Rule Engine**: `src/detection/rule_engine.py` — 18 rule types, 30+ parameters

This guide explains how to write custom detection rules for the CAN-IDS signature-based detection engine.

## Table of Contents

1. [Rule Basics](#rule-basics)
2. [Rule Structure](#rule-structure)
3. [All 18 Rule Types](#all-18-rule-types)
4. [Adaptive Timing Parameters](#adaptive-timing-parameters)
5. [Example Rules](#example-rules)
6. [Best Practices](#best-practices)
7. [Testing Rules](#testing-rules)
8. [Troubleshooting](#troubleshooting)

---

## Rule Basics

Detection rules are defined in YAML format and stored in `config/rules.yaml` (or `config/rules_adaptive.yaml` for auto-generated rules). Each rule describes a specific attack pattern, policy violation, or anomalous behavior to detect.

### Basic Concepts

- **Rules are stateful** — Track message history, timing, frequency, payloads
- **Rules are composable** — Multiple conditions are ANDed together
- **Rules generate alerts** — Matched rules create alerts with severity levels
- **Rules support priority** — Lower priority number = evaluated first, enables early exit

### Rule File Location

```
config/
├── rules.yaml              # Hand-written rules
├── rules_adaptive.yaml     # Auto-generated from baseline data
├── example_rules.yaml      # Templates
└── rules_generated.yaml    # Generated rules (alternative)
```

---

## Rule Structure

### Minimal Rule

Every rule must have these required fields:

```yaml
rules:
  - name: "My Detection Rule"
    severity: HIGH
    description: "Description of what this rule detects"
    action: alert
```

### Complete Rule Template

All available parameters (use only what you need — most are optional):

```yaml
rules:
  - name: "Rule Name"                    # Required: unique identifier
    severity: CRITICAL|HIGH|MEDIUM|LOW   # Required: alert severity
    description: "What this detects"     # Required: human-readable description
    action: alert|log                    # Required: action on match
    priority: 5                          # 0=critical, 5=normal, 10=low (early exit)
    
    # === CAN ID Matching ===
    can_id: 0x123                        # Exact CAN ID
    can_id_range: [0x100, 0x200]        # Range (inclusive)
    
    # === 1. Data Pattern Matching ===
    data_pattern: "DE AD BE EF"          # Hex pattern (* = wildcard)
    data_contains: ["27 01", "27 03"]   # Any-match list
    
    # === 2. Frequency Detection ===
    max_frequency: 1000                  # Max messages per time_window
    time_window: 1                       # Window in seconds
    
    # === 3. Timing Analysis ===
    check_timing: true
    expected_interval: 100               # Expected interval in ms
    interval_variance: 10                # Allowed variance in ms
    sigma_extreme: 3.0                   # Tier 1: extreme threshold (σ)
    sigma_moderate: 1.5                  # Tier 2: sustained threshold (σ)
    consecutive_required: 5              # Tier 2: consecutive violations needed
    payload_repetition_threshold: 0.55   # Tier 3: min repeated-payload fraction
    
    # === 4. Source Validation ===
    allowed_sources: [0x10, 0x20]       # Allowed source IDs
    
    # === 5. Checksum Validation ===
    check_checksum: true
    
    # === 6. Counter Validation ===
    check_counter: true
    
    # === 7. Entropy Analysis ===
    entropy_threshold: 7.5               # Max Shannon entropy (bits)
    
    # === 8. DLC Validation ===
    validate_dlc: true
    dlc_min: 0                           # Minimum DLC (0-8)
    dlc_max: 8                           # Maximum DLC (0-8)
    
    # === 9. Frame Format ===
    check_frame_format: true
    
    # === 10. Global Message Rate ===
    global_message_rate: 5000           # Max total msgs/window across all IDs
    
    # === 11. Enhanced Source Validation ===
    check_source: true                   # Diagnostic source validation (OBD-II/UDS)
    
    # === 12. Replay Detection ===
    check_replay: true
    replay_time_threshold: 1.0           # Max seconds between replays
    
    # === 13. Byte-Level Validation ===
    data_byte_0: 0xFF                    # Expected value for byte 0
    data_byte_1: 0x00                    # ... through data_byte_7
    
    # === 14. Whitelist Mode ===
    whitelist_mode: true
    allowed_can_ids: [0x100, 0x200]
    
    # === 15. Data Integrity ===
    check_data_integrity: true
    integrity_checksum_offset: 7         # Byte offset for XOR checksum (default: last byte)
    
    # === 16. Steering Range ===
    check_steering_range: true
    steering_min: -540.0                 # Min angle in degrees
    steering_max: 540.0                  # Max angle in degrees
    
    # === 17. Repetition Detection ===
    check_repetition: true
    repetition_threshold: 10             # Max consecutive identical messages
    
    # === 18. Frame Type ===
    frame_type: standard|extended        # Expected frame type
```

---

## All 18 Rule Types

### Original Rule Types (7)

#### 1. `data_pattern` — Pattern Matching

Match hex data patterns in message payloads. Use `*` as byte wildcard.

```yaml
- name: "UDS Programming Session"
  can_id: 0x7E0
  data_pattern: "10 02"
  severity: CRITICAL
  description: "ECU programming session initiated"
  action: alert
```

Use `data_contains` for any-of matching:

```yaml
- name: "Security Access Attempt"
  can_id_range: [0x7E0, 0x7E7]
  data_contains: ["27 01", "27 02", "27 03"]
  severity: HIGH
  description: "UDS security access request/response"
  action: alert
```

#### 2. `max_frequency` — Frequency Monitoring

Detect excessive message rates for a specific CAN ID. Catches DoS/flooding attacks.

```yaml
- name: "DoS Attack on Engine ECU"
  can_id: 0x7E0
  max_frequency: 500
  time_window: 1
  severity: CRITICAL
  description: "Excessive messages targeting engine ECU"
  action: alert
```

#### 3. `check_timing` — Timing Analysis

Detect irregular message intervals. Supports three-tier adaptive thresholds:

- **Tier 1** (`sigma_extreme`): Catches obvious attacks (DoS at 1ms vs 100ms baseline)
- **Tier 2** (`sigma_moderate` + `consecutive_required`): Catches subtle sustained attacks
- **Tier 3** (`payload_repetition_threshold`): Distinguishes attacks from normal jitter via payload analysis

```yaml
- name: "Interval Manipulation Attack"
  can_id: 0x316
  check_timing: true
  expected_interval: 100
  interval_variance: 10
  sigma_extreme: 3.0
  sigma_moderate: 1.5
  consecutive_required: 5
  payload_repetition_threshold: 0.55
  severity: HIGH
  description: "Message timing inconsistent with normal pattern"
  action: alert
```

See [Adaptive Timing Parameters](#adaptive-timing-parameters) for details on tuning.

#### 4. `allowed_sources` — Source Validation

Validate that messages come from authorized ECU sources.

```yaml
- name: "Fake Engine ECU"
  can_id: 0x7E8
  allowed_sources: [0x7E0]
  severity: CRITICAL
  description: "Message from unauthorized source impersonating ECU"
  action: alert
```

#### 5. `check_checksum` — Checksum Validation

Validate message checksums against expected values.

```yaml
- name: "Checksum Failure"
  can_id: 0x200
  check_checksum: true
  severity: MEDIUM
  description: "Message checksum validation failed"
  action: alert
```

#### 6. `check_counter` — Counter Validation

Detect sequence counter anomalies (skipped, repeated, or out-of-order counters).

```yaml
- name: "Counter Sequence Error"
  can_id: 0x300
  check_counter: true
  severity: MEDIUM
  description: "Message sequence counter out of order"
  action: alert
```

#### 7. `entropy_threshold` — Entropy Analysis

Detect encrypted, randomized, or fuzzing data by measuring Shannon entropy.

```yaml
- name: "High Entropy Data"
  entropy_threshold: 7.0
  severity: LOW
  description: "Unusually random data (possible encryption or fuzzing)"
  action: log
```

---

### Phase 1 — Critical Parameters (3)

Added December 2, 2025. Tests: `tests/test_rule_engine_phase1.py` (19/19 passing).

#### 8. `validate_dlc` — DLC Validation

Strict Data Length Code validation against CAN 2.0 specification:
- DLC must be 0–8
- Data length must match DLC
- Optional min/max constraints

```yaml
- name: "Invalid DLC"
  validate_dlc: true
  dlc_min: 8
  dlc_max: 8
  severity: HIGH
  description: "CAN frame with invalid or unexpected DLC"
  action: alert
```

#### 9. `check_frame_format` — Frame Format Validation

Validates CAN frame structure:
- CAN ID within valid range (11-bit standard, 29-bit extended)
- DLC range check
- Data length consistency
- Error frame detection
- Remote frame validation

```yaml
- name: "Malformed CAN Frame"
  check_frame_format: true
  severity: HIGH
  description: "CAN frame failed structural validation"
  action: alert
```

#### 10. `global_message_rate` — Bus Flooding Detection

Monitors total message rate across **all** CAN IDs (not per-ID). Detects bus-wide DoS/flooding.

```yaml
- name: "CAN Bus Flood"
  global_message_rate: 8000
  time_window: 1
  severity: CRITICAL
  description: "Total bus message rate exceeds safe threshold"
  action: alert
```

---

### Phase 2 — Important Parameters (4)

Added December 2, 2025. Tests: `tests/test_rule_engine_phase2.py` (21/21 passing).

#### 11. `check_source` — Enhanced Diagnostic Source Validation

Validates diagnostic message sources for OBD-II and UDS protocols. Automatically identifies diagnostic CAN IDs (0x7DF, 0x7E0–0x7EF) and checks source authorization.

```yaml
- name: "Unauthorized OBD-II Access"
  can_id: 0x7DF
  check_source: true
  allowed_sources: []  # No external sources allowed
  severity: HIGH
  description: "Diagnostic request from unauthorized source"
  action: alert
```

#### 12. `check_replay` — Replay Attack Detection

Detects replayed messages by tracking exact (CAN ID + data) signatures and timing.
Triggers when identical messages repeat within `replay_time_threshold` seconds.

```yaml
- name: "Replay Attack"
  can_id: 0x200
  check_replay: true
  replay_time_threshold: 1.0  # seconds
  severity: HIGH
  description: "Identical message replayed within suspicious time window"
  action: alert
```

**Detection logic**: Alerts on 3+ identical messages within the window, or any exact replay within 100ms.

#### 13. `data_byte_0`–`data_byte_7` — Byte-Level Validation

Check specific byte positions for expected values. Useful for safety-critical command validation.

```yaml
- name: "Emergency Brake Override"
  can_id: 0x220
  data_byte_0: 0xFF  # Override command byte
  data_byte_1: 0x00  # Should be zero in normal operation
  severity: CRITICAL
  description: "Brake system override command detected"
  action: alert
```

#### 14. `whitelist_mode` — CAN ID Whitelist

Only allow messages from known-good CAN IDs. Everything else triggers an alert.

```yaml
- name: "Unknown CAN ID"
  whitelist_mode: true
  allowed_can_ids: [0x100, 0x110, 0x120, 0x200, 0x316]
  severity: MEDIUM
  description: "Message from CAN ID not in whitelist"
  action: alert
```

---

### Phase 3 — Specialized Parameters (4)

Added December 2, 2025. Tests: `tests/test_rule_engine_phase3.py` (21/21 passing).

#### 15. `check_data_integrity` — Data Integrity Validation

XOR checksum validation for safety-critical messages (brake, steering, airbag). Configurable checksum byte offset (defaults to last byte).

```yaml
- name: "Brake Data Integrity Failure"
  can_id: 0x220
  check_data_integrity: true
  integrity_checksum_offset: 7  # Checksum in last byte
  severity: CRITICAL
  description: "Brake system message failed integrity check"
  action: alert
```

#### 16. `check_steering_range` — Steering Angle Validation

Validates steering angle values are within physically possible limits. Angle extracted from bytes 0–1 as signed 16-bit little-endian with 0.1° resolution.

```yaml
- name: "Steering Angle Manipulation"
  can_id: 0x0C6
  check_steering_range: true
  steering_min: -540.0  # degrees
  steering_max: 540.0   # degrees
  severity: CRITICAL
  description: "Steering angle outside physical limits"
  action: alert
```

| Vehicle Type | Typical Range |
|---|---|
| Standard | ±540° (1.5 turns) |
| Sports | ±720° (2 turns) |
| Truck | ±360° (1 turn) |

#### 17. `check_repetition` — Repetitive Pattern Detection

Detects excessive consecutive identical messages, catching stuck sensors, pattern-based DoS, and fuzzing attempts.

```yaml
- name: "Repeated Data Pattern"
  can_id: 0x100
  check_repetition: true
  repetition_threshold: 10  # Max identical messages in a row
  severity: MEDIUM
  description: "Excessive identical messages (stuck sensor or attack)"
  action: alert
```

#### 18. `frame_type` — Frame Type Validation

Enforces standard vs. extended frame type. Prevents attacks that switch frame formats to bypass filters.

```yaml
- name: "Extended Frame in Standard Network"
  frame_type: standard
  severity: HIGH
  description: "Extended frame detected on standard-only network"
  action: alert
```

---

## Adaptive Timing Parameters

The three-tier timing system (added Dec 9–14, 2025) provides fine-grained control over timing-based detection. These parameters are used alongside `check_timing: true`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sigma_extreme` | — | Tier 1: σ multiplier for extreme deviations (e.g., 2.5–3.3) |
| `sigma_moderate` | — | Tier 2: σ multiplier for sustained violations (e.g., 1.3–1.7) |
| `consecutive_required` | — | Tier 2: how many consecutive violations before alerting |
| `payload_repetition_threshold` | — | Tier 3: fraction of identical payloads in window (0.0–1.0) |

**How it works**:

1. **Tier 1** — Single message with timing deviation > `sigma_extreme` × σ → immediate alert (DoS/flood)
2. **Tier 2** — `consecutive_required` messages each deviating > `sigma_moderate` × σ → sustained attack alert
3. **Tier 3** — If Tier 2 triggers, check payload repetition. If < `payload_repetition_threshold`, suppress (normal jitter). If ≥ threshold, confirm attack.

**Tuning guidance**:
- High-traffic CAN IDs (>10 msg/s): tighter `sigma_moderate` (1.3–1.4)
- Low-traffic CAN IDs (<1 msg/s): looser `sigma_moderate` (1.7–1.9)
- Use `scripts/generate_rules_from_baseline.py` to auto-generate thresholds from baseline traffic

---

## Example Rules

### DoS Attack Detection (Multi-Layer)

```yaml
- name: "Per-ID DoS Attack"
  can_id: 0x7E0
  max_frequency: 500
  time_window: 1
  severity: CRITICAL
  description: "Excessive messages targeting engine ECU"
  action: alert

- name: "Bus-Wide Flood"
  global_message_rate: 8000
  time_window: 1
  severity: CRITICAL
  description: "Total CAN bus flooding detected"
  action: alert
```

### Safety System Protection

```yaml
- name: "Brake Override Attempt"
  can_id: 0x220
  data_byte_0: 0xFF
  check_data_integrity: true
  integrity_checksum_offset: 7
  severity: CRITICAL
  description: "Brake override with integrity check"
  action: alert

- name: "Steering Manipulation"
  can_id: 0x0C6
  check_steering_range: true
  steering_min: -540.0
  steering_max: 540.0
  severity: CRITICAL
  description: "Steering angle beyond physical limits"
  action: alert
```

### Replay + Timing Combined

```yaml
- name: "Replay Attack on Brake"
  can_id: 0x200
  check_replay: true
  replay_time_threshold: 0.5
  check_timing: true
  expected_interval: 50
  interval_variance: 5
  severity: HIGH
  description: "Replayed brake message with timing violation"
  action: alert
```

### Fuzzing Detection

```yaml
- name: "CAN Fuzzing"
  validate_dlc: true
  check_frame_format: true
  entropy_threshold: 7.0
  severity: MEDIUM
  description: "Malformed or random data suggesting fuzzing"
  action: alert
```

---

## Best Practices

### 1. Start Simple
Begin with basic rules and add complexity as needed.

### 2. Use Descriptive Names
```yaml
# Good
name: "Unauthorized Steering Angle Manipulation"
# Bad
name: "Rule 42"
```

### 3. Set Appropriate Severity
- **CRITICAL**: Safety-critical (steering, brakes, airbags)
- **HIGH**: Security violations (unauthorized access, impersonation)
- **MEDIUM**: Policy violations (unexpected messages)
- **LOW**: Anomalies that may be benign

### 4. Use Priority for Early Exit
```yaml
priority: 0   # Critical rules — evaluated first, can short-circuit
priority: 5   # Normal rules (default)
priority: 10  # Low-priority rules — skip if critical alert already found
```

### 5. Generate Thresholds from Data
Instead of guessing thresholds, use baseline data:
```bash
python scripts/generate_rules_from_baseline.py \
  --input data/raw/attack_free_traffic.csv \
  --output config/rules_adaptive.yaml
```

### 6. Combine Rule Types
Multiple conditions on one rule are ANDed — all must match:
```yaml
- name: "Targeted Engine Attack"
  can_id: 0x7E0
  data_pattern: "10 02"
  max_frequency: 100
  check_timing: true
  severity: CRITICAL
  action: alert
```

---

## Testing Rules

### Syntax Validation

```bash
python -c "import yaml; yaml.safe_load(open('config/rules.yaml'))"
```

### Load Test

```bash
python -c "
from src.detection.rule_engine import RuleEngine
re = RuleEngine('config/rules.yaml')
print(f'Loaded {len(re.rules)} rules')
"
```

### Run the Test Suites

```bash
# All rule engine tests (61 tests)
python -m pytest tests/test_rule_engine_phase1.py tests/test_rule_engine_phase2.py tests/test_rule_engine_phase3.py -v

# Quick check
python -m pytest tests/ -k "rule_engine" --tb=short
```

### Integration Testing

```bash
# Test with a real dataset
python scripts/test_rules_on_dataset.py --rules config/rules.yaml --data test_data/attack-free-1.csv
```

---

## Troubleshooting

### Rule Not Triggering

1. **Check CAN ID format**: Must be hex with `0x` prefix (`0x7DF`, not `7DF`)
2. **Check data patterns**: Space-separated hex (`"10 01"`, not `"1001"`)
3. **Check boolean parameters**: `check_timing: true`, not `check_timing: "yes"`
4. **Enable debug logging**: `python main.py -i can0 --log-level DEBUG`

### Too Many False Positives

1. **Increase thresholds** — Loosen `max_frequency`, widen `interval_variance`
2. **Add Tier 3 payload check** — Add `payload_repetition_threshold: 0.55` to timing rules
3. **Use baseline generation** — `scripts/generate_rules_from_baseline.py` for data-driven thresholds
4. **Test with `action: log`** first, then switch to `alert`

### Performance Issues

1. **Reduce stateful rules** — `check_timing`, `check_replay`, `check_repetition` track state
2. **Use smaller time windows** — `time_window: 1`, not `time_window: 60`
3. **Set priorities** — Critical rules first, low-priority rules can be skipped via early exit

---

## Resources

### Configuration Files
- `config/rules.yaml` — Production rules
- `config/rules_adaptive.yaml` — Auto-generated adaptive rules
- `config/example_rules.yaml` — Templates
- `config/fuzzing_detection_rules.yaml` — Fuzzing-specific rules

### Tools
- `scripts/generate_rules_from_baseline.py` — Generate rules from normal traffic
- `scripts/test_rules_on_dataset.py` — Test rules against datasets
- `scripts/benchmark.py` — Performance benchmarking

### Documentation
- [Implementation Status](../implementation/IMPLEMENTATION_STATUS.md) — Current feature status
- [Feature Inventory](../implementation/UNIMPLEMENTED_FEATURES.md) — All features with code references
- [Timing Tuning](../implementation/TIMING_DETECTION_TUNING.md) — Statistical details on timing detection

---

**Last Updated**: March 1, 2026  
**Version**: 2.0.0 — Covers all 18 rule types
