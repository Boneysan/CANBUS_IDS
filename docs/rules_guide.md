# CAN-IDS Detection Rules Writing Guide

This guide explains how to write custom detection rules for the CAN-IDS signature-based detection engine.

## Table of Contents

1. [Rule Basics](#rule-basics)
2. [Rule Structure](#rule-structure)
3. [Rule Parameters](#rule-parameters)
4. [Detection Methods](#detection-methods)
5. [Example Rules](#example-rules)
6. [Best Practices](#best-practices)
7. [Testing Rules](#testing-rules)
8. [Troubleshooting](#troubleshooting)

---

## Rule Basics

Detection rules are defined in YAML format and stored in `config/rules.yaml`. Each rule describes a specific attack pattern, policy violation, or anomalous behavior to detect.

### Basic Concepts

- **Rules are stateful** - Can track message history, timing, and frequency
- **Rules are composable** - Multiple conditions can be combined
- **Rules generate alerts** - When matched, create alerts with severity levels
- **Rules are hot-reloadable** - Can be updated without restarting the system

### Rule File Location

```
config/
├── rules.yaml          # Active rules (loaded by IDS)
└── example_rules.yaml  # Template/examples
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

```yaml
rules:
  - name: "Rule Name"                    # Required: Unique rule identifier
    severity: CRITICAL|HIGH|MEDIUM|LOW   # Required: Alert severity
    description: "What this detects"     # Required: Human-readable description
    action: alert|log|block              # Required: Action to take (block not yet implemented)
    
    # CAN ID Matching (choose one)
    can_id: 0x123                        # Exact CAN ID match
    can_id_range: [0x100, 0x200]        # Range of CAN IDs (inclusive)
    
    # Data Matching
    data_pattern: "DE AD BE EF"          # Hex pattern to match (space-separated)
    data_contains: ["27 01", "27 03"]   # List of patterns (any match)
    data_byte_0: 0xFF                    # Match specific byte position
    
    # Frame Validation
    dlc_min: 0                           # Minimum DLC (0-8)
    dlc_max: 8                           # Maximum DLC (0-8)
    validate_dlc: true                   # Enable DLC validation
    check_frame_format: true             # Check frame structure
    frame_type: standard|extended        # Expected frame type
    
    # Frequency Detection
    max_frequency: 1000                  # Max messages per second
    time_window: 1                       # Time window in seconds
    global_message_rate: 5000           # Global rate across all IDs
    
    # Timing Analysis
    check_timing: true                   # Enable timing checks
    expected_interval: 100               # Expected interval in ms
    interval_variance: 10                # Allowed variance in ms
    
    # Replay Detection
    check_replay: true                   # Enable replay detection
    replay_window: 5000                  # Detection window in ms
    
    # Source Validation
    check_source: true                   # Enable source checking
    allowed_sources: [0x10, 0x20]       # List of allowed source IDs
    
    # Data Integrity
    check_checksum: true                 # Validate message checksum
    check_counter: true                  # Validate sequence counter
    check_data_integrity: true           # General integrity check
    
    # Pattern Analysis
    entropy_threshold: 7.5               # Max Shannon entropy (bits)
    check_repetition: true               # Detect repeated patterns
    max_repetitions: 10                  # Max allowed repetitions
    
    # Network Topology
    whitelist_mode: true                 # Whitelist mode (vs blacklist)
    allowed_can_ids: [0x100, 0x200]     # Allowed CAN IDs in whitelist
    
    # Steering/Safety Specific
    check_steering_range: true           # Validate steering angle
    max_steering_angle: 540              # Max degrees
```

---

## Rule Parameters

### Required Parameters

#### `name` (string)
Unique identifier for the rule. Used in alerts and logs.

```yaml
name: "Unauthorized Diagnostic Access"
```

#### `severity` (string)
Alert severity level. Affects alert priority and filtering.

Options:
- `CRITICAL` - Immediate threat requiring urgent response
- `HIGH` - Serious security violation
- `MEDIUM` - Suspicious activity worth investigating
- `LOW` - Minor anomaly or policy violation

```yaml
severity: HIGH
```

#### `description` (string)
Human-readable description of what the rule detects.

```yaml
description: "Detects unauthorized OBD-II diagnostic requests"
```

#### `action` (string)
Action to take when rule matches.

Options:
- `alert` - Generate alert and send notifications
- `log` - Log to file only (no alerts)
- `block` - Block message (not yet implemented)

```yaml
action: alert
```

### CAN ID Matching

#### `can_id` (hex integer)
Match exact CAN identifier.

```yaml
can_id: 0x7DF  # OBD-II request ID
```

#### `can_id_range` (array)
Match range of CAN IDs (inclusive).

```yaml
can_id_range: [0x7E0, 0x7E7]  # UDS diagnostic range
```

**Note:** Use either `can_id` OR `can_id_range`, not both.

### Data Pattern Matching

#### `data_pattern` (string)
Match hex data pattern. Use `*` as wildcard.

```yaml
# Match exact pattern
data_pattern: "10 01"

# Match with wildcard
data_pattern: "27 * * *"  # Match 0x27 in first byte, any data after

# Match partial pattern
data_pattern: "DE AD BE EF"  # Must match first 4 bytes
```

#### `data_contains` (array)
Match if data contains any of the specified patterns.

```yaml
data_contains:
  - "27 01"  # Security Access - Request Seed
  - "27 02"  # Security Access - Send Key
  - "27 03"  # Security Access - Request Seed (Alt)
```

#### `data_byte_N` (hex integer)
Match specific byte at position N (0-7).

```yaml
data_byte_0: 0xFF  # First byte must be 0xFF
data_byte_1: 0x00  # Second byte must be 0x00
```

### Frequency Detection

#### `max_frequency` (integer)
Maximum allowed messages per `time_window`.

```yaml
can_id: 0x100
max_frequency: 100    # Max 100 messages
time_window: 1        # Per second
```

Detects DoS attacks and bus flooding.

#### `global_message_rate` (integer)
Maximum message rate across ALL CAN IDs.

```yaml
global_message_rate: 5000  # Max 5000 total messages/second
time_window: 1
```

### Timing Analysis

#### `check_timing` (boolean)
Enable timing analysis for this rule.

```yaml
check_timing: true
expected_interval: 100      # Messages should arrive every 100ms
interval_variance: 10       # ±10ms tolerance
```

Detects replay attacks and timing violations.

### Frame Validation

#### `dlc_min` / `dlc_max` (integer 0-8)
Validate Data Length Code.

```yaml
dlc_min: 8  # Must have at least 8 bytes
dlc_max: 8  # Must have exactly 8 bytes
```

#### `validate_dlc` (boolean)
Enable general DLC validation.

```yaml
validate_dlc: true
```

### Whitelist/Blacklist Mode

#### `whitelist_mode` (boolean)
Enable whitelist mode (only allowed IDs pass).

```yaml
whitelist_mode: true
allowed_can_ids: [0x100, 0x200, 0x300]  # Only these IDs allowed
```

Any CAN ID not in the list will trigger an alert.

---

## Detection Methods

### 1. Simple ID Matching

Detect messages from specific CAN ID:

```yaml
- name: "Suspicious ECU Communication"
  can_id: 0x666
  severity: HIGH
  description: "Unknown ECU with suspicious ID"
  action: alert
```

### 2. Pattern Matching

Detect specific data patterns:

```yaml
- name: "UDS Programming Session"
  can_id: 0x7E0
  data_pattern: "10 02"  # Programming session request
  severity: CRITICAL
  description: "ECU programming session initiated"
  action: alert
```

### 3. Frequency-Based Detection

Detect high-frequency attacks:

```yaml
- name: "DoS Attack on Engine ECU"
  can_id: 0x7E0
  max_frequency: 500
  time_window: 1
  severity: CRITICAL
  description: "Excessive messages targeting engine ECU"
  action: alert
```

### 4. Timing Analysis

Detect replay attacks via timing:

```yaml
- name: "Replay Attack Detection"
  can_id: 0x200
  check_timing: true
  expected_interval: 100  # Normal: every 100ms
  interval_variance: 5    # ±5ms tolerance
  severity: HIGH
  description: "Message timing inconsistent with normal pattern"
  action: alert
```

### 5. Data Integrity

Validate checksums and counters:

```yaml
- name: "Message Counter Error"
  can_id: 0x300
  check_counter: true
  severity: MEDIUM
  description: "Message sequence counter out of order"
  action: alert
```

### 6. Entropy Analysis

Detect encrypted or obfuscated data:

```yaml
- name: "High Entropy Data"
  entropy_threshold: 7.5  # High randomness
  severity: LOW
  description: "Unusually random data (possible encryption)"
  action: log
```

### 7. Whitelist Enforcement

Only allow known good IDs:

```yaml
- name: "Unknown CAN ID"
  whitelist_mode: true
  allowed_can_ids: [0x100, 0x110, 0x120, 0x200]
  severity: MEDIUM
  description: "Message from unauthorized CAN ID"
  action: alert
```

---

## Example Rules

### Example 1: OBD-II Attack Detection

```yaml
- name: "Unauthorized OBD-II Access"
  can_id: 0x7DF
  severity: HIGH
  description: "OBD-II diagnostic request detected"
  action: alert
  check_source: true
  allowed_sources: []  # No sources allowed = always alert
```

### Example 2: ECU Impersonation

```yaml
- name: "Fake Engine Control Unit"
  can_id: 0x7E8  # Engine ECU response ID
  allowed_sources: [0x7E0]  # Only from engine ECU
  severity: CRITICAL
  description: "Message from unauthorized source impersonating ECU"
  action: alert
```

### Example 3: Safety System Manipulation

```yaml
- name: "Brake System Override"
  can_id: 0x220  # Brake system ID
  data_byte_0: 0xFF  # Override command
  severity: CRITICAL
  description: "Potential brake system manipulation"
  action: alert
```

### Example 4: Fuzzing Detection

```yaml
- name: "CAN Fuzzing Attack"
  dlc_min: 0
  dlc_max: 8
  validate_dlc: true
  entropy_threshold: 7.0
  severity: MEDIUM
  description: "Random/malformed CAN frames (fuzzing)"
  action: alert
```

### Example 5: Bus Flooding

```yaml
- name: "CAN Bus Flood"
  global_message_rate: 8000
  time_window: 1
  severity: CRITICAL
  description: "CAN bus flooding attack detected"
  action: alert
```

---

## Best Practices

### 1. Start Simple
Begin with basic rules and add complexity as needed.

```yaml
# Good: Simple and clear
- name: "Diagnostic Request"
  can_id: 0x7DF
  severity: HIGH
  description: "OBD-II request detected"
  action: alert

# Avoid: Overly complex first rule
- name: "Complex Multi-Condition Rule"
  can_id_range: [0x7E0, 0x7EF]
  check_timing: true
  check_counter: true
  entropy_threshold: 6.5
  max_frequency: 100
  # ... too many conditions
```

### 2. Use Descriptive Names
Make rule names clear and searchable.

```yaml
# Good
name: "Unauthorized Steering Angle Manipulation"

# Bad
name: "Rule 42"
```

### 3. Set Appropriate Severity
Match severity to actual threat level.

- **CRITICAL**: Safety-critical systems (steering, brakes, airbags)
- **HIGH**: Security violations (unauthorized access, ECU impersonation)
- **MEDIUM**: Policy violations (unexpected messages)
- **LOW**: Anomalies that may be benign

### 4. Document Your Rules
Use clear descriptions that explain:
- What the rule detects
- Why it's important
- What normal behavior looks like

```yaml
- name: "Engine RPM Manipulation"
  can_id: 0x316
  severity: HIGH
  description: "Detects unauthorized engine RPM control messages. Normal RPM messages only come from ECU 0x7E0. This rule catches potential engine control hijacking attempts."
  action: alert
```

### 5. Test Before Deploying
Always test new rules on sample data before deploying to production.

```bash
# Generate test data
python scripts/generate_dataset.py --type normal

# Test rule against data
python main.py --mode replay --file data/synthetic/normal_traffic.json
```

### 6. Tune Thresholds
Start conservative and adjust based on false positives.

```yaml
# Initial rule (may have false positives)
max_frequency: 100

# After testing, adjust
max_frequency: 500  # Based on observed normal traffic
```

### 7. Use Comments
Document complex rules with YAML comments.

```yaml
- name: "UDS Security Access"
  can_id_range: [0x7E0, 0x7E7]
  # Service 0x27 with sub-functions:
  # 01, 03, 05 = Request Seed
  # 02, 04, 06 = Send Key
  data_pattern: "27 *"
  severity: HIGH
  description: "UDS security access attempt"
  action: alert
```

---

## Testing Rules

### 1. Syntax Validation

Check YAML syntax:

```bash
# Python YAML validation
python -c "import yaml; yaml.safe_load(open('config/rules.yaml'))"

# Or use yamllint
yamllint config/rules.yaml
```

### 2. Load Test

Test if rules load correctly:

```bash
python -c "from src.detection.rule_engine import RuleEngine; re = RuleEngine('config/rules.yaml'); print(f'Loaded {len(re.rules)} rules')"
```

### 3. Unit Testing

Create test cases for your rules:

```python
# tests/test_my_rules.py
def test_dos_detection():
    engine = RuleEngine('config/rules.yaml')
    
    # Simulate DoS attack
    for i in range(1500):
        message = {
            'timestamp': time.time(),
            'can_id': 0x100,
            'dlc': 8,
            'data': [0x00] * 8
        }
        alerts = engine.analyze_message(message)
    
    # Should detect high frequency
    assert len(alerts) > 0
    assert any('frequency' in a.description.lower() for a in alerts)
```

### 4. Integration Testing

Test with synthetic data:

```bash
# Generate attack data
python scripts/generate_dataset.py --type dos --duration 10

# Test detection
python main.py --mode replay --file data/synthetic/dos_attack.json

# Check alerts
cat logs/alerts.json | grep "High Frequency"
```

### 5. Benchmark Performance

Measure rule performance impact:

```bash
python scripts/benchmark.py --component rule-engine --data data/synthetic/normal_traffic.json
```

---

## Troubleshooting

### Rule Not Triggering

**Problem**: Rule never generates alerts even when expected.

**Solutions**:
1. Check CAN ID matching:
   ```yaml
   # Make sure CAN ID format is correct
   can_id: 0x7DF  # Not 7DF or 0x7df
   ```

2. Verify data pattern:
   ```yaml
   # Patterns are space-separated hex
   data_pattern: "10 01"  # Not "1001" or "10:01"
   ```

3. Check timing parameters:
   ```yaml
   # Ensure thresholds are reasonable
   max_frequency: 1000  # Not too high
   time_window: 1       # 1 second windows
   ```

4. Enable debug logging:
   ```bash
   python main.py -i can0 --log-level DEBUG
   ```

### Too Many False Positives

**Problem**: Rule triggers on normal traffic.

**Solutions**:
1. Increase thresholds:
   ```yaml
   # Before
   max_frequency: 100
   
   # After (more lenient)
   max_frequency: 500
   ```

2. Add more specific conditions:
   ```yaml
   # Before: Too broad
   can_id_range: [0x100, 0x7FF]
   
   # After: More specific
   can_id_range: [0x7E0, 0x7E7]
   data_pattern: "27 *"
   ```

3. Use `action: log` during testing:
   ```yaml
   action: log  # Log only, no alerts
   ```

### Performance Issues

**Problem**: Rules slow down detection.

**Solutions**:
1. Reduce stateful rules:
   ```yaml
   # Timing checks are expensive
   check_timing: true  # Use sparingly
   ```

2. Optimize frequency checks:
   ```yaml
   # Smaller time windows = less history
   time_window: 1  # Not 60
   ```

3. Profile specific rules:
   ```bash
   python scripts/benchmark.py --component rule-engine
   ```

### Rule Loading Errors

**Problem**: Rules fail to load.

**Solutions**:
1. Check YAML syntax:
   ```bash
   yamllint config/rules.yaml
   ```

2. Verify required fields:
   ```yaml
   # Must have: name, severity, description, action
   - name: "Test"
     severity: HIGH
     description: "Test rule"
     action: alert
   ```

3. Check for duplicate names:
   ```yaml
   # Each rule name must be unique
   - name: "Unique Name 1"
   - name: "Unique Name 2"
   ```

---

## Advanced Topics

### Combining Multiple Conditions

Rules with multiple conditions must match ALL conditions:

```yaml
- name: "Complex Attack Pattern"
  can_id: 0x7E0           # AND
  data_pattern: "27 01"   # AND
  max_frequency: 10       # AND
  time_window: 1          # All must match
  severity: HIGH
  description: "Complex multi-stage attack"
  action: alert
```

### Dynamic Rule Updates

Reload rules without restarting:

```python
# In code
rule_engine.reload_rules()

# Or via signal (future feature)
kill -HUP <pid>
```

### Rule Priority

Rules are evaluated in order. Put high-priority rules first:

```yaml
rules:
  # Critical rules first
  - name: "Safety System Attack"
    severity: CRITICAL
    # ...
    
  # Lower priority rules later
  - name: "Minor Anomaly"
    severity: LOW
    # ...
```

### Statistical Confidence

Some rules calculate confidence scores (0.0-1.0):

- Pattern matches: Higher confidence
- Frequency violations: Medium confidence
- Single condition: Lower confidence

---

## Resources

### Related Documentation
- [Main README](../README.md)
- [Getting Started Guide](../GETTING_STARTED.md)
- [Configuration Guide](configuration.md)
- [Test Suite](../tests/README.md)

### Example Rules
- `config/rules.yaml` - Production rules
- `config/example_rules.yaml` - Templates

### Tools
- `scripts/generate_dataset.py` - Generate test data
- `scripts/benchmark.py` - Performance testing
- `tests/test_detection.py` - Rule engine tests

### CAN Bus References
- [SocketCAN Documentation](https://www.kernel.org/doc/html/latest/networking/can.html)
- [UDS Protocol Specification](https://www.iso.org/standard/72439.html)
- [OBD-II Standards](https://www.sae.org/standards/content/j1979_202104/)

---

## Need Help?

- **Bug Reports**: GitHub Issues
- **Questions**: Check existing rules in `config/rules.yaml`
- **Performance**: Run benchmarks with `scripts/benchmark.py`
- **Testing**: See `tests/test_detection.py` for examples

---

**Last Updated**: October 24, 2025  
**Version**: 1.0.0
