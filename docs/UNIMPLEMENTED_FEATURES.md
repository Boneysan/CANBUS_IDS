# Unimplemented Features

## Detection Rule Parameters

The following advanced rule parameters are defined in `config/rules.yaml` but not yet implemented in the `DetectionRule` class (`src/detection/rule_engine.py`):

### 1. Source Validation
- **Parameter:** `check_source`
- **Used in:** Unauthorized OBD-II Diagnostic Request
- **Purpose:** Validate the source of diagnostic requests
- **Status:** Not implemented

### 2. Data Length Code Validation
- **Parameter:** `validate_dlc`
- **Used in:** Invalid Data Length Code
- **Purpose:** Validate DLC field against expected values
- **Status:** Not implemented

### 3. Frame Format Checking
- **Parameter:** `check_frame_format`
- **Used in:** Malformed CAN Frame
- **Purpose:** Verify CAN frame structure and format
- **Status:** Not implemented

### 4. Global Message Rate Monitoring
- **Parameter:** `global_message_rate`
- **Used in:** Bus Flooding Attack
- **Purpose:** Monitor overall bus message rate for flooding detection
- **Status:** Not implemented

### 5. Replay Attack Detection
- **Parameter:** `check_replay`
- **Used in:** Exact Message Replay
- **Purpose:** Detect identical message replays
- **Status:** Not implemented

### 6. Data Integrity Verification
- **Parameter:** `check_data_integrity`
- **Used in:** Brake System Manipulation
- **Purpose:** Verify data integrity for critical systems
- **Status:** Not implemented

### 7. Specific Byte Value Checking
- **Parameter:** `data_byte_0` (and similar)
- **Used in:** Emergency Brake Override
- **Purpose:** Check specific byte values in data field
- **Status:** Not implemented

### 8. Steering Range Validation
- **Parameter:** `check_steering_range`
- **Used in:** Steering Angle Manipulation
- **Purpose:** Validate steering angle values are within acceptable range
- **Status:** Not implemented

### 9. Repetition Pattern Detection
- **Parameter:** `check_repetition`
- **Used in:** Repeated Data Pattern
- **Purpose:** Detect repetitive data patterns that may indicate attacks
- **Status:** Not implemented

### 10. Frame Type Validation
- **Parameter:** `frame_type`
- **Used in:** Extended Frame in Standard Network
- **Purpose:** Validate extended vs standard CAN frame types
- **Status:** Not implemented

---

## Currently Implemented Rules

The following detection rules are **fully implemented and working**:

1. ✅ **High Entropy Data** - Detects unusual entropy in message data
2. ✅ **Unknown CAN ID** - Flags messages from unrecognized CAN IDs
3. ✅ **Checksum Validation Failure** - Validates message checksums
4. ✅ **Counter Sequence Error** - Detects sequence counter anomalies
5. ✅ **Seed Request Pattern** - Detects security access seed patterns

---

## Impact Assessment

### Current Performance
- **Detection Rate:** 100% recall on DoS attacks (no attacks missed)
- **False Positive Rate:** High (~90% for normal traffic)
- **Precision:** Low (8-10% for DoS, <1% for other attacks)

### Recommendations
1. **Priority 1:** Implement `check_source`, `validate_dlc`, and `check_frame_format` for basic validation
2. **Priority 2:** Implement `global_message_rate` for better bus flooding detection
3. **Priority 3:** Implement remaining features for specialized attack detection
4. **Alternative:** Calibrate existing rules to reduce false positives before adding new features

---

## Testing Notes

- Last tested: November 30, 2025
- Test dataset: Set 01 (12 CSV files, ~6.6M messages total)
- All tests completed successfully despite warnings about unimplemented features
- System performance: ~10,000 msg/s throughput, ~25% CPU usage on Raspberry Pi 4

---

## Next Steps

1. Review `config/rules.yaml` to understand parameter requirements
2. Extend `DetectionRule` dataclass in `src/detection/rule_engine.py`
3. Implement validation logic in `_evaluate_rule()` method
4. Add unit tests for new parameters
5. Re-run batch tests to measure improvement in precision
