# CRITICAL FINDING: ML Detection Not Enabled in Tests

**Date**: December 1, 2025  
**Discovered During**: Contamination parameter testing session

---

## Summary

**All testing conducted so far has been WITHOUT machine learning detection enabled.**

The comprehensive testing framework (`scripts/comprehensive_test.py`) and batch test script (`scripts/batch_test_set01.sh`) run with ML detection **disabled by default**, meaning:

- All 19.2 million messages tested were processed by **rule-based detection only**
- The ML contamination parameter (0.02 or 0.20) has **never been tested**
- All false positives observed are from **aggressive rules, not ML**
- The Vehicle_Models comparison is invalid (they likely run with ML enabled)

---

## Evidence

### 1. ML Disabled in comprehensive_test.py

**Line 371** in `scripts/comprehensive_test.py`:
```python
ml_detector = None
if config.get('enable_ml', False):  # ← Defaults to False!
    ml_detector = MLDetector()
```

### 2. Batch Script Doesn't Enable ML

**Lines 77-80** in `scripts/batch_test_set01.sh`:
```bash
python "$PROJECT_DIR/scripts/comprehensive_test.py" \
    "$DATA_DIR/$csv_file" \
    --output "$OUTPUT_DIR" \
    --rules "$RULES_FILE"
    # ← Missing: --enable-ml flag
```

### 3. Test Output Confirms It

From `comprehensive_summary.json`:
```json
{
    "test_info": {
        "config": {
            "enable_ml": false  // ← Explicitly disabled
        }
    }
}
```

---

## Impact on Previous Testing

### Test Run 1: contamination=0.02 (batch_set01_20251130_210252)
- **ML Status**: Disabled
- **Detections**: 100% from rule-based engine
- **Results**: 100% recall, 0.06-10% precision, high false positives

### Test Run 2: contamination=0.20 (batch_set01_20251130_231940)
- **ML Status**: Disabled
- **Detections**: 100% from rule-based engine
- **Results**: IDENTICAL to Test Run 1 (as expected with ML disabled)

### Why Results Were Identical
Both test runs produced the same results because:
1. ML detector was never initialized (`ml_detector = None`)
2. Only rule engine processed messages
3. Contamination parameter never used
4. Code path at line 445-446 never executed:
   ```python
   if ml_detector:  # ← Always False
       ml_alert = ml_detector.analyze_message(msg)
   ```

---

## Root Cause of False Positives

All false positives (90-100% FP rate) come from **overly aggressive rules**:

### High False Positive Rules
1. **"Unknown CAN ID"** - Flags every new CAN ID as suspicious
2. **"High Entropy Data"** - Triggers on normal randomized data patterns
3. **"Counter Sequence Error"** - Too strict on message ordering
4. **"Checksum Validation Failure"** - False positives on legitimate traffic

### Attack-Free Dataset Results
- **attack-free-1**: 1,952,833 false positives (100% FP rate)
- **attack-free-2**: 1,265,599 false positives (100% FP rate)

---

## What This Means for ML Contamination Parameter

### Vehicle_Models Comparison Invalid
The TONIGHT_SUMMARY.md and DETECTION_TUNING_COMPARISON.md claimed:
- Vehicle_Models: 95-100% recall, 74-100% precision ✅
- CANBUS_IDS: 0-10% recall, 0.06-10% precision ❌

**But this comparison is meaningless because:**
- Vehicle_Models likely runs with **ML enabled**
- CANBUS_IDS tests run with **ML disabled**
- We're comparing apples (ML+rules) to oranges (rules only)

### Contamination Parameter Never Tested
The contamination parameter (0.02 vs 0.20) has **zero relevance** to our tests because:
- Parameter only affects `IsolationForest` in `MLDetector`
- `MLDetector` was never instantiated
- No ML inference was ever performed

---

## Action Items

### To Actually Test ML Detection

1. **Enable ML in batch_test_set01.sh**:
   ```bash
   python "$PROJECT_DIR/scripts/comprehensive_test.py" \
       "$DATA_DIR/$csv_file" \
       --output "$OUTPUT_DIR" \
       --rules "$RULES_FILE" \
       --enable-ml  # ← ADD THIS FLAG
   ```

2. **Train ML Models First**:
   - Use attack-free-1.csv (1.9M messages) for training baseline
   - Use attack-free-2.csv (1.2M messages) for validation
   - Save trained model to `data/models/`

3. **Re-run All Tests with ML Enabled**:
   - Test contamination=0.02 (original)
   - Test contamination=0.20 (from Vehicle_Models)
   - Compare actual ML performance

### To Reduce Rule-Based False Positives

1. **Tune Rule Thresholds** (immediate priority):
   - Increase entropy threshold (less sensitive)
   - Widen counter sequence tolerance
   - Build CAN ID whitelist from attack-free data
   - Adjust checksum validation sensitivity

2. **Implement Ensemble Detection**:
   - Require both ML and rules to agree before alerting
   - Use ML to filter rule-based alerts
   - Reduce false positives while maintaining recall

3. **Add Vehicle Calibration**:
   - Learn normal CAN ID ranges from baseline data
   - Establish typical data patterns per CAN ID
   - Create per-vehicle baseline profiles

---

## Test Results Summary (ML Disabled)

### All 12 Datasets (19.2M messages total)

| Dataset | Precision | Recall | F1-Score | True Positives | False Positives |
|---------|-----------|--------|----------|----------------|-----------------|
| DoS-1 | 10.14% | 100% | 0.184 | 9,139 | 81,030 |
| DoS-2 | 8.34% | 100% | 0.154 | 25,954 | 285,091 |
| force-neutral-1 | 0.91% | 100% | 0.018 | 6,500 | 708,935 |
| force-neutral-2 | 0.05% | 100% | 0.001 | 492 | 997,993 |
| rpm-1 | 0.40% | 100% | 0.008 | 3,338 | 837,715 |
| rpm-2 | 0.05% | 100% | 0.001 | 374 | 822,085 |
| standstill-1 | 0.11% | 100% | 0.002 | 2,215 | 1,952,833 |
| standstill-2 | 0.12% | 100% | 0.002 | 1,545 | 1,265,599 |
| accessory-1 | 0.00% | 0% | 0.000 | 0 | 207,704 |
| accessory-2 | 0.00% | 0% | 0.000 | 0 | 226,166 |
| attack-free-1 | 0.00% | 0% | 0.000 | 0 | 1,952,833 |
| attack-free-2 | 0.00% | 0% | 0.000 | 0 | 1,265,599 |

**Key Observations**:
- ✅ **100% recall** on attack datasets (all attacks detected)
- ❌ **0.05-10% precision** (90-99.95% false positive rate)
- ❌ **100% FP rate** on attack-free datasets (every message flagged)

---

## Conclusions

1. **ML Detection Has Never Been Tested**: All 19.2M messages were processed by rules only
2. **Contamination Parameter Irrelevant**: Can't affect performance when ML is disabled
3. **Rules Are Too Aggressive**: 90-100% false positive rate on normal traffic
4. **Vehicle_Models Comparison Invalid**: Comparing different detection modes
5. **Next Step**: Enable ML and re-test to see actual ML+contamination impact

---

## References

- `scripts/comprehensive_test.py` - Lines 371-372 (ML initialization)
- `scripts/batch_test_set01.sh` - Lines 77-80 (test invocation)
- `docs/SESSION_LOG_20251130.md` - Original test results documentation
- `TONIGHT_SUMMARY.md` - Contamination parameter analysis (now known to be invalid)
- `docs/DETECTION_TUNING_COMPARISON.md` - Vehicle_Models comparison (apples-to-oranges)

---

**Status**: ML detection testing remains to be done. Current results only reflect rule-based detection performance.
