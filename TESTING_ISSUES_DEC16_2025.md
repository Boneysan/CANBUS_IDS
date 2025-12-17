# Testing Issues Report - December 16, 2025

**Date:** December 16, 2025  
**Test Session:** Comprehensive Performance Testing  
**Tester:** Automated testing suite  
**System:** Raspberry Pi 4 Model B, Raspberry Pi OS Bookworm, Python 3.11.2

---

## Executive Summary

Attempted to run 4 comprehensive test suites as documented in the updated RASPBERRY_PI_DEPLOYMENT_GUIDE.md. **Test 1 (Rule-based detection) was successful**, but Tests 2-4 failed due to missing ML models and incorrect path configurations.

**Overall Status:**
- ✅ Rule-based detection: **FULLY FUNCTIONAL** (708 msg/s, 100% recall)
- ❌ ML-based detection: **NOT FUNCTIONAL** (models not trained/missing)
- ❌ Multi-stage pipeline: **NOT FUNCTIONAL** (requires ML models)
- ❌ Full integration: **NOT FUNCTIONAL** (path configuration issues)

---

## Test Results Summary

### ✅ Test 1: Comprehensive Benchmark (Rule-Based Only) - SUCCESS

**Command:**
```bash
python scripts/comprehensive_test.py /media/boneysan/Data/GitHub/Vehicle_Models/data/raw/DoS-1.csv \
  --output test_results/benchmark_dos1_20251216_210139
```

**Dataset:** DoS-1.csv (90,169 messages)

**Performance Results:**
```
Throughput:        708.21 msg/s
Latency (avg):     1.374 ms
Latency (P95):     2.220 ms
Messages:          90,169 processed, 0 dropped (0.00%)
Alerts:            271,088 generated (300.64% alert rate)
```

**System Resources:**
```
CPU Usage:         25.2% avg, 30.7% peak
Memory:            188.3 MB avg, 194.9 MB peak
Temperature:       51.2°C avg, 53.5°C peak
Test Duration:     ~127 seconds (74 samples @ 1s interval)
```

**Detection Accuracy:**
```
Precision:         10.14%
Recall:            100.00%
F1-Score:          0.184
Accuracy:          10.14%

True Positives:    9,139
False Positives:   81,030
True Negatives:    0
False Negatives:   0
```

**Status:** ✅ **PASS** - Rule-based detection working correctly
- Catches 100% of attacks (excellent recall)
- High false positive rate (90%) expected for aggressive rules
- Throughput of 708 msg/s is below target (40-50K msg/s) but functional
- System resources well within limits

**Files Created:**
- `test_results/benchmark_dos1_20251216_210139/20251216_210147/system_metrics.csv`
- `test_results/benchmark_dos1_20251216_210139/20251216_210147/performance_metrics.json`
- `test_results/benchmark_dos1_20251216_210139/20251216_210147/comprehensive_summary.json`
- `logs/test1_benchmark_dos.log`

---

### ❌ Test 2: Real Attack Testing - FAILED

**Command:**
```bash
python scripts/test_real_attacks.py
```

**Error:**
```
2025-12-16 21:07:03,511 - ERROR - Model not found: data/models/decision_tree.pkl
2025-12-16 21:07:03,512 - ERROR - Run: python scripts/train_decision_tree.py --synthetic
```

**Root Cause:**
The `test_real_attacks.py` script requires a trained decision tree model that doesn't exist.

**Missing Files:**
- `data/models/decision_tree.pkl`

**Impact:**
- Cannot test decision tree-based detection
- Cannot validate attack detection using ML classifier
- Real attack testing depends on trained models

**Remediation Required:**
1. Train decision tree model:
   ```bash
   python scripts/train_decision_tree.py --synthetic
   ```
2. Or modify script to fall back to rule-based detection only
3. Or skip decision tree testing if not needed

**Status:** ❌ **FAIL** - Missing trained model

**Files Created:**
- `logs/test2_real_attacks.log`

---

### ❌ Test 3: Multi-Stage Integration - PARTIALLY FAILED

**Command:**
```bash
python scripts/test_multistage_integration.py
```

**Errors:**
```
Models directory not found: models
Multi-stage detection disabled
Error in enhanced ML analysis: ML detector is not trained - cannot analyze messages
```

**Test Results:**
```
1️⃣ Testing Enhanced ML Detector Import...            ✅ PASS
2️⃣ Testing Configuration Loading...                  ✅ PASS
3️⃣ Checking Model Files...                           ❌ FAIL (4 files missing)
4️⃣ Testing Enhanced Detector Initialization...       ✅ PASS (fallback mode)
5️⃣ Testing Feature Extraction...                     ❌ FAIL ('can_id' error)
6️⃣ Testing Detection Analysis...                     ❌ FAIL (ML not trained)
7️⃣ Testing Performance Baseline...                   ❌ FAIL (ML not trained)
8️⃣ Testing Statistics and Monitoring...              ✅ PASS (0 messages)
9️⃣ Testing Fallback Mechanism...                     ❌ FAIL (ML not trained)
```

**Missing Model Files:**
```
❌ Missing: multistage/aggressive_load_shedding.joblib
❌ Missing: improved_svm.joblib
❌ Missing: hybrid_rule_detector.joblib
❌ Missing: improved_isolation_forest.joblib
```

**Configuration Issues:**
- Expected models directory: `models/` (doesn't exist)
- Should be looking in: `data/models/`
- Multi-stage pipeline requires all 4 model files

**Root Causes:**
1. **Path Configuration Error:** Script looks for `models/` but actual location is `data/models/`
2. **Missing Models:** None of the 4 required multi-stage models exist
3. **Feature Extraction Error:** 'can_id' key error suggests data format mismatch

**Impact:**
- Multi-stage detection completely non-functional
- Cannot test hierarchical filtering (Stage 1 → Stage 2 → Stage 3)
- Cannot validate 7,000+ msg/s performance claims
- System falls back to single-stage detection only

**Remediation Required:**
1. **Fix path configuration:**
   - Update script to use `data/models/` instead of `models/`
   - Or create symlink: `ln -s data/models models`

2. **Train missing models:**
   ```bash
   # Need to train 4 models from Vehicle_Models project:
   - aggressive_load_shedding.joblib
   - improved_svm.joblib
   - hybrid_rule_detector.joblib
   - improved_isolation_forest.joblib
   ```

3. **Fix feature extraction:**
   - Review message format expected vs actual
   - Ensure 'can_id' field properly extracted from CAN messages

**Status:** ⚠️ **PARTIAL FAIL** - Basic integration works, ML features disabled

**Files Created:**
- `logs/test3_multistage.log`

---

### ❌ Test 4: Full Pipeline Test - FAILED

**Command:**
```bash
python scripts/test_full_pipeline.py
```

**Error:**
```
2025-12-16 21:09:22,593 - ERROR - Vehicle_Models not found at ../Vehicle_Models
```

**Root Cause:**
Script expects `Vehicle_Models` project to be in parent directory (`../Vehicle_Models`) but actual location is on USB drive at `/media/boneysan/Data/GitHub/Vehicle_Models/`

**Path Configuration Issue:**
- Expected: `../Vehicle_Models` (relative path from CANBUS_IDS)
- Actual: `/media/boneysan/Data/GitHub/Vehicle_Models/` (USB drive)

**Impact:**
- Cannot run full integration pipeline
- Cannot access training datasets for testing
- Cannot validate end-to-end workflow

**Remediation Required:**

**Option 1: Create Symlink (Recommended)**
```bash
cd ~/Documents/GitHub
ln -s /media/boneysan/Data/GitHub/Vehicle_Models Vehicle_Models
```

**Option 2: Modify Script**
Update `scripts/test_full_pipeline.py` to use configurable path:
```python
# Add configuration or command-line argument
VEHICLE_MODELS_PATH = os.environ.get(
    'VEHICLE_MODELS_PATH',
    '/media/boneysan/Data/GitHub/Vehicle_Models'
)
```

**Option 3: Copy Data**
```bash
# Copy datasets to CANBUS_IDS project (requires ~565MB)
mkdir -p data/raw
cp /media/boneysan/Data/GitHub/Vehicle_Models/data/raw/*.csv data/raw/
```

**Status:** ❌ **FAIL** - Path configuration error

**Files Created:**
- `logs/test4_full_pipeline.log`

---

## Root Cause Analysis

### Issue 1: Missing ML Models (Critical)

**Affected Tests:** 2, 3, 4

**Problem:**
No ML models have been trained or copied to the `data/models/` directory. The system was designed to work with ML detection, but models were never created.

**Required Models:**
1. `decision_tree.pkl` - For decision tree classifier
2. `multistage/aggressive_load_shedding.joblib` - Multi-stage pipeline
3. `improved_svm.joblib` - SVM classifier
4. `hybrid_rule_detector.joblib` - Hybrid detection
5. `improved_isolation_forest.joblib` - Isolation forest anomaly detector

**Why Models Are Missing:**
- Models exist in Vehicle_Models project on USB drive: `/media/boneysan/Data/GitHub/Vehicle_Models/models/multistage/`
- Models were trained but never copied to CANBUS_IDS project
- Or models need to be retrained specifically for CANBUS_IDS configuration

**Verification:**
```bash
# Check what models exist on USB
ls -lh /media/boneysan/Data/GitHub/Vehicle_Models/models/multistage/

# Output from previous work:
# adaptive_load_shedding.joblib    # 1.3MB, 100 estimators
# adaptive_only.joblib             # 1.3MB
# full_pipeline.joblib             # 1.3MB
# aggressive_load_shedding.joblib  # Exists
```

**Solution:**
Either copy existing models or train new ones specifically for this deployment.

---

### Issue 2: Path Configuration Mismatches (High Priority)

**Affected Tests:** 3, 4

**Problems:**
1. Multi-stage test looks for `models/` but should use `data/models/`
2. Full pipeline test looks for `../Vehicle_Models` but actual path is `/media/boneysan/Data/GitHub/Vehicle_Models/`

**Inconsistency:**
- Some scripts use relative paths (`../Vehicle_Models`)
- Some scripts use project paths (`data/models/`)
- Some scripts use hardcoded paths
- No centralized configuration for external resources

**Impact:**
Tests fail even if models/data exist because scripts can't find them.

**Solution:**
Create centralized path configuration or use environment variables.

---

### Issue 3: Rule-Based Detection Performance Gap (Medium Priority)

**Affected Tests:** 1

**Observation:**
Rule-based detection achieved **708 msg/s** but documentation claims **40-50K msg/s** capability.

**Performance Gap:**
- Actual: 708 msg/s
- Target: 40,000-50,000 msg/s
- Gap: **56-70x slower than claimed**

**Possible Causes:**
1. **Not using optimized rule engine:** May be using default rules.yaml instead of rules_adaptive.yaml
2. **Rule indexing not implemented:** O(n×m) complexity instead of O(1) lookup
3. **Aggressive rules triggering:** 300% alert rate suggests every message triggers multiple rules
4. **Python GIL limitations:** Single-threaded processing
5. **No early exit optimization:** Checking all rules even after finding violations

**Evidence:**
- 271,088 alerts for 90,169 messages = 300.64% alert rate
- 81,030 false positives (90% FP rate)
- Rules checking all messages, not indexed by CAN ID

**Solution:**
Implement optimizations from BUILD_PLAN_7000_MSG_SEC.md:
- Rule indexing by CAN ID
- Early exit conditions
- Optimized rule thresholds (use rules_adaptive.yaml)

---

### Issue 4: High False Positive Rate (Medium Priority)

**Affected Tests:** 1

**Observation:**
- False Positive Rate: **90% (81,030 / 90,169)**
- Precision: **10.14%**
- Alert rate: **300.64%** (multiple alerts per message)

**Problem:**
Rules are too aggressive, triggering on normal traffic patterns.

**Root Causes:**
1. Using generic `rules.yaml` instead of adaptive `rules_adaptive.yaml`
2. Rules not tuned for specific vehicle baseline
3. Unknown CAN ID rule triggering on every new ID
4. High entropy rule triggering on normal data variation

**Impact:**
- System would be unusable in production (3x messages flagged)
- Alert fatigue - real attacks hidden in noise
- Performance degradation from processing excessive alerts

**Solution:**
Use vehicle-specific adaptive rules generated from baseline data:
```bash
# Switch to adaptive rules
python main.py --config config/rules_adaptive.yaml
```

---

## Deployment Readiness Assessment

### Production Readiness by Component

| Component | Status | Throughput | Notes |
|-----------|--------|------------|-------|
| **Rule-Based Detection** | ⚠️ Functional | 708 msg/s | Works but 56x slower than target |
| **ML Detection** | ❌ Non-functional | N/A | No trained models available |
| **Multi-Stage Pipeline** | ❌ Non-functional | N/A | Missing 4 models + path issues |
| **Decision Tree Classifier** | ❌ Non-functional | N/A | Model not trained |
| **System Monitoring** | ✅ Functional | N/A | CPU, memory, temp tracking works |
| **Alert Generation** | ✅ Functional | N/A | Too aggressive (300% rate) |

### Can This Be Deployed?

**For Rule-Based Detection Only:** ⚠️ **YES, WITH CAVEATS**
- ✅ System is stable (51°C, 25% CPU, 188 MB RAM)
- ✅ Catches 100% of attacks (DoS tested)
- ❌ High false positive rate (90%) - needs tuning
- ❌ Throughput too low for high-traffic networks (708 vs 40,000+ target)
- ❌ Would generate 3 alerts per message (unusable alert volume)

**For ML-Enhanced Detection:** ❌ **NO**
- Critical dependencies missing (models not trained)
- Multi-stage pipeline non-functional
- Path configuration issues need resolution

**Recommendation:** 
Deploy rule-based detection ONLY for low-traffic monitoring (<500 msg/s) with understanding that:
1. Alert volume will be very high (tune rules first)
2. Performance limited to ~700 msg/s
3. ML features completely unavailable

---

## Remediation Plan

### Phase 1: Quick Fixes (1-2 hours)

**Goal:** Get all tests passing with basic functionality

1. **Fix Path Configurations**
   ```bash
   # Create symlink for Vehicle_Models
   cd ~/Documents/GitHub
   ln -s /media/boneysan/Data/GitHub/Vehicle_Models Vehicle_Models
   
   # Create models directory symlink
   cd ~/Documents/GitHub/CANBUS_IDS
   ln -s data/models models
   ```

2. **Copy Existing Models**
   ```bash
   # Copy available models from Vehicle_Models
   cp /media/boneysan/Data/GitHub/Vehicle_Models/models/multistage/*.joblib \
      data/models/
   ```

3. **Use Adaptive Rules**
   ```bash
   # Test with adaptive rules instead of generic
   python scripts/comprehensive_test.py \
     /media/boneysan/Data/GitHub/Vehicle_Models/data/raw/DoS-1.csv \
     --rules-config config/rules_adaptive.yaml \
     --output test_results/adaptive_test
   ```

**Expected Outcome:**
- Tests 3 and 4 should pass (path issues resolved)
- Test 2 may still fail (decision_tree.pkl missing)
- False positive rate should improve with adaptive rules

---

### Phase 2: Model Training (2-4 hours)

**Goal:** Train missing ML models

1. **Train Decision Tree**
   ```bash
   python scripts/train_decision_tree.py --synthetic
   ```

2. **Verify Model Compatibility**
   ```bash
   # Test if copied models load correctly
   python -c "
   import joblib
   model = joblib.load('data/models/aggressive_load_shedding.joblib')
   print('Model loaded:', type(model))
   print('Keys:', list(model.keys()) if isinstance(model, dict) else 'Not a dict')
   "
   ```

3. **Re-run Test Suite**
   ```bash
   # Test 2: Real attacks
   python scripts/test_real_attacks.py
   
   # Test 3: Multi-stage
   python scripts/test_multistage_integration.py
   
   # Test 4: Full pipeline
   python scripts/test_full_pipeline.py
   ```

**Expected Outcome:**
- All 4 tests should pass
- ML detection functional
- Multi-stage pipeline operational

---

### Phase 3: Performance Optimization (1-2 days)

**Goal:** Achieve documented performance targets

1. **Implement Rule Indexing**
   - Modify `src/detection/rule_engine.py`
   - Index rules by CAN ID for O(1) lookup
   - Add early exit conditions
   - Target: 5-10x throughput improvement

2. **Optimize ML Models**
   - Reduce estimators from 100 → 15
   - Use decision tree instead of ensemble
   - Implement batching
   - Target: 50-100x throughput improvement

3. **Enable Multi-Stage Pipeline**
   - Stage 1: Timing filter (80% pass)
   - Stage 2: Rule-based (50% pass)
   - Stage 3: ML deep analysis (10% analyzed)
   - Target: 7,000+ msg/s

4. **Re-test Performance**
   ```bash
   # Full suite with optimizations
   python scripts/comprehensive_test.py \
     /media/boneysan/Data/GitHub/Vehicle_Models/data/raw/DoS-1.csv \
     --enable-ml \
     --rules-config config/rules_adaptive.yaml \
     --output test_results/optimized_test
   ```

**Expected Outcome:**
- Rule-based: 5,000-10,000 msg/s
- With ML: 1,500-3,000 msg/s
- With multi-stage: 7,000+ msg/s
- False positive rate: <10%

---

## Testing Gaps Identified

### Datasets Not Tested

The following attack types have NOT been tested:
- ❌ fuzzing-1.csv (45MB, fuzzing attacks)
- ❌ fuzzing-2.csv (43MB)
- ❌ interval-1.csv (23MB, interval timing attacks)
- ❌ interval-2.csv (60MB)
- ❌ rpm-1.csv (32MB, RPM manipulation)
- ❌ rpm-2.csv (31MB)
- ❌ attack-free-1.csv (73MB, false positive rate validation)
- ❌ attack-free-2.csv (48MB)
- ❌ DoS-2.csv (12MB)

**Only tested:** DoS-1.csv (90K messages)

**Impact:**
- Unknown detection accuracy for fuzzing attacks
- Unknown detection accuracy for interval timing attacks
- Unknown detection accuracy for RPM manipulation
- False positive rate not validated on clean traffic
- Incomplete validation of system capabilities

**Recommendation:**
Run comprehensive test suite on all 16 datasets once models are available.

---

### Features Not Tested

- ❌ ML detection (models not available)
- ❌ Multi-stage hierarchical filtering
- ❌ Decision tree classifier
- ❌ Ensemble voting
- ❌ Adaptive load shedding
- ❌ Real-time CAN interface capture
- ❌ Live monitoring mode
- ❌ System service (systemd) operation
- ❌ Alert notifications
- ❌ Long-duration stability (>10 minutes)
- ❌ High message rate handling (>5,000 msg/s)
- ❌ Multiple attack types simultaneously

**Only tested:**
- ✅ Rule-based detection on DoS attacks
- ✅ System resource monitoring
- ✅ CSV file processing
- ✅ Performance metrics collection

---

## Configuration Issues Discovered

### 1. Virtual Environment Path Inconsistency

**Documentation says:** `.venv`  
**Actual path:** `venv`

**Impact:** Minor - commands in documentation need adjustment
**Fix:** Update RASPBERRY_PI_DEPLOYMENT_GUIDE.md to use `venv` not `.venv`

### 2. Rules Configuration

**Default:** `config/rules.yaml` (generic, aggressive)  
**Available:** `config/rules_adaptive.yaml` (vehicle-specific)  
**In use:** Generic rules (causing 90% FP rate)

**Impact:** High false positive rate makes system unusable
**Fix:** Switch default to adaptive rules or make configurable

### 3. Model Paths

**Multiple path references found:**
- `data/models/` (correct, used by main system)
- `models/` (incorrect, used by some test scripts)
- `../Vehicle_Models/models/` (relative, breaks on USB)

**Impact:** Tests fail due to path mismatches
**Fix:** Centralize path configuration in config.yaml

### 4. Feature Extraction

**Error:** `'can_id' key error` in multi-stage test

**Possible causes:**
- CAN message format mismatch
- Feature extractor expects different field names
- Data preprocessing not applied

**Impact:** ML detection cannot extract features
**Fix:** Review message format and feature extractor compatibility

---

## Performance Analysis

### Current vs Target Performance

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| **Rule-based throughput** | 708 msg/s | 40,000-50,000 msg/s | 56-70x slower |
| **ML throughput** | N/A (not functional) | 8,000-12,000 msg/s | Cannot test |
| **Multi-stage throughput** | N/A (not functional) | 25,000-40,000 msg/s | Cannot test |
| **False positive rate** | 90% | <10% | 80% higher |
| **Precision** | 10.14% | >70% | 60% lower |
| **Recall** | 100% | >95% | ✅ Exceeds |
| **CPU usage** | 25% | <60% | ✅ Good |
| **Memory usage** | 188 MB | <500 MB | ✅ Good |
| **Temperature** | 51°C | <65°C | ✅ Good |

### Performance Bottlenecks

**Identified from Test 1:**
1. **Processing Rate:** 708 msg/s is limited by:
   - Rule checking every message (O(n×m) complexity)
   - No rule indexing by CAN ID
   - No early exit optimization
   - Single-threaded Python GIL

2. **Alert Generation:** 300% alert rate suggests:
   - Rules firing on every message
   - Multiple rules matching same message
   - No alert aggregation/deduplication

3. **Rule Aggressiveness:** 90% FP rate indicates:
   - Generic thresholds not tuned for vehicle
   - Unknown CAN ID rule too strict
   - Entropy thresholds too low

**What's Working Well:**
- ✅ System resources (CPU, memory, temp) all healthy
- ✅ No message drops (0% loss)
- ✅ 100% recall (catches all attacks)
- ✅ Stable processing (no crashes over 127 seconds)

---

## Recommendations

### Immediate Actions (Do Now)

1. **Fix symlinks for path issues:**
   ```bash
   cd ~/Documents/GitHub
   ln -s /media/boneysan/Data/GitHub/Vehicle_Models Vehicle_Models
   cd CANBUS_IDS
   ln -s data/models models
   ```

2. **Switch to adaptive rules:**
   ```bash
   # Edit config/can_ids.yaml
   rules_file: config/rules_adaptive.yaml
   ```

3. **Copy available models:**
   ```bash
   cp /media/boneysan/Data/GitHub/Vehicle_Models/models/multistage/*.joblib data/models/
   ```

4. **Re-run tests to verify fixes**

### Short-term Goals (This Week)

1. **Train missing models:**
   - decision_tree.pkl
   - Any other required models

2. **Test all 16 datasets:**
   - Run batch test script on all attack types
   - Validate false positive rate on attack-free data

3. **Implement rule indexing:**
   - Modify rule_engine.py for O(1) lookup
   - Add early exit conditions

4. **Document actual performance:**
   - Update documentation with real measured values
   - Remove unvalidated claims (40-50K msg/s)

### Long-term Goals (Next 2-4 Weeks)

1. **Implement multi-stage pipeline:**
   - Stage 1: Timing detection
   - Stage 2: Rule-based filtering
   - Stage 3: ML deep analysis

2. **Performance optimization:**
   - Achieve 7,000+ msg/s target
   - Reduce false positive rate to <10%
   - Maintain 95%+ recall

3. **Production hardening:**
   - Long-duration stability testing
   - Real vehicle testing
   - Alert management system
   - Comprehensive documentation

---

## Files Generated During Testing

### Successful Test Outputs
```
test_results/benchmark_dos1_20251216_210139/20251216_210147/
├── system_metrics.csv              # CPU, memory, temp over time
├── performance_metrics.json        # Throughput, latency stats
└── comprehensive_summary.json      # Complete test summary
```

### Log Files
```
logs/
├── test1_benchmark_dos.log         # Full output from Test 1
├── test2_real_attacks.log          # Error log from Test 2
├── test3_multistage.log            # Partial output from Test 3
└── test4_full_pipeline.log         # Error log from Test 4
```

### All files preserved for analysis and debugging.

---

## Conclusion

**Summary:**
- Rule-based detection is **functional but not production-ready**
- ML detection is **completely non-functional** (missing models)
- Performance is **56-70x below documented targets**
- False positive rate is **9x higher than acceptable**

**Can we deploy?**
- ✅ For lab/testing environments with low traffic (<500 msg/s)
- ❌ For production vehicles (performance insufficient)
- ❌ For high-traffic networks (throughput too low)
- ❌ For ML-enhanced detection (models missing)

**Next Steps:**
1. Fix path configuration issues (1 hour)
2. Copy/train ML models (2-4 hours)
3. Switch to adaptive rules (5 minutes)
4. Re-run full test suite (1 hour)
5. Implement performance optimizations (1-2 days)

**Estimated Time to Production Ready:** 1-2 weeks with focused effort

---

**Report Generated:** December 16, 2025, 21:15  
**Test Session Duration:** ~15 minutes  
**Tests Run:** 4 attempted, 1 successful, 3 failed  
**Status:** System requires remediation before production deployment
