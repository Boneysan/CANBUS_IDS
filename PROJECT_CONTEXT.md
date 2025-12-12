# CAN-IDS Project Context & Status

**Project:** Controller Area Network Intrusion Detection System (CAN-IDS)  
**Platform:** Raspberry Pi 4 Model B  
**Language:** Python 3.11.2  
**Last Updated:** December 7, 2025  
**Status:** 🔧 **FUNCTIONAL - OPTIMIZATION PLANS READY**  

---

## Quick Status Summary

### ✅ What's Working
- Rule-based detection: 759 msg/s throughput, 100% attack recall
- 18/18 rule types implemented and functional
- ML models can now load successfully (bugs fixed Dec 3)
- System processes offline datasets correctly
- Comprehensive testing framework operational

### ⚠️ Critical Issues
- ML detection too slow: 15 msg/s (need 1,000-1,500 msg/s for real-world use)
- Rule-based detection insufficient for peak CAN loads (2,000-4,000 msg/s)
- High false positive rate: 81.7% (needs tuning)
- Not ready for production deployment in real vehicles

### 💡 Rule Tuning Strategy (December 7, 2025)
**Current Rules:** 20 configured rules with good coverage BUT too aggressive
- ✅ Coverage: All attack types (DoS, replay, fuzzing, ECU impersonation, etc.)
- ✅ Performance: 759 msg/s, 100% recall (catches all attacks)
- ⚠️ Problem: 81.7% false positives (only 18.28% precision)

**Root Cause:** Rules use generic thresholds, not vehicle-specific baselines

**Solution:** Extract timing/frequency parameters from ML training data
- ML models were trained on 10.6M attack-free messages from real vehicles
- Feature extractor already calculates per-CAN-ID statistics:
  * `interval_mean` - Average message interval
  * `interval_std` - Timing variance
  * `freq_last_1s` - Message frequency
- Can extract these baselines and auto-generate vehicle-specific rule thresholds
- Use mean ± 3×std for thresholds (covers 99.7% of normal traffic)

**Implementation:** Create script to analyze Vehicle_Models training data and output optimized `rules.yaml` with vehicle-specific timing/frequency thresholds instead of hardcoded generic values.

### 🎯 Implementation Roadmap (December 7, 2025)

**Two Approaches Available:**

1. **Conservative (IMPROVEMENT_ROADMAP.md):** 2-4 weeks, achieves 1,500 msg/s
   - Week 1: Fix false positives, lightweight ML, rule indexing → 750 msg/s
   - Week 2-3: Batching, multiprocessing → 1,500 msg/s
   - Week 4+: Production hardening

2. **Aggressive (BUILD_PLAN_7000_MSG_SEC.md):** 3 days, achieves 7,000+ msg/s
   - Day 1: Message cycle detection + rule optimization (6 hours)
   - Day 2: ML integration (4.5 hours)
   - Day 3: Testing (4 hours)
   - Research-validated, hierarchical 3-stage architecture

3. **Hybrid (Recommended):** Best of both
   - Week 1: Conservative approach → Working 750 msg/s system
   - Week 2: Test on real vehicle, measure actual traffic
   - Week 3: If needed, add cycle detection → 7,000+ msg/s

**Quick Wins Available Today:**
- Change `n_estimators=300` to `n_estimators=5` in ml_detector.py line 124 (5 min) → 100x ML speedup
- Add rule indexing by CAN ID in rule_engine.py (2 hours) → 3-5x rule speedup

---

## Project Structure & Key Files

### Documentation Files (Read These First)

#### 🎯 Implementation Plans (December 7, 2025)

| File | Purpose | Target Performance |
|------|---------|-------------------|
| **IMPROVEMENT_ROADMAP.md** | 4-phase optimization plan (2-4 weeks) | 750-1,500 msg/s |
| **BUILD_PLAN_7000_MSG_SEC.md** | Complete 3-day build plan with research citations | 7,000+ msg/s |
| **ACHIEVING_7000_MSG_PER_SEC.md** | Technical analysis & implementation guide | 7,000+ msg/s |
| **COMPARISON_ORIGINAL_VS_7K_PLAN.md** | Compare incremental vs architectural approaches | Both options |
| **HYBRID_APPROACH_CAPABILITIES.md** | Week-by-week capability progression | 750 → 7,000 msg/s |

#### 📋 Status & Testing (December 3, 2025)

| File | Purpose | Last Updated |
|------|---------|--------------|
| **DECEMBER_3_SESSION_SUMMARY.md** | Latest session - all changes made | Dec 3, 2025 |
| **TESTING_RESULTS.md** | Performance test results | Dec 3, 2025 |
| **PERFORMANCE_ISSUES.md** | Real-world requirements gap analysis | Dec 3, 2025 |
| **ML_OPTIMIZATION_GUIDE.md** | How to fix ML performance (7 strategies) | Dec 3, 2025 |
| **100_PERCENT_COMPLETE.md** | Feature completion status (18/18 rules) | Dec 2, 2025 |
| **TONIGHT_SUMMARY.md** | Nov 30 work - contamination testing | Nov 30, 2025 |
| **PROJECT_SUMMARY.md** | Original project setup & structure | Earlier |
| **README.md** | User-facing documentation | Earlier |

### Source Code Structure

```
src/
├── capture/
│   ├── can_sniffer.py          # Real-time SocketCAN monitoring
│   └── pcap_reader.py          # Offline PCAP/CSV analysis
├── detection/
│   ├── rule_engine.py          # Signature-based detection (18 rules)
│   ├── ml_detector.py          # ML anomaly detection [RECENTLY FIXED]
│   └── multistage_detector.py  # Multi-stage pipeline
├── preprocessing/
│   ├── feature_extractor.py    # CAN message feature extraction
│   └── normalizer.py           # Data normalization
└── alerts/
    ├── alert_manager.py        # Alert coordination
    └── notifiers.py            # Notification channels

scripts/
├── comprehensive_test.py       # Main testing framework [USE THIS]
├── benchmark.py                # Performance benchmarking
└── batch_test_set01.sh         # Batch testing

config/
├── can_ids.yaml                # General configuration
├── can_ids_rpi4.yaml          # Raspberry Pi 4 optimized
└── rules.yaml                  # Detection rules (18 types)

data/
└── models/                     # ML models [FIXED: now loads correctly]
    ├── adaptive_load_shedding.joblib
    ├── aggressive_load_shedding.joblib
    └── [other models...]
```

### External Resources (USB Drive)

**Location:** `/media/boneysan/Data/GitHub/Vehicle_Models/`

```
data/raw/                       # Training/testing datasets
├── attack-free-1.csv          # 73MB, ~1.9M messages, normal traffic
├── attack-free-2.csv          # 48MB, ~1.2M messages
├── DoS-1.csv                  # 3.4MB, DoS attack dataset
├── DoS-2.csv                  # 12MB
├── fuzzing-1.csv              # 45MB
├── fuzzing-2.csv              # 43MB
├── interval-1.csv             # 23MB
├── interval-2.csv             # 60MB
├── rpm-1.csv                  # 32MB
├── rpm-2.csv                  # 31MB
└── [12 more datasets...]

models/multistage/              # Pre-trained ML models
├── adaptive_load_shedding.joblib    # 1.3MB, 100 estimators
├── adaptive_only.joblib             # 1.3MB
├── full_pipeline.joblib             # 1.3MB
└── aggressive_load_shedding.joblib  # Exists but was corrupted locally
```

---

## Current Performance Metrics

### Rule-Based Detection (December 3, 2025 Test)

**Dataset:** DoS-1 (50,000 messages)

```
✅ Throughput:        759.22 msg/s
✅ Mean Latency:      1.284 ms
✅ P95 Latency:       2.038 ms
✅ CPU Usage:         25.3% avg, 28.7% peak
✅ Memory:            173.3 MB avg
✅ Temperature:       52.8°C avg
✅ Recall:            100% (caught ALL attacks!)
⚠️  Precision:        18.28% (81.7% false positives)
⚠️  F1-Score:         0.309
```

**Status:** Production-ready for low-medium traffic, needs tuning for false positives

### ML Detection (December 3, 2025 Test)

**Dataset:** DoS-1 (50,000 messages)  
**Model:** IsolationForest (**300 estimators**, 9 features)

```
❌ Throughput:        15.26 msg/s (100x slower than needed!)
❌ Mean Latency:      64.089 ms (49x slower!)
❌ P95 Latency:       101.861 ms
   CPU Usage:         27.2% avg
   Memory:            168.9 MB avg
   Bottleneck:        99% in sklearn IsolationForest.decision_function()
                      (loops through 300 trees per message)
```

**Why So Slow:**
```
Time per message = Feature extraction + (Trees × Time per tree)
                 = 0.1ms + (300 × 0.04ms)
                 = 12.1ms
Theoretical max  = 1000ms / 12.1ms = 83 msg/s
Actual observed  = 15 msg/s (with overhead)

Fix: Reduce to 5 trees → 0.3ms per message → 1,500 msg/s (100x faster!)
```

**Status:** NOT suitable for real-time, needs 50-100x speedup

### Real-World Requirements Gap

| Scenario | Required | Rule-Based | ML | Gap (Rules) | Gap (ML) |
|----------|----------|------------|----|----|---|
| **Idle/Parked** | 200-500 msg/s | 759 | 15 | ✅ +259 | ❌ -185 to -485 |
| **Normal Driving** | 1,000-1,500 msg/s | 759 | 15 | ❌ -241 to -741 | ❌ -985 to -1,485 |
| **Peak Activity** | 2,000-4,000 msg/s | 759 | 15 | ❌ -1,241 to -3,241 | ❌ -1,985 to -3,985 |

**Conclusion:** System cannot handle real-world traffic rates without optimization.

---

## Recent Bug Fixes (December 3, 2025)

### Bug Fix #1: ML Model Loading Failure

**File:** `src/detection/ml_detector.py` (Lines 536-565)  
**Symptom:** `_pickle.UnpicklingError: invalid load key, '\x02'`  
**Cause:** Custom unpickler couldn't handle joblib file format  
**Fix:** Use `joblib.load()` directly instead of custom unpickler  
**Status:** ✅ Fixed, models now load successfully

**Code Change:**
```python
# BEFORE (broken)
if VEHICLE_MODELS_COMPAT:
    with open(model_path, 'rb') as f:
        file_content = f.read()
    unpickler = VehicleModelsUnpickler(io.BytesIO(file_content))
    model_data = unpickler.load()  # FAILS

# AFTER (fixed)
model_data = joblib.load(model_path)  # Works!
```

### Bug Fix #2: Timing Tracker Performance

**File:** `src/detection/ml_detector.py` (Lines 95, 303-310)  
**Symptom:** Slow ML performance due to O(n) list operations  
**Cause:** Using `list.pop(0)` which shifts all elements  
**Fix:** Changed to `deque(maxlen=50)` for O(1) operations  
**Status:** ✅ Fixed, but ML model itself is still the bottleneck

**Code Change:**
```python
# BEFORE (slow)
self._timing_trackers = defaultdict(list)
if len(timing_list) > 50:
    timing_list.pop(0)  # O(n) - shifts all elements!

# AFTER (fast)
self._timing_trackers = defaultdict(lambda: deque(maxlen=50))
# Deque automatically removes oldest - O(1)
```

---

## How to Test the System

### Quick Test (Use This for Validation)

```bash
# 1. Activate virtual environment
cd /home/boneysan/Documents/Github/CANBUS_IDS
source venv/bin/activate

# 2. Create test dataset (or use existing)
head -50001 "/media/boneysan/Data/GitHub/Vehicle_Models/data/raw/DoS-1.csv" > /tmp/dos1_small.csv

# 3. Test rule-based detection only
python scripts/comprehensive_test.py /tmp/dos1_small.csv \
  --output test_results/test_$(date +%Y%m%d_%H%M%S)

# 4. Test with ML enabled
python scripts/comprehensive_test.py /tmp/dos1_small.csv \
  --enable-ml \
  --output test_results/ml_test_$(date +%Y%m%d_%H%M%S)
```

### Test Results Location

Results saved to: `test_results/{output_name}/YYYYMMDD_HHMMSS/`
- `system_metrics.csv` - CPU, memory, temperature over time
- `performance_metrics.json` - Throughput, latency statistics
- `comprehensive_summary.json` - Complete test summary

### Interpreting Results

**Good Performance:**
- Throughput: >1,500 msg/s for production use
- Mean Latency: <2 ms
- CPU: <70%
- Memory: <300 MB
- Temperature: <65°C
- No message drops

**Current Performance:**
- Rule-based: 759 msg/s (marginal)
- ML: 15 msg/s (unacceptable)

---

## Optimization Roadmap

### Phase 1: Quick Win (TODAY - 5 minutes)

**Goal:** Make ML usable right now  
**Strategy:** Implement message sampling  

**Implementation:**
```python
# File: src/detection/ml_detector.py
# Add to __init__:
self.sampling_rate = 10  # NEW parameter

# Add to analyze_message (before feature extraction):
if self._stats['messages_analyzed'] % self.sampling_rate != 0:
    self._update_message_state(message)
    return None
```

**Expected Result:** 150 msg/s (10x improvement)  
**Documentation:** See `ML_OPTIMIZATION_GUIDE.md` Strategy 1

### Phase 2: Production Viability (THIS WEEK - 3-4 hours)

**Goal:** Achieve real-time performance  
**Strategy:** Train lightweight ML model  

**Steps:**
1. Go to Vehicle_Models project on USB
2. Run retrain script (see ML_OPTIMIZATION_GUIDE.md)
3. Train with n_estimators=5 (down from 300) for 100x speedup
   OR n_estimators=15 (down from 300) for 50x speedup
4. Copy model to CANBUS_IDS/data/models/
5. Test and validate

**Expected Result:** 750-1,500 msg/s (50-100x improvement)  
**Documentation:** See `ML_OPTIMIZATION_GUIDE.md` Strategy 2

### Phase 3: Advanced Features (NEXT 2-3 WEEKS - 8-12 hours)

**Goal:** Intelligent adaptive system  
**Strategies:**
- Batch processing (50-100 messages at once)
- Adaptive load shedding (auto-adjust based on latency)
- Performance monitoring dashboard

**Expected Result:** 1,500-3,000 msg/s (handles all scenarios)  
**Documentation:** See `ML_OPTIMIZATION_GUIDE.md` Strategies 3-4

### Phase 4: Production Hardening (NEXT MONTH)

**Goal:** Deploy to real vehicles  
**Tasks:**
- Multi-processing architecture
- Real vehicle testing
- Hardware acceleration evaluation
- Comprehensive stress testing

**Documentation:** See `PERFORMANCE_ISSUES.md` Long-Term section

---

## Known Issues & Limitations

### Critical Issues

1. **ML Detection Too Slow for Real-Time**
   - Current: 15 msg/s
   - Required: 1,000-1,500 msg/s
   - Gap: 66-100x too slow
   - Fix: See Phase 1-3 above
   - Status: **BLOCKING PRODUCTION**

2. **Rule-Based Insufficient for Peak Loads**
   - Current: 759 msg/s
   - Peak requirement: 2,000-4,000 msg/s
   - Gap: 3-5x too slow
   - Fix: Multi-processing, C++ rewrite (Phase 4)
   - Status: **LIMITS DEPLOYMENT**

3. **High False Positive Rate**
   - Current: 81.7% of alerts are false positives
   - Precision: 18.28%
   - Cause: Aggressive rules (Unknown CAN ID, High Entropy)
   - Fix: Tune rule thresholds
   - Status: **NEEDS TUNING**

### Known Limitations

- **Python GIL:** Limits true parallelism
- **Interpreted Language:** 10-100x slower than C/C++
- **Raspberry Pi CPU:** Limited compute power for heavy ML
- **Single-threaded:** Most operations run on one core

### Testing Gaps

- [ ] No testing on fuzzing attacks
- [ ] No testing on interval timing attacks
- [ ] No testing on rpm manipulation attacks
- [ ] No sustained load testing (>10 minutes)
- [ ] No real vehicle capture under various conditions
- [ ] No attack-free baseline (false positive rate validation)

---

## Configuration Files

### Main Configuration: `config/can_ids.yaml`

```yaml
# Current settings
detection_modes:
  - rule_based    # Currently enabled
  # - ml_based    # Disabled due to performance

ml_model:
  path: data/models/aggressive_load_shedding.joblib
  contamination: 0.20
  # sampling_rate: 10    # ADD THIS for Phase 1 optimization

performance:
  max_cpu_percent: 70
  max_memory_mb: 300
  message_buffer_size: 500

capture:
  interface: can0
  buffer_size: 500
```

### Rules Configuration: `config/rules.yaml`

**18 Rule Types Implemented:**

1. Pattern matching (`data_pattern`)
2. Frequency monitoring (`max_frequency`)
3. Timing analysis (`check_timing`)
4. Source validation (`allowed_sources`)
5. Checksum validation (`check_checksum`)
6. Counter validation (`check_counter`)
7. Entropy analysis (`entropy_threshold`)
8. DLC validation (`validate_dlc`)
9. Frame format (`check_frame_format`)
10. Bus flooding (`global_message_rate`)
11. Diagnostic source (`check_source`)
12. Replay detection (`check_replay`)
13. Byte validation (`data_byte_0-7`)
14. Data integrity (`check_data_integrity`)
15. Steering range (`check_steering_range`)
16. Repetition patterns (`check_repetition`)
17. Frame type (`frame_type`)
18. Whitelist mode (`whitelist_mode`)

**Status:** All 18 implemented, but some too aggressive (causing false positives)

---

## Development Workflow

### Making Changes

1. **Always activate venv first:**
   ```bash
   cd /home/boneysan/Documents/Github/CANBUS_IDS
   source venv/bin/activate
   ```

2. **Test before committing:**
   ```bash
   # Quick test
   python scripts/comprehensive_test.py /tmp/dos1_small.csv --output test_results/validation
   
   # Check for errors
   python -m pytest tests/ -v  # (if tests exist)
   ```

3. **Document changes:**
   - Update relevant .md files
   - Add entry to session summary if significant

### Git Workflow

```bash
# Check status
git status

# Stage changes
git add src/detection/ml_detector.py
git add DECEMBER_3_SESSION_SUMMARY.md

# Commit with descriptive message
git commit -m "Fix ML model loading and timing tracker performance

- Fixed joblib unpickling error in ml_detector.py
- Changed timing_trackers from list to deque for O(1) operations
- Updated documentation with test results and optimization guide"

# Push to remote
git push origin main
```

---

## Quick Reference Commands

### Testing
```bash
# Activate venv
source venv/bin/activate

# Quick test (rules only)
python scripts/comprehensive_test.py /tmp/dos1_small.csv --output test_results/quick

# Test with ML
python scripts/comprehensive_test.py /tmp/dos1_small.csv --enable-ml --output test_results/ml

# Profile performance
python -m cProfile -o profile.stats scripts/comprehensive_test.py /tmp/dos1_small.csv
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(20)"
```

### System Monitoring
```bash
# Temperature
vcgencmd measure_temp

# Throttling check
vcgencmd get_throttled

# Resource usage
htop

# CAN interface stats
ip -s link show can0
```

### Data Access
```bash
# List available datasets
ls -lh /media/boneysan/Data/GitHub/Vehicle_Models/data/raw/

# List available models
ls -lh /media/boneysan/Data/GitHub/Vehicle_Models/models/multistage/

# Check model details
python3 << EOF
import joblib
m = joblib.load('/media/boneysan/Data/GitHub/Vehicle_Models/models/multistage/adaptive_load_shedding.joblib')
print(f"Model type: {type(m)}")
print(f"Keys: {list(m.keys())}")
print(f"Estimators: {m['stage1_model'].n_estimators}")
EOF
```

---

## Decision Points & Trade-offs

### Should I Enable ML Detection?

**NO - Not Yet**
- Current performance: 15 msg/s (too slow)
- Would cause message queue overflow
- System would fall behind and crash
- **Action:** Implement Phase 1 optimization first

**YES - After Phase 2 Optimization**
- With sampling + lightweight model: 750-1,500 msg/s
- Can handle normal driving scenarios
- Provides defense-in-depth

### Should I Use Rules or ML in Production?

**Current Recommendation:** Rules Only
- 759 msg/s is sufficient for low-medium traffic
- 100% attack recall
- Stable and predictable
- High false positives but manageable

**After Optimization:** Rules + ML (Hybrid)
- Rules for fast, high-confidence detection
- ML for novel attack detection
- Best of both worlds

### What Detection Rules Should I Tune?

**Priority 1 (Causing Most False Positives):**
1. Unknown CAN ID rule - too aggressive
2. High Entropy Data - triggers on normal variation
3. Counter Sequence Error - too sensitive

**Action:**
- Increase thresholds
- Add learning period
- Implement whitelist mode

---

## Success Metrics

### Current State (December 3, 2025)

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| **Throughput (Rules)** | 1,500 msg/s | 759 msg/s | ⚠️ Marginal |
| **Throughput (ML)** | 1,500 msg/s | 15 msg/s | ❌ Fail |
| **Recall** | >95% | 100% | ✅ Excellent |
| **Precision** | >70% | 18.28% | ❌ Poor |
| **Latency** | <5 ms | 1.3 ms (rules) | ✅ Good |
| **CPU Usage** | <70% | 25% | ✅ Excellent |
| **Memory** | <400 MB | 173 MB | ✅ Excellent |
| **Production Ready** | Yes | No | ❌ Not Yet |

### After Phase 1 (Projected)

| Metric | Target | Projected | Status |
|--------|--------|-----------|--------|
| **Throughput (ML)** | 1,500 msg/s | 150 msg/s | ⚠️ Marginal |

### After Phase 2 (Projected)

| Metric | Target | Projected | Status |
|--------|--------|-----------|--------|
| **Throughput (ML)** | 1,500 msg/s | 750-1,500 msg/s | ✅ Good |
| **Production Ready** | Yes | Yes (with caveats) | ✅ Ready |

---

## Important Contacts & Resources

### Documentation Hierarchy

**Start Here:**
1. THIS FILE (`PROJECT_CONTEXT.md`) - Overall status
2. `DECEMBER_3_SESSION_SUMMARY.md` - Latest changes
3. `ML_OPTIMIZATION_GUIDE.md` - How to fix ML
4. `PERFORMANCE_ISSUES.md` - Why it's slow
5. `TESTING_RESULTS.md` - Test data

**Background:**
- `100_PERCENT_COMPLETE.md` - Feature completion
- `PROJECT_SUMMARY.md` - Original setup
- `README.md` - User documentation

### External Resources

**Research Papers/Datasets:**
- Vehicle_Models project (USB drive)
- CAN bus datasets (academic research)

**Hardware:**
- Raspberry Pi 4 documentation
- MCP2515 CAN HAT datasheet

---

## Next Session Checklist

### Before You Start

- [ ] Read this file (PROJECT_CONTEXT.md)
- [ ] Review DECEMBER_3_SESSION_SUMMARY.md
- [ ] Check current branch: `git status`
- [ ] Activate venv: `source venv/bin/activate`
- [ ] Verify USB drive mounted: `ls /media/boneysan/Data/`

### If Implementing Phase 1 (Message Sampling)

- [ ] Review ML_OPTIMIZATION_GUIDE.md Strategy 1
- [ ] Modify src/detection/ml_detector.py
- [ ] Test with various sampling rates (5, 10, 25, 50)
- [ ] Document results in TESTING_RESULTS.md
- [ ] Update this file with new metrics

### If Implementing Phase 2 (Lightweight Model)

- [ ] Review ML_OPTIMIZATION_GUIDE.md Strategy 2
- [ ] Navigate to Vehicle_Models on USB
- [ ] Create retrain_lightweight.py script
- [ ] Train model with n_estimators=15
- [ ] Copy to CANBUS_IDS/data/models/
- [ ] Test and validate performance
- [ ] Update configuration files
- [ ] Document results

### If Testing Additional Attack Types

- [ ] Choose dataset from USB (fuzzing, interval, rpm)
- [ ] Run comprehensive_test.py
- [ ] Compare results to DoS baseline
- [ ] Update TESTING_RESULTS.md
- [ ] Note any new patterns or issues

---

## Troubleshooting Guide

### Problem: ML Model Won't Load

**Error:** `_pickle.UnpicklingError: invalid load key`

**Solution:** This was fixed on Dec 3. If you see this:
1. Check you're using latest ml_detector.py (after Dec 3)
2. Verify model file exists and isn't corrupted
3. Try loading directly: `joblib.load(model_path)`

### Problem: System Too Slow

**Symptom:** Messages processing <100 msg/s

**Diagnosis:**
1. Check if ML is enabled (should be disabled unless optimized)
2. Profile with cProfile to find bottleneck
3. Check CPU/memory usage with `htop`

**Solution:**
- Disable ML: Remove `ml_based` from config
- Implement sampling: See Phase 1
- Train lightweight model: See Phase 2

### Problem: High False Positive Rate

**Symptom:** 80%+ of alerts are false positives

**Diagnosis:**
1. Check which rules are triggering most
2. Review rule thresholds in config/rules.yaml

**Solution:**
1. Increase thresholds for aggressive rules
2. Disable "Unknown CAN ID" for learning period
3. Implement whitelist mode
4. Add normal traffic baseline training

### Problem: Tests Fail

**Symptom:** comprehensive_test.py crashes or errors

**Diagnosis:**
1. Check virtual environment active
2. Verify dataset file exists
3. Check for import errors

**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Verify imports
python -c "import sklearn; import pandas; import numpy; print('OK')"

# Check dataset
wc -l /tmp/dos1_small.csv  # Should show 50001
```

---

## Version History

| Date | Version | Changes | By |
|------|---------|---------|-----|
| Dec 3, 2025 | 1.0 | Initial context document created | Session |
| Dec 3, 2025 | 1.0 | Added bug fixes, test results, optimization roadmap | Session |

---

## Summary: Where to Start

### You're Looking At This File Because...

**Option A: You want to know project status**
→ Read: Executive Summary (top of this file)

**Option B: You want to continue optimization work**
→ Read: Optimization Roadmap → Phase 1
→ Implementation: ML_OPTIMIZATION_GUIDE.md

**Option C: You want to test the system**
→ Read: "How to Test the System" section
→ Run: Quick Test commands

**Option D: You want to know what changed recently**
→ Read: DECEMBER_3_SESSION_SUMMARY.md

**Option E: You want to understand why it's slow**
→ Read: PERFORMANCE_ISSUES.md

**Option F: You're deploying to production**
→ WARNING: Not ready yet! See "Critical Issues"
→ Complete: Phase 1 and Phase 2 first

---

**Last Updated:** December 3, 2025, 22:00  
**Next Review:** After Phase 1 implementation  
**Status:** Active Development - Optimization Required  
