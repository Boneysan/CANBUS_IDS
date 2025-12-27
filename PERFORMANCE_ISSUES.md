# CAN-IDS Performance Issues & Real-World Requirements

**Date:** December 3, 2025  
**Status:** ⚠️ Critical Performance Gap Identified  
**Priority:** High - Impacts Real-Time Deployment Viability  

---

## Executive Summary

Performance testing on December 3, 2025 revealed that the current CAN-IDS implementation **cannot handle real-world CAN bus traffic rates** in production vehicles. While rule-based detection shows promise, ML-based detection is 50-100x too slow for real-time use.

**Critical Finding:**
- **System Capacity:** 759 msg/s (rule-based) / 15 msg/s (with ML)
- **Real-World Requirement:** 1,000-4,000 msg/s
- **Gap:** System is **32-266x too slow** for ML, **1.3-5.3x too slow** for rules at peak

---

## Real-World CAN Bus Message Rates

### Typical Passenger Vehicle Rates

| Scenario | Message Rate | Duration | Total Messages |
|----------|--------------|----------|----------------|
| **Idle/Parked** | 200-500 msg/s | Continuous | - |
| **Normal Driving** | 1,000-1,500 msg/s | Typical | - |
| **Peak Activity** | 2,000-4,000 msg/s | Bursts | - |
| **Engine Start** | 3,000-4,000 msg/s | 5-10 seconds | 15,000-40,000 |
| **Heavy Braking (ABS)** | 2,500-3,500 msg/s | 2-5 seconds | 5,000-17,500 |
| **Highway Driving** | 1,200-1,800 msg/s | Hours | Millions |

### Dataset Analysis

**From Vehicle_Models Research Data:**

| Dataset | Messages | Duration | Avg Rate | Activity Type |
|---------|----------|----------|----------|---------------|
| attack-free-1 | 1,952,833 | ~30 min | ~1,087 msg/s | Normal driving |
| attack-free-2 | 1,265,599 | ~20 min | ~1,054 msg/s | Normal driving |
| DoS-1 | 89,968 | ~1.5 min | ~998 msg/s | Attack + normal |
| DoS-2 | 310,260 | ~5 min | ~1,034 msg/s | Attack + normal |
| standstill-1 | 1,952,833 | ~30 min | ~1,087 msg/s | Parked/idle |

**Your Captured Data:**
- can_capture_30min.csv: 240,375 messages / 30 min = **133 msg/s**
- **Analysis:** Extremely low activity - possibly vehicle off or minimal systems active

---

## Current System Performance

### Test Configuration
- **Platform:** Raspberry Pi 4 Model B
- **Python:** 3.11.2
- **Dataset:** DoS-1 (50,000 messages)
- **Test Date:** December 3, 2025

### Performance Results

#### Rule-Based Detection Only

```
┌───────────────────────────────────────────┐
│ Performance Metrics                       │
├───────────────────────────────────────────┤
│ Throughput:        759 msg/s             │
│ Mean Latency:      1.284 ms              │
│ P95 Latency:       2.038 ms              │
│ CPU Usage:         25.3% avg, 28.7% peak │
│ Memory:            173 MB avg            │
│ Temperature:       52.8°C avg            │
├───────────────────────────────────────────┤
│ Detection                                 │
├───────────────────────────────────────────┤
│ Recall:            100.00% ✅            │
│ Precision:         18.28% ⚠️             │
│ False Positives:   81.7%                 │
└───────────────────────────────────────────┘
```

**Status:** ⚠️ **Marginal for Real-Time Use**
- Can handle idle/parked scenarios (200-500 msg/s) ✅
- **Cannot handle normal driving** (1,000-1,500 msg/s) ❌
- **Cannot handle peak activity** (2,000-4,000 msg/s) ❌

#### Rule-Based + ML Detection

```
┌───────────────────────────────────────────┐
│ Performance Metrics                       │
├───────────────────────────────────────────┤
│ Throughput:        15 msg/s              │
│ Mean Latency:      64.089 ms             │
│ P95 Latency:       101.861 ms            │
│ CPU Usage:         27.2% avg             │
│ Memory:            169 MB avg            │
├───────────────────────────────────────────┤
│ Bottleneck Analysis                       │
├───────────────────────────────────────────┤
│ Time in ML:        99% (13.7s / 13.8s)   │
│ Function:          decision_function()    │
│ Model:             100 estimators         │
│ Operations:        2.6M function calls    │
└───────────────────────────────────────────┘
```

**Status:** ❌ **NOT VIABLE for Real-Time**
- **66x too slow** for normal driving
- **266x too slow** for peak activity
- Would cause message queue overflow and dropped packets

---

## Performance Gap Analysis

### Comparison to Real-World Requirements

| Scenario | Required Rate | System Capacity | Gap | Status |
|----------|---------------|-----------------|-----|--------|
| **Idle (Rules)** | 200-500 msg/s | 759 msg/s | ✅ +259 msg/s | **PASS** |
| **Normal (Rules)** | 1,000-1,500 msg/s | 759 msg/s | ❌ -241 to -741 | **FAIL** |
| **Peak (Rules)** | 2,000-4,000 msg/s | 759 msg/s | ❌ -1,241 to -3,241 | **FAIL** |
| **Idle (ML)** | 200-500 msg/s | 15 msg/s | ❌ -185 to -485 | **FAIL** |
| **Normal (ML)** | 1,000-1,500 msg/s | 15 msg/s | ❌ -985 to -1,485 | **FAIL** |
| **Peak (ML)** | 2,000-4,000 msg/s | 15 msg/s | ❌ -1,985 to -3,985 | **FAIL** |

### Message Processing Deficit

**At 1,500 msg/s (normal driving):**

| Detection Mode | Can Process | Incoming | Deficit | Queue Buildup |
|---------------|-------------|----------|---------|---------------|
| **Rules Only** | 759 msg/s | 1,500 msg/s | -741 msg/s | 44,460 msgs/min |
| **Rules + ML** | 15 msg/s | 1,500 msg/s | -1,485 msg/s | 89,100 msgs/min |

**Result:** System would fall behind almost immediately, leading to:
- Memory exhaustion (queue overflow)
- Dropped messages
- Delayed detection (alerts seconds/minutes late)
- System crash or restart

---

## Root Causes

### 1. ML Model Computational Complexity

**IsolationForest Configuration:**
- **Estimators:** 100 decision trees
- **Evaluations per message:** 100 trees × multiple operations
- **Time per message:** 64 ms average
- **Profiler results:** 99% of time in `sklearn.ensemble._iforest.decision_function()`

**Why It's Slow:**
```
For each message:
  1. Extract 9 features (timing, frequency, patterns)
  2. Normalize features
  3. Evaluate 100 decision trees
  4. Aggregate scores across all trees
  5. Calculate anomaly threshold
  6. Make prediction
  
Total: ~2.6 million function calls per 100 messages
```

### 2. Rule Engine Limitations

**Current Throughput:** 759 msg/s (1.32 ms/message)

**Bottlenecks Identified:**
- Complex rule evaluations (entropy calculations, checksums)
- Per-message state tracking for frequency/timing analysis
- Alert generation overhead
- False positive rate causing excessive alert processing (81.7%)

### 3. Python Performance Constraints

**GIL (Global Interpreter Lock):**
- Limits true parallelism
- Single-threaded execution for most operations
- Cannot fully utilize Raspberry Pi 4's quad-core CPU

**Interpreted Language Overhead:**
- ~10-100x slower than compiled languages (C/C++)
- Memory management overhead
- Function call overhead

---

## Impact on Deployment

### Scenarios Where System CANNOT Be Used

❌ **Real-time monitoring during normal driving**
- Messages would queue up and overflow
- Detection alerts would be delayed by minutes
- Critical attacks might not be detected in time

❌ **Production vehicle deployment**
- Cannot keep up with message rates
- Risk of system crash under load
- Unacceptable for safety-critical applications

❌ **High-activity scenarios (startup, braking, acceleration)**
- Peak rates of 2,000-4,000 msg/s
- System would be 3-5x too slow even with rules only
- Complete failure with ML enabled

### Scenarios Where System CAN Be Used

✅ **Low-activity monitoring (parked vehicles)**
- 200-500 msg/s is within capacity
- Suitable for overnight monitoring
- Can detect tampering attempts

✅ **Offline forensic analysis**
- Process captured PCAP files at system's pace
- No real-time constraints
- ML detection can be used (15 msg/s is acceptable)

✅ **Development and testing**
- Validate detection algorithms
- Train and evaluate ML models
- Test attack scenarios in controlled environment

⚠️ **Research and education**
- Demonstrate IDS concepts
- Study attack patterns
- Learn CAN bus security
- But: Not representative of production performance

---

## Solutions & Optimization Strategies

### Short-Term (Immediate)

#### 1. Disable ML Detection for Real-Time Use
```yaml
# config/can_ids.yaml
detection_modes:
  - rule_based     # Only use rules in real-time
  # - ml_based     # Disable ML
```
**Impact:** Throughput: 759 msg/s (still insufficient for normal driving)

#### 2. Implement Message Sampling
```python
# Only analyze every Nth message
if message_count % 10 == 0:
    ml_detector.analyze_message(msg)
```
**Impact:** 
- ML at 15 msg/s can handle 150 msg/s input (with 10x sampling)
- Rules handle all messages at 759 msg/s
- Trade-off: Might miss attacks between samples

#### 3. Optimize Rule Engine
- Disable most aggressive rules (Unknown CAN ID, High Entropy)
- Reduce frequency check windows
- Simplify entropy calculations
- Target: 1,500-2,000 msg/s

### Medium-Term (1-2 Weeks)

#### 4. Train Lightweight ML Model
```python
# Retrain with fewer estimators
model = IsolationForest(
    n_estimators=10,      # Down from 100
    max_samples=0.3,      # Reduce sample size
    max_features=0.5      # Use subset of features
)
```
**Expected Impact:** 10-20x speedup → 150-300 msg/s with ML

#### 5. Batch Processing
```python
# Process messages in batches
batch = []
for msg in messages:
    batch.append(msg)
    if len(batch) >= 100:
        ml_detector.analyze_batch(batch)
        batch = []
```
**Expected Impact:** Amortize overhead → 50-100 msg/s with ML

#### 6. Implement Adaptive Load Shedding
```python
# Drop ML analysis under high load
if queue_length > threshold:
    # Rules only
else:
    # Rules + ML
```

### Long-Term (1+ Months)

#### 7. Multi-Processing Architecture
- Separate processes for rules and ML
- Message distribution across CPU cores
- Target: 2,000-3,000 msg/s

#### 8. Hardware Acceleration
- Move to more powerful platform (x86 PC, Nvidia Jetson)
- Or: Keep Pi 4 for rules, offload ML to server
- Or: Use FPGA/ASIC for rule evaluation

#### 9. Rewrite Critical Paths in C/C++
- Python wrapper around native code
- 10-100x speedup for compute-intensive operations
- Maintain Python interface for ease of use

#### 10. Alternative ML Approaches
- One-class SVM (simpler than IsolationForest)
- Lightweight neural networks (optimized for inference)
- Lookup tables for common patterns
- Hybrid: Rules + lightweight scoring

---

## Testing Recommendations

### Required Performance Targets

| Deployment Scenario | Minimum Throughput | Recommended Throughput |
|--------------------|--------------------|------------------------|
| **Research/Demo** | 200 msg/s | 500 msg/s |
| **Parked Vehicle** | 500 msg/s | 800 msg/s |
| **Normal Driving** | 1,500 msg/s | 2,000 msg/s |
| **All Scenarios** | 2,000 msg/s | 4,000 msg/s |
| **Production Safety** | 4,000 msg/s | 6,000 msg/s (2x margin) |

### Test Suite Needed

1. **Sustained Load Test**
   - 10+ minutes at target rate
   - Monitor queue depth, CPU, memory
   - Ensure no message drops

2. **Burst Test**
   - Sudden spike to 4,000 msg/s
   - Measure recovery time
   - Validate graceful degradation

3. **Stress Test**
   - Push to system limits
   - Identify breaking point
   - Measure failure modes

4. **Real Vehicle Capture**
   - Capture during various driving scenarios
   - Startup sequence (highest rate)
   - Highway driving
   - City stop-and-go
   - Emergency braking

---

## Priority Action Items

### Critical (This Week)
1. ✅ Document performance issues (complete)
2. ⬜ Optimize rule engine to reach 1,500 msg/s minimum
3. ⬜ Implement message sampling for ML (10x-50x)
4. ⬜ Test with real vehicle data at various activity levels

### High Priority (Next 2 Weeks)
5. ⬜ Train lightweight ML model (n_estimators=10-20)
6. ⬜ Implement batch processing for ML
7. ⬜ Add adaptive load shedding
8. ⬜ Comprehensive performance testing

### Medium Priority (Next Month)
9. ⬜ Multi-processing architecture design
10. ⬜ Evaluate hardware acceleration options
11. ⬜ Investigate alternative ML algorithms
12. ⬜ Benchmark against production requirements

---

## Conclusion

The current CAN-IDS implementation is **not ready for real-world deployment** due to insufficient message processing capacity. The system can handle low-activity scenarios (parked vehicles, idle) but cannot keep up with normal driving conditions (1,000-1,500 msg/s) or peak activity (2,000-4,000 msg/s).

**Critical Issues:**
- ML detection is 50-266x too slow for real-time use
- Rule-based detection is 1.3-5.3x too slow for peak scenarios
- System would fall behind, queue messages, and eventually crash under real load

**Viable Use Cases (Current State):**
- ✅ Offline forensic analysis
- ✅ Research and development
- ✅ Low-activity monitoring (parked vehicles)
- ❌ Real-time production deployment

**Path Forward:**
Optimize rule engine to 1,500+ msg/s and implement lightweight ML with sampling for real-time viability, or redesign architecture for multi-processing/hardware acceleration to reach production-grade performance (4,000+ msg/s sustained).

---

**Last Updated:** December 3, 2025  
**Next Review:** After optimization implementation
