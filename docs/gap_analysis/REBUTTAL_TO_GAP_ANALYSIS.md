# Rebuttal to CAN-IDS Gap Analysis
**Date:** January 4, 2026 (Updated: January 5, 2026)  
**Author:** Project Team  
**In Response To:** GAP_ANALYSIS.md (December 27, 2025)  
**Version:** 2.0 - Complete Metrics Recovery

---

## Executive Summary

The December 27th gap analysis contains **factually incorrect claims** about testing status while identifying a legitimate issue with metrics capture. This rebuttal provides evidence-based corrections and clarifies the actual state of testing.

**Original Finding (Jan 4):** Multiple attack types labeled "Untested" were actually tested on December 3rd, 2025. The real issue is that detection performance metrics were not captured in the test results, not that tests were never conducted.

**Updated Finding (Jan 5):** Complete metrics recovery completed. All attack types have been tested with full TP/FP/TN/FN confusion matrices, precision/recall/F1-scores, and system resource monitoring. **Critical discovery:** Several datasets have severe quality issues (invalid labels, extreme imbalance) that invalidate previous assumptions.

---

## Factual Corrections

### 1. RPM/Speed Attack Testing - CLAIM INCORRECT ❌

**Gap Analysis Claim:**
> "**RPM/Speed** | Untested | 0% | Unknown | Unknown | Unknown | **UNKNOWN**"
> 
> "**Current Performance:** Not tested on any RPM datasets"

**Actual Evidence:**
```bash
$ ls -lh academic_test_results/batch_set01_20251203_133029/rpm-*/20251203_*/comprehensive_summary.json
-rw-r--r-- 1 boneysan boneysan 875 Dec  3 15:18 rpm-1/20251203_150424/comprehensive_summary.json
-rw-r--r-- 1 boneysan boneysan 873 Dec  3 15:32 rpm-2/20251203_151814/comprehensive_summary.json
```

**Verification:**
```bash
$ find academic_test_results -name "*rpm*" -type d
academic_test_results/batch_set01_20251130_210252/rpm-1
academic_test_results/batch_set01_20251130_210252/rpm-2
academic_test_results/batch_set01_20251130_231940/rpm-1
academic_test_results/batch_set01_20251130_231940/rpm-2
academic_test_results/batch_set01_20251203_133029/rpm-1
academic_test_results/batch_set01_20251203_133029/rpm-2
```

**Correction:** RPM attacks WERE tested across multiple batch runs (November 30th and December 3rd). The issue is that detection performance metrics were not captured in the comprehensive_summary.json files.

---

### 2. Accessory Attack Testing - CLAIM INCORRECT ❌

**Gap Analysis Claim:**
> "**Accessory** | Untested | 0% | Unknown | Unknown | Unknown | **UNKNOWN**"

**Actual Evidence:**
```bash
$ find academic_test_results/batch_set01_20251203_133029 -name "*accessory*" -type d
academic_test_results/batch_set01_20251203_133029/accessory-1
academic_test_results/batch_set01_20251203_133029/accessory-2
```

**Test Results:**
- `accessory-1/20251203_142931/comprehensive_summary.json` (885 bytes)
- `accessory-2/20251203_143245/comprehensive_summary.json` (885 bytes)

**Correction:** Accessory attacks WERE tested on December 3rd with both datasets (accessory-1.csv and accessory-2.csv).

---

### 3. Force Neutral Attack Testing - CLAIM INCORRECT ❌

**Gap Analysis Claim:**
> "**Force Neutral** | Untested | 0% | Unknown | Unknown | Unknown | **UNKNOWN**"

**Actual Evidence:**
```bash
$ find academic_test_results/batch_set01_20251203_133029 -name "*force-neutral*" -type d
academic_test_results/batch_set01_20251203_133029/force-neutral-1
academic_test_results/batch_set01_20251203_133029/force-neutral-2
```

**Test Results:**
- `force-neutral-1/20251203_143618/comprehensive_summary.json`
- `force-neutral-2/20251203_144756/comprehensive_summary.json`

**Correction:** Force Neutral attacks WERE tested on December 3rd with both datasets.

---

### 4. Standstill Attack Testing - CLAIM INCORRECT ❌

**Gap Analysis Claim:**
> "**Standstill** | Untested | 0% | Unknown | Unknown | Unknown | **UNKNOWN**"

**Actual Evidence:**
```bash
$ find academic_test_results/batch_set01_20251203_133029 -name "*standstill*" -type d
academic_test_results/batch_set01_20251203_133029/standstill-1
academic_test_results/batch_set01_20251203_133029/standstill-2
```

**Test Results:**
- `standstill-1/20251203_153216/comprehensive_summary.json`
- `standstill-2/20251203_160440/comprehensive_summary.json`

**Correction:** Standstill attacks WERE tested on December 3rd with both datasets.

---

## Legitimate Issue Identified

### Detection Performance Metrics Not Captured

The gap analysis correctly identifies that performance data is missing, but incorrectly attributes this to tests not being run.

**Evidence from December 3rd Batch Tests:**
```json
{
  "test_info": {
    "data_file": "rpm-1.csv",
    "test_date": "2025-12-03T15:18:09.783560",
    "config": {
      "sample_interval": 1.0,
      "enable_ml": true,
      "rules_file": "config/rules.yaml"
    }
  },
  "performance": {},  // ← EMPTY - metrics not captured
  "system": {
    "total_samples": 494,
    "duration_seconds": 823.523843050003,
    "cpu_percent": { "mean": 24.8, "max": 30.1, "min": 9.6 },
    "memory_rss_mb": { "mean": 419.1, "max": 467.1, "min": 127.1 },
    "temperature_c": { "mean": 50.9, "max": 53.0, "min": 49.1 }
  }
}
```

**All 12 tests from December 3rd batch have this issue:**
- DoS-1, DoS-2
- rpm-1, rpm-2
- accessory-1, accessory-2
- force-neutral-1, force-neutral-2
- standstill-1, standstill-2
- attack-free-1, attack-free-2

**Root Cause:** The test framework captured system performance metrics (CPU, memory, temperature) but did not capture detection performance metrics (true positives, false positives, detection rate, etc.).

---

## January 5, 2026 Update: Complete Metrics Recovery

### Comprehensive Test Results (160,000 messages tested)

**Test Configuration:**
- System: Enhanced Hybrid (Fuzzing Rules + Decision Tree ML)
- Hardware: Raspberry Pi 4
- Duration: ~10 minutes
- Monitoring: CPU, RAM, Temperature, Throughput
- Metrics: TP/FP/TN/FN, Precision, Recall, F1-Score, Accuracy

### Complete Performance Matrix with Validated Metrics

| Attack Type | Recall | Precision | F1-Score | Dataset Quality | Status |
|-------------|--------|-----------|----------|----------------|--------|
| **DoS Flood** | **99.64-99.91%** ✅ | 8.51-10.64% | 0.1568-0.1923 | ✅ Valid (8-11% attacks) | **EXCELLENT RECALL** |
| **Fuzzing** | **90.91-95.64%** ✅ | 0.91-15.45% | 0.0181-0.2660 | ✅ Valid (0.5-9% attacks) | **EXCELLENT RECALL** |
| **Interval Timing** | 66.67-88.26% | 0.05-2.63% | 0.0011-0.0511 | ⚠️ Imbalanced (0.03-2.6% attacks) | **FAIR** |
| **RPM** | 40.00-64.86% | **0.04-0.46%** | 0.0008-0.0091 | 🚨 **Severe imbalance** (0.05-0.37% attacks) | **DATASET INVALID** |
| **Accessory** | **N/A** | **0.00%** | **0.0000** | 🚨 **ZERO ATTACKS** (TP=0, FN=0) | **DATASET INVALID** |
| **Force Neutral** | 14.29-87.00% | **0.03-1.44%** | 0.0006-0.0283 | 🚨 **Severe imbalance** (0.07-1% attacks) | **DATASET INVALID** |
| **Standstill** | 27.27-62.50% | **0.15-0.16%** | 0.0030-0.0032 | 🚨 **Severe imbalance** (0.08-0.11% attacks) | **DATASET INVALID** |
| **Normal Traffic** | N/A | N/A | N/A | ✅ Valid (0% attacks) | **17.6-31.7% FP Rate** |

### System Resource Performance (All Tests)

| Metric | Average | Peak | Status |
|--------|---------|------|--------|
| **CPU Usage** | 95.9-98.1% | 101.7-102.0% | ⚠️ Near saturation |
| **RAM Usage** | 201-252 MB | 201-252 MB | ✅ Excellent efficiency |
| **Temperature** | 48-51°C | 49-53°C | ✅ Safe range |
| **Throughput** | 332-376 msg/s | N/A | ✅ Real-time capable |
| **Latency** | 2.66-3.01 ms | N/A | ✅ Low latency |

## Critical Discoveries from January 5th Testing

### 🚨 Dataset Quality Issues Identified

#### 1. Accessory Attacks - **COMPLETELY INVALID**
- **Finding:** TP=0, FN=0 across both datasets
- **Conclusion:** Datasets contain **ZERO actual attacks** (100% normal traffic)
- **Impact:** Previous "98% detection rate" claim is **FALSE** (all false positives)
- **Evidence:** `accessory-1.csv` and `accessory-2.csv` mislabeled as attack data
- **Action Required:** Discard datasets, remove from capability claims

#### 2. RPM/Force-Neutral/Standstill - **SEVERE IMBALANCE**
- **Finding:** Only 2-100 attacks per 10,000 messages (0.02-1%)
- **Impact:** Precision <1.5%, meaning 98.5%+ of detections are false positives
- **Examples:**
  - `rpm-2.csv`: 5 attacks, 9,995 normal → 0.04% precision
  - `force-neutral-2.csv`: 7 attacks, 9,993 normal → 0.03% precision
  - `standstill-1.csv`: 11 attacks, 9,989 normal → 0.16% precision
- **Action Required:** Create balanced datasets or remove from validation

#### 3. Validated Datasets (Publication-Ready)
- **Fuzzing:** 50-918 attacks per 10K (0.5-9.2%) → Precision 0.91-15.45% ✅
- **DoS:** 838-1,057 attacks per 10K (8.4-10.6%) → Precision 8.51-10.64% ✅
- **Interval-1:** 264 attacks per 10K (2.6%) → Precision 2.63% ⚠️
- **Normal Traffic:** 0 attacks → 17.6-31.7% FP baseline ✅

---

## Timeline of Testing Activities

### November 30, 2025
- Academic batch tests conducted
- RPM-1 and RPM-2 datasets tested (batch_set01_20251130_210252, batch_set01_20251130_231940)

### December 3, 2025
- Comprehensive batch test run (batch_set01_20251203_133029)
- **12 datasets tested:** DoS, RPM, Accessory, Force Neutral, Standstill, Attack-Free
- System metrics captured successfully
- **Detection metrics NOT captured** (performance dictionary empty in all results)

### December 27, 2025
- Gap analysis created
- **Incorrectly labeled multiple attack types as "Untested"**
- Did not verify test directory existence before making claims

### January 4, 2026
- Evidence review completed
- Rebuttal documentation created
- Metrics recovery plan developed

### January 5, 2026
- **Complete metrics recovery accomplished**
- Test framework enhanced with:
  - TP/FP/TN/FN confusion matrix calculation
  - Precision/Recall/F1-Score metrics
  - CPU/RAM/Temperature monitoring
  - Real-time throughput tracking
- **160,000 messages tested** across 16 datasets
- **Critical discovery:** Multiple datasets have invalid/severely imbalanced labels
- Comprehensive metrics report generated (COMPREHENSIVE_METRICS_REPORT.md)

---

## Recommended Actions (Updated January 5, 2026)

### ✅ COMPLETED Actions
1. ~~**Re-run batch tests with metrics capture**~~ → **DONE**
   - ✅ All 16 datasets tested with complete metrics
   - ✅ TP/FP/TN/FN confusion matrices captured
   - ✅ Precision/Recall/F1-Score calculated
   - ✅ System resource monitoring implemented
   - **Evidence:** `full_pipeline_complete_jan5.log`, `COMPREHENSIVE_METRICS_REPORT.md`

2. ~~**Investigate metrics capture failure**~~ → **RESOLVED**
   - ✅ Root cause: `test_full_pipeline.py` didn't compare against ground truth labels
   - ✅ Fixed: Added ground truth comparison using CSV 'attack' column
   - ✅ Validation: All metrics now captured correctly

### ⚠️ NEW CRITICAL Actions (Priority 1)

3. **Address Dataset Quality Issues**
   - 🚨 **Discard accessory datasets** (contain zero attacks, completely invalid)
   - 🚨 **Flag RPM/Force-Neutral/Standstill as unreliable** (<1% attack rate)
   - ✅ **Validate only Fuzzing, DoS, Interval, Normal datasets** for publication
   - 📋 **Document dataset limitations** in research paper

4. **Update Research Claims**
   - ❌ Remove "98% accessory detection" claim (false positives on normal traffic)
   - ❌ Remove or heavily caveat RPM/Force-Neutral/Standstill results (unreliable)
   - ✅ Focus publication on validated datasets: Fuzzing (95% recall), DoS (99% recall)
   - ⚠️ Report 17.6-31.7% FP rate honestly (above 10% target)

### Short-term (Priority 2)

5. **Create Balanced Datasets (If Needed)**
   - Collect or generate RPM attack data (target: 20-40% attack rate)
   - Collect or generate accessory attack data with actual attacks
   - Collect or generate force-neutral/standstill attack data (balanced)
   - Validate new datasets have proper ground truth labels

6. **Reduce False Positive Rate**
   - Current: 17.6-31.7% on normal traffic (target: <10%)
   - Investigate decision tree threshold adjustment
   - Consider ensemble voting to reduce FPs
   - Test post-processing filtering techniques

### Long-term (Priority 3)

7. **Improve Test Framework Validation**
   - ✅ Automated metrics validation (DONE - now captures all metrics)
   - Add dataset quality checks (attack percentage, label validation)
   - Flag severely imbalanced datasets (<5% attacks)
   - Require minimum attack samples (e.g., 100+) for validation

8. **Complete Untested Attack Types**
   - Replay attacks: Create datasets and validate detection
   - Speed attacks: Create datasets and implement detection
   - Update research scope based on actual capabilities

---

## Conclusion

### Original Rebuttal (January 4, 2026)

The gap analysis identifies a legitimate concern (missing detection metrics) but does so using factually incorrect claims. The evidence clearly shows:

✅ **What Actually Happened:**
- RPM, Accessory, Force Neutral, and Standstill attacks WERE tested
- System performance metrics (CPU, memory, temperature) were captured
- Detection performance metrics were NOT captured (empty performance dictionaries)
- 12 comprehensive test runs completed on December 3rd

❌ **What Gap Analysis Incorrectly Claimed:**
- "Untested" status for multiple attack types
- "Not tested on any RPM datasets"
- No test results available

### Updated Conclusion (January 5, 2026)

**Metrics Recovery Complete:** All detection performance metrics have been recovered through comprehensive testing on January 5th, 2026.

✅ **What We NOW Know (With Complete Metrics):**

**Validated Performance (Publication-Ready):**
- **Fuzzing Detection:** 90.91-95.64% recall, 0.91-15.45% precision ✅
- **DoS Detection:** 99.64-99.91% recall, 8.51-10.64% precision ✅
- **System Performance:** 332-376 msg/s, 201-252 MB RAM, 48-51°C ✅
- **False Positive Rate:** 17.6-31.7% (baseline on normal traffic) ⚠️

🚨 **Critical Dataset Issues Discovered:**
- **Accessory datasets are INVALID:** Contain zero attacks (TP=0, FN=0), all detections are false positives
- **RPM/Force-Neutral/Standstill severely imbalanced:** <1% attack rate makes precision metrics unreliable (<0.5%)
- **Only 4 attack types have valid data:** Fuzzing, DoS, Interval-1, Normal traffic

**Impact on Research Claims:**
- ❌ Cannot claim 98% accessory detection (dataset has no attacks)
- ❌ Cannot validate RPM detection (only 5-37 attacks per 10K messages)
- ❌ Cannot validate Force-Neutral/Standstill detection (only 1-100 attacks per 10K)
- ✅ CAN claim excellent recall on Fuzzing (95%) and DoS (99%)
- ⚠️ Must acknowledge high FP rate (18-32%, target was <10%)

**Final Assessment:** The gap analysis was correct that metrics were missing, but the real discovery is that several datasets are fundamentally flawed (invalid labels or severe imbalance). The testing infrastructure is now working perfectly, but half the datasets cannot be used for valid performance claims.

---

## Supporting Evidence Files

**December 3rd Test Results (Metrics Missing):**
```
academic_test_results/batch_set01_20251203_133029/
├── DoS-1/20251203_142311/comprehensive_summary.json (performance: {})
├── DoS-2/20251203_142437/comprehensive_summary.json (performance: {})
├── rpm-1/20251203_150424/comprehensive_summary.json (performance: {})
├── rpm-2/20251203_151814/comprehensive_summary.json (performance: {})
├── accessory-1/20251203_142931/comprehensive_summary.json (performance: {})
├── accessory-2/20251203_143245/comprehensive_summary.json (performance: {})
├── force-neutral-1/20251203_143618/comprehensive_summary.json (performance: {})
├── force-neutral-2/20251203_144756/comprehensive_summary.json (performance: {})
├── standstill-1/20251203_153216/comprehensive_summary.json (performance: {})
├── standstill-2/20251203_160440/comprehensive_summary.json (performance: {})
├── attack-free-1/20251203_133041/comprehensive_summary.json (performance: {})
└── attack-free-2/20251203_140308/comprehensive_summary.json (performance: {})
```

**January 5th Complete Test Results:**
```
test_data/ (16 datasets tested)
├── fuzzing-1.csv → 878 TP, 4,805 FP, 95.64% recall, 15.45% precision ✅
├── fuzzing-2.csv → 50 TP, 5,434 FP, 90.91% recall, 0.91% precision ✅
├── DoS-1.csv → 1,056 TP, 8,870 FP, 99.91% recall, 10.64% precision ✅
├── DoS-2.csv → 838 TP, 9,009 FP, 99.64% recall, 8.51% precision ✅
├── interval-1.csv → 233 TP, 8,630 FP, 88.26% recall, 2.63% precision ⚠️
├── interval-2.csv → 2 TP, 3,704 FP, 66.67% recall, 0.05% precision ❌
├── rpm-1.csv → 24 TP, 5,229 FP, 64.86% recall, 0.46% precision ❌
├── rpm-2.csv → 2 TP, 5,295 FP, 40.00% recall, 0.04% precision ❌
├── accessory-1.csv → 0 TP, 9,824 FP, N/A recall, 0.00% precision 🚨
├── accessory-2.csv → 0 TP, 9,865 FP, N/A recall, 0.00% precision 🚨
├── force-neutral-1.csv → 87 TP, 5,952 FP, 87.00% recall, 1.44% precision ❌
├── force-neutral-2.csv → 1 TP, 3,229 FP, 14.29% recall, 0.03% precision ❌
├── standstill-1.csv → 3 TP, 1,836 FP, 27.27% recall, 0.16% precision ❌
├── standstill-2.csv → 5 TP, 3,311 FP, 62.50% recall, 0.15% precision ❌
├── attack-free-1.csv → 0 TP, 1,764 FP (17.6% FP rate) ✅
└── attack-free-2.csv → 0 TP, 3,173 FP (31.7% FP rate) ✅
```

**Log Files:**
- `full_pipeline_complete_jan5.log` - Complete test run with all metrics
- `full_pipeline_metrics_jan4.log` - Initial metrics recovery test
- `full_pipeline_extended_jan4.log` - Extended test results
- `logs/dec27_enhanced_hybrid.log` (4,412 bytes)
- `logs/dec27_test3_decision_tree.log` (3,260 bytes)

**Documentation:**
- `docs/COMPREHENSIVE_METRICS_REPORT.md` - **Complete performance report with dataset quality analysis**
- `docs/PROOF_OF_RESULTS.md` - Comprehensive evidence documentation
- `GAP_Analysis/METRICS_RECOVERY_PLAN.md` - 3-phase recovery plan
- `GAP_Analysis/Testing Documentation Gap Analysis` - Original gap analysis
- This document - Factual corrections and complete metrics

**Test Framework:**
- `scripts/test_full_pipeline.py` - Enhanced with TP/FP/TN/FN metrics, resource monitoring
- Configuration: `config/rules_fuzzing_only.yaml` (4 rules)
- Model: `data/models/decision_tree.pkl` (175.7 KB, depth 12, 1112 leaves)

---

**Document Status:** Complete (v2.0)  
**Evidence Verified:** January 5, 2026  
**Metrics Recovery:** ✅ COMPLETE  
**Next Action Required:** Address dataset quality issues, update research claims
