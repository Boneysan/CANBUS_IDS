# Raspberry Pi 4 Validation Guide
**Date:** December 17, 2025  
**Purpose:** Validate adaptive rules fix and performance optimizations on Pi 4 hardware

---

## 🎯 What We're Testing

### Critical Issue: 100% False Positive Rate
- **Problem:** Generic rules flagged 38,839/40,000 normal messages (100% FP rate)
- **Solution:** Switched to `rules_adaptive.yaml` 
- **Expected Result:** FP rate should drop to ~8.43%
- **Test Goal:** Confirm the fix works on Pi 4 hardware

### Bonus Tests
- Pre-filter performance on ARM (not tested yet)
- PCA ML speedup (when models are trained)

---

## 📋 Prerequisites

### 1. Hardware Requirements
- ✅ Raspberry Pi 4 (4GB RAM recommended)
- ✅ MicroSD card with Raspberry Pi OS Bookworm
- ✅ CAN interface hardware (MCP2515 or similar)
- ✅ Network connection (for file transfer)
- ✅ Power supply (5V/3A minimum)

### 2. Software Requirements on Pi 4

**System packages:**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11+ (should be default on Bookworm)
python3 --version  # Should be 3.11+

# Install CAN utilities
sudo apt install -y can-utils

# Install system dependencies for Python packages
sudo apt install -y python3-dev python3-pip python3-venv
sudo apt install -y libatlas-base-dev  # For numpy/scipy on ARM
```

**Enable CAN interface:**
```bash
# Load kernel modules
sudo modprobe can
sudo modprobe can-raw
sudo modprobe mcp251x

# Configure CAN interface (replace with your actual interface)
sudo ip link set can0 type can bitrate 500000
sudo ip link set can0 up

# Verify
ip link show can0
# Should show: can0: <NOARP,UP,LOWER_UP,ECHO> mtu 16 qdisc pfifo_fast
```

**Make CAN interface permanent (optional):**
```bash
# Edit /etc/network/interfaces
sudo nano /etc/network/interfaces

# Add these lines:
auto can0
iface can0 can static
    bitrate 500000
    up /sbin/ip link set $IFACE up
    down /sbin/ip link set $IFACE down
```

### 3. Python Environment Setup

```bash
# Clone repository (if not already on Pi)
cd ~
git clone https://github.com/Boneysan/CANBUS_IDS.git
cd CANBUS_IDS

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify critical packages
python3 -c "import sklearn, numpy, pandas, joblib; print('✅ All packages imported successfully')"
```

---

## 🧪 Test 1: Validate Adaptive Rules Fix (CRITICAL)

**Goal:** Confirm FP rate drops from 100% → ~8.43%

### Step 1: Verify Configuration

```bash
# Check which rules file is configured
grep "rules_file:" config/can_ids.yaml

# Expected output:
# rules_file: config/rules_adaptive.yaml  # Switched to adaptive rules (Dec 16, 2025)

# Verify rules_adaptive.yaml exists
ls -lh config/rules_adaptive.yaml
```

### Step 2: Transfer Test Data to Pi

**On your Ubuntu machine:**
```bash
# Find test datasets in Vehicle_Models
cd ~/Documents/GitHub/Vehicle_Models
ls -lh data/raw/*.csv

# Transfer to Pi (replace PI_IP with your Pi's IP address)
scp data/raw/Normal-1.csv pi@PI_IP:~/CANBUS_IDS/test_data/
scp data/raw/Normal-2.csv pi@PI_IP:~/CANBUS_IDS/test_data/

# Optional: Transfer attack datasets for comparison
scp data/raw/DoS-1.csv pi@PI_IP:~/CANBUS_IDS/test_data/
scp data/raw/Fuzzing-1.csv pi@PI_IP:~/CANBUS_IDS/test_data/
```

**On Raspberry Pi:**
```bash
# Create test data directory
mkdir -p ~/CANBUS_IDS/test_data
```

### Step 3: Run False Positive Test

```bash
cd ~/CANBUS_IDS
source .venv/bin/activate

# Test with adaptive rules on normal traffic
python3 scripts/comprehensive_test.py \
    test_data/Normal-1.csv \
    --rules-only \
    --output test_results/adaptive_rules_validation.json

# Check results
cat test_results/adaptive_rules_validation.json | grep -A 5 "false_positive"
```

**Expected Output:**
```
Detection Results:
  Total Messages: 40,000
  Alerts Generated: ~3,372 (8.43% of normal traffic)
  False Positive Rate: ~8.43%  ← Should be MUCH LOWER than 100%!
  Throughput: ~750-1000 msg/s
```

**Success Criteria:**
- ✅ FP rate < 15% (ideally ~8.43%)
- ✅ No crashes or errors
- ✅ All 40,000 messages processed

### Step 4: Compare with Generic Rules (Optional)

```bash
# Temporarily switch to generic rules
sed -i 's/rules_adaptive.yaml/rules.yaml/' config/can_ids.yaml

# Run test again
python3 scripts/comprehensive_test.py \
    test_data/Normal-1.csv \
    --rules-only \
    --output test_results/generic_rules_comparison.json

# Switch back to adaptive rules
sed -i 's/rules.yaml/rules_adaptive.yaml/' config/can_ids.yaml

# Compare results
echo "=== Adaptive Rules ==="
cat test_results/adaptive_rules_validation.json | grep "false_positive_rate"
echo "=== Generic Rules ==="
cat test_results/generic_rules_comparison.json | grep "false_positive_rate"
```

**Expected Comparison:**
```
=== Adaptive Rules ===
  false_positive_rate: 8.43%  ← GOOD!

=== Generic Rules ===
  false_positive_rate: 97.10%  ← BAD! (100% on some datasets)
```

---

## 🧪 Test 2: Pre-Filter Performance on Pi (HIGH PRIORITY)

**Goal:** Measure pre-filter throughput on ARM architecture

### Step 1: Verify Pre-Filter is Enabled

```bash
# Check configuration
grep -A 10 "prefilter:" config/can_ids.yaml

# Should show:
# prefilter:
#   enabled: true
```

### Step 2: Run Pre-Filter Performance Test

```bash
# Test pre-filter on Pi 4 with real data
python3 scripts/test_prefilter_real.py \
    test_data/Normal-1.csv \
    --output test_results/prefilter_pi4_performance.json

# Check results
cat test_results/prefilter_pi4_performance.json
```

**Expected Output:**
```
Pre-Filter Performance (Raspberry Pi 4):
  Messages Processed: 40,000
  Pass Rate: ~99.9%
  Suspicious: ~0.1%
  Throughput: ??? msg/s  ← Need to measure!
  Avg Time per Message: ??? μs
  
Comparison to Ubuntu/x86:
  Ubuntu: 539,337 msg/s
  Pi 4: ??? msg/s
  Ratio: ???x slower (ARM vs x86)
```

**Success Criteria:**
- ✅ Throughput ≥ 7,000 msg/s (target goal)
- ✅ Pass rate ~99.9% (matches Ubuntu)
- ✅ No crashes or memory issues

---

## 🧪 Test 3: Full System Test (Rules + Pre-Filter)

**Goal:** Test complete optimized pipeline on Pi 4

### Step 1: Run Full Detection Pipeline

```bash
# Test with all optimizations enabled
python3 scripts/comprehensive_test.py \
    test_data/Normal-1.csv \
    --prefilter \
    --rules-only \
    --output test_results/full_system_pi4.json

# Monitor resource usage during test
# In another terminal:
watch -n 1 'ps aux | grep python && free -h && vcgencmd measure_temp'
```

**Expected Output:**
```
System Performance (Pi 4):
  Total Messages: 40,000
  Throughput: ??? msg/s  ← Target: ≥7,000 msg/s
  False Positive Rate: ~8.43%
  CPU Usage: <70%
  Memory Usage: <500 MB
  Temperature: <75°C
  
Pipeline Breakdown:
  Pre-filter: ~99.9% passed (bypassed deep analysis)
  Rule Engine: Analyzed ~0.1% suspicious messages
  Alerts: ~3,372 (8.43% of total)
```

**Success Criteria:**
- ✅ Throughput ≥ 7,000 msg/s
- ✅ FP rate < 15%
- ✅ No thermal throttling
- ✅ Stable operation

---

## 🧪 Test 4: ML Performance with PCA (When Models Are Trained)

**Goal:** Validate 3-5x ML speedup with PCA

### Step 1: Train PCA Models (Do this on Ubuntu first!)

**On Ubuntu machine:**
```bash
cd ~/Documents/GitHub/CANBUS_IDS
source .venv/bin/activate

# Train models with PCA
python3 scripts/train_with_pca.py \
    --data ../Vehicle_Models/data/raw/Normal-1.csv \
    --components 15 \
    --contamination 0.02 \
    --output data/models/

# Files created:
# - data/models/feature_reducer.joblib
# - data/models/model_with_pca.joblib
# - data/models/model_metadata.json
```

### Step 2: Transfer Models to Pi

```bash
# Transfer from Ubuntu to Pi
scp data/models/feature_reducer.joblib pi@PI_IP:~/CANBUS_IDS/data/models/
scp data/models/model_with_pca.joblib pi@PI_IP:~/CANBUS_IDS/data/models/
scp data/models/model_metadata.json pi@PI_IP:~/CANBUS_IDS/data/models/
```

### Step 3: Test ML Performance on Pi

**On Raspberry Pi:**
```bash
# Enable ML detection in config
sed -i 's/# - ml_based/  - ml_based/' config/can_ids.yaml

# Update model path to use PCA model
sed -i 's|path: data/models/.*|path: data/models/model_with_pca.joblib|' config/can_ids.yaml

# Run ML performance test
python3 scripts/comprehensive_test.py \
    test_data/Normal-1.csv \
    --ml-enabled \
    --output test_results/ml_with_pca_pi4.json

# Compare with non-PCA (if available)
# Previous test showed: 17.31 msg/s without PCA
# Expected with PCA: 50-85 msg/s (3-5x faster)
```

**Expected Output:**
```
ML Performance (Raspberry Pi 4):
  
Without PCA (baseline):
  Throughput: 17.31 msg/s  ← TOO SLOW!
  Inference Time: 57.7 ms/msg
  
With PCA (optimized):
  Throughput: 50-85 msg/s  ← Target: 3-5x improvement
  Inference Time: 12-20 ms/msg
  Feature Count: 15 (reduced from 58)
  Speedup: 3-5x  ✅
```

**Success Criteria:**
- ✅ ML throughput ≥ 50 msg/s
- ✅ 3-5x faster than baseline (17.31 msg/s)
- ✅ Accuracy maintained (within 5% of baseline)

---

## 🧪 Test 5: Attack Detection Validation

**Goal:** Ensure adaptive rules still detect real attacks

### Step 1: Test Attack Datasets

```bash
# Test DoS attack detection
python3 scripts/comprehensive_test.py \
    test_data/DoS-1.csv \
    --rules-only \
    --output test_results/dos_detection_pi4.json

# Test Fuzzing attack detection
python3 scripts/comprehensive_test.py \
    test_data/Fuzzing-1.csv \
    --rules-only \
    --output test_results/fuzzing_detection_pi4.json

# Check detection rates
echo "=== DoS Detection ==="
cat test_results/dos_detection_pi4.json | grep -A 3 "detection_rate"

echo "=== Fuzzing Detection ==="
cat test_results/fuzzing_detection_pi4.json | grep -A 3 "detection_rate"
```

**Expected Output:**
```
=== DoS Detection ===
  Total Attacks: ~20,000
  Detected: ~19,500+
  Detection Rate: ≥97.5%  ← Should be HIGH!
  
=== Fuzzing Detection ===
  Total Attacks: ~20,000
  Detected: ~18,000+
  Detection Rate: ≥90%  ← Should be HIGH!
```

**Success Criteria:**
- ✅ DoS detection rate ≥ 95%
- ✅ Fuzzing detection rate ≥ 85%
- ✅ No false negatives on critical attacks

---

## 📊 Summary Report Generation

### Generate Comprehensive Report

```bash
# Run all tests and generate summary
python3 scripts/generate_pi4_validation_report.py \
    --test-dir test_results/ \
    --output PI4_VALIDATION_RESULTS.md

# View results
cat PI4_VALIDATION_RESULTS.md
```

**Expected Report Sections:**
1. **Adaptive Rules Validation** ✅/❌
   - FP rate comparison (before/after)
   - Performance metrics
   
2. **Pre-Filter Performance** ✅/❌
   - Throughput on Pi 4 ARM
   - Comparison to Ubuntu x86
   
3. **Full System Performance** ✅/❌
   - End-to-end throughput
   - Resource utilization
   
4. **ML with PCA** ✅/❌ (when models trained)
   - Speedup factor
   - Accuracy retention
   
5. **Attack Detection** ✅/❌
   - Detection rates by attack type
   - False negative analysis

---

## 🚨 Troubleshooting

### Issue: CAN Interface Not Found

```bash
# Check interface status
ip link show can0

# If down, bring it up
sudo ip link set can0 type can bitrate 500000
sudo ip link set can0 up

# Check kernel modules
lsmod | grep can
```

### Issue: Python Package Installation Fails

```bash
# Install build dependencies
sudo apt install -y python3-dev libatlas-base-dev

# Try installing with --no-cache-dir
pip install --no-cache-dir -r requirements.txt

# If scikit-learn fails, try pre-built wheel
pip install --only-binary :all: scikit-learn
```

### Issue: Out of Memory

```bash
# Check swap space
free -h

# Increase swap if needed (on Pi)
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Set CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### Issue: Thermal Throttling

```bash
# Check temperature
vcgencmd measure_temp

# Check throttling status
vcgencmd get_throttled
# 0x0 = No throttling
# 0x50000 = Throttled!

# Solutions:
# 1. Add heatsink to Pi 4
# 2. Improve airflow/add fan
# 3. Reduce batch size in config
```

### Issue: Low Throughput

```bash
# Disable ML to test rules-only
sed -i 's/  - ml_based/# - ml_based/' config/can_ids.yaml

# Increase batch size
# Edit config/can_ids.yaml:
# batch_size: 200  # Increase from 100

# Check CPU frequency
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
# Should be ~1500000 (1.5 GHz) for Pi 4
```

---

## 📝 Checklist

### Before Testing
- [ ] Pi 4 has latest OS updates
- [ ] CAN interface is configured and up
- [ ] Virtual environment created and activated
- [ ] All dependencies installed
- [ ] Test data transferred to Pi
- [ ] Config points to rules_adaptive.yaml

### During Testing
- [ ] Monitor CPU temperature (should stay <75°C)
- [ ] Monitor memory usage (should stay <500 MB)
- [ ] Check for error messages in logs
- [ ] Record throughput measurements
- [ ] Note any thermal throttling

### After Testing
- [ ] Validate FP rate dropped from 100% → ~8.43%
- [ ] Confirm throughput ≥ 7,000 msg/s (with pre-filter)
- [ ] Verify attack detection rates ≥ 90%
- [ ] Document any issues or anomalies
- [ ] Generate summary report
- [ ] Update roadmap with actual Pi 4 results

---

## 🎯 Expected Final Results

| Metric | Before (Generic Rules) | After (Adaptive Rules) | Target |
|--------|----------------------|----------------------|--------|
| **False Positive Rate** | 97-100% ❌ | ~8.43% ✅ | <15% |
| **Throughput (Rules Only)** | 757 msg/s | ??? msg/s | ≥1,000 msg/s |
| **Throughput (Pre-Filter)** | Not tested | ??? msg/s | ≥7,000 msg/s |
| **ML Throughput** | 17.31 msg/s ❌ | 50-85 msg/s ✅ | ≥50 msg/s |
| **DoS Detection** | ~95% | ~95% ✅ | ≥95% |
| **Fuzzing Detection** | ~85% | ~85% ✅ | ≥85% |
| **Memory Usage** | <400 MB | <500 MB ✅ | <500 MB |
| **CPU Temp** | <70°C | <75°C ✅ | <75°C |

**Success Definition:** 
- ✅ FP rate < 15%
- ✅ Throughput ≥ 7,000 msg/s
- ✅ Attack detection ≥ 90%
- ✅ Stable operation for 1+ hour

---

## 📞 Support

If you encounter issues:

1. **Check logs**: `tail -f logs/can-ids.log`
2. **Review test output**: `cat test_results/*.json`
3. **Check system resources**: `htop` and `vcgencmd measure_temp`
4. **Verify configuration**: `cat config/can_ids.yaml`

Good luck with validation! 🚀
