# CAN-IDS Testing Results

**Test Date:** November 25, 2025  
**Platform:** Raspberry Pi 4 Model B  
**Hardware:** MCP2515 CAN HAT (16MHz oscillator)  
**OS:** Raspberry Pi OS Bookworm  
**Python Version:** 3.11.2  
**CAN Bitrate:** 500 kbps  

---

## Executive Summary

This document records the performance testing results for the CAN-IDS (Controller Area Network Intrusion Detection System) deployed on Raspberry Pi 4 hardware. Tests validate system capability to process real-time CAN bus traffic while maintaining acceptable resource usage and detection accuracy.

### Overall System Status: ⚠️ **Pending Testing**

Key findings will be documented below as tests are completed.

---

## Test Environment

### Hardware Configuration
- **Model:** Raspberry Pi 4 Model B
- **RAM:** [Record your RAM amount: 2GB/4GB/8GB]
- **Storage:** microSD card (Class 10)
- **CAN Interface:** MCP2515-based CAN HAT via SPI
- **Oscillator:** 16MHz
- **Cooling:** [Record: Passive/Active/None]
- **Power Supply:** 5V/3A USB-C

### Software Configuration
- **OS:** Raspberry Pi OS (Bookworm)
- **Kernel:** `uname -r` output: _______________
- **Python:** 3.11.2
- **CAN-IDS Version:** 1.0.0
- **Configuration:** `config/can_ids_rpi4.yaml`

### Optimization Status
- [x] System optimizations applied (`optimize_pi4.sh`)
- [x] Unnecessary services disabled
- [x] tmpfs configured for logs
- [x] Hardware watchdog enabled
- [x] Swap optimized (swappiness=10)

---

## Test 1: Interface Connectivity

**Date:** _______________  
**Duration:** 30 seconds  
**Test Command:**
```bash
python scripts/can_traffic_test.py --interface can0 --test-connectivity
```

### Results

**Interface Status:**
- [ ] ✓ Interface exists in system
- [ ] ✓ Successfully opened with python-can
- [ ] ✓ Receiving CAN messages
- [ ] ⚠️ Interface up but no traffic
- [ ] ✗ Interface not accessible

**First Message Received:**
```
ID: 0x___  
DLC: ___  
Data: __ __ __ __ __ __ __ __
```

**Status:** ☐ PASS ☐ FAIL

**Notes:**
```
[Record any observations about interface setup or connectivity issues]
```

---

## Test 2: Traffic Monitoring (30 seconds)

**Date:** _______________  
**Duration:** 30 seconds  
**Test Command:**
```bash
python scripts/can_traffic_test.py --interface can0 --monitor --duration 30
```

### Results

| Metric | Value |
|--------|-------|
| **Total Messages** | _________ |
| **Unique CAN IDs** | _________ |
| **Message Rate** | _________ msg/s |
| **Runtime** | _________ seconds |
| **Interface Status** | active_traffic / no_traffic |

**Sample Messages Observed:**
```
[Paste sample output from monitoring]
Timestamp    Interface  ID   [DLC]  Data
1732561234.5  can0      123  [8]    DE AD BE EF CA FE BA BE
```

**Traffic Pattern Analysis:**
- Periodic messages: ☐ Yes ☐ No
- Variable data: ☐ Yes ☐ No
- Consistent timing: ☐ Yes ☐ No

**Status:** ☐ PASS ☐ FAIL

**Notes:**
```
[Record observations about traffic patterns, message frequencies, etc.]
```

---

## Test 3: Performance Benchmark

**Date:** _______________  
**Test Data:** Synthetic / Real Traffic (circle one)  
**Message Count:** _________ messages  
**Test Command:**
```bash
python scripts/benchmark.py --component all --messages _____
```

### 3.1 Rule Engine Performance

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Throughput** | _____ msg/s | >5000 msg/s | ☐ Pass ☐ Fail |
| **Mean Latency** | _____ ms | <0.5 ms | ☐ Pass ☐ Fail |
| **P95 Latency** | _____ ms | <1.0 ms | ☐ Pass ☐ Fail |
| **P99 Latency** | _____ ms | <2.0 ms | ☐ Pass ☐ Fail |
| **CPU Usage** | _____ % | <70% | ☐ Pass ☐ Fail |
| **Memory Usage** | _____ MB | <512 MB | ☐ Pass ☐ Fail |
| **Alerts Generated** | _____ | N/A | N/A |
| **Alert Rate** | _____ % | N/A | N/A |

### 3.2 ML Detector Performance

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Throughput** | _____ msg/s | >3000 msg/s | ☐ Pass ☐ Fail |
| **Mean Latency** | _____ ms | <1.0 ms | ☐ Pass ☐ Fail |
| **P95 Latency** | _____ ms | <2.0 ms | ☐ Pass ☐ Fail |
| **P99 Latency** | _____ ms | <5.0 ms | ☐ Pass ☐ Fail |
| **CPU Usage** | _____ % | <70% | ☐ Pass ☐ Fail |
| **Memory Usage** | _____ MB | <512 MB | ☐ Pass ☐ Fail |
| **Anomalies Detected** | _____ | N/A | N/A |
| **Anomaly Rate** | _____ % | N/A | N/A |

### 3.3 End-to-End Pipeline Performance

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Throughput** | _____ msg/s | >2500 msg/s | ☐ Pass ☐ Fail |
| **Mean Latency** | _____ ms | <1.0 ms | ☐ Pass ☐ Fail |
| **P95 Latency** | _____ ms | <2.0 ms | ☐ Pass ☐ Fail |
| **P99 Latency** | _____ ms | <5.0 ms | ☐ Pass ☐ Fail |
| **CPU Usage (Avg)** | _____ % | <70% | ☐ Pass ☐ Fail |
| **CPU Usage (Max)** | _____ % | <80% | ☐ Pass ☐ Fail |
| **Memory (Avg)** | _____ MB | <400 MB | ☐ Pass ☐ Fail |
| **Memory (Max)** | _____ MB | <512 MB | ☐ Pass ☐ Fail |
| **Total Alerts** | _____ | N/A | N/A |

**Overall Benchmark Status:** ☐ PASS ☐ FAIL

**Notes:**
```
[Record any performance bottlenecks, unexpected results, or observations]
```

---

## Test 4: Live CAN-IDS Operation

**Date:** _______________  
**Duration:** 5 minutes  
**Configuration:** `config/can_ids_rpi4.yaml`  
**Detection Modes:** ☐ Rule-based ☐ ML-based ☐ Both  
**Test Command:**
```bash
timeout 300 python main.py -i can0 --config config/can_ids_rpi4.yaml
```

### 4.1 Processing Statistics

| Metric | Value |
|--------|-------|
| **Runtime** | _____ seconds |
| **Messages Processed** | _____ |
| **Messages per Second** | _____ msg/s |
| **Alerts Generated** | _____ |
| **Alert Rate** | _____ % |
| **Messages Dropped** | _____ |
| **Drop Rate** | _____ % |
| **Errors Encountered** | _____ |

### 4.2 Resource Usage

**CPU Utilization:**
```
[Record output from: top -b -n 1 | grep "python main.py"]
Average: _____ %
Peak: _____ %
```

**Memory Usage:**
```
[Record output from: ps aux | grep "python main.py" | grep -v grep]
RSS: _____ MB
VSZ: _____ MB
```

**Temperature:**
```
[Record output from: vcgencmd measure_temp]
Start: _____ °C
End: _____ °C
Peak: _____ °C
```

**Throttling Status:**
```
[Record output from: vcgencmd get_throttled]
0x0 = No throttling
Other = Throttling occurred: _____
```

### 4.3 CAN Interface Statistics

```bash
# Record output from: ip -s link show can0
```

```
RX Packets: _____
RX Dropped: _____
RX Errors: _____
TX Packets: _____
```

### 4.4 Alert Analysis

**Sample Alerts Generated:**
```json
[Paste 2-3 example alerts from logs/alerts.json]
```

**Alert Severity Breakdown:**
- CRITICAL: _____
- HIGH: _____
- MEDIUM: _____
- LOW: _____

**Status:** ☐ PASS ☐ FAIL

**Notes:**
```
[Record observations about system stability, alert quality, false positives, etc.]
```

---

## Test 5: Stress Test (High Load)

**Date:** _______________  
**Duration:** 2 minutes  
**Traffic Generator:** `cangen can0 -g 1 -I 100 -L 8 -D r`  
**Expected Load:** ~1000 msg/s  

### Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Messages Processed** | _____ | N/A | N/A |
| **Throughput** | _____ msg/s | >1000 msg/s | ☐ Pass ☐ Fail |
| **Messages Dropped** | _____ | <1% | ☐ Pass ☐ Fail |
| **CPU Usage (Peak)** | _____ % | <80% | ☐ Pass ☐ Fail |
| **Memory Usage (Peak)** | _____ MB | <512 MB | ☐ Pass ☐ Fail |
| **Temperature (Peak)** | _____ °C | <75°C | ☐ Pass ☐ Fail |
| **System Stability** | Stable / Crashed | Stable | ☐ Pass ☐ Fail |

**Buffer Statistics:**
```
Buffer Size: _____
Peak Utilization: _____ (_____%)
```

**Status:** ☐ PASS ☐ FAIL

**Notes:**
```
[Record system behavior under high load, any warnings, throttling events, etc.]
```

---

## Test 6: Attack Detection Validation

**Date:** _______________  
**Attack Types Tested:** ☐ DoS ☐ Replay ☐ Fuzzing ☐ Other: _____

### 6.1 DoS Attack Detection

**Test Setup:**
```bash
# High-frequency message injection
cangen can0 -g 0.1 -I 123 -L 8 -n 10000
```

| Metric | Value | Expected |
|--------|-------|----------|
| **Attack Messages Sent** | 10000 | 10000 |
| **Attack Detected** | ☐ Yes ☐ No | Yes |
| **Detection Time** | _____ seconds | <5 seconds |
| **Alert Severity** | _____ | HIGH/CRITICAL |
| **False Positives** | _____ | 0 |

**Status:** ☐ PASS ☐ FAIL

### 6.2 Frequency Anomaly Detection

**Test Setup:**
```bash
# Abnormal message frequency
while true; do cansend can0 456#1122334455667788; sleep 0.001; done
```

| Metric | Value | Expected |
|--------|-------|----------|
| **Anomaly Detected** | ☐ Yes ☐ No | Yes |
| **Detection Time** | _____ seconds | <10 seconds |
| **Alert Generated** | ☐ Yes ☐ No | Yes |

**Status:** ☐ PASS ☐ FAIL

### 6.3 Detection Accuracy Summary

| Metric | Value | Target |
|--------|-------|--------|
| **True Positives** | _____ | N/A |
| **False Positives** | _____ | <5% |
| **False Negatives** | _____ | 0 |
| **Detection Rate** | _____ % | >95% |

**Notes:**
```
[Record observations about detection accuracy, timing, alert quality]
```

---

## Test 7: Long-Duration Stability Test

**Date:** _______________  
**Duration:** _____ hours (Target: 24 hours)  
**Test Command:**
```bash
nohup python main.py -i can0 --config config/can_ids_rpi4.yaml > ids_longrun.log 2>&1 &
```

### 7.1 Memory Leak Analysis

**Memory Usage Over Time:**

| Time (hours) | Memory (MB) | Delta |
|--------------|-------------|-------|
| 0 | _____ | - |
| 1 | _____ | +_____ |
| 2 | _____ | +_____ |
| 4 | _____ | +_____ |
| 8 | _____ | +_____ |
| 12 | _____ | +_____ |
| 24 | _____ | +_____ |

**Memory Growth Rate:** _____ MB/hour

**Memory Leak Detected:** ☐ Yes ☐ No

### 7.2 Long-Term Statistics

| Metric | Value |
|--------|-------|
| **Total Runtime** | _____ hours |
| **Total Messages** | _____ |
| **Total Alerts** | _____ |
| **Crashes/Restarts** | _____ |
| **Error Count** | _____ |
| **Log File Size** | _____ MB |

### 7.3 System Stability

**Observations:**
- System remained responsive: ☐ Yes ☐ No
- Alert logging functional: ☐ Yes ☐ No
- No performance degradation: ☐ Yes ☐ No
- Temperature stable: ☐ Yes ☐ No

**Status:** ☐ PASS ☐ FAIL

**Notes:**
```
[Record any issues encountered during long-term operation]
```

---

## Test 8: Thermal Performance

**Date:** _______________  
**Ambient Temperature:** _____ °C  
**Cooling:** Passive / Active / None (circle one)  

### Temperature Under Different Loads

| Load Condition | Duration | Temp (°C) | Throttling |
|----------------|----------|-----------|------------|
| Idle | 5 min | _____ | ☐ Yes ☐ No |
| Low Load (~100 msg/s) | 10 min | _____ | ☐ Yes ☐ No |
| Normal Load (~500 msg/s) | 10 min | _____ | ☐ Yes ☐ No |
| High Load (~1000 msg/s) | 10 min | _____ | ☐ Yes ☐ No |
| Stress Test (max) | 5 min | _____ | ☐ Yes ☐ No |

**Peak Temperature:** _____ °C  
**Throttling Events:** _____ (0x0 = none)

**Thermal Management Status:** ☐ PASS ☐ FAIL

**Recommendations:**
```
[ ] No action needed (temp <65°C)
[ ] Monitor closely (temp 65-70°C)
[ ] Add passive cooling (temp 70-75°C)
[ ] Add active cooling (temp >75°C)
```

**Notes:**
```
[Record thermal behavior, throttling patterns, cooling effectiveness]
```

---

## Test 9: Configuration Validation

**Date:** _______________  

### 9.1 Buffer Size Testing

**Test Different Buffer Sizes:**

| Buffer Size | Drop Rate | CPU % | Memory (MB) | Status |
|-------------|-----------|-------|-------------|--------|
| 250 | _____% | _____ | _____ | ☐ Pass ☐ Fail |
| 500 (default) | _____% | _____ | _____ | ☐ Pass ☐ Fail |
| 1000 | _____% | _____ | _____ | ☐ Pass ☐ Fail |
| 2000 | _____% | _____ | _____ | ☐ Pass ☐ Fail |

**Optimal Buffer Size:** _____

### 9.2 Detection Mode Comparison

| Mode | Throughput | Latency | CPU % | Memory (MB) |
|------|------------|---------|-------|-------------|
| Rule-based only | _____ msg/s | _____ ms | _____ | _____ |
| ML-based only | _____ msg/s | _____ ms | _____ | _____ |
| Both enabled | _____ msg/s | _____ ms | _____ | _____ |

**Recommended Mode for Production:** _____________________

**Notes:**
```
[Record findings about optimal configuration settings]
```

---

## Test 10: SD Card Performance

**Date:** _______________  
**Card Type:** _______________  
**Logging Configuration:** tmpfs / SD card (circle one)

### Write Performance

| Test | Duration | Data Written | Write Speed | Status |
|------|----------|--------------|-------------|--------|
| Alert logging (high volume) | 5 min | _____ MB | _____ MB/s | ☐ Pass ☐ Fail |
| PCAP capture | 5 min | _____ MB | _____ MB/s | ☐ Pass ☐ Fail |
| Combined logging | 5 min | _____ MB | _____ MB/s | ☐ Pass ☐ Fail |

**SD Card Health:**
- No I/O errors: ☐ Yes ☐ No
- Acceptable write latency: ☐ Yes ☐ No
- tmpfs functioning: ☐ Yes ☐ No

**Status:** ☐ PASS ☐ FAIL

---

## Performance Summary

### Overall System Performance

| Category | Rating | Notes |
|----------|--------|-------|
| **Throughput** | ☐ Excellent ☐ Good ☐ Acceptable ☐ Poor | _____ msg/s sustained |
| **Latency** | ☐ Excellent ☐ Good ☐ Acceptable ☐ Poor | _____ ms avg |
| **Reliability** | ☐ Excellent ☐ Good ☐ Acceptable ☐ Poor | _____% uptime |
| **Resource Usage** | ☐ Excellent ☐ Good ☐ Acceptable ☐ Poor | CPU: _____% Mem: _____ MB |
| **Detection Accuracy** | ☐ Excellent ☐ Good ☐ Acceptable ☐ Poor | _____% accuracy |
| **Thermal Performance** | ☐ Excellent ☐ Good ☐ Acceptable ☐ Poor | _____ °C peak |

### Test Results Summary

| Test | Status | Critical Issues |
|------|--------|-----------------|
| 1. Interface Connectivity | ☐ Pass ☐ Fail | _____ |
| 2. Traffic Monitoring | ☐ Pass ☐ Fail | _____ |
| 3. Performance Benchmark | ☐ Pass ☐ Fail | _____ |
| 4. Live Operation | ☐ Pass ☐ Fail | _____ |
| 5. Stress Test | ☐ Pass ☐ Fail | _____ |
| 6. Attack Detection | ☐ Pass ☐ Fail | _____ |
| 7. Long-Duration Stability | ☐ Pass ☐ Fail | _____ |
| 8. Thermal Performance | ☐ Pass ☐ Fail | _____ |
| 9. Configuration Validation | ☐ Pass ☐ Fail | _____ |
| 10. SD Card Performance | ☐ Pass ☐ Fail | _____ |

**Tests Passed:** _____ / 10

---

## Issues and Limitations

### Issues Identified

**Issue #1:**
```
Description:
Impact: [ ] Critical [ ] High [ ] Medium [ ] Low
Resolution:
Status: [ ] Resolved [ ] In Progress [ ] Deferred
```

**Issue #2:**
```
Description:
Impact: [ ] Critical [ ] High [ ] Medium [ ] Low
Resolution:
Status: [ ] Resolved [ ] In Progress [ ] Deferred
```

**Issue #3:**
```
Description:
Impact: [ ] Critical [ ] High [ ] Medium [ ] Low
Resolution:
Status: [ ] Resolved [ ] In Progress [ ] Deferred
```

### Known Limitations

1. **Throughput Ceiling:**
   - Observed max: _____ msg/s
   - Limiting factor: ☐ CPU ☐ Memory ☐ I/O ☐ Network ☐ Other: _____

2. **Detection Blind Spots:**
   ```
   [Document any attack types or patterns that weren't detected]
   ```

3. **Resource Constraints:**
   ```
   [Document any resource bottlenecks encountered]
   ```

---

## Optimization Recommendations

### Immediate Actions Required

- [ ] **Priority 1:** _________________________________________________
- [ ] **Priority 2:** _________________________________________________
- [ ] **Priority 3:** _________________________________________________

### Configuration Tuning

**Recommended Changes to `config/can_ids_rpi4.yaml`:**
```yaml
# Based on test results, recommend:
performance:
  max_cpu_percent: _____  # Current: 70
  max_memory_mb: _____    # Current: 300

capture:
  buffer_size: _____      # Current: 500

# Other adjustments:
```

### Hardware Recommendations

- [ ] Add active cooling (if temp >70°C)
- [ ] Upgrade to higher capacity SD card
- [ ] Consider Pi 4 with more RAM
- [ ] Add RTC module for accurate timestamps
- [ ] Other: _________________________________________________

---

## Production Readiness Assessment

### Readiness Checklist

**Functional Requirements:**
- [ ] All detection rules working correctly
- [ ] Alerts generating and logging properly
- [ ] No critical errors during operation
- [ ] Configuration validated
- [ ] Documentation complete

**Performance Requirements:**
- [ ] Throughput meets expected CAN bus load
- [ ] Latency acceptable for real-time detection
- [ ] No message drops under normal load
- [ ] Resource usage within limits
- [ ] Thermal performance acceptable

**Reliability Requirements:**
- [ ] No crashes during extended testing
- [ ] No memory leaks detected
- [ ] Error handling validated
- [ ] Recovery mechanisms working
- [ ] Monitoring and alerting functional

**Overall Production Readiness:** ☐ READY ☐ NOT READY ☐ READY WITH CAVEATS

**Caveats/Conditions:**
```
[List any conditions or limitations for production deployment]
```

---

## Recommendations for Production Deployment

### Pre-Deployment Checklist

1. **System Configuration:**
   - [ ] Apply all system optimizations
   - [ ] Configure optimal buffer sizes
   - [ ] Enable appropriate detection modes
   - [ ] Set up log rotation
   - [ ] Configure alert notifications

2. **Monitoring Setup:**
   - [ ] Set up temperature monitoring
   - [ ] Configure resource alerts
   - [ ] Enable watchdog
   - [ ] Set up remote logging
   - [ ] Configure health checks

3. **Backup and Recovery:**
   - [ ] Backup configuration files
   - [ ] Document recovery procedures
   - [ ] Test service auto-restart
   - [ ] Prepare SD card image backup

### Ongoing Monitoring

**Metrics to Track:**
- Messages processed per hour
- Drop rate trends
- CPU/memory utilization
- Temperature patterns
- Alert frequency
- False positive rate

**Monitoring Frequency:**
- Real-time: Temperature, drop rate
- Hourly: Resource usage, throughput
- Daily: Alert summary, error logs
- Weekly: Performance trends, capacity planning

---

## Conclusion

**Test Date Range:** _______________  
**Testing Duration:** _______________  
**Tester:** _______________  

**Final Verdict:**

☐ **PASS** - System ready for production deployment  
☐ **CONDITIONAL PASS** - Ready with noted limitations  
☐ **FAIL** - Significant issues require resolution  

**Summary Statement:**
```
[Provide 2-3 paragraph summary of testing outcomes, key findings, 
and overall readiness of the system for production use]






```

**Next Steps:**
1. _________________________________________________
2. _________________________________________________
3. _________________________________________________

---

## Appendices

### Appendix A: Test Commands Reference

```bash
# Interface connectivity test
python scripts/can_traffic_test.py --interface can0 --test-connectivity

# Traffic monitoring
python scripts/can_traffic_test.py --interface can0 --monitor --duration 30

# Performance benchmark
python scripts/benchmark.py --component all --messages 5000

# Live IDS operation
python main.py -i can0 --config config/can_ids_rpi4.yaml

# Stress test traffic generation
cangen can0 -g 1 -I 100 -L 8 -D r

# Temperature monitoring
watch -n 5 'vcgencmd measure_temp && vcgencmd get_throttled'

# Resource monitoring
htop -p $(pgrep -f "python main.py")

# CAN interface statistics
watch -n 2 'ip -s link show can0'
```

### Appendix B: Configuration Files Used

**Primary Configuration:** `config/can_ids_rpi4.yaml`
```yaml
[Attach or reference the configuration file version used for testing]
```

**Detection Rules:** `config/rules.yaml`
```yaml
[Note the rules version/checksum used for testing]
```

### Appendix C: Raw Test Data

**Location:** `performance_reports/[date]/`

Files generated:
- `benchmark.json` - Benchmark results
- `resources.csv` - Resource usage over time
- `temperature.log` - Temperature readings
- `live_test.log` - Live operation logs

### Appendix D: System Information Snapshot

```bash
# System info
uname -a

# Python packages
pip list

# CAN interface details
ip -d link show can0

# Memory info
free -h

# Disk usage
df -h
```

---

**Document Version:** 1.0  
**Last Updated:** November 25, 2025  
**Document Status:** Template - Awaiting Test Results
