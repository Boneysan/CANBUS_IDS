# CAN-IDS Performance Testing & Monitoring Guide

**Document Created:** November 25, 2025  
**Target Platform:** Raspberry Pi 4 with CAN Interface  
**Purpose:** Test system performance under CAN traffic load and optimize for production use  

---

## Overview

This guide provides comprehensive testing procedures to validate CAN-IDS performance when ingesting real CAN bus traffic. It covers CPU utilization, memory usage, dropped frames, detection latency, and other critical metrics.

## Available Testing Tools

The project includes several built-in testing utilities:

### 1. **benchmark.py** - Comprehensive Performance Benchmarking
- Measures throughput (messages/second)
- Detection latency (mean, P95, P99)
- CPU and memory usage
- Alert generation rates
- Component-specific and end-to-end testing

### 2. **can_traffic_test.py** - Live Traffic Monitoring
- Tests interface connectivity
- Monitors real-time CAN traffic
- Tracks message rates and unique IDs
- Validates IDS can see live traffic

### 3. **CANSniffer Statistics** - Built-in Metrics
- Messages received counter
- Messages dropped (buffer overflows)
- Error count
- Messages per second
- Buffer utilization

---

## Quick Performance Test

### Test 1: Interface Connectivity Check

Verify the CAN interface is working and receiving traffic:

```bash
cd /home/boneysan/Documents/Github/CANBUS_IDS
source venv/bin/activate
python scripts/can_traffic_test.py --interface can0 --test-connectivity
```

**Expected Output:**
```
✓ Interface can0 exists in system
✓ Successfully opened can0 with python-can
✓ Received CAN message: ID=0x123, DLC=8
```

### Test 2: Quick Traffic Monitor (30 seconds)

Monitor live CAN traffic and see message statistics:

```bash
python scripts/can_traffic_test.py --interface can0 --monitor --duration 30
```

**Metrics Displayed:**
- Total messages received
- Unique CAN IDs detected
- Message rate (msg/s)
- Interface status

### Test 3: Run IDS with Statistics

Run CAN-IDS and observe real-time statistics:

```bash
python main.py -i can0 --config config/can_ids_rpi4.yaml
```

When you stop with `Ctrl+C`, it displays:
- Total runtime
- Messages processed
- Messages per second
- Alerts generated
- Dropped frames (if any)

---

## Comprehensive Performance Testing

### Setting Up Test Environment

#### Option A: Using Virtual CAN for Controlled Testing

Create a virtual CAN interface for repeatable tests:

```bash
# Setup virtual CAN
sudo modprobe vcan
sudo ip link add dev vcan0 type vcan
sudo ip link set up vcan0

# Generate test traffic (in separate terminal)
while true; do
    cansend vcan0 123#DEADBEEF
    cansend vcan0 456#12345678
    cansend vcan0 789#CAFEBABE
    sleep 0.01  # 100 msg/s per ID = 300 msg/s total
done
```

#### Option B: Using Real CAN Bus

Connect to your actual CAN bus (already configured):
```bash
# Verify can0 is up
ip link show can0

# Monitor raw traffic first
candump can0 -n 100
```

---

## Detailed Benchmarking

### Generate Test Dataset

First, create synthetic test data:

```bash
source venv/bin/activate

# Generate 10,000 normal CAN messages
python scripts/generate_dataset.py \
    --type normal \
    --count 10000 \
    --output data/synthetic/normal_traffic.json

# Generate attack scenarios
python scripts/generate_dataset.py \
    --type attack \
    --count 2000 \
    --output data/synthetic/attack_traffic.json
```

### Run Benchmark Tests

#### Full System Benchmark

Test all components (Rule Engine, ML Detector, End-to-End):

```bash
python scripts/benchmark.py \
    --data data/synthetic/normal_traffic.json \
    --component all \
    --output benchmark_results.json
```

**Output Includes:**
```
Benchmarking Rule Engine...
  Messages: 10000
  Results:
    Throughput:  4523.45 msg/s
    Mean Latency: 0.221 ms
    P95 Latency:  0.385 ms
    P99 Latency:  0.512 ms
    CPU Usage:    45.2%
    Memory:       85.3 MB

Benchmarking ML Detector...
  Messages: 10000
  Results:
    Throughput:  3876.12 msg/s
    Mean Latency: 0.258 ms
    ...

Benchmarking End-to-End Pipeline...
  Messages: 10000
  Results:
    Throughput:  2543.89 msg/s
    Mean Latency: 0.393 ms
    P95 Latency:  0.687 ms
    P99 Latency:  1.234 ms
    CPU Usage:    58.7% (avg)
    Memory:       156.4 MB (avg)
```

#### Benchmark Individual Components

**Rule Engine Only:**
```bash
python scripts/benchmark.py \
    --component rule-engine \
    --messages 5000
```

**ML Detector Only:**
```bash
python scripts/benchmark.py \
    --component ml-detector \
    --messages 5000
```

---

## Live Performance Monitoring

### Monitor System Resources During Operation

#### Terminal 1: Run CAN-IDS

```bash
source venv/bin/activate
python main.py -i can0 --config config/can_ids_rpi4.yaml
```

#### Terminal 2: Monitor System Resources

```bash
# Watch CPU and memory in real-time
watch -n 1 'ps aux | grep "python main.py" | grep -v grep'
```

Or use htop for detailed view:
```bash
htop -p $(pgrep -f "python main.py")
```

#### Terminal 3: Monitor Temperature (Raspberry Pi)

```bash
# Check temperature every 5 seconds
watch -n 5 'vcgencmd measure_temp && vcgencmd get_throttled'
```

**Temperature Guidelines:**
- **Normal:** 40-60°C
- **Warning:** 60-70°C
- **Throttling:** 70°C+ (system will reduce performance)

#### Terminal 4: Monitor CAN Interface

```bash
# Watch CAN interface statistics
watch -n 2 'ip -s link show can0'
```

**Key Metrics:**
- RX packets: Messages received
- RX dropped: Frames dropped by kernel
- RX errors: Receive errors

---

## Key Performance Metrics

### 1. **Throughput (Messages/Second)**

Maximum sustained message processing rate.

**Target Values:**
- Rule Engine: 5,000+ msg/s
- ML Detector: 3,000+ msg/s
- End-to-End: 2,500+ msg/s

**Pi4 Expected:** 2,000-3,000 msg/s sustained

**How to Measure:**
```bash
# Built into benchmark.py
python scripts/benchmark.py --component end-to-end
```

### 2. **Detection Latency (ms)**

Time from message arrival to detection result.

**Target Values:**
- Mean: <0.5 ms
- P95: <1.0 ms
- P99: <2.0 ms

**Critical:** Latency must be lower than CAN bus message interval to avoid backlog.

### 3. **CPU Utilization (%)**

Processor usage during operation.

**Pi4 Configuration Limit:** 70% (thermal management)

**Monitor:**
```bash
# During operation
top -p $(pgrep -f "python main.py")

# Or detailed stats
pidstat -p $(pgrep -f "python main.py") 1
```

**Optimization:**
- If consistently >70%: Increase detection thresholds, reduce rule complexity
- If <30%: Consider enabling more detection features

### 4. **Memory Usage (MB)**

RAM consumption during operation.

**Pi4 Configuration Limit:** 512 MB (systemd service)
**Target Range:** 150-300 MB

**Monitor:**
```bash
# During operation
ps aux | grep "python main.py" | grep -v grep | awk '{print $6/1024 " MB"}'

# Or systemd service
systemctl status can-ids.service | grep Memory
```

**Warning Signs:**
- Memory growing continuously = memory leak
- Sudden spikes = inefficient processing
- Hitting limit = potential dropped messages

### 5. **Dropped CAN Frames**

Messages lost due to buffer overflow or processing backlog.

**Target:** 0 dropped frames
**Acceptable:** <0.1% drop rate under peak load

**Check Built-in Statistics:**
```python
# Add to main.py or check during operation
sniffer.get_statistics()
# Returns: {'messages_dropped': X, ...}
```

**Check Kernel-Level Drops:**
```bash
ip -s link show can0 | grep -A 2 RX
```

**If Dropping Frames:**
1. Reduce detection rule complexity
2. Increase buffer size in config
3. Disable ML detection temporarily
4. Check for thermal throttling

### 6. **Alert Generation Rate**

Number of alerts generated per message processed.

**Normal Traffic:** 0-2% alert rate
**Attack Traffic:** 5-50% alert rate (depends on attack type)

**Monitor:**
```bash
# Check alert log
tail -f logs/alerts.json | grep -c "severity"
```

### 7. **Buffer Utilization**

Internal message queue usage.

**Target:** <80% buffer usage
**Danger:** >90% indicates approaching capacity

**Check:**
```python
# In CANSniffer
info = sniffer.get_can_info()
print(f"Buffer: {info['buffer_usage']}/{info['buffer_size']}")
```

---

## Performance Testing Scenarios

### Scenario 1: Normal Traffic Load Test

Simulate typical vehicle operation.

**Setup:**
```bash
# Generate realistic traffic (100-500 msg/s)
while true; do
    for id in 100 180 200 320 400 500; do
        data=$(printf "%02X%02X%02X%02X%02X%02X%02X%02X" \
               $RANDOM $RANDOM $RANDOM $RANDOM \
               $RANDOM $RANDOM $RANDOM $RANDOM)
        cansend can0 ${id}#${data:0:16}
    done
    sleep 0.01
done &
```

**Run IDS for 5 minutes and monitor:**
```bash
timeout 300 python main.py -i can0 --config config/can_ids_rpi4.yaml
```

**Success Criteria:**
- 0 dropped frames
- CPU <70%
- Memory stable
- Temperature <70°C

### Scenario 2: High Load Stress Test

Test maximum capacity.

**Setup:**
```bash
# Generate high-speed traffic (1000+ msg/s)
cangen can0 -g 1 -I 100 -L 8 -D r &
```

**Monitor during test:**
```bash
python scripts/can_traffic_test.py --interface can0 --monitor --duration 60
```

**Success Criteria:**
- Processing rate >1000 msg/s
- Drop rate <1%
- No system crashes
- Temperature <75°C

### Scenario 3: Attack Detection Performance

Test detection accuracy under attack scenarios.

**Setup:**
```bash
# DoS attack simulation
cangen can0 -g 0.1 -I 123 -L 8 -n 10000 &
```

**Run IDS:**
```bash
python main.py -i can0 --config config/can_ids_rpi4.yaml
```

**Success Criteria:**
- DoS attack detected within 1 second
- Alert generated with correct severity
- System remains stable during attack
- No false negatives

### Scenario 4: Long-Duration Stability Test

Test for memory leaks and long-term stability.

**Run IDS for 24 hours:**
```bash
# Start in background
nohup python main.py -i can0 --config config/can_ids_rpi4.yaml > ids_longrun.log 2>&1 &

# Monitor memory every hour
while true; do
    echo "$(date): $(ps aux | grep 'python main.py' | grep -v grep | awk '{print $6/1024 " MB"}')" >> memory_log.txt
    sleep 3600
done
```

**Success Criteria:**
- Memory usage stable (not growing)
- No crashes or restarts
- Alert log continues functioning
- System responsive after 24h

---

## Optimization Based on Test Results

### If CPU Usage is Too High (>70%)

1. **Reduce Rule Complexity:**
   ```yaml
   # config/rules.yaml
   # Disable less critical rules or increase thresholds
   ```

2. **Disable ML Detection Temporarily:**
   ```yaml
   # config/can_ids_rpi4.yaml
   detection_modes:
     - rule_based
     # - ml_based  # Commented out
   ```

3. **Increase Processing Interval:**
   ```yaml
   # config/can_ids_rpi4.yaml
   performance:
     batch_size: 50  # Process in larger batches
   ```

### If Dropping Frames

1. **Increase Buffer Size:**
   ```yaml
   # config/can_ids_rpi4.yaml
   capture:
     buffer_size: 1000  # Increase from 500
   ```

2. **Reduce Alert Verbosity:**
   ```yaml
   alerts:
     rate_limit: 20  # Increase rate limit
     console_output: false  # Disable console
   ```

3. **Optimize Logging:**
   ```yaml
   logging:
     level: WARNING  # Reduce logging
   ```

### If Memory Usage is Growing

1. **Limit History Tracking:**
   ```python
   # Check ml_detector.py feature_window setting
   feature_window: 50  # Reduce from 100
   ```

2. **Enable Garbage Collection:**
   ```python
   # Add to main loop
   import gc
   if message_count % 1000 == 0:
       gc.collect()
   ```

### If Temperature is Too High (>70°C)

1. **Verify Thermal Settings:**
   ```yaml
   raspberry_pi:
     thermal_throttling_temp: 70
   ```

2. **Reduce CPU Quota:**
   ```bash
   # Edit systemd service
   sudo nano /etc/systemd/system/can-ids.service
   # Change: CPUQuota=70% → CPUQuota=50%
   sudo systemctl daemon-reload
   ```

3. **Physical Cooling:**
   - Add heatsinks
   - Install active cooling fan
   - Improve airflow

---

## Automated Testing Script

Create a comprehensive test suite:

```bash
#!/bin/bash
# test_canids_performance.sh

echo "CAN-IDS Performance Test Suite"
echo "================================"

source venv/bin/activate

# Test 1: Interface Connectivity
echo "Test 1: Interface Connectivity"
python scripts/can_traffic_test.py --interface can0 --test-connectivity
if [ $? -eq 0 ]; then
    echo "✓ PASS"
else
    echo "✗ FAIL"
    exit 1
fi

# Test 2: Short Traffic Monitor
echo -e "\nTest 2: Traffic Monitor (10s)"
timeout 10 python scripts/can_traffic_test.py --interface can0 --monitor --duration 10
echo "✓ PASS"

# Test 3: Benchmark (if test data exists)
if [ -f "data/synthetic/normal_traffic.json" ]; then
    echo -e "\nTest 3: Performance Benchmark"
    python scripts/benchmark.py --data data/synthetic/normal_traffic.json --messages 1000
    echo "✓ PASS"
else
    echo -e "\nTest 3: SKIPPED (no test data)"
fi

# Test 4: Live IDS Test
echo -e "\nTest 4: Live IDS Test (30s)"
timeout 30 python main.py -i can0 --config config/can_ids_rpi4.yaml
if [ $? -eq 124 ]; then  # timeout exit code
    echo "✓ PASS (ran for 30s)"
else
    echo "✗ FAIL"
    exit 1
fi

# Test 5: System Resources Check
echo -e "\nTest 5: System Resources"
echo "Temperature: $(vcgencmd measure_temp)"
echo "Throttling: $(vcgencmd get_throttled)"
echo "Memory: $(free -h | grep Mem:)"
echo "✓ PASS"

echo -e "\n================================"
echo "All tests completed!"
```

Make it executable and run:
```bash
chmod +x test_canids_performance.sh
./test_canids_performance.sh
```

---

## Performance Baseline (Raspberry Pi 4)

Expected performance metrics on Pi4 with MCP2515 CAN HAT:

| Metric | Target | Good | Acceptable | Poor |
|--------|--------|------|------------|------|
| **Throughput** | 3000+ msg/s | 2000-3000 | 1000-2000 | <1000 |
| **Mean Latency** | <0.5 ms | 0.5-1.0 ms | 1.0-2.0 ms | >2.0 ms |
| **P99 Latency** | <2.0 ms | 2.0-5.0 ms | 5.0-10 ms | >10 ms |
| **CPU Usage** | 30-50% | 50-70% | 70-80% | >80% |
| **Memory** | 150-250 MB | 250-350 MB | 350-450 MB | >450 MB |
| **Drop Rate** | 0% | 0-0.1% | 0.1-1% | >1% |
| **Temperature** | 50-60°C | 60-70°C | 70-75°C | >75°C |

---

## Troubleshooting Performance Issues

### Issue: Low Throughput (<1000 msg/s)

**Diagnostics:**
```bash
# Check CPU throttling
vcgencmd get_throttled
# 0x0 = normal, any other value = throttling occurred

# Check actual CAN bus speed
ip -d link show can0 | grep bitrate
```

**Solutions:**
- Verify CAN bitrate matches network
- Check for thermal throttling
- Review systemd service limits

### Issue: High Latency (>5ms)

**Diagnostics:**
```bash
# Check system load
uptime

# Check for competing processes
top -b -n 1 | head -20
```

**Solutions:**
- Stop unnecessary services
- Increase process priority: `nice -n -10 python main.py ...`
- Disable swap: `sudo swapoff -a`

### Issue: Dropped Frames

**Check Kernel Buffers:**
```bash
# Increase kernel receive buffer
sudo ip link set can0 txqueuelen 1000
```

**Check Application Buffer:**
```yaml
# config/can_ids_rpi4.yaml
capture:
  buffer_size: 2000  # Increase
```

---

## Collecting Performance Data for Analysis

### Generate Full Performance Report

```bash
#!/bin/bash
# generate_performance_report.sh

REPORT_DIR="performance_reports/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$REPORT_DIR"

echo "Generating performance report in $REPORT_DIR"

# System information
echo "=== System Info ===" > "$REPORT_DIR/system_info.txt"
uname -a >> "$REPORT_DIR/system_info.txt"
vcgencmd version >> "$REPORT_DIR/system_info.txt"
free -h >> "$REPORT_DIR/system_info.txt"

# CAN interface info
echo "=== CAN Interface ===" > "$REPORT_DIR/can_info.txt"
ip -d link show can0 >> "$REPORT_DIR/can_info.txt"

# Run benchmark
python scripts/benchmark.py \
    --component all \
    --output "$REPORT_DIR/benchmark.json"

# Run live test with monitoring
timeout 60 python main.py -i can0 --config config/can_ids_rpi4.yaml \
    > "$REPORT_DIR/live_test.log" 2>&1 &

IDS_PID=$!
sleep 5  # Let IDS start

# Monitor resources
for i in {1..55}; do
    echo "$(date +%s),$(ps -p $IDS_PID -o %cpu,%mem | tail -1)" >> "$REPORT_DIR/resources.csv"
    vcgencmd measure_temp >> "$REPORT_DIR/temperature.log"
    sleep 1
done

wait $IDS_PID

echo "Report generated in $REPORT_DIR"
```

---

## Summary

### Quick Testing Checklist

- [ ] Interface connectivity verified
- [ ] Live traffic monitoring successful
- [ ] Benchmark tests completed
- [ ] CPU usage within limits (<70%)
- [ ] Memory usage stable (<512MB)
- [ ] No dropped frames under normal load
- [ ] Temperature acceptable (<70°C)
- [ ] Alerts generating correctly
- [ ] Long-term stability tested (>1 hour)

### Key Metrics to Monitor in Production

1. **Messages per second** - Ensure keeping up with bus
2. **Dropped frames** - Should be zero
3. **CPU usage** - Keep under 70% for thermal management
4. **Memory usage** - Monitor for leaks
5. **Temperature** - Stay under 70°C
6. **Alert rate** - Validate detection working

### Performance Optimization Priority

1. **First:** Ensure no dropped frames
2. **Second:** Keep CPU <70% for thermal stability
3. **Third:** Minimize latency for real-time response
4. **Fourth:** Optimize memory for long-term operation

---

**Document Version:** 1.0  
**Last Updated:** November 25, 2025  
**Tested On:** Raspberry Pi 4 Model B, MCP2515 CAN HAT, Python 3.11.2
