# CAN Traffic Monitoring and Connectivity Testing Guide

**Document Version:** 1.0  
**Date:** October 28, 2025  
**Purpose:** Testing CAN-IDS connectivity and monitoring capabilities  

---

## Overview

Your CAN-IDS system now includes comprehensive traffic monitoring and connectivity testing functions that allow you to verify if the system can "see" CAN traffic, similar to being plugged into a real CAN bus network.

## Available Testing Functions

### 1. **Built-in Main Application Tests**

#### **Interface Connectivity Test**
```bash
# Test if CAN interface is accessible
python main.py --test-interface can0

# Test virtual CAN interface
python main.py --test-interface vcan0
```

**What it tests:**
- ✅ Interface exists in system
- ✅ python-can can open the interface
- ✅ Traffic detection (5-second timeout)
- ✅ Basic message reception

**Example output:**
```
CAN Interface Connectivity Test
==================================================
Testing CAN interface connectivity: can0
✓ Interface can0 exists in system
✓ Successfully opened can0 with python-can
  Testing message reception (5 second timeout)...
✓ Received CAN message: ID=0x123, DLC=8
✓ Interface test successful
```

#### **Traffic Monitoring Mode**
```bash
# Monitor traffic for 30 seconds (default)
python main.py --monitor-traffic can0

# Monitor for specific duration
python main.py --monitor-traffic can0 --duration 60

# Monitor with debug logging
python main.py --monitor-traffic vcan0 --duration 10 --log-level DEBUG
```

**What it shows:**
- 📊 Real-time message count
- 📊 Unique CAN IDs detected
- 📊 Message rate (messages/second)
- 📊 Traffic activity timeline

**Example output:**
```
CAN Traffic Monitoring
==================================================
Monitoring can0 for 30 seconds...
Messages: 50, Rate: 25.0 msg/s, IDs: 12
Messages: 100, Rate: 33.3 msg/s, IDs: 18
...
Monitoring complete: 847 messages, 23 unique IDs, 28.2 msg/s

Monitoring Results:
Messages received: 847
Unique CAN IDs: 23
Average rate: 28.2 msg/s
```

### 2. **Dedicated Testing Script** (`scripts/can_traffic_test.py`)

#### **Comprehensive Connectivity Test**
```bash
# Full interface connectivity test
python scripts/can_traffic_test.py --test-connectivity -i can0

# Test with statistics saving
python scripts/can_traffic_test.py --test-connectivity -i vcan0 --save-results test_results.json
```

#### **Live Traffic Monitoring**
```bash
# Monitor with message display
python scripts/can_traffic_test.py --monitor -i can0 --duration 30

# Monitor quietly (no individual messages)
python scripts/can_traffic_test.py --monitor -i can0 --duration 60 --quiet
```

#### **CAN-IDS Detection Testing**
```bash
# Test detection engines with live traffic
python scripts/can_traffic_test.py --test-canids -i can0 --duration 15

# Test detection with generated traffic
python scripts/can_traffic_test.py --generate-traffic --test-canids -i vcan0 --duration 10
```

#### **Test Traffic Generation**
```bash
# Generate 100 test messages
python scripts/can_traffic_test.py --generate-traffic -i vcan0 --count 100

# Generate traffic and monitor simultaneously
python scripts/can_traffic_test.py --generate-traffic --monitor -i vcan0 --duration 10
```

### 3. **Virtual CAN Setup** (`scripts/setup_vcan.py`)

For testing without real CAN hardware:

```bash
# Setup virtual CAN interface
sudo python scripts/setup_vcan.py

# Setup with custom interface name
sudo python scripts/setup_vcan.py --interface vcan1

# Setup persistent interface (survives reboot)
sudo python scripts/setup_vcan.py --persistent

# Remove virtual interface
sudo python scripts/setup_vcan.py --remove
```

---

## Complete Testing Workflow

### **Step 1: Setup Test Environment**

#### **Option A: Virtual CAN (No Hardware)**
```bash
# Create virtual CAN interface
sudo python scripts/setup_vcan.py --interface vcan0

# Verify interface is up
ip link show vcan0
```

#### **Option B: Real CAN Interface**
```bash
# Check if interface exists
ip link show can0

# Bring up interface (if needed)
sudo ip link set can0 up type can bitrate 500000
```

### **Step 2: Basic Connectivity Test**
```bash
# Test interface accessibility
python main.py --test-interface can0

# Expected results:
# ✓ Interface accessible
# ✓ Traffic detected (if present)
# ✓ No errors
```

### **Step 3: Generate Test Traffic** (Virtual CAN)
```bash
# Generate test messages
python scripts/can_traffic_test.py --generate-traffic -i vcan0 --count 200

# Or use manual cansend
cansend vcan0 123#DEADBEEF
cansend vcan0 456#12345678
cansend vcan0 789#CAFEBABE
```

### **Step 4: Monitor Traffic**
```bash
# Monitor for 30 seconds
python main.py --monitor-traffic vcan0 --duration 30

# Expected results:
# Messages received: 200+
# Unique CAN IDs: 3-10
# Average rate: Variable based on test
```

### **Step 5: Test Detection Capabilities**
```bash
# Test CAN-IDS detection engines
python scripts/can_traffic_test.py --test-canids -i vcan0 --duration 15

# Expected results:
# Messages processed: 100+
# Detection engines: rule_engine, ml_detector
# Alerts generated: Depends on rules/data
```

### **Step 6: Full Integration Test**
```bash
# Run normal CAN-IDS with live traffic
python main.py -i vcan0

# Should show:
# ✓ Interface connectivity test
# ✓ Real-time message processing
# ✓ Alert generation (if attacks detected)
# ✓ Statistics display
```

---

## Troubleshooting Guide

### **Common Issues and Solutions**

#### **1. "Interface not found"**
```
Error: Interface can0 not found in system
```

**Solutions:**
```bash
# Check available interfaces
ip link show

# For virtual CAN
sudo python scripts/setup_vcan.py

# For real CAN hardware
sudo modprobe can
sudo modprobe can_raw
sudo ip link set can0 up type can bitrate 500000
```

#### **2. "Permission denied"**
```
Error: Cannot open can0: [Errno 1] Operation not permitted
```

**Solutions:**
```bash
# Add user to dialout group
sudo usermod -a -G dialout $USER

# Or run with sudo (not recommended for production)
sudo python main.py --test-interface can0

# Check interface permissions
ls -la /sys/class/net/can0/
```

#### **3. "No traffic detected"**
```
Warning: No traffic detected (interface may be up but no messages)
```

**Solutions:**
```bash
# Generate test traffic
cansend can0 123#DEADBEEF

# Use traffic generator
python scripts/can_traffic_test.py --generate-traffic -i can0

# Check if other applications are using interface
lsof | grep can0
```

#### **4. "python-can not available"**
```
Warning: python-can not available. Some features will be limited.
```

**Solutions:**
```bash
# Install python-can
pip install python-can

# Or install from requirements
pip install -r requirements.txt

# Verify installation
python -c "import can; print('python-can available')"
```

#### **5. "CAN-IDS modules not available"**
```
Warning: CAN-IDS modules not available
```

**Solutions:**
```bash
# Ensure running from project root
cd /home/mike/Documents/GitHub/CANBUS_IDS

# Check Python path
python -c "import sys; print(sys.path)"

# Install in development mode
pip install -e .
```

---

## Performance Benchmarking

### **Expected Performance Metrics**

#### **Raspberry Pi 4 (Target Platform)**
```
Interface Test: < 2 seconds
Traffic Detection: < 5 seconds
Message Processing: 5,000-10,000 msg/s
Memory Usage: < 300MB
CPU Usage: < 70%
```

#### **Development Machine (x86_64)**
```
Interface Test: < 1 second
Traffic Detection: < 2 seconds
Message Processing: 20,000+ msg/s
Memory Usage: < 500MB
CPU Usage: < 50%
```

### **Benchmark Commands**
```bash
# Test processing performance
python scripts/can_traffic_test.py --generate-traffic --test-canids -i vcan0 --count 1000 --duration 30 --save-results benchmark.json

# Monitor resource usage
top -p $(pgrep -f "main.py")
htop
iostat 1

# On Raspberry Pi
vcgencmd measure_temp
vcgencmd get_throttled
```

---

## Integration with Real Networks

### **Production Deployment Checklist**

#### **1. Hardware Interface Validation**
```bash
# Test real CAN interface
python main.py --test-interface can0

# Monitor real traffic
python main.py --monitor-traffic can0 --duration 60

# Verify message rates and patterns
# Automotive: 100-5000 msg/s typical
# Industrial: 10-1000 msg/s typical
```

#### **2. Detection Engine Validation**
```bash
# Test with real traffic
python scripts/can_traffic_test.py --test-canids -i can0 --duration 300

# Check alert rates
# Normal: < 1% false positive rate expected
# Attack scenarios: > 90% detection rate expected
```

#### **3. Performance Validation**
```bash
# Extended operation test
python main.py -i can0 --duration 3600  # 1 hour

# Memory leak check
while true; do
    ps aux | grep main.py | grep -v grep >> memory_usage.log
    sleep 60
done
```

#### **4. Raspberry Pi Deployment**
```bash
# Deploy with systemd service
sudo systemctl start can-ids
sudo systemctl status can-ids

# Monitor logs
sudo journalctl -u can-ids -f

# Check thermal status
vcgencmd measure_temp
```

---

## Advanced Testing Scenarios

### **1. High-Traffic Stress Test**
```bash
# Generate high-rate traffic
for i in {1..10}; do
    python scripts/can_traffic_test.py --generate-traffic -i vcan0 --count 1000 &
done

# Monitor performance
python main.py --monitor-traffic vcan0 --duration 60
```

### **2. Attack Pattern Testing**
```bash
# Import real attack data
python scripts/import_real_dataset.py /home/mike/Downloads/cantrainandtest/can-train-and-test/

# Test with real DoS attacks
python main.py --mode replay --file data/real_dataset/set_01/train_01/DoS-1.json

# Test with real replay attacks
python main.py --mode replay --file data/real_dataset/set_01/train_01/rpm-1.json
```

### **3. Multi-Interface Testing**
```bash
# Setup multiple virtual interfaces
sudo python scripts/setup_vcan.py --interface vcan0
sudo python scripts/setup_vcan.py --interface vcan1

# Test each interface
python main.py --test-interface vcan0
python main.py --test-interface vcan1

# Run concurrent monitoring
python main.py -i vcan0 &
python main.py -i vcan1 &
```

---

## Summary

Your CAN-IDS now includes comprehensive connectivity testing that allows you to:

✅ **Verify interface accessibility** before starting detection  
✅ **Monitor live traffic** to ensure data is flowing  
✅ **Test detection engines** with real or simulated traffic  
✅ **Generate test patterns** for validation  
✅ **Benchmark performance** under various conditions  
✅ **Troubleshoot connectivity** issues systematically  

This ensures your IDS can reliably "see" CAN traffic just like being plugged into a real automotive or industrial network.