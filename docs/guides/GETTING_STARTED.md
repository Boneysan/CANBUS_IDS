# Getting Started with CAN-IDS

**Last Updated**: March 1, 2026

This guide walks you through installation, first run on virtual CAN, and Raspberry Pi 4 deployment. Every command in this guide has been verified against the current codebase.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Quick Test with Virtual CAN](#quick-test-with-virtual-can)
4. [Understanding the 3-Stage Pipeline](#understanding-the-3-stage-pipeline)
5. [Training the Decision Tree (Optional)](#training-the-decision-tree-optional)
6. [Generating Vehicle-Specific Rules](#generating-vehicle-specific-rules)
7. [Analyzing Captured Traffic](#analyzing-captured-traffic)
8. [Raspberry Pi 4 Deployment](#raspberry-pi-4-deployment)
9. [Understanding Output](#understanding-output)
10. [Next Steps](#next-steps)
11. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware

- **Desktop/Laptop**: Linux with kernel 2.6.25+ (for SocketCAN) — or any OS for offline analysis
- **Raspberry Pi 4**: 2 GB+ RAM recommended
- **CAN hardware** (for real vehicles): MCP2515 HAT, USB-to-CAN adapter (PCAN, CANtact, SLCAN), or PiCAN shield

### Software

- Python 3.8+
- Git
- `can-utils` (Linux, for virtual CAN and `candump`/`cansend`)

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Boneysan/CANBUS_IDS.git
cd CANBUS_IDS
```

### 2. Create and Activate Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 4. Verify Installation

```bash
python --version          # Should be 3.8+
python main.py --version  # Should print CAN-IDS 1.0.0
python main.py --help     # Show all CLI options
```

---

## Quick Test with Virtual CAN

No CAN hardware needed — test the full pipeline on a virtual interface.

### 1. Set Up Virtual CAN

```bash
# Install can-utils
sudo apt-get install -y can-utils

# Create virtual CAN interface
sudo modprobe vcan
sudo ip link add dev vcan0 type vcan
sudo ip link set up vcan0

# Verify
ip link show vcan0
```

### 2. Generate Test Traffic (Terminal 1)

Open a terminal and send CAN messages in a loop:

```bash
# Normal traffic + a simulated attack burst
while true; do
    cansend vcan0 316#$(openssl rand -hex 8)   # Simulated engine data
    cansend vcan0 0C6#$(openssl rand -hex 8)   # Simulated steering
    sleep 0.01
done
```

### 3. Run CAN-IDS (Terminal 2)

```bash
cd CANBUS_IDS
source venv/bin/activate
python main.py -i vcan0
```

You should see:
- Component initialization messages (rule engine, pre-filter)
- Alerts when test traffic triggers rules
- Press `Ctrl+C` to stop and see runtime statistics

### 4. Try Different Log Levels

```bash
python main.py -i vcan0 --log-level DEBUG    # Verbose output
python main.py -i vcan0 --log-level WARNING  # Alerts only
```

---

## Understanding the 3-Stage Pipeline

CAN-IDS uses a hierarchical detection architecture:

```
Every CAN Message
        │
   Stage 1: Fast Pre-Filter
        │  Filters 80–95% of known-benign traffic
        │  Config: prefilter.enabled (default: true)
        │
   Stage 2: Rule Engine
        │  18 rule types (pattern, timing, frequency, etc.)
        │  Config: detection_modes: [rule_based]
        │
   Stage 3: Decision Tree ML (optional)
        │  Lightweight classifier on suspicious messages only
        │  Config: decision_tree.enabled (default: true)
        │  Requires: data/models/decision_tree.pkl (train locally)
        │
   Alert Manager → logs/alerts.json + console
```

**Out of the box**, Stages 1+2 are fully active. Stage 3 requires training the Decision Tree model (see below). Without it, Stage 3 simply doesn't run — detection still works via Stages 1+2.

---

## Training the Decision Tree (Optional)

Stage 3 adds an ML classifier that processes messages still suspicious after Stages 1+2. The model is not shipped in the repo because it should be trained on data representative of your target environment.

### Quick Start (Synthetic Data)

```bash
python scripts/train_decision_tree.py --synthetic
```

This creates:
- `data/models/decision_tree.pkl` — the trained model
- `data/models/decision_tree_rules.txt` — human-readable tree rules

### Train from Real Data

If you have the Vehicle_Models dataset or similar labeled CAN data:

```bash
python scripts/train_decision_tree.py --vehicle-models /path/to/Vehicle_Models
```

### Train from the Bundled Test Data

The `test_data/` directory includes 16 labeled CSV files (attack-free, DoS, fuzzing, etc.):

```bash
python scripts/train_decision_tree.py \
  --vehicle-models . \
  --output data/models/decision_tree.pkl
```

### Verify Stage 3 is Active

After training, restart CAN-IDS. Look for this log line:

```
✅ STAGE 3 ML DETECTION ENABLED
```

If you see `Stage 3 ML DETECTION INITIALIZATION FAILED`, the model file is missing or corrupt.

---

## Generating Vehicle-Specific Rules

The default rules use generic thresholds that work for testing but cause false positives on real vehicles. Generate tuned rules from baseline (attack-free) traffic:

### From Bundled Test Data

```bash
python scripts/generate_rules_from_baseline.py \
  --input test_data/attack-free-1.csv test_data/attack-free-2.csv \
  --output config/rules_my_vehicle.yaml
```

### From Your Own Captures

```bash
# 1. Capture normal traffic (let run for 10+ minutes on a real vehicle)
candump -l can0

# 2. Convert candump log to CSV
python scripts/convert_candump.py candump-*.log data/raw/baseline.csv

# 3. Generate tuned rules
python scripts/generate_rules_from_baseline.py \
  --input data/raw/baseline.csv \
  --output config/rules_my_vehicle.yaml

# 4. Use the tuned rules
python main.py -i can0 --config config/can_ids.yaml
# (edit config to set rules_file: config/rules_my_vehicle.yaml)
```

---

## Analyzing Captured Traffic

You don't need a live CAN bus to use CAN-IDS. Replay captured traffic:

```bash
# Replay a candump log file
python main.py --mode replay --file data/raw/traffic.log

# Replay with a specific config
python main.py --mode replay --file data/raw/traffic.log --config config/can_ids_rpi4.yaml
```

### Test Against the Bundled Datasets

The `test_data/` directory contains 16 labeled CSV files from real vehicle CAN traffic:

| Dataset | File | Messages |
|---------|------|----------|
| Attack-free | `attack-free-1.csv`, `attack-free-2.csv` | Normal driving traffic |
| DoS | `DoS-1.csv`, `DoS-2.csv` | Denial of service attack |
| Fuzzing | `fuzzing-1.csv`, `fuzzing-2.csv` | Random CAN message fuzzing |
| RPM | `rpm-1.csv`, `rpm-2.csv` | Engine RPM manipulation |
| Interval | `interval-1.csv`, `interval-2.csv` | Timing manipulation |
| Force neutral | `force-neutral-1.csv`, `force-neutral-2.csv` | Gear forcing |
| Standstill | `standstill-1.csv`, `standstill-2.csv` | Forced standstill |
| Accessory | `accessory-1.csv`, `accessory-2.csv` | Accessory manipulation |

Run the rule-testing script against all of them:

```bash
python scripts/test_rules_on_dataset.py --rules config/rules_adaptive.yaml --data test_data/DoS-1.csv
```

---

## Raspberry Pi 4 Deployment

### 1. Prepare the Pi

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3 python3-pip python3-venv can-utils git -y
```

### 2. Clone and Install

```bash
git clone https://github.com/Boneysan/CANBUS_IDS.git
cd CANBUS_IDS
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 3. Set Up CAN Hardware

For MCP2515 CAN HAT:

```bash
sudo bash raspberry-pi/scripts/setup_mcp2515.sh
sudo reboot
```

After reboot:

```bash
# Configure CAN interface (500 kbps is typical for OBD-II)
sudo bash raspberry-pi/scripts/setup_can_interface.sh can0 500000

# Verify
ip -details link show can0
candump can0   # Should show traffic if bus is connected
```

### 4. Run with Pi-Optimized Config

```bash
python main.py -i can0 --config config/can_ids_rpi4.yaml
```

The Pi config reduces buffer sizes, limits threads, and manages thermal throttling.

### 5. Install as System Service

```bash
# Copy service file
sudo cp raspberry-pi/systemd/can-ids.service /etc/systemd/system/

# Edit paths if your install directory differs
sudo nano /etc/systemd/system/can-ids.service

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable can-ids.service
sudo systemctl start can-ids.service

# Check status
sudo systemctl status can-ids.service

# Follow logs
sudo journalctl -u can-ids.service -f
```

---

## Understanding Output

### Console Output

```
[2026-03-01 10:30:45] [HIGH] [RULE_ENGINE] CAN ID: 0x7E0 | Rule: DoS Attack | Description: Excessive messages targeting engine ECU
```

### JSON Alert Log (`logs/alerts.json`)

```json
{
  "timestamp": 1740825045.123,
  "severity": "HIGH",
  "rule_name": "DoS Attack",
  "can_id": "0x7E0",
  "description": "Excessive messages targeting engine ECU",
  "confidence": 0.95,
  "source": "rule_engine"
}
```

### Shutdown Statistics

Press `Ctrl+C` for a summary:

```
==========================================
CAN-IDS STATISTICS
==========================================
Runtime: 60.0 seconds
Messages processed: 45600
Alerts generated: 12
Processing rate: 760.00 messages/second
==========================================
```

---

## Next Steps

1. **Write custom rules** — See [rules_guide.md](rules_guide.md) for all 18 rule types with YAML examples
2. **Tune for your vehicle** — Generate rules from baseline traffic (`scripts/generate_rules_from_baseline.py`)
3. **Train Stage 3 ML** — `python scripts/train_decision_tree.py --synthetic`
4. **Review configuration** — See [configuration.md](configuration.md) for every parameter
5. **Benchmark performance** — `python scripts/benchmark.py`
6. **Deploy to production** — Install as a systemd service on Raspberry Pi 4

---

## Troubleshooting

### CAN Interface Not Found

```bash
# Check interfaces
ip link show

# Load kernel modules
sudo modprobe can
sudo modprobe can_raw
sudo modprobe vcan       # For virtual CAN

# For MCP2515 on Pi
sudo modprobe spi-bcm2835
```

### Permission Denied on CAN Interface

```bash
# Add user to can group (re-login required)
sudo usermod -a -G dialout $USER

# Or use the setup script
sudo python scripts/setup_vcan.py
```

### No Traffic Detected

```bash
# Is the interface up?
ip link show can0

# Is there traffic?
candump can0

# Wrong bitrate? Try 250000 or 500000
sudo ip link set can0 type can bitrate 500000
sudo ip link set up can0
```

### Stage 3 ML Not Activating

```bash
# Check if model exists
ls -lh data/models/decision_tree.pkl

# If missing, train it
python scripts/train_decision_tree.py --synthetic

# Check logs for error
python main.py -i vcan0 --log-level DEBUG 2>&1 | grep -i "stage 3"
```

### Too Many False Positives

1. Generate vehicle-specific rules from baseline data (see [above](#generating-vehicle-specific-rules))
2. Switch to adaptive rules: edit config to set `rules_file: config/rules_adaptive.yaml`
3. Increase thresholds in your rules file
4. See [rules_guide.md](rules_guide.md) > Troubleshooting for detailed guidance

### Import Errors

```bash
# Make sure you're in the venv
source venv/bin/activate

# Reinstall in dev mode
pip install -e .

# Check scikit-learn (needed for ML)
pip install scikit-learn numpy
```

---

## Safety Notice

- This is a **detection-only** system — it does not block or modify CAN traffic
- Always test on non-critical systems or virtual CAN first
- Use read-only CAN interfaces when possible on production vehicles
- Unauthorized access to vehicle networks may be illegal in your jurisdiction