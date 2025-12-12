# CAN-IDS: Controller Area Network Intrusion Detection System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-4-red.svg)](https://www.raspberrypi.org/)

A real-time intrusion detection system for Controller Area Network (CAN) bus, optimized for Raspberry Pi 4 deployment. Combines signature-based detection with machine learning anomaly detection to identify and alert on suspicious or malicious traffic in automotive and industrial control systems.

## Overview

CAN-IDS is a Python-based network security monitoring tool designed specifically for CAN bus networks. Inspired by traditional network IDS solutions like Snort, CAN-IDS adapts intrusion detection principles to the unique characteristics of automotive and industrial CAN bus communications.

The system employs a dual-detection approach:
- **Rule-based detection** for known attack patterns and policy violations
- **Machine learning-based anomaly detection** for identifying novel or zero-day attacks

**Optimized for Raspberry Pi 4**: This project is specifically designed to run efficiently on Raspberry Pi 4 hardware, making it ideal for embedded automotive security applications, research projects, and educational purposes.

## Problem Statement

Modern vehicles and industrial systems rely heavily on CAN bus for critical communications between electronic control units (ECUs). However, CAN bus was designed without security in mind, making it vulnerable to various attacks:

- **Message injection attacks** - Unauthorized messages controlling critical systems (brakes, steering, engine)
- **Denial of Service (DoS)** - Flooding the bus with high-priority messages to block legitimate traffic
- **Replay attacks** - Retransmitting captured legitimate messages at inappropriate times
- **Fuzzing attacks** - Sending malformed or unexpected data to trigger vulnerabilities
- **ECU impersonation** - Spoofing messages from legitimate controllers
- **Data manipulation** - Altering message contents in transit

CAN-IDS addresses these threats by monitoring CAN bus traffic in real-time and detecting anomalous patterns that may indicate malicious activity or system compromise.

## Key Features

### Core Capabilities
- **Real-time traffic monitoring** on live CAN interfaces (socketcan)
- **PCAP file analysis** for offline investigation and forensic analysis
- **Signature-based detection** with customizable YAML rule definitions
- **ML-based anomaly detection** using Isolation Forest algorithm
- **Multi-level alerting** with configurable severity thresholds (CRITICAL, HIGH, MEDIUM, LOW)
- **Comprehensive logging** of all alerts and suspicious activities
- **Raspberry Pi 4 optimized** with reduced memory footprint and ARM-specific optimizations

### Detection Techniques
- **Frequency analysis** - Identify abnormal message transmission rates
- **Timing analysis** - Detect irregular message intervals and timing violations
- **Data pattern matching** - Recognize malicious payload signatures
- **Statistical modeling** - Baseline normal behavior and flag deviations
- **Entropy analysis** - Identify encrypted or randomized data patterns
- **ID monitoring** - Track unauthorized or unexpected CAN identifiers

### Architecture Benefits
- **Modular design** for easy extension and customization
- **Low latency** suitable for real-time automotive applications
- **Scalable** to handle high-traffic CAN networks
- **Portable** across Linux systems with socketcan support
- **Training framework** for adapting to specific vehicle/system profiles
- **Resource-efficient** for embedded deployment on Raspberry Pi 4

## System Requirements

### Platform Support

**CAN-IDS runs on multiple platforms:**

| Platform | Support Level | Notes |
|----------|---------------|-------|
| **Linux PC/Server** | ✅ Full Support | Native SocketCAN, best performance |
| **Raspberry Pi 4** | ✅ Optimized | Special configs for embedded deployment |
| **Windows PC** | ⚠️ Partial | Via WSL2 or USB CAN adapters |
| **macOS** | ⚠️ Partial | USB CAN adapters, PCAP analysis |

### General Requirements
- **Operating System**: Linux (Ubuntu, Debian, Fedora, Raspberry Pi OS) recommended
  - Windows: Requires WSL2 for SocketCAN or USB CAN adapter
  - macOS: Requires USB CAN adapter
- **Python**: 3.8 or higher
- **CAN Interface**: 
  - Linux: SocketCAN-compatible interface
  - Windows/Mac: PCAN-USB, CANtact, SLCAN, or other python-can supported devices
  - All platforms: PCAP file analysis (no hardware needed)
- **Storage**: 100MB+ available for logs and models
- **Memory**: 2GB+ RAM (512MB minimum for basic operation)

### Raspberry Pi 4 Specific
- **Hardware**: Raspberry Pi 4 Model B (4GB or 8GB RAM recommended, 2GB minimum)
- **Storage**: 32GB+ microSD card (Class 10 or UHS-I for best performance)
- **OS**: Raspberry Pi OS Lite (64-bit) or Ubuntu Server 20.04+
- **CAN Interface**: One of the following:
  - MCP2515-based CAN HAT (connects via SPI) - e.g., Waveshare RS485/CAN HAT
  - USB-to-CAN adapter (PCAN-USB, CANtact, SLCAN devices)
  - PiCAN 2/3 or compatible CAN-BUS shields
- **Cooling**: Heatsinks or fan recommended for continuous operation
- **Power**: Official Raspberry Pi 4 USB-C power supply (5V/3A minimum)

### Optional Components
- Real-Time Clock (RTC) module for accurate timestamps without network (Pi only)
- Ethernet connection for stable remote monitoring
- UPS/battery backup for uninterrupted operation
- Protective enclosure for vehicle/industrial deployment

## Platform-Specific Notes

### Linux PC
- **Best performance** - full SocketCAN support
- Ideal for development, high-throughput analysis, and ML training
- All features fully supported
- Recommended for production deployments handling >500k messages/sec

### Windows PC
- **Live monitoring**: Requires WSL2 (for SocketCAN) or USB CAN adapter
- **PCAP analysis**: Fully supported without additional setup
- **Development**: Excellent for rule development and offline analysis
- **USB adapters supported**: PCAN-USB, CANtact, IXXAT, Vector, SLCAN

### macOS
- Similar to Windows - requires USB CAN adapter for live monitoring
- PCAP analysis and development fully supported
- Best for PCAP forensics and rule testing

### Raspberry Pi 4
- Optimized configuration included (`can_ids_rpi4.yaml`)
- Perfect for embedded, in-vehicle deployment
- Lower power consumption, compact form factor
- Automotive-grade reliability when properly configured

## Installation

### Quick Start (Linux PC or Raspberry Pi 4)

### Quick Start (Linux PC or Raspberry Pi 4)

```bash
# 1. Update system
sudo apt update && sudo apt upgrade -y

# 2. Install dependencies
sudo apt install python3 python3-pip python3-venv can-utils git -y

# 3. For Raspberry Pi with MCP2515: Enable SPI
sudo raspi-config
# Navigate to: Interface Options → SPI → Enable
# (Skip this step on regular Linux PC)

# 4. For Raspberry Pi with MCP2515: Configure device tree overlay
sudo nano /boot/config.txt
# Add these lines:
# dtparam=spi=on
# dtoverlay=mcp2515-can0,oscillator=12000000,interrupt=25
# dtoverlay=spi-bcm2835
# (Skip this step on regular Linux PC with other CAN hardware)

# 5. Reboot (Raspberry Pi only, if configured SPI)
sudo reboot

# 6. Clone repository
git clone https://github.com/yourusername/can-ids.git
cd can-ids

# 7. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 8. Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 9. Install CAN-IDS
pip install -e .

# 10. Configure CAN interface (500 kbps for automotive)
sudo ip link set can0 type can bitrate 500000
sudo ip link set up can0

# 11. Test CAN interface
candump can0  # Should show CAN traffic if connected

# 12. Run CAN-IDS
python main.py -i can0
```

## Usage

### Basic Usage

```bash
# Monitor live CAN traffic
python main.py -i can0

# Analyze PCAP file
python main.py --mode replay --file suspicious_traffic.pcap

# Use specific configuration
python main.py -i can0 --config config/can_ids_rpi4.yaml

# Enable debug logging
python main.py -i can0 --log-level DEBUG
```

## Configuration

See the `config/` directory for configuration examples:
- `can_ids.yaml` - Main configuration file
- `can_ids_rpi4.yaml` - Raspberry Pi 4 optimized configuration
- `rules.yaml` - Detection rule definitions
- `example_rules.yaml` - Sample rules for common attacks

## Project Structure

```
can-ids/
├── README.md                 # This file
├── LICENSE                   # MIT License
├── requirements.txt          # Python dependencies
├── requirements-dev.txt      # Development dependencies
├── setup.py                  # Package installation script
├── .gitignore               # Git ignore rules
│
├── config/                   # Configuration files
│   ├── can_ids.yaml         # Main system configuration
│   ├── can_ids_rpi4.yaml    # Raspberry Pi 4 optimized config
│   ├── rules.yaml           # Detection rule definitions
│   └── example_rules.yaml   # Sample rules for common attacks
│
├── src/                      # Source code
│   ├── __init__.py
│   ├── capture/             # Traffic capture modules
│   ├── detection/           # Detection engines
│   ├── preprocessing/       # Feature engineering
│   ├── models/              # ML model management
│   └── alerts/              # Alert handling
│
├── raspberry-pi/            # Raspberry Pi specific files
│   ├── systemd/            # Systemd service files
│   ├── boot_config/        # Boot configuration examples
│   └── scripts/            # Pi-specific utility scripts
│
├── data/                    # Data directory (git-ignored)
│   ├── raw/                # Raw CAN captures
│   ├── processed/          # Processed feature datasets
│   ├── models/             # Trained ML models
│   └── samples/            # Example datasets for testing
│
├── logs/                    # Log files (git-ignored)
├── tests/                   # Unit and integration tests
├── docs/                    # Additional documentation
├── scripts/                 # Utility scripts
└── main.py                  # Main entry point
```

## Recent Developments (December 2025)

### Dual-Sigma Adaptive Timing Detection (December 11, 2025)

**Major breakthrough**: Implemented separate threshold controls for extreme vs. sustained timing violations, achieving 94.81% recall with 70% reduction in false positives.

**Problem Solved**: Previous single-sigma approach couldn't simultaneously:
- Avoid false positives on normal timing variation (requires loose thresholds)
- Catch subtle interval manipulation attacks (requires tight thresholds)

**Solution - Two-Tier Threshold System**:
- **Tier 1 (Extreme)**: σ = 2.5-3.3 - Very loose, catches obvious attacks (DoS flooding at 1ms intervals)
- **Tier 2 (Moderate)**: σ = 1.3-1.7 - Tight, catches subtle attacks (interval manipulation at 20ms vs 10.92ms baseline)
- **Per-CAN-ID adaptation**: High-traffic IDs get tighter Tier 2 (1.3-1.4σ), low-traffic get looser (1.7-1.9σ)

**Critical Bug Fixed**: Tier 2 was hardcoded to 1.0σ instead of using adaptive `sigma_extreme`, causing all threshold tuning to have zero effect. Now uses separate `sigma_moderate` parameter.

**Test Results**:
- Interval attacks: **94.81% recall**, 25.94% FPR (down from 54.66%)
- Attack-free data: **23.38% FPR** (down from 93.45%)
- **70% reduction in false positives** while maintaining >94% recall

**Technical Details**:
- Attack characteristics: 20ms interval vs 10.92ms baseline (1.56σ deviation)
- Sophisticated attack design: falls within statistical noise
- Files modified: `src/detection/rule_engine.py`, `scripts/generate_rules_from_baseline.py`
- New field: `sigma_moderate` added to `DetectionRule` dataclass

**Documentation**: See [TONIGHT_SUMMARY.md](TONIGHT_SUMMARY.md) for complete implementation details.

---

### Adaptive Timing Detection System (December 9, 2025)

Implementation of per-CAN-ID adaptive timing thresholds for improved detection accuracy:

- **Hybrid multi-tier detection** combining extreme violation detection with sustained pattern analysis
- **Per-CAN-ID threshold adaptation** based on traffic rate and timing variability
- **Zero performance overhead** (<0.002% CPU at 7,000 msg/s)
- **Dual-sigma architecture** with separate controls for extreme and moderate violations

**Documentation:**
- [Tonight's Work Summary](TONIGHT_SUMMARY.md) - Latest dual-sigma implementation (Dec 11)
- [Adaptive Timing Implementation Summary](ADAPTIVE_TIMING_IMPLEMENTATION_SUMMARY.md) - Complete implementation overview
- [Timing Detection Tuning Guide](TIMING_DETECTION_TUNING.md) - Technical deep-dive with statistical analysis
- [7K msg/s Architecture Plan](BUILD_PLAN_7000_MSG_SEC.md) - Performance optimization roadmap
- [Rule Generation Summary](RULE_GENERATION_SUMMARY.md) - Data-driven rule creation methodology

**Key Features:**
- Automatic rule generation from baseline (attack-free) CAN traffic
- Traffic-aware thresholds: tighter for high-frequency, looser for low-frequency CAN IDs
- Jitter compensation: adjusts sensitivity based on natural timing variance
- Windowed detection: tolerates occasional normal messages during sustained attacks

**Current Status:** Dual-sigma implementation complete. 94.81% recall, 23% FPR. Next: test on DoS attacks, consider consecutive_required tuning.

--- Performance Optimization for High-Throughput

Research-validated architecture for 7,000 msg/s sustained throughput:

- **3-stage hierarchical filtering** reduces ML computational load by 90%
- **Stage 1:** Adaptive timing filter (current implementation)
- **Stage 2:** Optimized rule engine with CAN ID indexing
- **Stage 3:** Lightweight ML analysis on suspicious traffic only

**References:**
- Yu et al. (2023) - TCE-IDS cross-check filter architecture
- Ming et al. (2023) - Threshold-adaptive message cycle detection
- See [BUILD_PLAN_7000_MSG_SEC.md](BUILD_PLAN_7000_MSG_SEC.md) for complete design

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

**Test rule-based detection on datasets:**
```bash
# Test adaptive timing rules on attack dataset
python3 scripts/test_rules_on_dataset.py data/interval-1.csv --rules config/rules_adaptive.yaml

# Test on attack-free baseline
python3 scripts/test_rules_on_dataset.py data/attack-free-1.csv --rules config/rules_adaptive.yaml
```

**Generate vehicle-specific rules from baseline:**
```bash
# Analyze attack-free traffic and generate optimized rules
python3 scripts/generate_rules_from_baseline.py \
  --confidence 0.683 \
  --output config/rules_custom.yaml

# Confidence levels:
# 0.997 (3-sigma): Lowest FPR, may miss subtle attacks
# 0.954 (2-sigma): Balanced approach
# 0.683 (1-sigma): Highest sensitivity, adaptive thresholds recommended
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Security Considerations

- CAN-IDS is a detection system only – cannot prevent or block attacks in real-time
- Run on dedicated hardware separate from critical ECUs
- Use read-only CAN interfaces to prevent accidental injection
- Regularly update detection rules and ML models

## Disclaimer

This tool is for research, education, and legitimate security testing purposes only. Unauthorized access to vehicle networks or industrial control systems may be illegal in your jurisdiction. Always obtain proper authorization before conducting security assessments.