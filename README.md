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

### General Requirements
- Linux operating system with kernel 2.6.25+ (for SocketCAN support)
- Python 3.8 or higher
- SocketCAN-compatible CAN interface
- 100MB+ available storage for logs and models

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
- Real-Time Clock (RTC) module for accurate timestamps without network
- Ethernet connection for stable remote monitoring
- UPS/battery backup for uninterrupted operation
- Protective enclosure for vehicle/industrial deployment

## Installation

### Quick Start (Raspberry Pi 4)

```bash
# 1. Update system
sudo apt update && sudo apt upgrade -y

# 2. Install dependencies
sudo apt install python3 python3-pip python3-venv can-utils git -y

# 3. Enable SPI (for MCP2515 CAN HATs)
sudo raspi-config
# Navigate to: Interface Options → SPI → Enable

# 4. Configure device tree overlay for MCP2515
sudo nano /boot/config.txt
# Add these lines:
# dtparam=spi=on
# dtoverlay=mcp2515-can0,oscillator=12000000,interrupt=25
# dtoverlay=spi-bcm2835

# 5. Reboot
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
python main.py -i can0 --config config/can_ids_rpi4.conf

# Enable debug logging
python main.py -i can0 --log-level DEBUG
```

## Configuration

See the `config/` directory for configuration examples:
- `can_ids.conf` - Main configuration file
- `can_ids_rpi4.conf` - Raspberry Pi 4 optimized configuration
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
│   ├── can_ids.conf         # Main system configuration
│   ├── can_ids_rpi4.conf    # Raspberry Pi 4 optimized config
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

## Testing

Run the test suite:
```bash
pytest tests/ -v
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