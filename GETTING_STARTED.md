# Getting Started with CAN-IDS

Welcome to CAN-IDS! This guide will help you get up and running quickly.

## Prerequisites

### Hardware
- Linux system with kernel 2.6.25+ (for SocketCAN)
- OR Raspberry Pi 4 (2GB+ RAM recommended)
- CAN interface hardware:
  - MCP2515 CAN HAT for Raspberry Pi
  - USB-to-CAN adapter (PCAN, CANtact, SLCAN)
  - PiCAN shield
- CAN bus network to monitor

### Software
- Python 3.8 or higher
- Git
- Virtual environment support

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/can-ids.git
cd can-ids
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install CAN-IDS in development mode
pip install -e .
```

### 4. Verify Installation

```bash
# Check Python version
python --version  # Should be 3.8+

# Test import
python -c "from src.capture import CANSniffer; print('OK')"

# View help
python main.py --help
```

## Quick Test with Virtual CAN

Before connecting to real hardware, test with a virtual CAN interface:

### 1. Setup Virtual CAN (Linux)

```bash
# Load vcan kernel module
sudo modprobe vcan

# Create virtual CAN interface
sudo ip link add dev vcan0 type vcan
sudo ip link set up vcan0

# Verify
ip link show vcan0
```

### 2. Generate Test Traffic

In one terminal, generate CAN messages:

```bash
# Install can-utils if not already installed
sudo apt-get install can-utils

# Send random CAN messages
while true; do
    cansend vcan0 123#DEADBEEF
    sleep 0.1
done
```

### 3. Run CAN-IDS

In another terminal:

```bash
# Monitor virtual CAN interface
python main.py -i vcan0

# You should see alerts for the test messages
```

### 4. Stop Test

Press `Ctrl+C` to stop CAN-IDS. It will display statistics before exiting.

## Raspberry Pi 4 Setup

### 1. Prepare Raspberry Pi

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install python3 python3-pip python3-venv can-utils git -y
```

### 2. Setup CAN Hardware

For MCP2515 CAN HAT:

```bash
# Run setup script
sudo bash raspberry-pi/scripts/setup_mcp2515.sh

# Reboot
sudo reboot
```

After reboot, verify CAN interface:

```bash
# Check interface
ip link show can0

# Should show: can0: <NOARP,ECHO> mtu 16 qdisc noop state DOWN mode DEFAULT
```

### 3. Configure CAN Interface

```bash
# Setup CAN0 with 500kbps bitrate
sudo bash raspberry-pi/scripts/setup_can_interface.sh can0 500000

# Verify
ip -details link show can0
```

### 4. Test with Real CAN Bus

```bash
# Monitor CAN traffic
candump can0

# In another terminal, run CAN-IDS
python main.py -i can0 --config config/can_ids_rpi4.conf
```

### 5. Install as System Service

```bash
# Copy service file
sudo cp raspberry-pi/systemd/can-ids.service /etc/systemd/system/

# Edit paths if needed
sudo nano /etc/systemd/system/can-ids.service

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable can-ids.service
sudo systemctl start can-ids.service

# Check status
sudo systemctl status can-ids.service

# View logs
sudo journalctl -u can-ids.service -f
```

## Configuration

### Basic Configuration

Edit `config/can_ids.conf`:

```yaml
# CAN interface
interface: can0
bustype: socketcan

# Detection modes
detection_modes:
  - rule_based
  - ml_based  # Requires trained model

# Alert settings
alerts:
  log_file: logs/alerts.json
  console_output: true
  rate_limit: 10
```

### Customizing Detection Rules

Edit `config/rules.yaml` to add your own detection rules:

```yaml
rules:
  - name: "My Custom Rule"
    can_id: 0x123
    max_frequency: 100
    time_window: 1
    severity: HIGH
    description: "Custom frequency check"
    action: alert
```

See `config/example_rules.yaml` for more examples.

## Analyzing PCAP Files

If you have captured CAN traffic as PCAP:

```bash
# Analyze PCAP file
python main.py --mode replay --file path/to/capture.pcap

# With custom configuration
python main.py --mode replay --file capture.pcap --config config/can_ids_rpi4.conf
```

## Training ML Models

To use ML-based detection, you need to train a model:

```bash
# 1. Collect baseline normal traffic
candump -l can0  # Let run for 24-48 hours

# 2. Convert to CSV (script to be created)
python scripts/convert_candump.py candump-*.log data/raw/baseline.csv

# 3. Train model (module to be created)
python -m src.models.train_model \
    --input data/raw/baseline.csv \
    --output data/models/my_model.pkl \
    --contamination 0.02

# 4. Use the model
python main.py -i can0 --model data/models/my_model.pkl
```

## Understanding Output

### Console Output

```
[2025-01-20 10:30:45] [HIGH] [RULE_ENGINE] CAN ID: 0x123 | Rule: High Frequency Attack | Description: Possible denial of service attack
```

### JSON Alerts

Check `logs/alerts.json` for structured alert data:

```json
{
  "timestamp": 1705747845.123,
  "severity": "HIGH",
  "rule_name": "High Frequency Attack",
  "can_id": "0x123",
  "description": "Possible denial of service attack",
  "confidence": 0.95
}
```

### Statistics

Press `Ctrl+C` to see runtime statistics:

```
==========================================
CAN-IDS STATISTICS
==========================================
Runtime: 3600.0 seconds
Messages processed: 1500000
Alerts generated: 25
Processing rate: 416.67 messages/second
==========================================
```

## Common Use Cases

### 1. Monitor Vehicle CAN Bus

```bash
python main.py -i can0 --log-level INFO
```

### 2. Analyze Suspicious Traffic

```bash
python main.py --mode replay --file suspicious.pcap --log-level DEBUG
```

### 3. Development and Testing

```bash
# Use virtual CAN for safe testing
python main.py -i vcan0 --config config/can_ids.conf
```

### 4. Production Deployment on Pi

```bash
# Run as system service
sudo systemctl start can-ids.service
```

## Troubleshooting

### CAN Interface Not Found

```bash
# Check if interface exists
ip link show

# Load can modules
sudo modprobe can
sudo modprobe can_raw
sudo modprobe vcan  # For virtual CAN

# For MCP2515
sudo modprobe spi-bcm2835
```

### Permission Denied

```bash
# Add user to dialout group (may require re-login)
sudo usermod -a -G dialout $USER

# Or run with sudo (not recommended for production)
sudo python main.py -i can0
```

### No Traffic Detected

```bash
# Verify bus is active
candump can0

# Check bitrate matches your network
sudo ip link set can0 type can bitrate 250000  # Try different rates
sudo ip link set up can0
```

### ML Model Errors

```bash
# Install scikit-learn if missing
pip install scikit-learn numpy

# Check model file exists
ls -lh data/models/

# Train a new model or disable ML detection
# Edit config: detection_modes: [rule_based]
```

## Next Steps

1. **Customize Rules**: Edit `config/rules.yaml` for your specific CAN network
2. **Train ML Model**: Collect baseline traffic and train anomaly detector
3. **Optimize Performance**: Tune buffer sizes and rate limits in config
4. **Add Notifications**: Configure email or webhook alerts
5. **Monitor Logs**: Set up log rotation and monitoring
6. **Deploy to Production**: Install as systemd service

## Getting Help

- Check `docs/troubleshooting.md` for common issues
- Review `config/example_rules.yaml` for rule examples
- See `PROJECT_SUMMARY.md` for architecture overview
- Open GitHub issue for bugs or feature requests

## Safety and Security

⚠️ **Important**: 
- This is a **detection-only** system - it cannot block attacks
- Always test on non-critical systems first
- Use read-only CAN interfaces when possible
- Follow responsible disclosure for security issues
- Unauthorized access to vehicle networks may be illegal

## Contributing

Contributions welcome! Please see CONTRIBUTING.md (to be created) for guidelines.

---

**Happy CAN bus security monitoring!** 🚀