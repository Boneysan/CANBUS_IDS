# Raspberry Pi Deployment Guide - Latest Updates

**Last Updated:** December 16, 2025  
**CAN-IDS Version:** 100% Complete (18/18 Rule Types + ML Detection)  
**Target Hardware:** Raspberry Pi 4 Model B (2GB+ RAM recommended)  
**Status:** Production Ready

---

## 🎯 What's New in This Release

This deployment includes all the latest improvements:

✅ **Complete Detection System** (100% Implementation)
- All 18 rule types implemented and tested
- ML detection with 97.20% recall, 100% precision
- Adaptive timing detection for sophisticated attacks
- Enhanced fuzzing detection
- Cross-vehicle model support

✅ **Performance Optimizations**
- 40-50K messages/second throughput capability
- Optimized for Raspberry Pi 4 hardware
- Efficient memory usage
- Reduced false positive rates

✅ **Advanced Features**
- Ensemble model support
- Decision tree classifier integration
- Enhanced feature engineering (58 features)
- Novel attack detection

---

## 📋 Prerequisites

### Hardware Required
- **Raspberry Pi 4 Model B** (2GB+ RAM, 4GB recommended for ML features)
- **MCP2515-based CAN HAT** (16MHz oscillator) or compatible interface
- **32GB+ microSD card** (Class 10 or UHS-I recommended)
- **Cooling solution** (heatsinks/fan - important for ML processing)
- **5V/3A USB-C power supply** (official Raspberry Pi adapter recommended)
- **CAN bus connection** to your target network

### Software Prerequisites
- Raspberry Pi OS (Bookworm or later)
- Python 3.8+ (Python 3.11+ recommended)
- Internet connection for initial setup

---

## 🚀 Fresh Installation (First Time Setup)

### Step 1: System Preparation

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install required system packages
sudo apt install python3 python3-pip python3-venv can-utils git -y

# Install build dependencies (needed for some Python packages)
sudo apt install python3-dev build-essential -y
```

### Step 2: Clone or Pull Repository

#### If Installing Fresh:
```bash
cd ~/Documents
mkdir -p GitHub
cd GitHub
git clone https://github.com/yourusername/CANBUS_IDS.git
cd CANBUS_IDS
```

#### If Updating Existing Installation:
```bash
cd ~/Documents/GitHub/CANBUS_IDS

# Make sure you're on the correct branch
git branch

# Pull latest changes
git pull origin main
```

### Step 3: CAN Hardware Setup

#### Configure MCP2515 CAN HAT:
```bash
# Run automated setup script
sudo bash raspberry-pi/scripts/setup_mcp2515.sh

# Reboot to apply changes
sudo reboot
```

After reboot, verify CAN interface:
```bash
# Check interface exists
ip link show can0

# Should show: can0: <NOARP,ECHO> mtu 16 qdisc noop state DOWN

# Check kernel messages
dmesg | grep -i mcp251

# Should see: mcp251x spi0.0 can0: MCP2515 successfully initialized
```

#### Bring up CAN interface:
```bash
# Configure CAN0 with appropriate bitrate (adjust for your network)
sudo bash raspberry-pi/scripts/setup_can_interface.sh can0 500000

# Verify it's UP
ip -details link show can0
```

**Common bitrates:** 125000, 250000, 500000, 1000000 (adjust to match your CAN network)

### Step 4: Python Virtual Environment

```bash
cd ~/Documents/GitHub/CANBUS_IDS

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

**Important:** The virtual environment path is `.venv` (with a dot). This matches your current setup.

### Step 5: Verify Installation

```bash
# Verify Python packages
pip list | grep -E "python-can|scikit-learn|numpy|joblib|PyYAML|colorlog"

# Test import
python -c "from src.capture.can_sniffer import CANSniffer; print('✅ Import successful')"

# View help
python main.py --help
```

### Step 6: Configuration

#### Check/Edit Configuration:
```bash
# View current config
cat config/can_ids.yaml

# Edit if needed
nano config/can_ids.yaml
```

**Key settings to verify:**
```yaml
interface: can0  # Match your CAN interface
bustype: socketcan

# Enable ML detection (if you have models)
ml_detection:
  enabled: true  # Set to false if no models available
  contamination: 0.02
  model_path: data/models/your_model.pkl

# Rule engine (should be enabled)
rule_engine:
  enabled: true
  rules_file: config/rules.yaml
```

---

## 🔄 Updating Existing Installation

If you already have CAN-IDS installed and want to pull the latest updates:

```bash
# Navigate to project
cd ~/Documents/GitHub/CANBUS_IDS

# Activate virtual environment
source .venv/bin/activate

# Pull latest code
git pull origin main

# Update dependencies (in case new ones were added)
pip install --upgrade -r requirements.txt

# Verify everything works
python main.py --help
```

**Important:** After pulling updates, always check if there are new configuration options in `config/can_ids.yaml`.

---

## 🧪 Testing the Installation

### Test 1: Virtual CAN (Safe Testing)

```bash
# Create virtual CAN interface
sudo modprobe vcan
sudo ip link add dev vcan0 type vcan
sudo ip link set up vcan0

# In one terminal, generate test traffic
while true; do
    cansend vcan0 123#DEADBEEF
    sleep 0.1
done

# In another terminal, run CAN-IDS
source .venv/bin/activate
python main.py -i vcan0
```

### Test 2: Real CAN Interface

```bash
# Monitor CAN traffic first
candump can0

# If you see traffic, run CAN-IDS
source .venv/bin/activate
python main.py -i can0
```

### Test 3: Check All Components Load

```bash
# Run with verbose logging
python main.py -i can0 --log-level DEBUG
```

Look for these startup messages:
- ✅ Rule engine initialized
- ✅ ML detection enabled (if configured)
- ✅ Alert manager initialized
- ✅ Starting monitoring...

---

## 🎛️ Running CAN-IDS

### Basic Usage

```bash
# Make sure virtual environment is active
source .venv/bin/activate

# Start monitoring
python main.py -i can0

# With custom config
python main.py -i can0 --config config/my_config.yaml

# With specific log level
python main.py -i can0 --log-level INFO

# Analyze PCAP file
python main.py --mode replay --file capture.pcap
```

### Understanding the Output

You'll see:
1. **Startup banner** - System initialization
2. **Detection stages** - Rule engine, ML detector status
3. **Real-time alerts** - Format: `[SEVERITY] [DETECTOR] CAN ID: 0x123 | Description`
4. **Statistics** - Press Ctrl+C to see summary

Example alert:
```
[HIGH] [RULE_ENGINE] CAN ID: 0x123 | Rule: High Frequency Attack | Description: Message rate 500/s exceeds threshold
```

---

## 🔧 Installing as System Service

To run CAN-IDS automatically on boot:

### Step 1: Edit Service File

```bash
# Copy service file
sudo cp can-ids.service /etc/systemd/system/

# Edit paths to match your installation
sudo nano /etc/systemd/system/can-ids.service
```

**Verify these paths in the service file:**
```ini
WorkingDirectory=/home/YOUR_USERNAME/Documents/GitHub/CANBUS_IDS
ExecStart=/home/YOUR_USERNAME/Documents/GitHub/CANBUS_IDS/.venv/bin/python main.py -i can0
User=YOUR_USERNAME
```

### Step 2: Enable and Start Service

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service (start on boot)
sudo systemctl enable can-ids.service

# Start service now
sudo systemctl start can-ids.service

# Check status
sudo systemctl status can-ids.service
```

### Step 3: View Logs

```bash
# Follow service logs
sudo journalctl -u can-ids.service -f

# View recent logs
sudo journalctl -u can-ids.service -n 100

# View logs since boot
sudo journalctl -u can-ids.service -b
```

### Managing the Service

```bash
# Stop service
sudo systemctl stop can-ids.service

# Restart service
sudo systemctl restart can-ids.service

# Disable service (don't start on boot)
sudo systemctl disable can-ids.service
```

---

## 📊 ML Models (Optional)

The latest version includes ML detection capabilities. Models are **not** included in the repository and must be trained separately.

### If You Have Pre-trained Models:

```bash
# Create models directory
mkdir -p data/models

# Copy your trained models
cp /path/to/your/model.pkl data/models/

# Update config to point to your model
nano config/can_ids.yaml
```

In `config/can_ids.yaml`:
```yaml
ml_detection:
  enabled: true
  contamination: 0.02
  model_path: data/models/your_model.pkl
```

### If You Don't Have Models:

The system will work fine with rule-based detection only:

```yaml
ml_detection:
  enabled: false  # Disable ML detection
```

### Training Your Own Models:

Refer to the `Vehicle_Models` workspace for model training scripts. You'll need:
1. Baseline CAN traffic from your network
2. Attack samples (or use synthetic attacks)
3. Run training scripts from Vehicle_Models project
4. Copy resulting `.joblib` or `.pkl` files to `data/models/`

---

## 🐛 Troubleshooting

### CAN Interface Issues

**Problem:** `can0` not found
```bash
# Check loaded modules
lsmod | grep -i can

# Load required modules
sudo modprobe can
sudo modprobe can_raw
sudo modprobe mcp2515

# Check hardware
dmesg | grep -i spi
dmesg | grep -i mcp251
```

**Problem:** Interface won't come up
```bash
# Check bitrate matches your network
sudo ip link set can0 type can bitrate 250000  # Try different rates
sudo ip link set up can0

# Check CAN HAT is properly seated
# Verify /boot/firmware/config.txt has correct overlay
```

### Python/Package Issues

**Problem:** Import errors
```bash
# Make sure virtual environment is activated
source .venv/bin/activate

# Verify you're in the right directory
pwd  # Should be .../CANBUS_IDS

# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

**Problem:** scikit-learn version errors
```bash
# Check version
pip show scikit-learn

# Should be 1.3.0 or higher
# If not, upgrade:
pip install --upgrade scikit-learn numpy scipy
```

### Permission Issues

**Problem:** Permission denied accessing CAN
```bash
# Add user to dialout group
sudo usermod -a -G dialout $USER

# Log out and log back in for changes to take effect

# Alternatively (not recommended for production):
sudo python main.py -i can0
```

### Performance Issues

**Problem:** System running slow
```bash
# Check CPU usage
htop

# Monitor temperature
vcgencmd measure_temp

# If overheating, verify cooling is adequate
# Disable ML detection if Pi struggles:
# Set ml_detection: enabled: false in config
```

### No Traffic Detected

```bash
# Verify bus is actually active
candump can0

# If no messages, check:
# 1. CAN connections (CANH, CANL, GND)
# 2. Termination resistors (120Ω at each end)
# 3. Bitrate matches network
# 4. Power to CAN transceiver
```

---

## 📁 Important Files and Directories

```
CANBUS_IDS/
├── .venv/                          # Python virtual environment
├── config/
│   ├── can_ids.yaml               # Main configuration
│   └── rules.yaml                 # Detection rules
├── data/
│   ├── models/                    # ML models (if used)
│   └── raw/                       # Training data
├── logs/                          # Application logs
├── src/                           # Source code
├── raspberry-pi/
│   ├── scripts/
│   │   ├── setup_mcp2515.sh      # Hardware setup
│   │   └── setup_can_interface.sh # CAN interface config
│   └── systemd/
│       └── can-ids.service        # Service file
├── main.py                        # Main entry point
├── requirements.txt               # Python dependencies
└── can-ids.service                # Systemd service file
```

---

## 🔐 Security Considerations

⚠️ **Important:**
- This is a **detection-only** system - it cannot block attacks
- Always test on non-production systems first
- Use read-only CAN interfaces when possible
- Keep logs secure (may contain sensitive vehicle data)
- Follow responsible disclosure for any security findings
- Unauthorized access to vehicle networks may be **illegal**

---

## 📈 Performance Expectations

On Raspberry Pi 4 (4GB):
- **Rule-based only:** 40-50K msg/s
- **With ML detection:** 8-12K msg/s (depends on model complexity)
- **Memory usage:** 200-500MB
- **CPU usage:** 25-60% (single core)

**Tips for better performance:**
- Use decision tree models instead of ensemble models
- Disable debug logging in production
- Ensure adequate cooling
- Consider overclocking (with proper cooling)

---

## 📚 Additional Documentation

- `GETTING_STARTED.md` - General usage guide
- `100_PERCENT_COMPLETE.md` - Feature completion status
- `PERFORMANCE_TESTING_GUIDE.md` - Performance testing procedures
- `PROJECT_SUMMARY.md` - Architecture overview
- `HYBRID_APPROACH_CAPABILITIES.md` - Detection capabilities

---

## 🆘 Getting Help

1. Check existing documentation in the project
2. Review logs in `logs/` directory
3. Run with `--log-level DEBUG` for detailed output
4. Check GitHub issues for similar problems
5. Join project discussions (if available)

---

## ✅ Quick Reference Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Update from git
git pull origin main

# Install/update dependencies
pip install -r requirements.txt

# Bring up CAN interface
sudo bash raspberry-pi/scripts/setup_can_interface.sh can0 500000

# Start monitoring
python main.py -i can0

# Start as service
sudo systemctl start can-ids.service

# View service logs
sudo journalctl -u can-ids.service -f

# Stop service
sudo systemctl stop can-ids.service
```

---

## 🎉 You're Ready!

Your Raspberry Pi is now ready to run the latest CAN-IDS with all improvements:
- ✅ Complete rule-based detection (18 rule types)
- ✅ ML anomaly detection (if models configured)
- ✅ Adaptive timing analysis
- ✅ Enhanced fuzzing detection
- ✅ Production-ready performance

Happy CAN bus monitoring! 🚀🔒

