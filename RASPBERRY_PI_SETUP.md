# Raspberry Pi Setup Guide - CAN-IDS Project

**Document Created:** November 14, 2025  
**Target Hardware:** Raspberry Pi 4 Model B  
**CAN Interface:** MCP2515-based CAN HAT (16MHz oscillator)  
**User:** boneysan  
**Project Path:** /home/boneysan/Documents/Github/CANBUS_IDS  

---

## Overview

This document outlines all the steps performed to configure a Raspberry Pi 4 to run the CAN-IDS (Controller Area Network Intrusion Detection System) project. The setup includes hardware configuration, system optimization, Python environment setup, and automatic service configuration.

---

## Prerequisites

### Hardware Required
- Raspberry Pi 4 Model B (2GB+ RAM recommended)
- MCP2515-based CAN HAT or PiCAN shield
- 32GB+ microSD card (Class 10 or UHS-I)
- Proper cooling (heatsinks/fan recommended)
- 5V/3A USB-C power supply

### Software Requirements
- Raspberry Pi OS (tested with Bookworm)
- Python 3.8+ (tested with Python 3.11.2)
- Internet connection for package installation

---

## Setup Steps

### 1. Initial System Update

```bash
sudo apt update && sudo apt upgrade -y
```

Install required system packages:
```bash
sudo apt install python3 python3-pip python3-venv can-utils git -y
```

### 2. CAN Hardware Setup

#### 2.1 Configure MCP2515 CAN HAT

Run the automated setup script:
```bash
sudo bash raspberry-pi/scripts/setup_mcp2515.sh
```

This script performs the following:
- Enables SPI interface via raspi-config
- Backs up `/boot/firmware/config.txt`
- Adds device tree overlay configuration:
  ```
  dtparam=spi=on
  dtoverlay=mcp2515-can0,oscillator=16000000,interrupt=25
  dtoverlay=spi-bcm2835
  ```
- Installs can-utils package
- Creates `/etc/network/interfaces.d/can0` for interface auto-configuration

**Note:** The MCP2515 was detected with a 16MHz oscillator (not the default 12MHz in the script).

#### 2.2 Reboot System

```bash
sudo reboot
```

#### 2.3 Verify CAN Interface

After reboot, check that the interface exists:
```bash
ip link show can0
```

Expected output:
```
3: can0: <NOARP,ECHO> mtu 16 qdisc noop state DOWN mode DEFAULT group default qlen 10
    link/can
```

Verify hardware initialization in kernel messages:
```bash
dmesg | grep -i mcp251
```

Expected output:
```
[    6.674413] CAN device driver interface
[    6.777376] mcp251x spi0.0 can0: MCP2515 successfully initialized.
```

#### 2.4 Configure CAN Interface

Set up the CAN interface with 500kbps bitrate:
```bash
sudo bash raspberry-pi/scripts/setup_can_interface.sh can0 500000
```

Verify configuration:
```bash
ip -details link show can0
```

Expected output should show:
- State: UP
- Bitrate: 500000
- CAN state: ERROR-ACTIVE
- restart-ms: 100

### 3. Python Environment Setup

#### 3.1 Create Virtual Environment

```bash
cd /home/boneysan/Documents/Github/CANBUS_IDS
python3 -m venv venv
```

#### 3.2 Activate Virtual Environment

```bash
source venv/bin/activate
```

#### 3.3 Upgrade pip

```bash
pip install --upgrade pip
```

#### 3.4 Install Project Dependencies

```bash
pip install -r requirements.txt
```

Installed packages:
- python-can (4.6.1)
- scikit-learn (1.7.2)
- numpy (2.3.4)
- scipy (1.16.3)
- PyYAML (6.0.3)
- colorlog (6.10.1)
- joblib (1.5.2)
- threadpoolctl (3.6.0)
- typing_extensions (4.15.0)
- wrapt (1.17.3)
- packaging (25.0)

#### 3.5 Install CAN-IDS Package

```bash
pip install -e .
```

This installs the package in development mode, allowing code changes without reinstallation.

### 4. System Optimization for Raspberry Pi 4

#### 4.1 Run Optimization Script

```bash
sudo bash raspberry-pi/scripts/optimize_pi4.sh
```

This script performs the following optimizations:

**Step 1: Disable Unnecessary Services**
- bluetooth.service
- hciuart.service
- triggerhappy.service
- avahi-daemon.service

**Step 2: Boot Configuration Optimization**
Adds to `/boot/firmware/config.txt`:
```
# CAN-IDS Optimizations
dtoverlay=disable-bt
#dtoverlay=disable-wifi  # Optional, for wired-only setups
dtparam=audio=off
dtparam=watchdog=on
gpu_mem=16
```

**Step 3: Configure tmpfs for Logs**
Adds to `/etc/fstab`:
```
tmpfs /var/log tmpfs defaults,noatime,nosuid,mode=0755,size=50m 0 0
```

**Step 4: Install and Configure Hardware Watchdog**
- Installs watchdog package
- Configures `/etc/watchdog.conf`:
  ```
  watchdog-device = /dev/watchdog
  watchdog-timeout = 15
  max-load-1 = 24
  ```
- Enables and starts watchdog service

**Step 5: Optimize Swap Usage**
Adds to `/etc/sysctl.conf`:
```
vm.swappiness=10
```

**Step 6: Set Up Log Rotation**
Creates `/etc/logrotate.d/can-ids` for automatic log rotation.

### 5. Create Logs Directory

```bash
mkdir -p logs
```

### 6. Test CAN-IDS Operation

Test the system manually before setting up the service:
```bash
source venv/bin/activate
timeout 5 python main.py -i can0 --config config/can_ids_rpi4.yaml
```

If no errors appear, the system is working correctly.

### 7. Systemd Service Configuration

#### 7.1 Create Custom Service File

Create a customized service file with correct paths:

**File:** `/home/boneysan/Documents/Github/CANBUS_IDS/can-ids.service`

```ini
[Unit]
Description=CAN-IDS - Controller Area Network Intrusion Detection System
Documentation=https://github.com/Boneysan/CANBUS_IDS
After=network.target multi-user.target

[Service]
Type=simple
User=boneysan
Group=boneysan
WorkingDirectory=/home/boneysan/Documents/Github/CANBUS_IDS
Environment="PATH=/home/boneysan/Documents/Github/CANBUS_IDS/venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStartPre=/bin/sleep 10
ExecStartPre=/sbin/ip link set can0 type can bitrate 500000
ExecStartPre=/sbin/ip link set up can0
ExecStart=/home/boneysan/Documents/Github/CANBUS_IDS/venv/bin/python /home/boneysan/Documents/Github/CANBUS_IDS/main.py -i can0 --config /home/boneysan/Documents/Github/CANBUS_IDS/config/can_ids_rpi4.yaml
ExecStop=/sbin/ip link set down can0
Restart=on-failure
RestartSec=30
StandardOutput=journal
StandardError=journal
SyslogIdentifier=can-ids

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=/home/boneysan/Documents/Github/CANBUS_IDS/logs /home/boneysan/Documents/Github/CANBUS_IDS/data

# Resource limits
MemoryLimit=512M
CPUQuota=70%

[Install]
WantedBy=multi-user.target
```

#### 7.2 Install Service

```bash
sudo cp can-ids.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable can-ids.service
```

The service is now configured to start automatically on boot.

---

## Service Management

### Start the Service

```bash
sudo systemctl start can-ids.service
```

### Stop the Service

```bash
sudo systemctl stop can-ids.service
```

### Check Service Status

```bash
sudo systemctl status can-ids.service
```

### View Live Logs

```bash
sudo journalctl -u can-ids.service -f
```

### View Historical Logs

```bash
sudo journalctl -u can-ids.service --since "1 hour ago"
```

### Restart Service

```bash
sudo systemctl restart can-ids.service
```

### Disable Auto-start

```bash
sudo systemctl disable can-ids.service
```

---

## Manual Operation (For Testing)

To run CAN-IDS manually without the service:

```bash
cd /home/boneysan/Documents/Github/CANBUS_IDS
source venv/bin/activate
python main.py -i can0 --config config/can_ids_rpi4.yaml
```

Press `Ctrl+C` to stop.

---

## Configuration Files

### Primary Configuration

**File:** `config/can_ids_rpi4.yaml`

Key Raspberry Pi 4 optimizations in this config:
- `max_cpu_percent: 70` - Thermal management
- `max_memory_mb: 300` - Conservative memory limit
- `processing_threads: 1` - Prevents overheating
- `buffer_size: 500` - Reduced from desktop config
- `thermal_throttling_temp: 70` - Temperature monitoring
- `use_tmpfs_logs: true` - RAM-based logging

### Detection Rules

**File:** `config/rules.yaml`

Contains detection rules for:
- DoS attacks
- Replay attacks
- Fuzzing attempts
- ECU impersonation
- Frequency anomalies
- Timing violations

---

## Monitoring System Health

### Check Temperature

```bash
vcgencmd measure_temp
```

Keep temperature below 70°C during operation.

### Check Throttling Status

```bash
vcgencmd get_throttled
```

Output `0x0` means no throttling occurred.

### Monitor Resources

```bash
htop
```

### Check CAN Interface Statistics

```bash
ip -s link show can0
```

### Monitor CAN Traffic

```bash
candump can0
```

---

## Troubleshooting

### CAN Interface Not Coming Up

Check kernel messages:
```bash
dmesg | grep -i "mcp251\|can\|spi"
```

Verify SPI is enabled:
```bash
ls /dev/spi*
```

### Service Fails to Start

Check service logs:
```bash
sudo journalctl -u can-ids.service -n 50
```

Verify CAN interface manually:
```bash
sudo ip link set can0 type can bitrate 500000
sudo ip link set can0 up
ip link show can0
```

### Python Import Errors

Ensure virtual environment is activated:
```bash
source venv/bin/activate
which python
```

Reinstall dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

### High Temperature

Check cooling:
```bash
vcgencmd measure_temp
```

Verify CPU throttling is configured:
```bash
grep CPUQuota /etc/systemd/system/can-ids.service
```

Install heatsinks or add a fan if temperature exceeds 70°C.

### Memory Issues

Check current memory usage:
```bash
free -h
```

Review service memory limit:
```bash
grep MemoryLimit /etc/systemd/system/can-ids.service
```

Adjust if needed and reload:
```bash
sudo systemctl daemon-reload
sudo systemctl restart can-ids.service
```

---

## Post-Setup Recommendations

### 1. Reboot After Initial Setup

Apply all system optimizations:
```bash
sudo reboot
```

### 2. Monitor First 24 Hours

Check logs regularly:
```bash
sudo journalctl -u can-ids.service -f
```

Monitor system resources:
```bash
htop
watch -n 5 vcgencmd measure_temp
```

### 3. Backup Configuration

```bash
tar -czf canids-backup-$(date +%Y%m%d).tar.gz \
  config/ \
  can-ids.service \
  /etc/systemd/system/can-ids.service
```

### 4. Update Detection Rules

Customize `config/rules.yaml` for your specific CAN bus environment.

### 5. Train ML Model (Optional)

If using ML-based detection:
```bash
# Collect baseline normal traffic (24-48 hours)
candump -l can0

# Train model
source venv/bin/activate
python src/models/train_model.py --input candump-*.log
```

---

## Performance Benchmarks

Expected performance on Raspberry Pi 4 (4GB):

- **CAN Message Processing:** ~5000 messages/second
- **CPU Usage:** 30-50% (single core)
- **Memory Usage:** 150-300MB
- **Temperature:** 50-65°C (with heatsink)
- **Detection Latency:** <10ms per message

---

## Security Considerations

The systemd service is configured with security hardening:
- Runs as unprivileged user (boneysan)
- `NoNewPrivileges=true` - Prevents privilege escalation
- `PrivateTmp=true` - Isolated /tmp directory
- `ProtectSystem=strict` - Read-only system directories
- `ProtectHome=read-only` - Read-only home directories (except logs/data)

---

## Maintenance

### Regular Updates

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Update Python packages
source venv/bin/activate
pip install --upgrade -r requirements.txt
```

### Log Management

Logs are automatically rotated by the optimization script. Manual cleanup if needed:
```bash
cd logs
rm *.log.*.gz  # Remove old compressed logs
```

### SD Card Health

Monitor SD card writes:
```bash
sudo iotop
```

The tmpfs configuration reduces writes significantly.

---

## Summary

Your Raspberry Pi 4 is now configured with:

✅ MCP2515 CAN HAT (16MHz oscillator, 500kbps)  
✅ Python 3.11.2 with all dependencies  
✅ CAN-IDS installed in virtual environment  
✅ System optimizations for embedded operation  
✅ Automatic startup via systemd service  
✅ Resource limits (512MB RAM, 70% CPU)  
✅ Hardware watchdog for reliability  
✅ Reduced SD card wear (tmpfs logs)  
✅ Security hardening  

The system is ready for production use and will automatically start monitoring the CAN bus on every boot.

---

## Additional Resources

- **Project Documentation:** `docs/README.md`
- **Getting Started Guide:** `GETTING_STARTED.md`
- **Raspberry Pi Optimization:** `docs/raspberry_pi4_optimization_guide.md`
- **Rules Configuration:** `docs/rules_guide.md`
- **Testing Guide:** `docs/testing_results.md`

---

**Setup Completed:** November 14, 2025  
**System Status:** Ready for Production  
