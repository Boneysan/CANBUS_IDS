#!/bin/bash
# Setup script for MCP2515 CAN HAT on Raspberry Pi 4

set -e

echo "=========================================="
echo "CAN-IDS MCP2515 Setup Script"
echo "=========================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root (use sudo)"
  exit 1
fi

echo "Step 1: Enabling SPI interface..."
raspi-config nonint do_spi 0

echo "Step 2: Configuring device tree overlay..."
CONFIG_FILE="/boot/config.txt"

# Backup config
cp $CONFIG_FILE ${CONFIG_FILE}.backup.$(date +%Y%m%d_%H%M%S)

# Check if already configured
if grep -q "dtoverlay=mcp2515-can0" $CONFIG_FILE; then
    echo "MCP2515 overlay already configured"
else
    echo "Adding MCP2515 overlay to config.txt..."
    cat << EOF >> $CONFIG_FILE

# CAN-IDS MCP2515 Configuration
dtparam=spi=on
dtoverlay=mcp2515-can0,oscillator=12000000,interrupt=25
dtoverlay=spi-bcm2835
EOF
fi

echo "Step 3: Installing CAN utilities..."
apt-get update
apt-get install -y can-utils

echo "Step 4: Creating CAN interface configuration..."
cat << 'EOF' > /etc/network/interfaces.d/can0
auto can0
iface can0 inet manual
    pre-up /sbin/ip link set can0 type can bitrate 500000
    up /sbin/ifconfig can0 up
    down /sbin/ifconfig can0 down
EOF

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo "Please reboot your Raspberry Pi for changes to take effect:"
echo "  sudo reboot"
echo ""
echo "After reboot, test the CAN interface with:"
echo "  ip link show can0"
echo "  candump can0"
echo "=========================================="