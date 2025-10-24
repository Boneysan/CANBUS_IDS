#!/bin/bash
# CAN interface setup script for various bitrates

set -e

INTERFACE=${1:-can0}
BITRATE=${2:-500000}

echo "Setting up CAN interface: $INTERFACE at $BITRATE bps"

# Check if interface exists
if ! ip link show $INTERFACE &> /dev/null; then
    echo "Error: CAN interface $INTERFACE not found"
    echo "Available interfaces:"
    ip link show | grep can
    exit 1
fi

# Bring down interface if already up
if ip link show $INTERFACE | grep -q "UP"; then
    echo "Bringing down $INTERFACE..."
    sudo ip link set $INTERFACE down
fi

# Configure interface
echo "Configuring $INTERFACE with bitrate $BITRATE..."
sudo ip link set $INTERFACE type can bitrate $BITRATE

# Optional: Set restart delay
sudo ip link set $INTERFACE type can restart-ms 100

# Bring up interface
echo "Bringing up $INTERFACE..."
sudo ip link set $INTERFACE up

# Show interface status
echo ""
echo "CAN interface status:"
ip -details link show $INTERFACE

echo ""
echo "Setup complete! Test with: candump $INTERFACE"