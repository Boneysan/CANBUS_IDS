#!/bin/bash
# Raspberry Pi 4 optimization script for CAN-IDS

set -e

echo "=========================================="
echo "CAN-IDS Raspberry Pi 4 Optimization"
echo "=========================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root (use sudo)"
  exit 1
fi

echo "Step 1: Disabling unnecessary services..."
# Disable services that aren't needed for headless operation
services_to_disable=(
    "bluetooth.service"
    "hciuart.service"
    "triggerhappy.service"
    "avahi-daemon.service"
)

for service in "${services_to_disable[@]}"; do
    if systemctl is-enabled $service &> /dev/null; then
        echo "  Disabling $service..."
        systemctl disable $service
        systemctl stop $service
    fi
done

echo "Step 2: Optimizing boot configuration..."
CONFIG_FILE="/boot/config.txt"

# Backup config
cp $CONFIG_FILE ${CONFIG_FILE}.backup.$(date +%Y%m%d_%H%M%S)

# Add optimization settings if not present
if ! grep -q "# CAN-IDS Optimizations" $CONFIG_FILE; then
    cat << EOF >> $CONFIG_FILE

# CAN-IDS Optimizations
# Disable Bluetooth
dtoverlay=disable-bt

# Disable WiFi (uncomment if using wired connection only)
#dtoverlay=disable-wifi

# Disable audio
dtparam=audio=off

# Enable hardware watchdog
dtparam=watchdog=on

# GPU memory (reduce for headless)
gpu_mem=16
EOF
fi

echo "Step 3: Configuring tmpfs for logs..."
# Use tmpfs to reduce SD card writes
if ! grep -q "/var/log.*tmpfs" /etc/fstab; then
    echo "tmpfs /var/log tmpfs defaults,noatime,nosuid,mode=0755,size=50m 0 0" >> /etc/fstab
fi

echo "Step 4: Installing watchdog..."
apt-get update
apt-get install -y watchdog

# Configure watchdog
cat << EOF > /etc/watchdog.conf
watchdog-device = /dev/watchdog
watchdog-timeout = 15
max-load-1 = 24
EOF

systemctl enable watchdog
systemctl start watchdog

echo "Step 5: Optimizing swap..."
# Reduce swap usage
if ! grep -q "vm.swappiness" /etc/sysctl.conf; then
    echo "vm.swappiness=10" >> /etc/sysctl.conf
    sysctl -p
fi

echo "Step 6: Setting up log rotation..."
cat << EOF > /etc/logrotate.d/can-ids
/home/pi/can-ids/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 pi pi
}

/home/pi/can-ids/logs/*.json {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 pi pi
}
EOF

echo ""
echo "=========================================="
echo "Optimization Complete!"
echo "=========================================="
echo "Recommended next steps:"
echo "1. Reboot to apply all changes:"
echo "   sudo reboot"
echo ""
echo "2. Monitor system resources:"
echo "   htop"
echo "   vcgencmd measure_temp"
echo "   vcgencmd get_throttled"
echo ""
echo "3. Test CAN-IDS performance:"
echo "   python main.py -i can0"
echo "=========================================="