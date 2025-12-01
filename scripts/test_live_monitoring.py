#!/usr/bin/env python3
"""
Live CAN traffic monitoring test script.
Generates traffic while monitoring to validate the IDS can see live traffic.
"""

import subprocess
import time
import threading
import signal
import sys

def generate_traffic():
    """Generate CAN traffic in background thread."""
    print("Starting traffic generation...")
    time.sleep(2)  # Let monitoring start first
    
    for i in range(20):
        try:
            # Send various test messages
            can_id = 0x100 + (i % 8)
            data = f"{i:02X}{i:02X}{i:02X}{i:02X}{i:02X}{i:02X}{i:02X}{i:02X}"
            cmd = f"cansend vcan0 {can_id:03X}#{data}"
            subprocess.run(cmd, shell=True, check=True)
            print(f"Sent message {i+1}/20: ID=0x{can_id:03X}")
            time.sleep(0.5)
        except Exception as e:
            print(f"Error sending message {i+1}: {e}")
            break
    
    print("Traffic generation complete")

def main():
    print("CAN-IDS Live Traffic Test")
    print("=" * 50)
    
    # Start traffic generation in background
    traffic_thread = threading.Thread(target=generate_traffic, daemon=True)
    traffic_thread.start()
    
    # Start monitoring
    print("Starting CAN-IDS monitoring...")
    cmd = [
        "python3", "main.py", 
        "--monitor-traffic", "vcan0", 
        "--duration", "15"
    ]
    
    try:
        result = subprocess.run(cmd, timeout=20)
        print(f"Monitoring completed with exit code: {result.returncode}")
    except subprocess.TimeoutExpired:
        print("Monitoring timed out")
    except KeyboardInterrupt:
        print("Test interrupted by user")
    
    # Wait for traffic thread to complete
    traffic_thread.join(timeout=5)
    print("Live traffic test completed")

if __name__ == "__main__":
    main()