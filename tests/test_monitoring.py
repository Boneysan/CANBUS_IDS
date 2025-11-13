#!/usr/bin/env python3
"""
Test script for resource monitoring functionality.

Tests metrics collection, logging, and alerting.
"""

import sys
import time
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitoring.resource_monitor import ResourceMonitor, MetricsCollector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_metrics_collector():
    """Test basic metrics collection."""
    print("\n" + "="*60)
    print("Testing MetricsCollector")
    print("="*60)
    
    collector = MetricsCollector()
    
    # Collect metrics
    metrics = collector.collect()
    
    print("\nCollected metrics:")
    for key, value in metrics.items():
        if key != 'datetime':
            print(f"  {key}: {value}")
    
    # Test temperature reading
    if collector.enable_temperature:
        temp = collector.get_temperature()
        print(f"\nCPU Temperature: {temp}°C" if temp else "\nTemperature unavailable")
    
    print(f"\nCollector stats: {collector.get_statistics()}")


def test_resource_monitor():
    """Test resource monitor with live collection."""
    print("\n" + "="*60)
    print("Testing ResourceMonitor (10 second test)")
    print("="*60)
    
    # Create temporary log file
    log_file = Path("logs/test_metrics.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize monitor
    monitor = ResourceMonitor(
        sample_interval=2.0,  # Sample every 2 seconds
        log_interval=5.0,  # Write every 5 seconds
        log_file=str(log_file),
        console_output=True,  # Print metrics
        enable_alerts=True
    )
    
    # Start monitoring
    print("\nStarting resource monitor...")
    monitor.start()
    
    # Run for 10 seconds
    print("Monitoring for 10 seconds...\n")
    time.sleep(10)
    
    # Stop monitoring
    monitor.stop()
    
    # Print summary
    monitor.print_summary()
    
    # Check log file
    if log_file.exists():
        with open(log_file, 'r') as f:
            lines = f.readlines()
        print(f"\nLog file created: {log_file}")
        print(f"Log entries: {len(lines)}")
        
        if lines:
            print("\nFirst log entry:")
            print(lines[0][:200] + "...")
    
    # Cleanup
    if log_file.exists():
        log_file.unlink()
        print(f"\nCleaned up test log file")


def test_alert_thresholds():
    """Test alert threshold functionality."""
    print("\n" + "="*60)
    print("Testing Alert Thresholds")
    print("="*60)
    
    # Set very low thresholds to trigger alerts
    monitor = ResourceMonitor(
        sample_interval=1.0,
        enable_alerts=True,
        alert_thresholds={
            'cpu_percent': 0.1,  # Very low to trigger
            'memory_percent': 0.1,
            'cpu_temp_celsius': 0.1
        }
    )
    
    print("\nSet very low thresholds to test alerting...")
    monitor.start()
    
    print("Running for 5 seconds to generate alerts...\n")
    time.sleep(5)
    
    monitor.stop()
    
    stats = monitor.get_statistics()
    print(f"\nAlerts generated: {stats['alerts_generated']}")
    
    if stats['alerts_generated'] > 0:
        print("✅ Alert system working correctly")
    else:
        print("⚠️ No alerts generated (thresholds might need adjustment)")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("CAN-IDS Resource Monitoring Test Suite")
    print("="*70)
    
    try:
        # Test 1: Basic metrics collection
        test_metrics_collector()
        
        # Test 2: Resource monitor
        test_resource_monitor()
        
        # Test 3: Alert thresholds
        test_alert_thresholds()
        
        print("\n" + "="*70)
        print("✅ All tests completed successfully")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
