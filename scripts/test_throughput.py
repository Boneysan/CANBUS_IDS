#!/usr/bin/env python3
"""
Throughput testing script for batch processing optimization.

Tests actual messages/second performance to validate 5-10x improvement.
"""

import sys
import time
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from typing import List, Dict, Any
import statistics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_vcan():
    """Setup virtual CAN interface for testing."""
    try:
        # Check if vcan0 exists
        result = subprocess.run(['ip', 'link', 'show', 'vcan0'], 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.info("Setting up vcan0...")
            subprocess.run(['sudo', 'modprobe', 'vcan'], check=True)
            subprocess.run(['sudo', 'ip', 'link', 'add', 'dev', 'vcan0', 'type', 'vcan'], check=True)
            subprocess.run(['sudo', 'ip', 'link', 'set', 'up', 'vcan0'], check=True)
            logger.info("✅ vcan0 setup complete")
        else:
            logger.info("✅ vcan0 already exists")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to setup vcan0: {e}")
        return False


def generate_test_traffic(duration: int = 10, rate: int = 1000):
    """
    Generate CAN test traffic using cansend.
    
    Args:
        duration: How long to generate traffic (seconds)
        rate: Target messages per second
    """
    logger.info(f"Generating test traffic: {rate} msg/s for {duration}s...")
    
    try:
        # Use cangen for high-speed traffic generation
        cmd = [
            'cangen', 'vcan0',
            '-g', str(int(1000 / rate)),  # Gap in milliseconds
            '-n', str(duration * rate),    # Total messages
            '-I', '123',                   # CAN ID
            '-L', '8'                      # DLC=8
        ]
        
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.info(f"✅ Traffic generator started")
        return True
        
    except FileNotFoundError:
        logger.error("cangen not found. Install can-utils: sudo apt install can-utils")
        return False
    except Exception as e:
        logger.error(f"Failed to generate traffic: {e}")
        return False


def test_batch_processing(duration: int = 10) -> Dict[str, Any]:
    """
    Test batch processing performance.
    
    Args:
        duration: Test duration in seconds
        
    Returns:
        Performance statistics
    """
    from src.capture.can_sniffer import CANSniffer
    from src.detection.rule_engine import RuleEngine
    
    logger.info("Testing BATCH processing mode...")
    
    # Initialize components
    sniffer = CANSniffer('vcan0')
    sniffer.start()
    
    # Load rules (lightweight for testing)
    rule_engine = RuleEngine('config/rules.yaml')
    
    # Performance tracking
    messages_processed = 0
    batch_count = 0
    start_time = time.time()
    batch_sizes = []
    
    try:
        while time.time() - start_time < duration:
            # Read batch
            batch = sniffer.read_batch(batch_size=100, timeout=0.1)
            
            if batch:
                batch_count += 1
                messages_processed += len(batch)
                batch_sizes.append(len(batch))
                
                # Process batch through rule engine
                alerts = rule_engine.analyze_batch(batch)
                
    except KeyboardInterrupt:
        pass
    finally:
        sniffer.stop()
    
    # Calculate statistics
    elapsed = time.time() - start_time
    throughput = messages_processed / elapsed if elapsed > 0 else 0
    
    stats = {
        'mode': 'BATCH',
        'messages': messages_processed,
        'duration': elapsed,
        'throughput': throughput,
        'batches': batch_count,
        'avg_batch_size': statistics.mean(batch_sizes) if batch_sizes else 0,
        'min_batch_size': min(batch_sizes) if batch_sizes else 0,
        'max_batch_size': max(batch_sizes) if batch_sizes else 0
    }
    
    logger.info(f"✅ Batch test complete: {throughput:.0f} msg/s")
    return stats


def test_individual_processing(duration: int = 10) -> Dict[str, Any]:
    """
    Test individual message processing (baseline).
    Simulates old behavior by reading batch_size=1.
    
    Args:
        duration: Test duration in seconds
        
    Returns:
        Performance statistics
    """
    from src.capture.can_sniffer import CANSniffer
    from src.detection.rule_engine import RuleEngine
    
    logger.info("Testing INDIVIDUAL processing mode (baseline)...")
    
    # Initialize components
    sniffer = CANSniffer('vcan0')
    sniffer.start()
    
    # Load rules
    rule_engine = RuleEngine('config/rules.yaml')
    
    # Performance tracking
    messages_processed = 0
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration:
            # Read single message (batch_size=1 simulates old behavior)
            batch = sniffer.read_batch(batch_size=1, timeout=0.001)
            
            if batch:
                for msg in batch:
                    messages_processed += 1
                    
                    # Process individually through rule engine
                    alerts = rule_engine.analyze_message(msg)
                
    except KeyboardInterrupt:
        pass
    finally:
        sniffer.stop()
    
    # Calculate statistics
    elapsed = time.time() - start_time
    throughput = messages_processed / elapsed if elapsed > 0 else 0
    
    stats = {
        'mode': 'INDIVIDUAL',
        'messages': messages_processed,
        'duration': elapsed,
        'throughput': throughput,
        'batches': messages_processed,  # Each message is a "batch" of 1
        'avg_batch_size': 1,
        'min_batch_size': 1,
        'max_batch_size': 1
    }
    
    logger.info(f"✅ Individual test complete: {throughput:.0f} msg/s")
    return stats


def print_results(baseline: Dict[str, Any], optimized: Dict[str, Any]):
    """Print comparison results."""
    
    print("\n" + "="*70)
    print("THROUGHPUT TEST RESULTS")
    print("="*70)
    
    print(f"\n{'Metric':<30} {'Baseline':<20} {'Batch Processing':<20}")
    print("-"*70)
    
    print(f"{'Mode':<30} {baseline['mode']:<20} {optimized['mode']:<20}")
    print(f"{'Messages Processed':<30} {baseline['messages']:<20,} {optimized['messages']:<20,}")
    print(f"{'Duration (s)':<30} {baseline['duration']:<20.2f} {optimized['duration']:<20.2f}")
    print(f"{'Throughput (msg/s)':<30} {baseline['throughput']:<20.0f} {optimized['throughput']:<20.0f}")
    print(f"{'Avg Batch Size':<30} {baseline['avg_batch_size']:<20.1f} {optimized['avg_batch_size']:<20.1f}")
    
    print("\n" + "-"*70)
    
    # Calculate improvement
    if baseline['throughput'] > 0:
        improvement = optimized['throughput'] / baseline['throughput']
        improvement_pct = (improvement - 1) * 100
        
        print(f"\n🚀 PERFORMANCE IMPROVEMENT: {improvement:.2f}x ({improvement_pct:+.1f}%)")
        
        if improvement >= 5.0:
            print("✅ SUCCESS: Achieved 5x+ improvement target!")
        elif improvement >= 2.0:
            print("⚠️  PARTIAL: 2x+ improvement, below 5x target")
        else:
            print("❌ BELOW TARGET: Less than 2x improvement")
            
        print(f"\n   Baseline:  {baseline['throughput']:>8.0f} msg/s")
        print(f"   Optimized: {optimized['throughput']:>8.0f} msg/s")
        print(f"   Gain:      {optimized['throughput'] - baseline['throughput']:>8.0f} msg/s")
    
    print("\n" + "="*70)


def main():
    """Main testing procedure."""
    
    print("\n" + "="*70)
    print("CAN-IDS THROUGHPUT TESTING")
    print("Testing batch processing optimization")
    print("="*70 + "\n")
    
    # Step 1: Setup vcan0
    if not setup_vcan():
        logger.error("Failed to setup vcan0. Exiting.")
        return 1
    
    # Wait for interface to be ready
    time.sleep(1)
    
    # Test duration
    test_duration = 10  # seconds
    traffic_rate = 2000  # msg/s
    
    print(f"\nTest Configuration:")
    print(f"  Duration: {test_duration} seconds")
    print(f"  Traffic rate: {traffic_rate} msg/s")
    print(f"  Target improvement: 5-10x")
    print()
    
    # Step 2: Test baseline (individual processing)
    print("\n" + "-"*70)
    print("TEST 1: BASELINE (Individual Message Processing)")
    print("-"*70)
    
    if not generate_test_traffic(duration=test_duration + 2, rate=traffic_rate):
        logger.error("Failed to generate traffic. Exiting.")
        return 1
    
    time.sleep(0.5)  # Let traffic generator start
    baseline_stats = test_individual_processing(duration=test_duration)
    
    # Wait a bit between tests
    time.sleep(2)
    
    # Step 3: Test optimized (batch processing)
    print("\n" + "-"*70)
    print("TEST 2: OPTIMIZED (Batch Processing)")
    print("-"*70)
    
    if not generate_test_traffic(duration=test_duration + 2, rate=traffic_rate):
        logger.error("Failed to generate traffic. Exiting.")
        return 1
    
    time.sleep(0.5)  # Let traffic generator start
    optimized_stats = test_batch_processing(duration=test_duration)
    
    # Step 4: Print results
    print_results(baseline_stats, optimized_stats)
    
    print("\n✅ Testing complete!\n")
    
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)
