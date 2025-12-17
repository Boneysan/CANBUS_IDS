#!/usr/bin/env python3
"""
Quick throughput test - measures actual message processing rate.
"""

import sys
import time
import subprocess
import signal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.capture.can_sniffer import CANSniffer
from src.detection.rule_engine import RuleEngine


def test_throughput(batch_size: int, duration: int = 10):
    """Test throughput with specific batch size."""
    
    print(f"\n{'='*70}")
    print(f"Testing with batch_size={batch_size}")
    print(f"{'='*70}")
    
    # Start traffic generator in background
    # Use very fast rate (gap=0 for maximum speed)
    print("Starting traffic generator (maximum speed)...")
    traffic_proc = subprocess.Popen(
        ['cangen', 'vcan0', '-g', '0', '-I', '123', '-L', '8'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    time.sleep(1)  # Give it more time to start generating
    
    # Initialize components
    sniffer = CANSniffer('vcan0')
    sniffer.start()
    
    rule_engine = RuleEngine('config/rules.yaml')
    
    # Run test
    messages_processed = 0
    batches_read = 0
    start_time = time.time()
    
    print(f"Running test for {duration} seconds...")
    
    try:
        while time.time() - start_time < duration:
            if batch_size == 1:
                # Simulate individual processing
                batch = sniffer.read_batch(batch_size=1, timeout=0.001)
                if batch:
                    for msg in batch:
                        alerts = rule_engine.analyze_message(msg)
                        messages_processed += 1
                        batches_read += 1
            else:
                # Batch processing
                batch = sniffer.read_batch(batch_size=batch_size, timeout=0.01)
                if batch:
                    alerts = rule_engine.analyze_batch(batch)
                    messages_processed += len(batch)
                    batches_read += 1
                    
    except KeyboardInterrupt:
        pass
    finally:
        elapsed = time.time() - start_time
        sniffer.stop()
        traffic_proc.terminate()
        traffic_proc.wait()
    
    # Results
    throughput = messages_processed / elapsed if elapsed > 0 else 0
    
    print(f"\nResults:")
    print(f"  Messages processed: {messages_processed:,}")
    print(f"  Batches read: {batches_read:,}")
    print(f"  Duration: {elapsed:.2f}s")
    print(f"  Throughput: {throughput:.0f} msg/s")
    
    if batch_size > 1 and batches_read > 0:
        avg_batch = messages_processed / batches_read
        print(f"  Avg batch size: {avg_batch:.1f}")
    
    return throughput


def main():
    print("\n" + "="*70)
    print("CAN-IDS BATCH PROCESSING THROUGHPUT TEST")
    print("="*70)
    
    # Test 1: Baseline (batch_size=1, simulating individual)
    baseline = test_throughput(batch_size=1, duration=10)
    
    time.sleep(2)
    
    # Test 2: Optimized (batch_size=100)
    optimized = test_throughput(batch_size=100, duration=10)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nBaseline (individual):  {baseline:>8.0f} msg/s")
    print(f"Optimized (batch=100):  {optimized:>8.0f} msg/s")
    
    if baseline > 0:
        improvement = optimized / baseline
        print(f"\n🚀 Performance gain:    {improvement:>8.2f}x")
        
        if improvement >= 5.0:
            print("\n✅ SUCCESS: Achieved 5x+ improvement target!")
        elif improvement >= 2.0:
            print("\n⚠️  PARTIAL: 2x+ improvement, but below 5x target")
        else:
            print("\n❌ BELOW TARGET: Less than 2x improvement")
    
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted")
        sys.exit(1)
