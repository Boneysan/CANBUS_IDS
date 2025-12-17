#!/usr/bin/env python3
"""
Pure CAN reading throughput test - no rule processing.
This isolates the batch reading optimization.
"""

import sys
import time
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.capture.can_sniffer import CANSniffer


def test_reading(batch_size: int, duration: int = 10):
    """Test pure reading throughput."""
    
    print(f"\nTesting with batch_size={batch_size}")
    print("-" * 50)
    
    # Start traffic
    traffic_proc = subprocess.Popen(
        ['cangen', 'vcan0', '-g', '0', '-I', '123', '-L', '8'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    time.sleep(1)
    
    # Test reading
    sniffer = CANSniffer('vcan0')
    sniffer.start()
    
    messages_read = 0
    batches = 0
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration:
            batch = sniffer.read_batch(batch_size=batch_size, timeout=0.01)
            if batch:
                messages_read += len(batch)
                batches += 1
                
    finally:
        elapsed = time.time() - start_time
        sniffer.stop()
        traffic_proc.terminate()
        traffic_proc.wait()
    
    throughput = messages_read / elapsed if elapsed > 0 else 0
    
    print(f"  Messages: {messages_read:,}")
    print(f"  Duration: {elapsed:.2f}s")
    print(f"  Throughput: {throughput:.0f} msg/s")
    if batches > 0:
        print(f"  Batches: {batches:,} (avg size: {messages_read/batches:.1f})")
    
    return throughput


def main():
    print("\n" + "="*70)
    print("PURE CAN READING THROUGHPUT TEST")
    print("(No rule processing - tests batch reading optimization only)")
    print("="*70)
    
    # Test individual
    baseline = test_reading(batch_size=1, duration=10)
    
    time.sleep(2)
    
    # Test batch
    optimized = test_reading(batch_size=100, duration=10)
    
    # Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nIndividual reading (batch=1):  {baseline:>8.0f} msg/s")
    print(f"Batch reading (batch=100):     {optimized:>8.0f} msg/s")
    
    if baseline > 0:
        improvement = optimized / baseline
        print(f"\nReading speedup:               {improvement:>8.2f}x")
        
        if improvement >= 1.5:
            print("\n✅ Batch reading is faster!")
        else:
            print("\n⚠️  Similar performance - bottleneck is elsewhere")
    
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
