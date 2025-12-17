#!/usr/bin/env python3
"""
End-to-end throughput test comparing overall system performance.
This shows the REAL throughput improvement from batch processing.
"""

import sys
import time
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.capture.can_sniffer import CANSniffer
from src.detection.rule_engine import RuleEngine


def test_system_throughput(use_batch: bool, duration: int = 10):
    """Test end-to-end system throughput."""
    
    mode = "BATCH PROCESSING" if use_batch else "INDIVIDUAL PROCESSING"
    print(f"\n{'='*70}")
    print(f"{mode}")
    print("="*70)
    
    # Start traffic generator
    print("Starting traffic generator...")
    traffic_proc = subprocess.Popen(
        ['cangen', 'vcan0', '-g', '0', '-I', '123', '-L', '8'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    time.sleep(1)
    
    # Initialize
    sniffer = CANSniffer('vcan0')
    sniffer.start()
    rule_engine = RuleEngine('config/rules.yaml')
    
    messages_processed = 0
    alerts_generated = 0
    start_time = time.time()
    
    print(f"Running for {duration} seconds...")
    
    try:
        if use_batch:
            # BATCH MODE: Read and process in batches
            while time.time() - start_time < duration:
                # Read batch
                batch = sniffer.read_batch(batch_size=100, timeout=0.01)
                
                if batch:
                    # Process batch
                    alerts = rule_engine.analyze_batch(batch)
                    
                    messages_processed += len(batch)
                    alerts_generated += len(alerts)
        else:
            # INDIVIDUAL MODE: Read batch but process one-by-one
            while time.time() - start_time < duration:
                # Still read in batches for fair comparison of reading
                batch = sniffer.read_batch(batch_size=100, timeout=0.01)
                
                if batch:
                    # Process individually
                    for msg in batch:
                        alerts = rule_engine.analyze_message(msg)
                        messages_processed += 1
                        alerts_generated += len(alerts)
                
    finally:
        elapsed = time.time() - start_time
        sniffer.stop()
        traffic_proc.terminate()
        traffic_proc.wait()
    
    throughput = messages_processed / elapsed if elapsed > 0 else 0
    
    print(f"\nResults:")
    print(f"  Messages processed: {messages_processed:,}")
    print(f"  Alerts generated: {alerts_generated:,}")
    print(f"  Duration: {elapsed:.2f}s")
    print(f"  Throughput: {throughput:.0f} msg/s")
    
    return throughput


def main():
    print("\n" + "="*70)
    print("END-TO-END SYSTEM THROUGHPUT TEST")
    print("Comparing individual vs batch message processing")
    print("="*70)
    
    # Test 1: Individual processing
    baseline = test_system_throughput(use_batch=False, duration=10)
    
    time.sleep(2)
    
    # Test 2: Batch processing  
    optimized = test_system_throughput(use_batch=True, duration=10)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nIndividual processing:  {baseline:>8.0f} msg/s")
    print(f"Batch processing:       {optimized:>8.0f} msg/s")
    
    if baseline > 0:
        improvement = optimized / baseline
        diff = optimized - baseline
        
        print(f"\n🚀 Performance change:  {improvement:>8.2f}x ({diff:+.0f} msg/s)")
        
        if improvement >= 1.5:
            print("\n✅ Batch processing is faster!")
        elif improvement >= 0.9:
            print("\n⚠️  Similar performance - implementation needs optimization")
        else:
            print("\n❌ Batch processing is slower - needs investigation")
    
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
