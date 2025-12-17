#!/usr/bin/env python3
"""
Test end-to-end system performance WITH Fast Pre-Filter enabled.

This measures the combined impact of:
1. Batch Processing (Milestone 1.1) - 3.8x gain
2. Fast Pre-Filter (Milestone 1.2) - Expected 2-3x additional gain

Target: 6,000-8,000 msg/s (to exceed 7K target)
"""

import sys
import time
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.capture.can_sniffer import CANSniffer
from src.detection.rule_engine import RuleEngine
from src.detection.prefilter import FastPreFilter


def test_with_prefilter():
    """Test system throughput with pre-filter enabled."""
    print("=" * 60)
    print("END-TO-END TEST: BATCH + PRE-FILTER")
    print("=" * 60)
    
    # Initialize components
    print("\n1. Initializing components...")
    sniffer = CANSniffer('vcan0')
    engine = RuleEngine('config/rules.yaml')
    
    # Extract known IDs from rules
    known_ids = set()
    for rule in engine.rules:
        if rule.can_id:
            known_ids.add(rule.can_id)
    
    print(f"   Known good IDs: {len(known_ids)}")
    
    prefilter = FastPreFilter(known_ids, 0.3)
    
    print("   ✓ CAN Sniffer initialized")
    print("   ✓ Rule Engine initialized")
    print("   ✓ Pre-Filter initialized")
    
    # Start traffic generator
    print("\n2. Starting CAN traffic generator...")
    traffic_proc = subprocess.Popen(
        ['cangen', 'vcan0', '-g', '0'],  # Maximum speed
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(0.5)  # Let traffic start
    print("   ✓ Traffic generator running")
    
    try:
        # Start sniffer
        sniffer.start()
        
        # Test parameters
        test_duration = 10
        batch_size = 100
        timeout = 0.1
        
        messages_processed = 0
        batches_read = 0
        messages_prefiltered = 0
        messages_analyzed = 0
        alerts_generated = 0
        
        print(f"\n3. Processing messages for {test_duration} seconds...")
        print("   (Using Batch Processing + Pre-Filter)")
        
        start_time = time.time()
        
        while time.time() - start_time < test_duration:
            # Read batch (Milestone 1.1)
            batch = sniffer.read_batch(batch_size, timeout)
            
            if not batch:
                continue
            
            batches_read += 1
            messages_processed += len(batch)
            
            # Pre-filter (Milestone 1.2)
            passed, suspicious = prefilter.filter_batch(batch)
            messages_prefiltered += len(passed)
            messages_analyzed += len(suspicious)
            
            # Only analyze suspicious messages
            if suspicious:
                alerts = engine.analyze_batch(suspicious)
                alerts_generated += len(alerts)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Stop sniffer
        sniffer.stop()
        
        # Calculate results
        throughput = messages_processed / duration
        prefilter_rate = (messages_prefiltered / messages_processed * 100) if messages_processed > 0 else 0
        
        print(f"\n4. Results:")
        print(f"   Duration: {duration:.2f} seconds")
        print(f"   Batches read: {batches_read}")
        print(f"   Messages processed: {messages_processed}")
        print(f"   Pre-filtered (passed): {messages_prefiltered} ({prefilter_rate:.1f}%)")
        print(f"   Analyzed (suspicious): {messages_analyzed} ({100-prefilter_rate:.1f}%)")
        print(f"   Alerts generated: {alerts_generated}")
        print(f"\n   📊 THROUGHPUT: {throughput:,.0f} msg/s")
        
        # Compare to baseline
        baseline = 708  # From DoS-1 test
        improvement = throughput / baseline
        
        print(f"\n5. Performance Comparison:")
        print(f"   Baseline (original): {baseline} msg/s")
        print(f"   With optimizations: {throughput:,.0f} msg/s")
        print(f"   Improvement: {improvement:.1f}x")
        
        # Target assessment
        target = 7000
        if throughput >= target:
            print(f"\n   ✅ TARGET MET! ({throughput:,.0f} >= {target:,} msg/s)")
        else:
            gap = target - throughput
            print(f"\n   ⚠️  Need {gap:,.0f} more msg/s to reach {target:,} target")
        
        # Get pre-filter stats
        prefilter_stats = prefilter.get_stats()
        print(f"\n6. Pre-Filter Statistics:")
        print(f"   Messages processed: {prefilter_stats['messages_processed']}")
        print(f"   Pass rate: {prefilter_stats['pass_rate_percent']:.1f}%")
        print(f"   Avg time: {prefilter_stats['avg_time_microseconds']:.2f} μs/msg")
        
    finally:
        # Stop traffic generator
        print("\n7. Cleaning up...")
        traffic_proc.terminate()
        traffic_proc.wait()
        print("   ✓ Traffic generator stopped")


def main():
    """Run the test."""
    print("\n" + "=" * 60)
    print("MILESTONE 1.2: FAST PRE-FILTER TEST")
    print("Testing: Batch Processing + Pre-Filter")
    print("Target: 7,000 msg/s")
    print("=" * 60)
    
    try:
        test_with_prefilter()
        
        print("\n" + "=" * 60)
        print("✅ TEST COMPLETE")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
