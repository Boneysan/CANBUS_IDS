#!/usr/bin/env python3
"""
Test Fast Pre-Filter performance and effectiveness.

This script measures:
1. Pre-filter processing speed
2. Pass/flag rates for normal vs attack traffic
3. Impact on overall system throughput
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import after path is set
from src.detection.prefilter import FastPreFilter


def test_prefilter_speed():
    """Test pre-filter processing speed."""
    print("=" * 60)
    print("TEST 1: Pre-Filter Processing Speed")
    print("=" * 60)
    
    # Create pre-filter with some known IDs
    known_ids = {0x100, 0x200, 0x200, 0x316, 0x324}
    prefilter = FastPreFilter(known_ids, 0.3)
    
    # Generate test messages (normal traffic)
    num_messages = 10000
    messages = []
    base_time = time.time()
    
    for i in range(num_messages):
        can_id = list(known_ids)[i % len(known_ids)]
        messages.append({
            'can_id': can_id,
            'timestamp': base_time + (i * 0.01),  # 10ms intervals
            'dlc': 8,
            'data': [0] * 8
        })
    
    print(f"Processing {num_messages} normal messages...")
    
    # Test batch processing
    start = time.time()
    passed, flagged = prefilter.filter_batch(messages)
    duration = time.time() - start
    
    print(f"\nResults:")
    print(f"  Total messages: {num_messages}")
    print(f"  Passed: {len(passed)} ({len(passed)/num_messages*100:.1f}%)")
    print(f"  Flagged: {len(flagged)} ({len(flagged)/num_messages*100:.1f}%)")
    print(f"  Processing time: {duration*1000:.2f} ms")
    print(f"  Throughput: {num_messages/duration:,.0f} msg/s")
    print(f"  Avg time per message: {duration*1000000/num_messages:.2f} μs")
    
    stats = prefilter.get_stats()
    print(f"\nPre-filter stats:")
    print(f"  Pass rate: {stats['pass_rate_percent']:.1f}%")
    print(f"  Avg time: {stats['avg_time_microseconds']:.2f} μs/msg")
    
    return prefilter, passed, flagged


def test_attack_detection():
    """Test pre-filter on attack traffic."""
    print("\n" + "=" * 60)
    print("TEST 2: Attack Traffic Detection")
    print("=" * 60)
    
    # Create pre-filter with known IDs
    known_ids = {0x100, 0x200, 0x300}
    prefilter = FastPreFilter(known_ids, 0.3)
    
    # Train on normal traffic first
    normal_messages = []
    base_time = time.time()
    for i in range(100):
        normal_messages.append({
            'can_id': 0x100,
            'timestamp': base_time + (i * 0.01),  # Regular 10ms intervals
            'dlc': 8,
            'data': [0] * 8
        })
    
    prefilter.calibrate(normal_messages)
    print("Pre-filter calibrated on normal traffic")
    
    # Test 1: Unknown CAN ID (should be flagged)
    unknown_msg = {
        'can_id': 0x999,  # Unknown ID
        'timestamp': time.time(),
        'dlc': 8,
        'data': [0xFF] * 8
    }
    
    passed, flagged = prefilter.filter_batch([unknown_msg])
    print(f"\nTest 1 - Unknown CAN ID (0x999):")
    print(f"  Result: {'FLAGGED ✓' if len(flagged) > 0 else 'PASSED ✗'}")
    
    # Test 2: Timing anomaly (too fast)
    fast_msg = {
        'can_id': 0x100,  # Known ID
        'timestamp': base_time + 100.001,  # Only 0.001s interval (expected ~0.01s)
        'dlc': 8,
        'data': [0] * 8
    }
    
    passed, flagged = prefilter.filter_batch([fast_msg])
    print(f"\nTest 2 - Timing Anomaly (too fast):")
    print(f"  Result: {'FLAGGED ✓' if len(flagged) > 0 else 'PASSED ✗'}")
    
    # Test 3: Normal message (should pass)
    normal_msg = {
        'can_id': 0x100,
        'timestamp': base_time + 101.01,  # Normal 10ms interval
        'dlc': 8,
        'data': [0] * 8
    }
    
    passed, flagged = prefilter.filter_batch([normal_msg])
    print(f"\nTest 3 - Normal Message:")
    print(f"  Result: {'PASSED ✓' if len(passed) > 0 else 'FLAGGED ✗'}")


def test_with_real_traffic():
    """Test pre-filter with realistic mixed traffic."""
    print("\n" + "=" * 60)
    print("TEST 3: Realistic Mixed Traffic")
    print("=" * 60)
    
    known_ids = {0x100, 0x200, 0x300, 0x316, 0x324}
    prefilter = FastPreFilter(known_ids, 0.3)
    
    # Generate mixed traffic
    messages = []
    base_time = time.time()
    
    # 90% normal traffic
    for i in range(9000):
        can_id = list(known_ids)[i % len(known_ids)]
        messages.append({
            'can_id': can_id,
            'timestamp': base_time + (i * 0.01),
            'dlc': 8,
            'data': [0] * 8
        })
    
    # 10% attack traffic (unknown IDs)
    for i in range(1000):
        messages.append({
            'can_id': 0x700 + (i % 100),  # Unknown IDs
            'timestamp': base_time + (9000 + i) * 0.01,
            'dlc': 8,
            'data': [0xFF] * 8
        })
    
    print(f"Processing {len(messages)} messages (90% normal, 10% attacks)...")
    
    start = time.time()
    passed, flagged = prefilter.filter_batch(messages)
    duration = time.time() - start
    
    print(f"\nResults:")
    print(f"  Total messages: {len(messages)}")
    print(f"  Passed (benign): {len(passed)} ({len(passed)/len(messages)*100:.1f}%)")
    print(f"  Flagged (suspicious): {len(flagged)} ({len(flagged)/len(messages)*100:.1f}%)")
    print(f"  Processing time: {duration*1000:.2f} ms")
    print(f"  Throughput: {len(messages)/duration:,.0f} msg/s")
    
    # Check detection accuracy
    expected_flagged = 1000  # We added 1000 attack messages
    detection_rate = (len(flagged) / expected_flagged) * 100
    print(f"\nDetection accuracy:")
    print(f"  Expected to flag: {expected_flagged}")
    print(f"  Actually flagged: {len(flagged)}")
    print(f"  Detection rate: {detection_rate:.1f}%")


def main():
    """Run all pre-filter tests."""
    print("\n" + "=" * 60)
    print("FAST PRE-FILTER PERFORMANCE TEST")
    print("=" * 60)
    
    # Run tests
    test_prefilter_speed()
    test_attack_detection()
    test_with_real_traffic()
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS COMPLETE")
    print("=" * 60)
    
    print("\nExpected performance with pre-filter:")
    print("  - 80-95% of normal traffic filtered out")
    print("  - <0.1ms per message processing time")
    print("  - 2-3x overall system throughput improvement")
    print("  - Combined with batch processing: 6,000-8,000 msg/s target")


if __name__ == '__main__':
    main()
