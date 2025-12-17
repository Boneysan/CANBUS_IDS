#!/usr/bin/env python3
"""
Test pre-filter with REAL vehicle CAN data to measure actual performance gain.

Uses actual training data with real CAN IDs and timing patterns.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.detection.rule_engine import RuleEngine
from src.detection.prefilter import FastPreFilter


def test_prefilter_real_data():
    """Test pre-filter using real vehicle CAN data."""
    
    print("=" * 60)
    print("PRE-FILTER TEST: REAL VEHICLE DATA")
    print("=" * 60)
    
    # Load real data (use RAW data with proper timestamps)
    print("\n1. Loading real vehicle CAN data...")
    data_file = Path('../Vehicle_Models/data/raw/attack-free-1.csv')
    
    if not data_file.exists():
        print(f"❌ Data not found: {data_file}")
        print("   Make sure Vehicle_Models data is available")
        return
    
    df = pd.read_csv(data_file)
    print(f"   ✓ Loaded {len(df)} messages from attack-free-1.csv")
    
    # Extract CAN IDs (raw data uses hex strings)
    # Convert hex strings to integers
    can_ids = set()
    for hex_str in df['arbitration_id'].unique():
        can_id = int(hex_str, 16)  # Convert hex string to int
        can_ids.add(can_id)
    
    print(f"   ✓ Found {len(can_ids)} unique CAN IDs")
    print(f"   IDs: {sorted([hex(x) for x in list(can_ids)[:10]])}{'...' if len(can_ids) > 10 else ''}")
    
    # Convert to messages
    print("\n2. Preparing test messages...")
    messages = []
    
    for idx, row in df.head(10000).iterrows():  # Use 10K messages
        msg = {
            'can_id': int(row['arbitration_id'], 16),  # Convert hex to int
            'timestamp': float(row['timestamp']),  # Use real timestamps
            'dlc': 8,  # Standard CAN DLC
            'data': [0] * 8
        }
        messages.append(msg)
    
    print(f"   ✓ Prepared {len(messages)} test messages")
    
    # Initialize components
    print("\n3. Initializing components...")
    engine = RuleEngine('config/rules.yaml')
    prefilter = FastPreFilter(can_ids, 0.3)
    
    # Calibrate on first 1000 messages
    prefilter.calibrate(messages[:1000])
    print(f"   ✓ Pre-filter calibrated")
    
    # Test messages
    test_msgs = messages[1000:]
    print(f"\n4. Testing {len(test_msgs)} messages...")
    
    # WITHOUT pre-filter
    print("\n   WITHOUT pre-filter:")
    start = time.time()
    for msg in test_msgs:
        engine.analyze_message(msg)
    duration_without = time.time() - start
    throughput_without = len(test_msgs) / duration_without
    print(f"   Throughput: {throughput_without:,.0f} msg/s")
    
    # WITH pre-filter
    print("\n   WITH pre-filter:")
    start = time.time()
    passed_count = 0
    for msg in test_msgs:
        p, f = prefilter.filter_batch([msg])
        if p:
            passed_count += 1
        else:
            engine.analyze_message(msg)
    duration_with = time.time() - start
    throughput_with = len(test_msgs) / duration_with
    
    print(f"   Throughput: {throughput_with:,.0f} msg/s")
    print(f"   Filtered: {passed_count/len(test_msgs)*100:.1f}%")
    
    # Results
    improvement = throughput_with / throughput_without
    print(f"\n5. RESULTS:")
    print(f"   Improvement: {improvement:.2f}x")
    print(f"   Pass rate: {passed_count/len(test_msgs)*100:.1f}%")
    
    # Estimate full system
    baseline = 2715  # From batch test
    estimated = baseline * improvement
    print(f"\n6. ESTIMATED FULL SYSTEM:")
    print(f"   Batch processing: {baseline} msg/s")
    print(f"   + Pre-filter: {estimated:,.0f} msg/s")
    
    if estimated >= 7000:
        print(f"   ✅ TARGET MET!")
    
    # Output config
    print(f"\n7. CONFIG (add to config/can_ids.yaml):")
    print("known_good_ids:")
    for cid in sorted(list(can_ids)[:20]):
        print(f"  - 0x{cid:03X}")
    if len(can_ids) > 20:
        print(f"  # ... and {len(can_ids)-20} more")


if __name__ == '__main__':
    test_prefilter_real_data()
