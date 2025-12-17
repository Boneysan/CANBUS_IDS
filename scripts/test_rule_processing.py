#!/usr/bin/env python3
"""
Rule processing throughput test.
Tests analyze_message() vs analyze_batch() performance.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.rule_engine import RuleEngine


def generate_test_messages(count: int):
    """Generate test CAN messages."""
    messages = []
    for i in range(count):
        messages.append({
            'timestamp': time.time(),
            'can_id': 0x123 + (i % 10),  # Vary CAN IDs
            'dlc': 8,
            'data': [i & 0xFF] * 8,
            'is_extended_id': False,
            'is_error_frame': False,
            'is_remote_frame': False
        })
    return messages


def test_individual_processing(messages, rule_engine):
    """Test individual message processing."""
    
    print("\nIndividual Processing (analyze_message)")
    print("-" * 50)
    
    alerts_total = 0
    start_time = time.time()
    
    for msg in messages:
        alerts = rule_engine.analyze_message(msg)
        alerts_total += len(alerts)
    
    elapsed = time.time() - start_time
    throughput = len(messages) / elapsed if elapsed > 0 else 0
    
    print(f"  Messages: {len(messages):,}")
    print(f"  Duration: {elapsed:.3f}s")
    print(f"  Throughput: {throughput:.0f} msg/s")
    print(f"  Alerts: {alerts_total}")
    
    return throughput


def test_batch_processing(messages, rule_engine):
    """Test batch message processing."""
    
    print("\nBatch Processing (analyze_batch)")
    print("-" * 50)
    
    start_time = time.time()
    alerts = rule_engine.analyze_batch(messages)
    elapsed = time.time() - start_time
    
    throughput = len(messages) / elapsed if elapsed > 0 else 0
    
    print(f"  Messages: {len(messages):,}")
    print(f"  Duration: {elapsed:.3f}s")
    print(f"  Throughput: {throughput:.0f} msg/s")
    print(f"  Alerts: {len(alerts)}")
    
    return throughput


def main():
    print("\n" + "="*70)
    print("RULE PROCESSING THROUGHPUT TEST")
    print("(Tests analyze_message vs analyze_batch performance)")
    print("="*70)
    
    # Load rules
    rule_engine = RuleEngine('config/rules.yaml')
    print(f"\nLoaded {len(rule_engine.rules)} rules")
    
    # Generate test data
    message_count = 10000
    print(f"Generating {message_count:,} test messages...")
    messages = generate_test_messages(message_count)
    
    # Test individual processing
    baseline = test_individual_processing(messages, rule_engine)
    
    # Test batch processing
    optimized = test_batch_processing(messages, rule_engine)
    
    # Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nIndividual processing:  {baseline:>8.0f} msg/s")
    print(f"Batch processing:       {optimized:>8.0f} msg/s")
    
    if baseline > 0:
        improvement = optimized / baseline
        print(f"\n🚀 Speedup:            {improvement:>8.2f}x")
        
        if improvement >= 5.0:
            print("\n✅ SUCCESS: Achieved 5x+ improvement!")
        elif improvement >= 2.0:
            print("\n✅ GOOD: 2x+ improvement achieved")
        else:
            print("\n⚠️  Moderate improvement")
    
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
