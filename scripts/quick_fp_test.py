#!/usr/bin/env python3
"""Quick false positive rate test on normal traffic."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.detection.rule_engine import RuleEngine

def test_fp_rate(data_file, rules_file, sample_size=10000):
    """Test false positive rate on normal traffic."""
    
    print(f"Testing FP rate on {sample_size} normal messages...")
    print(f"Data: {data_file}")
    print(f"Rules: {rules_file}")
    print()
    
    # Load rules
    engine = RuleEngine(str(rules_file))
    print(f"Loaded {len(engine.rules)} rules")
    print()
    
    # Load data
    df = pd.read_csv(data_file)
    # Filter for normal traffic only
    df_normal = df[df['attack'] == 0].head(sample_size)
    print(f"Testing on {len(df_normal):,} normal messages\n")
    
    # Test each message
    alerts = 0
    for idx, row in df_normal.iterrows():
        message = {
            'timestamp': float(row['timestamp']),
            'can_id': int(row['arbitration_id'], 16) if isinstance(row['arbitration_id'], str) else int(row['arbitration_id']),
            'data': bytes.fromhex(row['data_field']) if isinstance(row['data_field'], str) else row['data_field'],
            'dlc': 8
        }
        
        result = engine.analyze_message(message)
        if result:  # If any alerts returned
            alerts += 1
        
        if (idx + 1) % 1000 == 0:
            current_fp = (alerts / (idx + 1)) * 100
            print(f"  {idx + 1:,} messages: {alerts:,} alerts ({current_fp:.2f}% FP)")
    
    fp_rate = (alerts / len(df_normal)) * 100
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Total messages: {len(df_normal):,}")
    print(f"Alerts:         {alerts:,}")
    print(f"FP Rate:        {fp_rate:.2f}%")
    print(f"{'='*60}")
    
    return fp_rate

if __name__ == "__main__":
    data_file = "test_data/attack-free-1.csv"
    rules_file = "config/rules_adaptive.yaml"
    
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    if len(sys.argv) > 2:
        rules_file = sys.argv[2]
    
    test_fp_rate(data_file, rules_file, sample_size=50000)
