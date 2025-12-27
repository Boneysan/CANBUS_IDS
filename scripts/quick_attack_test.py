#!/usr/bin/env python3
"""Quick attack detection test."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.detection.rule_engine import RuleEngine

def test_attack_detection(data_file, rules_file, sample_size=10000):
    """Test attack detection rate."""
    
    print(f"Testing attack detection on {sample_size} messages...")
    print(f"Data: {data_file}")
    print(f"Rules: {rules_file}")
    print()
    
    # Load rules
    engine = RuleEngine(str(rules_file))
    print(f"Loaded {len(engine.rules)} rules")
    print()
    
    # Load data
    df = pd.read_csv(data_file)
    # Filter for attack traffic only
    df_attacks = df[df['attack'] == 1].head(sample_size)
    print(f"Testing on {len(df_attacks):,} attack messages\n")
    
    # Test each message
    detected = 0
    for idx, row in df_attacks.iterrows():
        message = {
            'timestamp': float(row['timestamp']),
            'can_id': int(row['arbitration_id'], 16) if isinstance(row['arbitration_id'], str) else int(row['arbitration_id']),
            'data': bytes.fromhex(row['data_field']) if isinstance(row['data_field'], str) else row['data_field'],
            'dlc': 8
        }
        
        result = engine.analyze_message(message)
        if result:  # If any alerts returned
            detected += 1
        
        if (idx + 1) % 1000 == 0:
            current_recall = (detected / (idx + 1)) * 100
            print(f"  {idx + 1:,} messages: {detected:,} detected ({current_recall:.2f}% recall)")
    
    recall = (detected / len(df_attacks)) * 100
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Total attacks: {len(df_attacks):,}")
    print(f"Detected:      {detected:,}")
    print(f"Recall:        {recall:.2f}%")
    print(f"Missed:        {len(df_attacks) - detected:,}")
    print(f"{'='*60}")
    
    return recall

if __name__ == "__main__":
    data_file = "test_data/DoS-1.csv"
    rules_file = "config/rules_adaptive.yaml"
    
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    if len(sys.argv) > 2:
        rules_file = sys.argv[2]
    
    test_attack_detection(data_file, rules_file, sample_size=10000)
