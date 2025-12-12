#!/usr/bin/env python3
"""
Test CAN-IDS detection rules on CSV datasets.

Directly processes CSV files without requiring PCAP format conversion.
"""

import sys
import pandas as pd
import yaml
from pathlib import Path
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.rule_engine import RuleEngine


def test_rules(csv_path: str, rules_path: str, verbose: bool = False):
    """Test rules on CSV dataset."""
    
    print(f"\n{'='*60}")
    print(f"Testing Rules on Dataset")
    print(f"{'='*60}")
    print(f"Dataset: {csv_path}")
    print(f"Rules: {rules_path}")
    
    # Load data
    print("\nLoading dataset...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df):,} messages")
    
    # Check for attack labels
    has_labels = 'attack' in df.columns
    if has_labels:
        attack_count = df['attack'].sum()
        normal_count = len(df) - attack_count
        print(f"  Normal: {normal_count:,} | Attacks: {attack_count:,}")
    
    # Initialize rule engine
    print("\nInitializing rule engine...")
    rule_engine = RuleEngine(rules_file=rules_path)
    print(f"  Loaded {len(rule_engine.rules)} rules")
    
    # Process messages
    print("\nProcessing messages...")
    alerts = []
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    for idx, row in df.iterrows():
        # Convert to CAN message format
        can_id_raw = row['arbitration_id']
        if isinstance(can_id_raw, str):
            can_id = int(can_id_raw, 16)
        else:
            can_id = int(can_id_raw)
        
        message = {
            'timestamp': row['timestamp'],
            'can_id': can_id,
            'data': bytes.fromhex(row['data_field']),
            'dlc': len(row['data_field']) // 2
        }
        
        # Analyze with rules
        result = rule_engine.analyze_message(message)
        is_attack = row.get('attack', 0) == 1
        detected = len(result) > 0  # result is a list of Alert objects
        
        if detected:
            alerts.append({
                'timestamp': message['timestamp'],
                'can_id': f"0x{can_id:03X}",
                'severity': result[0].severity if result else 'UNKNOWN',
                'rules_matched': [alert.rule_name for alert in result],
                'is_attack': is_attack
            })
        
        # Calculate metrics if labels available
        if has_labels:
            if is_attack and detected:
                true_positives += 1
            elif is_attack and not detected:
                false_negatives += 1
            elif not is_attack and detected:
                false_positives += 1
            elif not is_attack and not detected:
                true_negatives += 1
        
        if (idx + 1) % 10000 == 0:
            print(f"  Processed {idx + 1:,} messages...")
    
    print(f"  Completed: {len(df):,} messages processed")
    
    # Results
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Total Alerts: {len(alerts)}")
    
    if has_labels:
        print(f"\nConfusion Matrix:")
        print(f"  True Positives:  {true_positives:,} (attacks caught)")
        print(f"  False Positives: {false_positives:,} (normal flagged as attack)")
        print(f"  True Negatives:  {true_negatives:,} (normal correctly ignored)")
        print(f"  False Negatives: {false_negatives:,} (attacks missed)")
        
        # Metrics
        total = len(df)
        if attack_count > 0:
            recall = true_positives / attack_count * 100
            print(f"\n✓ Recall: {recall:.2f}% (caught {true_positives}/{attack_count} attacks)")
        
        if (true_positives + false_positives) > 0:
            precision = true_positives / (true_positives + false_positives) * 100
            print(f"✓ Precision: {precision:.2f}% ({false_positives:,} false alarms)")
        
        if normal_count > 0:
            fpr = false_positives / normal_count * 100
            print(f"✓ False Positive Rate: {fpr:.2f}% (flagged {false_positives}/{normal_count} normal messages)")
    
    # Alert breakdown
    if alerts and verbose:
        print(f"\nAlert Breakdown by Severity:")
        severity_counts = defaultdict(int)
        for alert in alerts:
            severity_counts[alert['severity']] += 1
        
        for severity, count in sorted(severity_counts.items()):
            print(f"  {severity}: {count:,}")
        
        print(f"\nFirst 10 Alerts:")
        for i, alert in enumerate(alerts[:10], 1):
            attack_flag = "⚠️ ATTACK" if alert['is_attack'] else "❌ FALSE POSITIVE"
            print(f"  {i}. {alert['timestamp']:.4f} | {alert['can_id']} | {alert['severity']} | {attack_flag}")
            if alert['rules_matched']:
                print(f"     Rules: {', '.join(alert['rules_matched'][:3])}")
    
    print(f"\n{'='*60}\n")
    
    return {
        'total_messages': len(df),
        'total_alerts': len(alerts),
        'true_positives': true_positives if has_labels else None,
        'false_positives': false_positives if has_labels else None,
        'recall': (true_positives / attack_count * 100) if has_labels and attack_count > 0 else None,
        'precision': (true_positives / (true_positives + false_positives) * 100) if has_labels and (true_positives + false_positives) > 0 else None,
        'fpr': (false_positives / normal_count * 100) if has_labels and normal_count > 0 else None
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test CAN-IDS rules on CSV dataset')
    parser.add_argument('dataset', help='CSV file with CAN messages')
    parser.add_argument('--rules', default='config/rules_generated.yaml', help='Rules file to test')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed alert breakdown')
    
    args = parser.parse_args()
    
    test_rules(args.dataset, args.rules, args.verbose)
