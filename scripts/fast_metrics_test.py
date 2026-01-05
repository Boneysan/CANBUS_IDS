#!/usr/bin/env python3
"""
Fast metrics test for current CAN-IDS detection system.
Tests enhanced hybrid approach (fuzzing rules + decision tree) on all attack types.
Captures detection metrics (TP, FP, TN, FN, Precision, Recall, F1).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import time
import json
from datetime import datetime
from src.detection.rule_engine import RuleEngine
from src.detection.decision_tree_detector import DecisionTreeDetector

def test_dataset(data_file, rule_engine, ml_detector, sample_size=10000):
    """
    Test detection on a dataset and return metrics.
    
    Returns dict with TP, FP, TN, FN, precision, recall, F1, etc.
    """
    
    # Load data
    df = pd.read_csv(data_file)
    
    # Sample if needed
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    print(f"  Testing {len(df):,} messages...")
    
    # Metrics
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    stage2_detections = 0  # Fuzzing rules
    stage3_detections = 0  # ML
    
    start_time = time.time()
    
    # Process each message
    for idx, row in df.iterrows():
        # Ground truth
        is_attack = int(row.get('attack', row.get('Attack', 0))) == 1
        
        # Convert to message format
        message = {
            'timestamp': float(row['timestamp']),
            'can_id': int(row['arbitration_id'], 16) if isinstance(row['arbitration_id'], str) else int(row['arbitration_id']),
            'data': bytes.fromhex(row['data_field']) if isinstance(row['data_field'], str) else row['data_field'],
            'dlc': 8
        }
        
        # Stage 2: Fuzzing rules (0% FP)
        rule_result = rule_engine.analyze_message(message)
        detected_by_rules = bool(rule_result)
        
        # Stage 3: ML detector
        ml_result = None
        detected_by_ml = False
        if not detected_by_rules and ml_detector:
            ml_result = ml_detector.analyze_message(message)
            detected_by_ml = bool(ml_result)
        
        # Combined detection
        detected = detected_by_rules or detected_by_ml
        
        # Update metrics
        if is_attack:
            if detected:
                true_positives += 1
                if detected_by_rules:
                    stage2_detections += 1
                elif detected_by_ml:
                    stage3_detections += 1
            else:
                false_negatives += 1
        else:
            if detected:
                false_positives += 1
                if detected_by_rules:
                    stage2_detections += 1
                elif detected_by_ml:
                    stage3_detections += 1
            else:
                true_negatives += 1
    
    duration = time.time() - start_time
    throughput = len(df) / duration if duration > 0 else 0
    
    # Calculate metrics
    total = true_positives + false_positives + true_negatives + false_negatives
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    f1_score = 2 * (precision * recall) / max(precision + recall, 0.001)
    accuracy = (true_positives + true_negatives) / max(total, 1)
    
    # Detection rate and FP rate
    total_attacks = true_positives + false_negatives
    total_normal = true_negatives + false_positives
    detection_rate = recall
    fp_rate = false_positives / max(total_normal, 1)
    
    return {
        'messages_tested': len(df),
        'duration_seconds': duration,
        'throughput_msg_per_sec': throughput,
        
        'true_positives': true_positives,
        'false_positives': false_positives,
        'true_negatives': true_negatives,
        'false_negatives': false_negatives,
        
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy,
        
        'detection_rate': detection_rate,
        'false_positive_rate': fp_rate,
        
        'total_attacks': total_attacks,
        'total_normal': total_normal,
        
        'stage2_detections': stage2_detections,
        'stage3_detections': stage3_detections,
    }


def main():
    """Run fast metrics test on all available datasets."""
    
    print("="*80)
    print("CAN-IDS ENHANCED HYBRID DETECTION - FAST METRICS TEST")
    print("="*80)
    print(f"Test Date: {datetime.now().isoformat()}")
    print(f"Configuration: Fuzzing Rules + Decision Tree ML")
    print()
    
    # Initialize detection components
    print("Loading detection components...")
    rule_engine = RuleEngine('config/rules_fuzzing_only.yaml')
    print(f"  ✅ Fuzzing rules loaded: {len(rule_engine.rules)} rules")
    
    ml_detector = None
    try:
        ml_detector = DecisionTreeDetector()
        print(f"  ✅ Decision tree loaded: {ml_detector.model_info}")
    except Exception as e:
        print(f"  ⚠️  ML detector failed to load: {e}")
        print(f"     Testing with rules only")
    
    print()
    
    # Test datasets
    datasets = [
        ('DoS-1', 'test_data/DoS-1.csv'),
        ('DoS-2', 'test_data/DoS-2.csv'),
        ('rpm-1', 'test_data/rpm-1.csv'),
        ('rpm-2', 'test_data/rpm-2.csv'),
        ('accessory-1', 'test_data/accessory-1.csv'),
        ('accessory-2', 'test_data/accessory-2.csv'),
        ('force-neutral-1', 'test_data/force-neutral-1.csv'),
        ('force-neutral-2', 'test_data/force-neutral-2.csv'),
        ('standstill-1', 'test_data/standstill-1.csv'),
        ('standstill-2', 'test_data/standstill-2.csv'),
        ('attack-free-1', 'test_data/attack-free-1.csv'),
        ('attack-free-2', 'test_data/attack-free-2.csv'),
    ]
    
    results = {}
    
    for name, filepath in datasets:
        filepath = Path(filepath)
        
        if not filepath.exists():
            print(f"⚠️  Skipping {name}: File not found")
            continue
        
        print(f"\n{'='*80}")
        print(f"Testing: {name}")
        print(f"File: {filepath}")
        print(f"{'='*80}")
        
        try:
            metrics = test_dataset(filepath, rule_engine, ml_detector, sample_size=10000)
            results[name] = metrics
            
            # Display results
            print(f"\n  Results:")
            print(f"    Messages tested:   {metrics['messages_tested']:,}")
            print(f"    Throughput:        {metrics['throughput_msg_per_sec']:.1f} msg/s")
            print(f"    Duration:          {metrics['duration_seconds']:.1f}s")
            print()
            print(f"  Detection Metrics:")
            print(f"    Detection Rate:    {metrics['detection_rate']*100:.2f}% ({metrics['true_positives']}/{metrics['total_attacks']})")
            print(f"    False Positive:    {metrics['false_positive_rate']*100:.2f}% ({metrics['false_positives']}/{metrics['total_normal']})")
            print(f"    Precision:         {metrics['precision']*100:.2f}%")
            print(f"    Recall:            {metrics['recall']*100:.2f}%")
            print(f"    F1-Score:          {metrics['f1_score']:.4f}")
            print(f"    Accuracy:          {metrics['accuracy']*100:.2f}%")
            print()
            print(f"  Confusion Matrix:")
            print(f"    TP: {metrics['true_positives']:,} | FP: {metrics['false_positives']:,}")
            print(f"    FN: {metrics['false_negatives']:,} | TN: {metrics['true_negatives']:,}")
            print()
            print(f"  Detection Stages:")
            print(f"    Stage 2 (Rules):   {metrics['stage2_detections']:,}")
            print(f"    Stage 3 (ML):      {metrics['stage3_detections']:,}")
            
        except Exception as e:
            print(f"  ❌ Error testing {name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results to JSON
    output_dir = Path('test_results/fast_metrics')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'metrics_{timestamp}.json'
    
    summary = {
        'test_info': {
            'test_date': datetime.now().isoformat(),
            'test_type': 'fast_metrics_test',
            'configuration': 'Enhanced Hybrid (Fuzzing Rules + Decision Tree)',
            'rules_file': 'config/rules_fuzzing_only.yaml',
            'ml_model': 'data/models/decision_tree.pkl',
            'sample_size': 10000,
        },
        'results': results,
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Datasets tested: {len(results)}/{len(datasets)}")
    print(f"Results saved to: {output_file}")
    print()
    
    # Summary table
    print(f"\n{'Dataset':<20} {'Detection':>10} {'FP Rate':>10} {'Precision':>10} {'Recall':>10} {'F1':>8}")
    print(f"{'-'*80}")
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['detection_rate']*100:>9.1f}% {metrics['false_positive_rate']*100:>9.1f}% "
              f"{metrics['precision']*100:>9.1f}% {metrics['recall']*100:>9.1f}% {metrics['f1_score']:>8.4f}")
    
    print(f"\n✅ Fast metrics test complete!")
    return summary


if __name__ == '__main__':
    main()
