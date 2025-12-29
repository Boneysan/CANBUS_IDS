#!/usr/bin/env python3
"""Test 3-stage ensemble approach: Fuzzing Rules + Ensemble ML + Adaptive Rules."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import joblib
import numpy as np
from src.detection.rule_engine import RuleEngine
from src.detection.improved_detectors import EnsembleHybridDetector
import math
def preprocess_can_data(df):
    """Preprocess raw CAN data to add basic features needed for feature engineering."""
    df_processed = df.copy()
    
    # Convert arbitration_id to numeric
    df_processed['arb_id_numeric'] = df_processed['arbitration_id'].apply(
        lambda x: int(x, 16) if isinstance(x, str) else int(x)
    )
    
    # Extract data field length
    df_processed['data_length'] = df_processed['data_field'].str.len()
    
    # Convert timestamp if needed
    if not pd.api.types.is_datetime64_any_dtype(df_processed['timestamp']):
        df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'], unit='s')
    
    # Time-based features
    df_processed['hour'] = df_processed['timestamp'].dt.hour
    df_processed['minute'] = df_processed['timestamp'].dt.minute
    df_processed['second'] = df_processed['timestamp'].dt.second
    
    # Time delta (time between messages)
    df_processed = df_processed.sort_values('timestamp')
    df_processed['time_delta'] = df_processed['timestamp'].diff().dt.total_seconds()
    df_processed['time_delta'] = df_processed['time_delta'].fillna(0)
    
    # Count frequency of each CAN ID
    id_counts = df_processed.groupby('arbitration_id').size().to_dict()
    df_processed['id_frequency'] = df_processed['arbitration_id'].map(id_counts)
    
    # Statistical features per CAN ID
    df_processed['id_mean_time_delta'] = df_processed.groupby('arbitration_id')['time_delta'].transform('mean')
    df_processed['id_std_time_delta'] = df_processed.groupby('arbitration_id')['time_delta'].transform('std')
    
    # Fill NaN values
    df_processed = df_processed.fillna(0)
    
    return df_processed

def extract_ensemble_features(df):
    """Extract the specific features that the ensemble detector expects."""
    df_features = df.copy()
    
    # Basic features that should already be present
    required_features = [
        'arb_id_numeric', 'data_length', 'id_frequency', 'time_delta', 
        'id_mean_time_delta', 'id_std_time_delta', 'hour', 'minute', 'second'
    ]
    
    # Check if basic features exist
    missing_basic = [f for f in required_features if f not in df_features.columns]
    if missing_basic:
        raise ValueError(f"Missing basic features: {missing_basic}")
    
    # Add payload entropy
    def calculate_entropy(data_str):
        if not isinstance(data_str, str) or len(data_str) == 0:
            return 0
        try:
            data_bytes = bytes.fromhex(data_str)
            if len(data_bytes) == 0:
                return 0
            # Calculate Shannon entropy
            from collections import Counter
            counts = Counter(data_bytes)
            total = len(data_bytes)
            entropy = 0
            for count in counts.values():
                p = count / total
                entropy -= p * math.log2(p)
            return entropy
        except:
            return 0
    
    df_features['payload_entropy'] = df_features['data_field'].apply(calculate_entropy)
    
    # Add Hamming distance (simplified - distance from previous message)
    df_features['hamming_distance'] = 0  # Placeholder - would need previous message comparison
    
    # Add IAT z-score (standardized time_delta)
    df_features['iat_zscore'] = (df_features['time_delta'] - df_features['time_delta'].mean()) / (df_features['time_delta'].std() + 1e-6)
    
    # Add unknown bigram/trigram (simplified)
    df_features['unknown_bigram'] = 0  # Placeholder
    df_features['unknown_trigram'] = 0  # Placeholder
    
    # Add bit time features (simplified)
    df_features['bit_time_mean'] = df_features['time_delta'] * 8  # Rough estimate
    df_features['bit_time_rms'] = df_features['bit_time_mean']  # Simplified
    df_features['bit_time_energy'] = df_features['bit_time_mean'] ** 2  # Simplified
    
    # Select only the expected features
    expected_features = [
        'arb_id_numeric', 'data_length', 'id_frequency', 'time_delta', 
        'id_mean_time_delta', 'id_std_time_delta', 'hour', 'minute', 'second',
        'payload_entropy', 'hamming_distance', 'iat_zscore', 
        'unknown_bigram', 'unknown_trigram', 
        'bit_time_mean', 'bit_time_rms', 'bit_time_energy'
    ]
    
    # Fill any NaN values
    df_features = df_features.fillna(0)
    
    return df_features[expected_features]

def test_ensemble_hybrid(df, ensemble_detector, rule_engine, sample_size=10000):
    """Test 3-stage ensemble on dataset."""
    df_sample = df.head(sample_size)
    
    # Preprocess data to add basic features
    print("Preprocessing data...")
    df_processed = preprocess_can_data(df_sample)
    
    predictions = []
    rule_alerts = 0
    ensemble_alerts = 0

    # Extract features for the entire sample at once
    try:
        df_with_features = extract_ensemble_features(df_processed)
        print(f"✅ Extracted {len(df_with_features.columns)} features from {len(df_processed)} messages")
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        return np.zeros(len(df_sample)), 0, 0

    # Stage 1: Apply rules to all messages at once (where possible)
    rule_predictions = []
    for idx, row in df_sample.iterrows():
        message = {
            'can_id': int(row['arbitration_id'], 16) if isinstance(row['arbitration_id'], str) else int(row['arbitration_id']),
            'timestamp': float(row['timestamp']),
            'data': bytes.fromhex(row['data_field']) if isinstance(row['data_field'], str) else row['data_field'],
            'dlc': 8
        }

        rule_result = rule_engine.analyze_message(message)
        if rule_result:  # Rule alerts
            rule_predictions.append(1)
            rule_alerts += 1
        else:
            rule_predictions.append(0)
    
    # Stage 2: Apply ensemble to messages that passed rules
    ensemble_predictions = []
    ensemble_indices = []
    
    for i, (rule_pred, (idx, row)) in enumerate(zip(rule_predictions, df_sample.iterrows())):
        if rule_pred == 0:  # No rule alert, check ensemble
            ensemble_indices.append(i)
    
    if ensemble_indices:
        # Get features for messages that need ensemble checking
        ensemble_features = df_with_features.iloc[ensemble_indices]
        
        try:
            # Decision tree prediction
            ensemble_preds = ensemble_detector.predict(ensemble_features.values)  # Convert to numpy array
            ensemble_predictions = ensemble_preds.tolist()
            
            # Count ensemble alerts (decision tree returns 1 for attack, 0 for normal)
            ensemble_alerts = sum(1 for pred in ensemble_preds if pred == 1)
        except Exception as e:
            print(f"⚠️  Decision tree prediction failed: {e}")
            ensemble_predictions = [0] * len(ensemble_indices)
    else:
        ensemble_predictions = []
    
    # Combine predictions
    final_predictions = []
    rule_idx = 0
    ensemble_idx = 0
    
    for rule_pred in rule_predictions:
        if rule_pred == 1:
            final_predictions.append(1)  # Rule alert
        else:
            # Use ensemble prediction
            if ensemble_idx < len(ensemble_predictions):
                final_predictions.append(ensemble_predictions[ensemble_idx])
                ensemble_idx += 1
            else:
                final_predictions.append(0)  # Fallback
    
    return np.array(final_predictions), rule_alerts, ensemble_alerts

def main():
    print("="*80)
    print("3-STAGE ENSEMBLE TEST")
    print("Fuzzing Rules + Ensemble ML + Adaptive Rules")
    print("="*80)

    # Load components
    print("\nLoading components...")

    # Fuzzing rules (Stage 2)
    rule_engine = RuleEngine('config/rules_fuzzing_only.yaml')
    print(f"✅ Loaded fuzzing rules: {len(rule_engine.rules)} rules")

    # Ensemble ML (Stage 3) - Use decision tree instead of complex ensemble
    try:
        dt_dict = joblib.load('data/models/decision_tree.pkl')
        ensemble = dt_dict['tree']  # Extract the actual tree model
        print(f"✅ Loaded decision tree: {type(ensemble).__name__}")
    except Exception as e:
        print(f"❌ Failed to load decision tree: {e}")
        return

    # Test datasets
    datasets = {
        'DoS-1': ('test_data/DoS-1.csv', True),
        'Fuzzing-1': ('test_data/fuzzing-1.csv', True),
        'Interval-1': ('test_data/interval-1.csv', True),
        'Attack-free-1': ('test_data/attack-free-1.csv', False),
        'RPM-1': ('test_data/rpm-1.csv', True),
        'RPM-2': ('test_data/rpm-2.csv', True),
    }

    results = []

    for name, (path, is_attack) in datasets.items():
        if not Path(path).exists():
            print(f"\n⚠️  Skipping {name} - file not found")
            continue

        print(f"\n{'='*80}")
        print(f"Testing: {name}")
        print(f"{'='*80}")

        # Load data
        df = pd.read_csv(path)
        actual_full = df['attack'].values if 'attack' in df.columns else np.zeros(len(df))

        print(f"Samples: {len(df):,}")
        print(f"Attacks: {actual_full.sum():,} ({actual_full.mean()*100:.1f}%)")

        # Test 3-stage ensemble
        print("\nTesting 3-stage ensemble...")
        try:
            preds, rule_alerts, ensemble_alerts = test_ensemble_hybrid(df, ensemble, rule_engine, sample_size=1000)
            
            # Use actual labels for the sampled messages
            actual = actual_full[:len(preds)]

            tp = ((preds == 1) & (actual == 1)).sum()
            fp = ((preds == 1) & (actual == 0)).sum()
            detection = tp / actual.sum() * 100 if actual.sum() > 0 else 0
            fp_rate = fp / (actual == 0).sum() * 100 if (actual == 0).sum() > 0 else 0

            print(f"  Detection: {detection:.1f}%")
            print(f"  FP Rate: {fp_rate:.1f}%")
            print(f"  Stage 2 (Rules): {rule_alerts} alerts")
            print(f"  Stage 3 (Ensemble): {ensemble_alerts} alerts")

            results.append({
                'dataset': name,
                'detection': detection,
                'fp_rate': fp_rate,
                'rule_alerts': rule_alerts,
                'ensemble_alerts': ensemble_alerts,
                'is_attack': is_attack
            })

        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue

    # Summary
    print(f"\n{'='*80}")
    print("3-STAGE ENSEMBLE SUMMARY")
    print(f"{'='*80}\n")

    attack_results = [r for r in results if r['is_attack']]
    normal_results = [r for r in results if not r['is_attack']]

    if attack_results:
        avg_detection = np.mean([r['detection'] for r in attack_results])
        print(f"Average Attack Detection: {avg_detection:.1f}%")

    if normal_results:
        avg_fp = np.mean([r['fp_rate'] for r in normal_results])
        print(f"Average FP Rate: {avg_fp:.1f}%")

    print(f"\n{'Dataset':<15} | {'Detection':<10} | {'FP Rate':<10} | {'Rules':<8} | {'Ensemble':<10}")
    print("-"*70)
    for r in results:
        print(f"{r['dataset']:<15} | {r['detection']:>8.1f}% | {r['fp_rate']:>8.1f}% | {r['rule_alerts']:>6} | {r['ensemble_alerts']:>8}")

if __name__ == "__main__":
    main()