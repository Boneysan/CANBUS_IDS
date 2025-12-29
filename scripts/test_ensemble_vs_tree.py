#!/usr/bin/env python3
"""Compare ensemble detector vs decision tree on all attack types."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import joblib
import numpy as np
from src.detection.decision_tree_detector import DecisionTreeDetector

def extract_features(row, detector):
    """Extract features from CSV row."""
    message = {
        'can_id': int(row['arbitration_id'], 16) if isinstance(row['arbitration_id'], str) else int(row['arbitration_id']),
        'timestamp': float(row['timestamp']),
        'data': bytes.fromhex(row['data_field']) if isinstance(row['data_field'], str) else row['data_field'],
        'dlc': 8
    }
    return detector.extract_features(message)

def test_detector(detector, df, name):
    """Test detector on dataset."""
    predictions = []
    for idx, row in df.iterrows():
        features = extract_features(row, DecisionTreeDetector())
        
        if name == "Ensemble":
            # Ensemble expects dict format
            pred = detector.predict(features)
        else:
            # Decision tree expects array
            pred = detector.predict(features.reshape(1, -1))[0]
        
        predictions.append(1 if pred == -1 else 0)  # -1 = anomaly/attack
    
    return np.array(predictions)

def main():
    print("="*80)
    print("ENSEMBLE vs DECISION TREE COMPARISON")
    print("="*80)
    
    # Load models
    print("\nLoading models...")
    try:
        ensemble = joblib.load('data/models/ensemble_detector.joblib')
        print(f"✅ Loaded ensemble: {type(ensemble).__name__}")
    except Exception as e:
        print(f"❌ Failed to load ensemble: {e}")
        return
    
    tree_detector = DecisionTreeDetector(model_path=Path('data/models/decision_tree.pkl'))
    print(f"✅ Loaded decision tree")
    
    # Test datasets
    datasets = {
        'DoS-1': 'test_data/DoS-1.csv',
        'Fuzzing-1': 'test_data/fuzzing-1.csv',
        'Interval-1': 'test_data/interval-1.csv',
        'Attack-free-1': 'test_data/attack-free-1.csv'
    }
    
    results = []
    
    for name, path in datasets.items():
        if not Path(path).exists():
            print(f"\n⚠️  Skipping {name} - file not found")
            continue
        
        print(f"\n{'='*80}")
        print(f"Testing: {name}")
        print(f"{'='*80}")
        
        # Load data (sample for speed)
        df = pd.read_csv(path).head(5000)
        actual = df['attack'].values if 'attack' in df.columns else np.zeros(len(df))
        
        print(f"Samples: {len(df):,}")
        print(f"Attacks: {actual.sum():,} ({actual.mean()*100:.1f}%)")
        
        # Test ensemble
        print("\nTesting ensemble...")
        try:
            ensemble_preds = test_detector(ensemble, df, "Ensemble")
            ens_tp = ((ensemble_preds == 1) & (actual == 1)).sum()
            ens_fp = ((ensemble_preds == 1) & (actual == 0)).sum()
            ens_detection = ens_tp / actual.sum() * 100 if actual.sum() > 0 else 0
            ens_fp_rate = ens_fp / (actual == 0).sum() * 100 if (actual == 0).sum() > 0 else 0
            print(f"  Detection: {ens_detection:.1f}%")
            print(f"  FP Rate: {ens_fp_rate:.1f}%")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            ens_detection, ens_fp_rate = 0, 0
        
        # Test decision tree
        print("\nTesting decision tree...")
        tree_preds = test_detector(tree_detector, df, "Tree")
        tree_tp = ((tree_preds == 1) & (actual == 1)).sum()
        tree_fp = ((tree_preds == 1) & (actual == 0)).sum()
        tree_detection = tree_tp / actual.sum() * 100 if actual.sum() > 0 else 0
        tree_fp_rate = tree_fp / (actual == 0).sum() * 100 if (actual == 0).sum() > 0 else 0
        print(f"  Detection: {tree_detection:.1f}%")
        print(f"  FP Rate: {tree_fp_rate:.1f}%")
        
        results.append({
            'dataset': name,
            'ensemble_det': ens_detection,
            'ensemble_fp': ens_fp_rate,
            'tree_det': tree_detection,
            'tree_fp': tree_fp_rate
        })
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    print(f"{'Dataset':<15} | {'Ensemble Det':<12} | {'Tree Det':<12} | {'Ensemble FP':<12} | {'Tree FP':<12} | Winner")
    print("-"*90)
    for r in results:
        ens_better = r['ensemble_det'] > r['tree_det'] and r['ensemble_fp'] < r['tree_fp']
        winner = "Ensemble ✅" if ens_better else "Tree ✅"
        print(f"{r['dataset']:<15} | {r['ensemble_det']:>10.1f}% | {r['tree_det']:>10.1f}% | "
              f"{r['ensemble_fp']:>10.1f}% | {r['tree_fp']:>10.1f}% | {winner}")

if __name__ == "__main__":
    main()
