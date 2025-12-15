#!/usr/bin/env python3
"""
Test Decision Tree detector on real attack traffic from Vehicle_Models.

Tests against multiple attack types:
- Fuzzing attacks (fuzzing-1.csv, fuzzing-2.csv)
- DoS attacks (DoS-1.csv, DoS-2.csv)
- Interval timing attacks (interval-1.csv, interval-2.csv)
- Normal traffic (attack-free-1.csv, attack-free-2.csv)

Evaluates:
- Detection rates per attack type
- False positive rate on normal traffic
- Feature importance for each attack type
- Overall system performance
"""

import sys
import time
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.decision_tree_detector import DecisionTreeDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_can_data(filepath: Path, max_samples: int = 10000) -> pd.DataFrame:
    """Load CAN data from CSV file."""
    logger.info(f"Loading {filepath.name}...")
    
    try:
        # Read CSV
        df = pd.read_csv(filepath)
        
        # Sample if too large
        if len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42)
            logger.info(f"  Sampled {max_samples} from {len(df)} total messages")
        else:
            logger.info(f"  Loaded {len(df)} messages")
        
        return df
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return pd.DataFrame()


def convert_to_messages(df: pd.DataFrame) -> List[Dict]:
    """Convert DataFrame to CAN message dictionaries."""
    messages = []
    
    for idx, row in df.iterrows():
        # Parse data bytes
        if 'data' in row:
            # Data might be in hex string format
            data_str = str(row['data']).replace(' ', '')
            if len(data_str) >= 16:  # At least 8 bytes (16 hex chars)
                data = [int(data_str[i:i+2], 16) for i in range(0, min(16, len(data_str)), 2)]
            else:
                data = [0] * 8
        else:
            # Try byte_X columns
            data = []
            for i in range(8):
                col = f'byte_{i}'
                if col in row:
                    data.append(int(row[col]))
                else:
                    data.append(0)
        
        # Ensure 8 bytes
        data = (data + [0] * 8)[:8]
        
        # Get CAN ID (might be hex string)
        can_id = row.get('arbitration_id', row.get('can_id', 0))
        if isinstance(can_id, str):
            can_id = int(can_id, 16)  # Parse hex
        else:
            can_id = int(can_id)
        
        message = {
            'can_id': can_id,
            'timestamp': float(row.get('timestamp', idx * 0.01)),
            'data': data,
            'dlc': int(row.get('dlc', len(data)))
        }
        
        messages.append(message)
    
    return messages


def test_attack_type(detector: DecisionTreeDetector, 
                     data_path: Path,
                     attack_name: str,
                     is_attack: bool) -> Dict:
    """Test detector on specific attack type."""
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"Testing: {attack_name}")
    logger.info("=" * 80)
    
    # Load data
    df = load_can_data(data_path)
    if df.empty:
        return {}
    
    messages = convert_to_messages(df)
    logger.info(f"Converted {len(messages)} messages")
    
    # Test detection
    start_time = time.time()
    detections = []
    feature_importance_agg = defaultdict(float)
    
    for message in messages:
        alert = detector.analyze_message(message)
        detected = alert is not None
        detections.append(detected)
        
        if alert:
            # Aggregate feature importance
            for feature, importance in alert.feature_importance.items():
                feature_importance_agg[feature] += importance
    
    elapsed = time.time() - start_time
    
    # Calculate metrics
    num_detected = sum(detections)
    num_total = len(messages)
    detection_rate = (num_detected / num_total * 100) if num_total > 0 else 0
    throughput = num_total / elapsed if elapsed > 0 else 0
    latency = (elapsed / num_total * 1000) if num_total > 0 else 0
    
    # Normalize feature importance
    if feature_importance_agg:
        total_importance = sum(feature_importance_agg.values())
        for feature in feature_importance_agg:
            feature_importance_agg[feature] /= total_importance
    
    # Log results
    logger.info(f"")
    logger.info(f"Results:")
    logger.info(f"  Messages: {num_total}")
    logger.info(f"  Detected: {num_detected} ({detection_rate:.1f}%)")
    logger.info(f"  Time: {elapsed:.2f}s")
    logger.info(f"  Throughput: {throughput:,.0f} msg/s")
    logger.info(f"  Latency: {latency:.3f} ms/msg")
    
    if feature_importance_agg:
        logger.info(f"")
        logger.info(f"Top features for {attack_name}:")
        for feature, importance in sorted(
            feature_importance_agg.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]:
            logger.info(f"    {feature}: {importance:.3f}")
    
    # Evaluate performance
    if is_attack:
        if detection_rate >= 90:
            status = "✅ EXCELLENT"
        elif detection_rate >= 70:
            status = "✅ GOOD"
        elif detection_rate >= 50:
            status = "⚠️ FAIR"
        else:
            status = "❌ POOR"
        logger.info(f"")
        logger.info(f"{status} - Detection rate: {detection_rate:.1f}%")
    else:
        # For normal traffic, lower detection is better (FPR)
        fpr = detection_rate
        if fpr <= 5:
            status = "✅ EXCELLENT"
        elif fpr <= 10:
            status = "✅ GOOD"
        elif fpr <= 20:
            status = "⚠️ FAIR"
        else:
            status = "❌ HIGH FPR"
        logger.info(f"")
        logger.info(f"{status} - False positive rate: {fpr:.1f}%")
    
    return {
        'attack_name': attack_name,
        'is_attack': is_attack,
        'num_messages': num_total,
        'num_detected': num_detected,
        'detection_rate': detection_rate,
        'throughput': throughput,
        'latency': latency,
        'feature_importance': dict(feature_importance_agg),
        'status': status
    }


def main():
    """Run comprehensive attack testing."""
    logger.info("")
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║" + " " * 15 + "DECISION TREE - REAL ATTACK TRAFFIC TEST" + " " * 23 + "║")
    logger.info("╚" + "=" * 78 + "╝")
    logger.info("")
    
    # Load model
    model_path = Path("data/models/decision_tree.pkl")
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        logger.error("Run: python scripts/train_decision_tree.py --synthetic")
        return 1
    
    detector = DecisionTreeDetector(str(model_path))
    logger.info(f"✅ Model loaded: {model_path}")
    logger.info(f"   Tree depth: {detector.tree.get_depth()}")
    logger.info(f"   Tree leaves: {detector.tree.get_n_leaves()}")
    
    # Define test datasets
    vehicle_models_path = Path("../Vehicle_Models/data/raw")
    
    test_cases = [
        # Attack types
        (vehicle_models_path / "fuzzing-1.csv", "Fuzzing Attack (Set 1)", True),
        (vehicle_models_path / "fuzzing-2.csv", "Fuzzing Attack (Set 2)", True),
        (vehicle_models_path / "DoS-1.csv", "DoS Attack (Set 1)", True),
        (vehicle_models_path / "DoS-2.csv", "DoS Attack (Set 2)", True),
        (vehicle_models_path / "interval-1.csv", "Interval Timing Attack (Set 1)", True),
        (vehicle_models_path / "interval-2.csv", "Interval Timing Attack (Set 2)", True),
        
        # Normal traffic
        (vehicle_models_path / "attack-free-1.csv", "Normal Traffic (Set 1)", False),
        (vehicle_models_path / "attack-free-2.csv", "Normal Traffic (Set 2)", False),
    ]
    
    # Run tests
    results = []
    for data_path, attack_name, is_attack in test_cases:
        if not data_path.exists():
            logger.warning(f"Dataset not found: {data_path}")
            continue
        
        result = test_attack_type(detector, data_path, attack_name, is_attack)
        if result:
            results.append(result)
    
    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE TEST SUMMARY")
    logger.info("=" * 80)
    
    # Attack detection summary
    logger.info("")
    logger.info("Attack Detection Rates:")
    logger.info("-" * 80)
    logger.info(f"{'Attack Type':<40} {'Detected':<12} {'Rate':<10} {'Status':<15}")
    logger.info("-" * 80)
    
    attack_results = [r for r in results if r['is_attack']]
    for result in attack_results:
        logger.info(f"{result['attack_name']:<40} "
                   f"{result['num_detected']:>5}/{result['num_messages']:<5} "
                   f"{result['detection_rate']:>6.1f}%   "
                   f"{result['status']}")
    
    # Normal traffic FPR
    logger.info("")
    logger.info("False Positive Rate on Normal Traffic:")
    logger.info("-" * 80)
    logger.info(f"{'Traffic Type':<40} {'Flagged':<12} {'FPR':<10} {'Status':<15}")
    logger.info("-" * 80)
    
    normal_results = [r for r in results if not r['is_attack']]
    for result in normal_results:
        logger.info(f"{result['attack_name']:<40} "
                   f"{result['num_detected']:>5}/{result['num_messages']:<5} "
                   f"{result['detection_rate']:>6.1f}%   "
                   f"{result['status']}")
    
    # Overall statistics
    if attack_results:
        avg_detection = np.mean([r['detection_rate'] for r in attack_results])
        avg_throughput = np.mean([r['throughput'] for r in results])
        
        logger.info("")
        logger.info("Overall Statistics:")
        logger.info(f"  Average Attack Detection: {avg_detection:.1f}%")
        if normal_results:
            avg_fpr = np.mean([r['detection_rate'] for r in normal_results])
            logger.info(f"  Average False Positive Rate: {avg_fpr:.1f}%")
        logger.info(f"  Average Throughput: {avg_throughput:,.0f} msg/s")
    
    # Feature importance across attack types
    logger.info("")
    logger.info("Feature Importance by Attack Type:")
    logger.info("-" * 80)
    
    for result in attack_results:
        if result['feature_importance']:
            logger.info(f"\n{result['attack_name']}:")
            for feature, importance in sorted(
                result['feature_importance'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]:
                logger.info(f"  {feature:15s}: {importance:.3f}")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("TEST COMPLETE")
    logger.info("=" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
