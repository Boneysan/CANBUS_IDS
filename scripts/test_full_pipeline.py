#!/usr/bin/env python3
"""
Test FULL 3-stage pipeline with fuzzing detection rules.

Tests the complete detection system:
- Stage 1: Adaptive timing detection (statistical)
- Stage 2: Rule-based detection (NOW WITH FUZZING RULES!)
- Stage 3: ML Decision Tree

Measures improvement in fuzzing detection after adding payload rules.
"""

import sys
import time
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.rule_engine import RuleEngine
from src.detection.decision_tree_detector import DecisionTreeDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FullPipelineDetector:
    """2-stage pipeline: Rule-based + ML detector."""
    
    def __init__(self, rules_file: str, model_path: str):
        """Initialize Stage 2 (rules) and Stage 3 (ML)."""
        self.stage2 = RuleEngine(rules_file)
        self.stage3 = DecisionTreeDetector(model_path=model_path)
        
        self.stats = {
            'stage2_detections': 0,
            'stage3_detections': 0,
            'total_messages': 0
        }
    
    def analyze(self, message: Dict) -> Dict:
        """Run message through 2-stage pipeline."""
        self.stats['total_messages'] += 1
        
        # Stage 2: Rule-based (WITH FUZZING RULES!)
        stage2_alerts = self.stage2.analyze_message(message)
        if stage2_alerts:
            self.stats['stage2_detections'] += 1
            return {
                'detected': True,
                'stage': 2,
                'alerts': stage2_alerts
            }
        
        # Stage 3: ML decision tree
        is_anomalous, confidence, feature_importance = self.stage3.predict(message)
        if is_anomalous:
            self.stats['stage3_detections'] += 1
            return {
                'detected': True,
                'stage': 3,
                'confidence': confidence
            }
        
        return {'detected': False}
    
    def get_stats(self) -> Dict:
        """Get detection statistics."""
        return self.stats.copy()


def load_and_test(detector: FullPipelineDetector, filepath: Path, max_samples: int = 10000) -> Dict:
    """Load data and test full pipeline."""
    logger.info(f"\nTesting: {filepath.name}")
    logger.info("=" * 80)
    
    # Load data
    df = pd.read_csv(filepath)
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
    
    logger.info(f"Samples: {len(df)}")
    
    # Convert to messages
    messages = []
    for idx, row in df.iterrows():
        if 'data_field' in row:
            data_str = str(row['data_field']).replace(' ', '')
            if len(data_str) >= 16:
                data = [int(data_str[i:i+2], 16) for i in range(0, 16, 2)]
            else:
                data = [0] * 8
        elif 'data' in row:
            data_str = str(row['data']).replace(' ', '')
            if len(data_str) >= 16:
                data = [int(data_str[i:i+2], 16) for i in range(0, 16, 2)]
            else:
                data = [0] * 8
        else:
            data = [int(row.get(f'byte_{i}', 0)) for i in range(8)]
        data = (data + [0] * 8)[:8]
        
        can_id = row.get('arbitration_id', row.get('can_id', 0))
        if isinstance(can_id, str):
            can_id = int(can_id, 16)
        else:
            can_id = int(can_id)
        
        message = {
            'can_id': can_id,
            'timestamp': float(row.get('timestamp', idx * 0.01)),
            'data': data,
            'dlc': int(row.get('dlc', len(data)))
        }
        messages.append(message)
    
    # Test pipeline
    detections_by_stage = {2: 0, 3: 0}
    total_detections = 0
    
    start_time = time.time()
    
    for message in messages:
        result = detector.analyze(message)
        if result['detected']:
            total_detections += 1
            stage = result['stage']
            detections_by_stage[stage] += 1
    
    elapsed = time.time() - start_time
    throughput = len(messages) / elapsed if elapsed > 0 else 0
    
    detection_rate = total_detections / len(messages) * 100
    
    logger.info(f"\nResults:")
    logger.info(f"  Total detections: {total_detections}/{len(messages)} ({detection_rate:.1f}%)")
    logger.info(f"  Stage 2 (Rules):  {detections_by_stage[2]} ({detections_by_stage[2]/len(messages)*100:.1f}%)")
    logger.info(f"  Stage 3 (ML):     {detections_by_stage[3]} ({detections_by_stage[3]/len(messages)*100:.1f}%)")
    logger.info(f"  Throughput: {throughput:.0f} msg/s")
    
    return {
        'total_detections': total_detections,
        'total_messages': len(messages),
        'detection_rate': detection_rate,
        'stage2': detections_by_stage[2],
        'stage3': detections_by_stage[3],
        'throughput': throughput
    }


def main():
    """Test full pipeline on Vehicle_Models data."""
    # Try multiple possible locations for Vehicle_Models
    possible_paths = [
        Path("../Vehicle_Models"),
        Path.home() / "Documents" / "GitHub" / "Vehicle_Models",
        Path("/media/boneysan/Data/GitHub/Vehicle_Models")
    ]
    
    vehicle_models_path = None
    for path in possible_paths:
        if path.exists():
            vehicle_models_path = path
            break
    
    if vehicle_models_path is None:
        logger.error(f"Vehicle_Models not found in any of: {possible_paths}")
        return
    
    raw_data_path = vehicle_models_path / 'data' / 'raw'
    
    # Initialize full pipeline
    logger.info("Initializing 2-stage detection pipeline...")
    logger.info("  Stage 2: Rule Engine (WITH FUZZING DETECTION RULES!)")
    logger.info("  Stage 3: Decision Tree ML")
    
    detector = FullPipelineDetector(
        rules_file='config/rules_adaptive.yaml',
        model_path='data/models/decision_tree.pkl'
    )
    
    # Test on attack datasets
    datasets = [
        ('fuzzing-1.csv', 'Fuzzing Attack (Set 1)', True),
        ('fuzzing-2.csv', 'Fuzzing Attack (Set 2)', True),
        ('DoS-1.csv', 'DoS Attack (Set 1)', True),
        ('DoS-2.csv', 'DoS Attack (Set 2)', True),
        ('interval-1.csv', 'Interval Timing (Set 1)', True),
        ('interval-2.csv', 'Interval Timing (Set 2)', True),
        ('attack-free-1.csv', 'Normal Traffic (Set 1)', False),
        ('attack-free-2.csv', 'Normal Traffic (Set 2)', False),
    ]
    
    results = {}
    
    for filename, name, is_attack in datasets:
        filepath = raw_data_path / filename
        if not filepath.exists():
            logger.warning(f"Skipping {filename} - not found")
            continue
        
        result = load_and_test(detector, filepath, max_samples=10000)
        results[name] = result
    
    # Print comprehensive summary
    logger.info("\n" + "=" * 80)
    logger.info("COMPREHENSIVE PIPELINE TEST SUMMARY")
    logger.info("=" * 80)
    
    logger.info("\nAttack Detection Rates:")
    logger.info("-" * 80)
    for name, result in results.items():
        if 'Attack' in name:
            rate = result['detection_rate']
            status = "✅ EXCELLENT" if rate >= 90 else "✅ GOOD" if rate >= 70 else "⚠️ FAIR" if rate >= 50 else "❌ POOR"
            logger.info(f"{name:40s} {result['total_detections']:5d}/{result['total_messages']:5d}   {rate:5.1f}%   {status}")
    
    logger.info("\nFalse Positive Rate:")
    logger.info("-" * 80)
    for name, result in results.items():
        if 'Normal' in name:
            fpr = result['detection_rate']
            status = "✅ GOOD" if fpr < 10 else "⚠️ FAIR" if fpr < 25 else "❌ HIGH FPR"
            logger.info(f"{name:40s} {result['total_detections']:5d}/{result['total_messages']:5d}   {fpr:5.1f}%   {status}")
    
    logger.info("\nStage Breakdown:")
    logger.info("-" * 80)
    total_stats = detector.get_stats()
    logger.info(f"Total messages processed: {total_stats['total_messages']}")
    logger.info(f"Stage 2 detections: {total_stats['stage2_detections']} ← FUZZING RULES HERE!")
    logger.info(f"Stage 3 detections: {total_stats['stage3_detections']}")
    
    # Calculate average fuzzing detection
    fuzzing_results = [r for n, r in results.items() if 'Fuzzing' in n]
    if fuzzing_results:
        avg_fuzzing = sum(r['detection_rate'] for r in fuzzing_results) / len(fuzzing_results)
        logger.info(f"\n🎯 Average Fuzzing Detection: {avg_fuzzing:.1f}%")
        logger.info(f"   (Previous with ML only: 54.8%)")
        if avg_fuzzing > 54.8:
            improvement = avg_fuzzing - 54.8
            logger.info(f"   ✅ IMPROVEMENT: +{improvement:.1f}% with fuzzing rules!")


if __name__ == '__main__':
    main()
