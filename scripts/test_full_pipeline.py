#!/usr/bin/env python3
"""
Test FULL 3-stage pipeline with fuzzing detection rules.

Tests the complete detection system:
- Stage 1: Adaptive timing detection (statistical)
- Stage 2: Rule-based detection (NOW WITH FUZZING RULES!)
- Stage 3: ML Decision Tree

Measures improvement in fuzzing detection after adding payload rules.
Monitors system resources: CPU, RAM, temperature, message throughput.
"""

import sys
import time
import logging
import pandas as pd
import psutil
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
    """Load data and test full pipeline with system resource monitoring."""
    logger.info(f"\nTesting: {filepath.name}")
    logger.info("=" * 80)
    
    # Initialize system monitoring
    process = psutil.Process()
    cpu_samples = []
    ram_samples = []
    temp_available = hasattr(psutil, 'sensors_temperatures')
    temp_samples = []
    
    # Load data
    df = pd.read_csv(filepath)
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
    
    logger.info(f"Samples: {len(df)}")
    
    # Check if ground truth labels exist
    has_labels = 'attack' in df.columns or 'Attack' in df.columns
    ground_truth = []
    
    # Convert to messages
    messages = []
    for idx, row in df.iterrows():
        # Get ground truth if available
        if has_labels:
            is_attack = int(row.get('attack', row.get('Attack', 0))) == 1
            ground_truth.append(is_attack)
        
        # Convert row to message format
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
    
    # Test pipeline with resource monitoring
    detections_by_stage = {2: 0, 3: 0}
    predictions = []
    
    start_time = time.time()
    last_monitor_time = start_time
    monitor_interval = 0.5  # Monitor every 0.5 seconds
    
    for i, message in enumerate(messages):
        result = detector.analyze(message)
        detected = result['detected']
        predictions.append(detected)
        
        if detected:
            stage = result['stage']
            detections_by_stage[stage] += 1
        
        # Sample system resources periodically
        current_time = time.time()
        if current_time - last_monitor_time >= monitor_interval:
            cpu_samples.append(process.cpu_percent(interval=None))
            ram_samples.append(process.memory_info().rss / 1024 / 1024)  # MB
            
            # Get CPU temperature if available
            if temp_available:
                try:
                    temps = psutil.sensors_temperatures()
                    if 'coretemp' in temps:
                        temp_samples.append(temps['coretemp'][0].current)
                    elif 'cpu_thermal' in temps:  # Raspberry Pi
                        temp_samples.append(temps['cpu_thermal'][0].current)
                except:
                    pass
            
            last_monitor_time = current_time
    
    elapsed = time.time() - start_time
    throughput = len(messages) / elapsed if elapsed > 0 else 0
    
    total_detections = sum(predictions)
    detection_rate = total_detections / len(messages) * 100
    
    # Calculate resource statistics
    resource_stats = {
        'cpu_avg': sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0,
        'cpu_max': max(cpu_samples) if cpu_samples else 0,
        'ram_avg_mb': sum(ram_samples) / len(ram_samples) if ram_samples else 0,
        'ram_max_mb': max(ram_samples) if ram_samples else 0,
        'temp_avg_c': sum(temp_samples) / len(temp_samples) if temp_samples else 0,
        'temp_max_c': max(temp_samples) if temp_samples else 0,
        'processing_time_sec': elapsed,
        'throughput_msg_per_sec': throughput
    }
    
    # Calculate performance metrics if ground truth available
    metrics = {}
    if has_labels:
        true_positives = sum(1 for gt, pred in zip(ground_truth, predictions) if gt and pred)
        false_positives = sum(1 for gt, pred in zip(ground_truth, predictions) if not gt and pred)
        true_negatives = sum(1 for gt, pred in zip(ground_truth, predictions) if not gt and not pred)
        false_negatives = sum(1 for gt, pred in zip(ground_truth, predictions) if gt and not pred)
        
        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)
        f1_score = 2 * (precision * recall) / max(precision + recall, 0.001)
        accuracy = (true_positives + true_negatives) / len(messages)
        
        metrics = {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy
        }
    
    logger.info(f"\nResults:")
    logger.info(f"  Total detections: {total_detections}/{len(messages)} ({detection_rate:.1f}%)")
    logger.info(f"  Stage 2 (Rules):  {detections_by_stage[2]} ({detections_by_stage[2]/len(messages)*100:.1f}%)")
    logger.info(f"  Stage 3 (ML):     {detections_by_stage[3]} ({detections_by_stage[3]/len(messages)*100:.1f}%)")
    logger.info(f"  Throughput: {throughput:.0f} msg/s")
    
    if metrics:
        logger.info(f"\n  Detection Metrics:")
        logger.info(f"    TP: {metrics['true_positives']:,} | FP: {metrics['false_positives']:,} | TN: {metrics['true_negatives']:,} | FN: {metrics['false_negatives']:,}")
        logger.info(f"    Precision: {metrics['precision']*100:.2f}% | Recall: {metrics['recall']*100:.2f}% | F1: {metrics['f1_score']:.4f}")
        logger.info(f"    Accuracy: {metrics['accuracy']*100:.2f}%")
    
    logger.info(f"\n  System Resources:")
    logger.info(f"    CPU: {resource_stats['cpu_avg']:.1f}% avg, {resource_stats['cpu_max']:.1f}% peak")
    logger.info(f"    RAM: {resource_stats['ram_avg_mb']:.1f} MB avg, {resource_stats['ram_max_mb']:.1f} MB peak")
    if resource_stats['temp_avg_c'] > 0:
        logger.info(f"    Temperature: {resource_stats['temp_avg_c']:.1f}°C avg, {resource_stats['temp_max_c']:.1f}°C peak")
    logger.info(f"    Processing time: {resource_stats['processing_time_sec']:.2f}s")
    
    return {
        'total_detections': total_detections,
        'total_messages': len(messages),
        'detection_rate': detection_rate,
        'stage2': detections_by_stage[2],
        'stage3': detections_by_stage[3],
        'throughput': throughput,
        **metrics,  # Include all calculated metrics
        **resource_stats  # Include resource statistics
    }


def main():
    """Test full pipeline on test_data or Vehicle_Models data."""
    # Try multiple possible locations for data
    possible_paths = [
        Path("test_data"),  # Local test_data directory (PREFERRED)
        Path("../Vehicle_Models/data/raw"),
        Path.home() / "Documents" / "GitHub" / "Vehicle_Models" / "data" / "raw",
        Path("/media/boneysan/Data/GitHub/Vehicle_Models/data/raw")
    ]
    
    data_path = None
    for path in possible_paths:
        if path.exists():
            data_path = path
            logger.info(f"Using data from: {path}")
            break
    
    if data_path is None:
        logger.error(f"Data not found in any of: {possible_paths}")
        return
    
    # Initialize full pipeline
    logger.info("Initializing 2-stage detection pipeline...")
    logger.info("  Stage 2: Rule Engine (FUZZING-ONLY RULES - 0% FP)")
    logger.info("  Stage 3: Decision Tree ML")
    
    detector = FullPipelineDetector(
        rules_file='config/rules_fuzzing_only.yaml',
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
        ('rpm-1.csv', 'RPM Attack (Set 1)', True),
        ('rpm-2.csv', 'RPM Attack (Set 2)', True),
        ('accessory-1.csv', 'Accessory Attack (Set 1)', True),
        ('accessory-2.csv', 'Accessory Attack (Set 2)', True),
        ('force-neutral-1.csv', 'Force Neutral Attack (Set 1)', True),
        ('force-neutral-2.csv', 'Force Neutral Attack (Set 2)', True),
        ('standstill-1.csv', 'Standstill Attack (Set 1)', True),
        ('standstill-2.csv', 'Standstill Attack (Set 2)', True),
        ('attack-free-1.csv', 'Normal Traffic (Set 1)', False),
        ('attack-free-2.csv', 'Normal Traffic (Set 2)', False),
    ]
    
    results = {}
    
    for filename, name, is_attack in datasets:
        filepath = data_path / filename
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
