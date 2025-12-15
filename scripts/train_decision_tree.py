#!/usr/bin/env python3
"""
Train decision tree ML detector for Phase 3.

Loads training data from Vehicle_Models workspace, trains a DecisionTreeClassifier,
and saves the model for production use in the CAN-IDS system.

Expected Performance:
    - Throughput: 8,000+ msg/s
    - Training Time: ~10-15 minutes on full dataset
    - Model Size: ~2 MB
    - Accuracy: 85-88% on test set
"""

import sys
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Any
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from detection.decision_tree_detector import DecisionTreeDetector, SKLEARN_AVAILABLE

if SKLEARN_AVAILABLE:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_vehicle_models_data(vehicle_models_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load training data from Vehicle_Models workspace.
    
    Uses real CAN traffic CSVs:
    - Normal: attack-free-1.csv, attack-free-2.csv
    - Attacks: fuzzing-1.csv, DoS-1.csv, interval-1.csv
    
    Args:
        vehicle_models_path: Path to Vehicle_Models directory
        
    Returns:
        Tuple of (features, labels)
    """
    logger.info(f"Loading real CAN data from {vehicle_models_path}")
    
    raw_data_path = vehicle_models_path / 'data' / 'raw'
    
    if not raw_data_path.exists():
        logger.warning(f"Raw data path not found: {raw_data_path}")
        logger.warning("Falling back to synthetic data generation...")
        return generate_synthetic_data(num_normal=10000, num_attack=2000)
    
    # Load real normal traffic (attack-free)
    normal_files = [
        raw_data_path / 'attack-free-1.csv',
        raw_data_path / 'attack-free-2.csv'
    ]
    
    # Load real attack traffic (ALL sets for better coverage)
    attack_files = [
        raw_data_path / 'fuzzing-1.csv',
        raw_data_path / 'fuzzing-2.csv',
        raw_data_path / 'DoS-1.csv',
        raw_data_path / 'DoS-2.csv',
        raw_data_path / 'interval-1.csv',
        raw_data_path / 'interval-2.csv'
    ]
    
    # Initialize detector for feature extraction
    detector = DecisionTreeDetector()
    
    all_features = []
    all_labels = []
    
    # Load normal traffic
    for csv_file in normal_files:
        if not csv_file.exists():
            logger.warning(f"Normal traffic file not found: {csv_file.name}")
            continue
        
        logger.info(f"Loading normal traffic: {csv_file.name}")
        df = pd.read_csv(csv_file)
        
        # Sample to avoid imbalance (increased from 20k to 30k per file)
        sample_size = min(30000, len(df))
        df_sample = df.sample(n=sample_size, random_state=42)
        
        for idx, row in df_sample.iterrows():
            # Parse CAN message
            can_id = row.get('arbitration_id', row.get('can_id', 0))
            if isinstance(can_id, str):
                can_id = int(can_id, 16)
            else:
                can_id = int(can_id)
            
            # Parse data bytes
            if 'data' in row:
                data_str = str(row['data']).replace(' ', '')
                if len(data_str) >= 16:
                    data = [int(data_str[i:i+2], 16) for i in range(0, min(16, len(data_str)), 2)]
                else:
                    data = [0] * 8
            else:
                data = []
                for i in range(8):
                    col = f'byte_{i}'
                    if col in row:
                        data.append(int(row[col]))
                    else:
                        data.append(0)
            
            data = (data + [0] * 8)[:8]
            
            message = {
                'can_id': can_id,
                'timestamp': float(row.get('timestamp', idx * 0.01)),
                'data': data,
                'dlc': int(row.get('dlc', len(data)))
            }
            
            features = detector.extract_features(message)
            all_features.append(features)
            all_labels.append(0)  # Normal = 0
        
        logger.info(f"  Loaded {len(df_sample)} normal samples from {csv_file.name}")
    
    # Load attack traffic
    for csv_file in attack_files:
        if not csv_file.exists():
            logger.warning(f"Attack traffic file not found: {csv_file.name}")
            continue
        
        logger.info(f"Loading attack traffic: {csv_file.name}")
        df = pd.read_csv(csv_file)
        
        # Sample to balance with normal traffic (increased from 5k to 10k per attack file)
        sample_size = min(10000, len(df))
        df_sample = df.sample(n=sample_size, random_state=42)
        
        for idx, row in df_sample.iterrows():
            # Parse CAN message
            can_id = row.get('arbitration_id', row.get('can_id', 0))
            if isinstance(can_id, str):
                can_id = int(can_id, 16)
            else:
                can_id = int(can_id)
            
            # Parse data bytes
            if 'data' in row:
                data_str = str(row['data']).replace(' ', '')
                if len(data_str) >= 16:
                    data = [int(data_str[i:i+2], 16) for i in range(0, min(16, len(data_str)), 2)]
                else:
                    data = [0] * 8
            else:
                data = []
                for i in range(8):
                    col = f'byte_{i}'
                    if col in row:
                        data.append(int(row[col]))
                    else:
                        data.append(0)
            
            data = (data + [0] * 8)[:8]
            
            message = {
                'can_id': can_id,
                'timestamp': float(row.get('timestamp', idx * 0.01)),
                'data': data,
                'dlc': int(row.get('dlc', len(data)))
            }
            
            features = detector.extract_features(message)
            all_features.append(features)
            all_labels.append(1)  # Attack = 1
        
        logger.info(f"  Loaded {len(df_sample)} attack samples from {csv_file.name}")
    
    if not all_features:
        logger.warning("No real data loaded. Falling back to synthetic data...")
        return generate_synthetic_data(num_normal=10000, num_attack=2000)
    
    # Convert to numpy arrays
    features = np.array(all_features, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int32)
    
    logger.info(f"Total samples loaded: {len(features)}")
    logger.info(f"  Normal: {np.sum(labels == 0)}")
    logger.info(f"  Attacks: {np.sum(labels == 1)}")
    
    return features, labels


def generate_synthetic_data(num_normal: int = 10000, num_attack: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic CAN data for training.
    
    Args:
        num_normal: Number of normal samples
        num_attack: Number of attack samples
        
    Returns:
        Tuple of (features, labels)
    """
    logger.info(f"Generating {num_normal} normal + {num_attack} attack samples")
    
    # Initialize detector for feature extraction
    detector = DecisionTreeDetector()
    
    # Generate normal traffic
    normal_features = []
    for i in range(num_normal):
        # Simulate normal CAN message
        can_id = np.random.choice([0x100, 0x200, 0x300, 0x400, 0x500])
        timestamp = i * 0.01  # 10ms intervals
        data = list(np.random.randint(0, 256, 8))
        dlc = 8
        
        message = {
            'can_id': can_id,
            'timestamp': timestamp,
            'data': data,
            'dlc': dlc
        }
        
        features = detector.extract_features(message)
        normal_features.append(features)
    
    # Generate attack traffic (anomalies)
    attack_features = []
    for i in range(num_attack):
        # Simulate various attack patterns
        attack_type = np.random.choice(['fuzzing', 'replay', 'dos', 'timing'])
        
        if attack_type == 'fuzzing':
            # Random CAN ID and data
            can_id = np.random.randint(0x000, 0x7FF)
            data = list(np.random.randint(0, 256, 8))
        elif attack_type == 'replay':
            # Replayed message with high frequency
            can_id = np.random.choice([0x100, 0x200, 0x300])
            data = [0xFF] * 8  # Suspicious constant pattern
        elif attack_type == 'dos':
            # DoS attack with flooding
            can_id = 0x000
            data = [0x00] * 8
        else:  # timing attack
            # Abnormal timing interval
            can_id = np.random.choice([0x100, 0x200, 0x300])
            data = list(np.random.randint(0, 256, 8))
        
        timestamp = (num_normal + i) * 0.01
        dlc = 8
        
        message = {
            'can_id': can_id,
            'timestamp': timestamp,
            'data': data,
            'dlc': dlc
        }
        
        features = detector.extract_features(message)
        attack_features.append(features)
    
    # Combine and create labels
    all_features = np.array(normal_features + attack_features, dtype=np.float32)
    all_labels = np.array([0] * num_normal + [1] * num_attack, dtype=np.int32)
    
    # Shuffle
    indices = np.random.permutation(len(all_features))
    all_features = all_features[indices]
    all_labels = all_labels[indices]
    
    logger.info(f"Generated {len(all_features)} total samples")
    logger.info(f"  Normal: {np.sum(all_labels == 0)}")
    logger.info(f"  Attacks: {np.sum(all_labels == 1)}")
    
    return all_features, all_labels


def train_model(
    features: np.ndarray,
    labels: np.ndarray,
    max_depth: int = 10,
    min_samples_split: int = 20,
    min_samples_leaf: int = 10,
    class_weight: str = 'balanced',
    test_size: float = 0.2
) -> DecisionTreeDetector:
    """
    Train decision tree model.
    
    Args:
        features: Feature matrix
        labels: Label vector
        max_depth: Maximum tree depth
        min_samples_split: Minimum samples for split
        test_size: Test set proportion
        
    Returns:
        Trained DecisionTreeDetector
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels,
        test_size=test_size,
        random_state=42,
        stratify=labels
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Initialize and train detector
    detector = DecisionTreeDetector()
    
    # Prepare class weight
    cw = class_weight if class_weight != 'None' else None
    
    detector.train(
        X_train, y_train,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=cw
    )
    detector.train(X_train, y_train, max_depth=max_depth, min_samples_split=min_samples_split)
    
    # Evaluate on test set
    logger.info("\n" + "=" * 80)
    logger.info("Evaluating on test set...")
    logger.info("=" * 80)
    
    # Scale test features
    X_test_scaled = detector.scaler.transform(X_test)
    
    # Predictions
    y_pred = detector.tree.predict(X_test_scaled)
    y_proba = detector.tree.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    test_accuracy = np.mean(y_pred == y_test)
    logger.info(f"Test Accuracy: {test_accuracy:.2%}")
    
    # Classification report
    logger.info("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info("\nConfusion Matrix:")
    logger.info(f"                Predicted Normal  Predicted Attack")
    logger.info(f"Actual Normal   {cm[0][0]:15d}  {cm[0][1]:15d}")
    logger.info(f"Actual Attack   {cm[1][0]:15d}  {cm[1][1]:15d}")
    
    # ROC AUC
    if len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, y_proba)
        logger.info(f"\nROC AUC Score: {auc:.4f}")
    
    # Performance benchmark
    logger.info("\n" + "=" * 80)
    logger.info("Performance Benchmark")
    logger.info("=" * 80)
    
    start_time = time.time()
    num_predictions = 10000
    
    for i in range(num_predictions):
        _ = detector.tree.predict([X_test_scaled[i % len(X_test_scaled)]])
    
    elapsed = time.time() - start_time
    throughput = num_predictions / elapsed
    latency = (elapsed / num_predictions) * 1000  # ms
    
    logger.info(f"Predictions: {num_predictions}")
    logger.info(f"Time: {elapsed:.2f}s")
    logger.info(f"Throughput: {throughput:,.0f} msg/s")
    logger.info(f"Latency: {latency:.3f} ms per message")
    
    return detector


def main():
    parser = argparse.ArgumentParser(description='Train decision tree ML detector')
    parser.add_argument(
        '--vehicle-models',
        type=str,
        default='../Vehicle_Models',
        help='Path to Vehicle_Models workspace'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/models/decision_tree.pkl',
        help='Output path for trained model'
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        default=10,
        help='Maximum tree depth'
    )
    parser.add_argument(
        '--min-samples-split',
        type=int,
        default=20,
        help='Minimum samples for split (lower = more detailed tree)'
    )
    parser.add_argument(
        '--min-samples-leaf',
        type=int,
        default=10,
        help='Minimum samples per leaf (lower = more detailed tree)'
    )
    parser.add_argument(
        '--class-weight',
        type=str,
        default='balanced',
        help='Class weight strategy (balanced or None)'
    )
    parser.add_argument(
        '--tree-viz',
        type=str,
        default='data/models/decision_tree_rules.txt',
        help='Output path for tree visualization'
    )
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Force use of synthetic data'
    )
    
    args = parser.parse_args()
    
    if not SKLEARN_AVAILABLE:
        logger.error("scikit-learn not installed. Install with: pip install scikit-learn")
        return 1
    
    logger.info("=" * 80)
    logger.info("Decision Tree ML Detector Training")
    logger.info("=" * 80)
    logger.info(f"Max depth: {args.max_depth}")
    logger.info(f"Min samples split: {args.min_samples_split}")
    logger.info(f"Output model: {args.output}")
    logger.info(f"Tree visualization: {args.tree_viz}")
    logger.info("")
    
    # Load or generate data
    if args.synthetic:
        features, labels = generate_synthetic_data()
    else:
        vehicle_models_path = Path(args.vehicle_models).resolve()
        try:
            features, labels = load_vehicle_models_data(vehicle_models_path)
        except Exception as e:
            logger.warning(f"Could not load Vehicle_Models data: {e}")
            logger.info("Falling back to synthetic data generation")
            features, labels = generate_synthetic_data()
    
    # Train model
    detector = train_model(
        features, labels,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split
    )
    
    # Save model
    logger.info("\n" + "=" * 80)
    logger.info("Saving Model")
    logger.info("=" * 80)
    
    output_path = Path(args.output)
    detector.save_model(str(output_path))
    
    # Export tree visualization
    logger.info(f"Exporting tree visualization to {args.tree_viz}")
    detector.export_tree_visualization(args.tree_viz)
    
    logger.info("\n" + "=" * 80)
    logger.info("Training Complete!")
    logger.info("=" * 80)
    logger.info(f"Model saved to: {output_path.absolute()}")
    logger.info(f"Tree rules saved to: {Path(args.tree_viz).absolute()}")
    logger.info("\nNext steps:")
    logger.info("1. Review tree visualization for interpretability")
    logger.info("2. Integrate detector into main.py pipeline")
    logger.info("3. Test on live CAN traffic")
    logger.info("4. Validate 8,000+ msg/s throughput on Stage 3")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
