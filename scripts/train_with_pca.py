#!/usr/bin/env python3
"""
Train ML model with PCA feature reduction for Raspberry Pi 4 deployment.

Research basis: IJRASET 2025 - "Intrusion Detection System using Raspberry Pi for IoT Devices"
                Reduces 58 features to 10-15 using PCA for 3-5x faster inference.

Expected performance improvement on Pi 4:
- Current: 17.31 msg/s (57.7ms per message) with 58 features
- Target: 50-85 msg/s (15-20ms per message) with 10-15 features
- Speedup: 3-5x faster with <5% accuracy loss

Usage:
    python scripts/train_with_pca.py [--components 15] [--variance 0.95]
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import time
import logging

from src.preprocessing.feature_reduction import FeatureReducer
from src.preprocessing.feature_extractor import FeatureExtractor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_training_data(data_path: str) -> pd.DataFrame:
    """
    Load training data from CSV file.
    
    Args:
        data_path: Path to training data CSV
        
    Returns:
        DataFrame with training data
    """
    logger.info(f"Loading training data from {data_path}...")
    
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df):,} samples")
    
    # Check for required columns
    if 'label' not in df.columns:
        logger.warning("No 'label' column found, assuming all data is normal (label=0)")
        df['label'] = 0
    
    # Show class distribution
    if 'label' in df.columns:
        class_dist = df['label'].value_counts()
        logger.info(f"Class distribution:")
        for label, count in class_dist.items():
            label_name = "Attack" if label == 1 else "Normal"
            pct = (count / len(df)) * 100
            logger.info(f"  {label_name} (label={label}): {count:,} ({pct:.2f}%)")
    
    return df


def extract_features_from_data(df: pd.DataFrame, extractor: FeatureExtractor) -> tuple:
    """
    Extract features from raw CAN message data.
    
    Args:
        df: DataFrame with CAN message data
        extractor: FeatureExtractor instance
        
    Returns:
        (X, y, feature_names) tuple
    """
    logger.info("Extracting features from raw data...")
    
    X = []
    y = []
    feature_names = None
    
    start_time = time.time()
    for i, row in enumerate(df.iterrows()):
        idx, data = row
        msg = data.to_dict()
        
        try:
            features = extractor.extract_features(msg)
            if features:
                X.append(list(features.values()))
                y.append(data.get('label', 0))
                
                # Get feature names from first successful extraction
                if feature_names is None:
                    feature_names = list(features.keys())
        except Exception as e:
            logger.debug(f"Error extracting features from row {idx}: {e}")
            continue
        
        # Progress indicator
        if (i + 1) % 10000 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            logger.info(f"  Processed {i+1:,} messages ({rate:.0f} msg/s)")
    
    duration = time.time() - start_time
    logger.info(f"Feature extraction complete: {len(X):,} samples in {duration:.1f}s")
    
    X = np.array(X)
    y = np.array(y)
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Feature names: {len(feature_names)} features")
    
    return X, y, feature_names


def train_pca_reducer(X: np.ndarray, feature_names: list, 
                     n_components: int, variance_threshold: float) -> FeatureReducer:
    """
    Train PCA feature reducer.
    
    Args:
        X: Feature matrix
        feature_names: List of feature names
        n_components: Target number of components
        variance_threshold: Minimum variance to explain
        
    Returns:
        Fitted FeatureReducer
    """
    logger.info(f"\n{'='*60}")
    logger.info("TRAINING PCA FEATURE REDUCER")
    logger.info(f"{'='*60}")
    
    reducer = FeatureReducer(n_components=n_components, 
                            variance_threshold=variance_threshold)
    
    start_time = time.time()
    reducer.fit(X, feature_names)
    duration = time.time() - start_time
    
    logger.info(f"PCA training complete in {duration:.2f}s")
    
    # Show variance explained per component
    logger.info("\nVariance explained by each component:")
    for i, (var, cum_var) in enumerate(zip(reducer.explained_variance_ratio, 
                                           reducer.cumulative_variance)):
        logger.info(f"  PC{i+1}: {var*100:6.2f}% (cumulative: {cum_var*100:6.2f}%)")
    
    # Show top contributing features
    logger.info("\nTop 15 most important features:")
    for name, importance in reducer.get_feature_importance(15):
        logger.info(f"  {name:40s}: {importance:.4f}")
    
    return reducer


def train_model_with_pca(X_train: np.ndarray, X_test: np.ndarray, 
                        y_train: np.ndarray, y_test: np.ndarray,
                        contamination: float = 0.02) -> IsolationForest:
    """
    Train Isolation Forest model on PCA-reduced features.
    
    Args:
        X_train: Training features (already PCA-reduced)
        X_test: Test features (already PCA-reduced)
        y_train: Training labels
        y_test: Test labels
        contamination: Expected proportion of outliers
        
    Returns:
        Trained IsolationForest model
    """
    logger.info(f"\n{'='*60}")
    logger.info("TRAINING MODEL ON REDUCED FEATURES")
    logger.info(f"{'='*60}")
    logger.info(f"Training samples: {X_train.shape[0]:,}")
    logger.info(f"Test samples: {X_test.shape[0]:,}")
    logger.info(f"Features: {X_train.shape[1]}")
    logger.info(f"Contamination: {contamination}")
    
    # Train model with fewer trees for faster inference on Pi
    model = IsolationForest(
        n_estimators=50,      # Reduced from typical 100 for speed
        contamination=contamination,
        max_samples='auto',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    logger.info("Training Isolation Forest...")
    start_time = time.time()
    model.fit(X_train)
    duration = time.time() - start_time
    
    logger.info(f"Model training complete in {duration:.2f}s")
    
    # Evaluate model
    logger.info("\nEvaluating model on test set...")
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred == -1).astype(int)  # Convert to 0/1
    
    # Show classification report
    logger.info("\nClassification Report:")
    print(classification_report(y_test, y_pred_binary, 
                               target_names=['Normal', 'Attack'],
                               zero_division=0))
    
    # Show confusion matrix
    cm = confusion_matrix(y_test, y_pred_binary)
    logger.info("\nConfusion Matrix:")
    logger.info(f"  {'':10s} {'Pred Normal':15s} {'Pred Attack':15s}")
    logger.info(f"  {'True Normal':10s} {cm[0,0]:15d} {cm[0,1]:15d}")
    logger.info(f"  {'True Attack':10s} {cm[1,0]:15d} {cm[1,1]:15d}")
    
    return model


def save_models(reducer: FeatureReducer, model: IsolationForest, 
               output_dir: str = 'data/models'):
    """
    Save PCA reducer and trained model to disk.
    
    Args:
        reducer: Fitted FeatureReducer
        model: Trained IsolationForest
        output_dir: Directory to save models
    """
    logger.info(f"\n{'='*60}")
    logger.info("SAVING MODELS")
    logger.info(f"{'='*60}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save feature reducer
    reducer_path = output_path / 'feature_reducer.joblib'
    reducer.save(str(reducer_path))
    logger.info(f"✅ Feature reducer saved: {reducer_path}")
    
    # Save model
    model_path = output_path / 'model_with_pca.joblib'
    joblib.dump(model, str(model_path))
    logger.info(f"✅ Model saved: {model_path}")
    
    # Save metadata
    metadata = {
        'n_features_original': len(reducer.feature_names_in),
        'n_components': reducer.n_components,
        'variance_explained': float(reducer.cumulative_variance[-1]),
        'model_type': 'IsolationForest',
        'n_estimators': model.n_estimators,
        'contamination': model.contamination,
        'trained_date': pd.Timestamp.now().isoformat()
    }
    
    metadata_path = output_path / 'model_metadata.json'
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✅ Metadata saved: {metadata_path}")


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(
        description='Train ML model with PCA feature reduction for Pi deployment'
    )
    parser.add_argument('--data', type=str, 
                       default='../Vehicle_Models/data/processed/training_data.csv',
                       help='Path to training data CSV')
    parser.add_argument('--components', type=int, default=15,
                       help='Number of PCA components (default: 15)')
    parser.add_argument('--variance', type=float, default=0.95,
                       help='Minimum variance to explain (default: 0.95)')
    parser.add_argument('--contamination', type=float, default=0.02,
                       help='Expected proportion of outliers (default: 0.02)')
    parser.add_argument('--output', type=str, default='data/models',
                       help='Output directory for models')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("PCA FEATURE REDUCTION TRAINING")
    logger.info("="*60)
    logger.info(f"Configuration:")
    logger.info(f"  Data path: {args.data}")
    logger.info(f"  PCA components: {args.components}")
    logger.info(f"  Variance threshold: {args.variance}")
    logger.info(f"  Contamination: {args.contamination}")
    logger.info(f"  Test size: {args.test_size}")
    logger.info(f"  Output directory: {args.output}")
    logger.info("")
    
    try:
        # 1. Load data
        df = load_training_data(args.data)
        
        # 2. Extract features
        extractor = FeatureExtractor()
        X, y, feature_names = extract_features_from_data(df, extractor)
        
        logger.info(f"\nOriginal feature count: {len(feature_names)}")
        logger.info(f"Target feature count: {args.components}")
        logger.info(f"Expected reduction: {len(feature_names) / args.components:.1f}x")
        
        # 3. Train PCA reducer
        reducer = train_pca_reducer(X, feature_names, 
                                   args.components, args.variance)
        
        # 4. Transform features
        logger.info("\nTransforming features with PCA...")
        X_reduced = reducer.transform(X)
        logger.info(f"Reduced shape: {X_reduced.shape}")
        
        # 5. Split data
        logger.info(f"\nSplitting data (test_size={args.test_size})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_reduced, y, 
            test_size=args.test_size, 
            random_state=42,
            stratify=y if len(np.unique(y)) > 1 else None
        )
        
        # 6. Train model
        model = train_model_with_pca(X_train, X_test, y_train, y_test,
                                    contamination=args.contamination)
        
        # 7. Save models
        save_models(reducer, model, args.output)
        
        # 8. Summary
        logger.info(f"\n{'='*60}")
        logger.info("TRAINING COMPLETE ✅")
        logger.info(f"{'='*60}")
        logger.info("\nExpected Performance Improvement on Raspberry Pi 4:")
        logger.info(f"  Before PCA: ~17.31 msg/s (57.7ms per message)")
        logger.info(f"  After PCA:  ~50-85 msg/s (15-20ms per message)")
        logger.info(f"  Speedup:    3-5x faster")
        logger.info(f"  Features:   {len(feature_names)} → {args.components} "
                   f"({(1 - args.components/len(feature_names))*100:.0f}% reduction)")
        logger.info(f"  Variance:   {reducer.cumulative_variance[-1]*100:.2f}% explained")
        logger.info("\nNext Steps:")
        logger.info("  1. Copy models to Raspberry Pi 4:")
        logger.info(f"     scp {args.output}/feature_reducer.joblib pi@raspberrypi:~/CANBUS_IDS/data/models/")
        logger.info(f"     scp {args.output}/model_with_pca.joblib pi@raspberrypi:~/CANBUS_IDS/data/models/")
        logger.info("  2. Update ML detector to use PCA (use_pca=True)")
        logger.info("  3. Test on Pi 4 to validate speedup")
        
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
