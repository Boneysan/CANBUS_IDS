"""
Machine learning model training module.

Provides utilities for training the Isolation Forest anomaly detection
model on CAN bus traffic data.
"""

import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not available. Model training will be disabled.")
    SKLEARN_AVAILABLE = False


class ModelTrainer:
    """
    Train and manage ML models for CAN-IDS.
    
    Handles data preparation, model training, validation,
    and model persistence.
    """
    
    def __init__(self, contamination: float = 0.02, random_state: int = 42):
        """
        Initialize model trainer.
        
        Args:
            contamination: Expected proportion of anomalies (0.01-0.1)
            random_state: Random seed for reproducibility
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for model training")
            
        self.contamination = contamination
        self.random_state = random_state
        
        self.isolation_forest: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        self.training_stats = {}
        
    def prepare_data(self, features: List[Dict[str, float]], 
                    labels: Optional[List[int]] = None,
                    test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Prepare feature data for training.
        
        Args:
            features: List of feature dictionaries
            labels: Optional labels for supervised evaluation (1=anomaly, 0=normal)
            test_size: Proportion of data for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Preparing {len(features)} samples for training")
        
        # Convert feature dicts to numpy array
        feature_names = sorted(features[0].keys())
        X = np.array([[f.get(name, 0.0) for name in feature_names] for f in features])
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Split data
        if labels is not None:
            y = np.array(labels)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )
        else:
            # Unsupervised - no labels
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = None, None
            
        logger.info(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
        
    def train(self, X_train: np.ndarray, 
             n_estimators: int = 100,
             max_samples: str = 'auto',
             max_features: float = 1.0,
             bootstrap: bool = False,
             n_jobs: int = -1) -> Dict[str, Any]:
        """
        Train Isolation Forest model.
        
        Args:
            X_train: Training feature matrix
            n_estimators: Number of trees in the forest
            max_samples: Number of samples to draw for each tree
            max_features: Number of features to draw for each tree
            bootstrap: Whether to use bootstrap sampling
            n_jobs: Number of parallel jobs (-1 = all cores)
            
        Returns:
            Dictionary containing training statistics
        """
        logger.info("Training Isolation Forest model...")
        start_time = time.time()
        
        # Initialize scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Initialize and train Isolation Forest
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=self.random_state,
            verbose=0
        )
        
        self.isolation_forest.fit(X_scaled)
        
        training_time = time.time() - start_time
        
        # Training statistics
        self.training_stats = {
            'training_time': training_time,
            'n_samples': len(X_train),
            'n_features': X_train.shape[1],
            'n_estimators': n_estimators,
            'contamination': self.contamination,
            'timestamp': time.time()
        }
        
        logger.info(f"Model trained in {training_time:.2f} seconds")
        logger.info(f"Features: {X_train.shape[1]}, Samples: {len(X_train)}")
        
        return self.training_stats
        
    def evaluate(self, X_test: np.ndarray, 
                y_test: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Evaluate trained model on test data.
        
        Args:
            X_test: Test feature matrix
            y_test: Optional true labels for supervised evaluation
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.isolation_forest is None or self.scaler is None:
            raise RuntimeError("Model not trained. Call train() first.")
            
        logger.info(f"Evaluating model on {len(X_test)} test samples")
        
        # Scale test data
        X_scaled = self.scaler.transform(X_test)
        
        # Get predictions (-1 for anomaly, 1 for normal)
        predictions = self.isolation_forest.predict(X_scaled)
        anomaly_scores = -self.isolation_forest.score_samples(X_scaled)
        
        # Count anomalies
        n_anomalies = np.sum(predictions == -1)
        anomaly_rate = n_anomalies / len(predictions)
        
        metrics = {
            'n_test_samples': len(X_test),
            'n_anomalies_detected': int(n_anomalies),
            'anomaly_rate': float(anomaly_rate),
            'mean_anomaly_score': float(np.mean(anomaly_scores)),
            'std_anomaly_score': float(np.std(anomaly_scores))
        }
        
        # If labels provided, calculate supervised metrics
        if y_test is not None:
            y_pred = (predictions == -1).astype(int)
            
            # Calculate metrics
            true_positives = np.sum((y_test == 1) & (y_pred == 1))
            false_positives = np.sum((y_test == 0) & (y_pred == 1))
            true_negatives = np.sum((y_test == 0) & (y_pred == 0))
            false_negatives = np.sum((y_test == 1) & (y_pred == 0))
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (true_positives + true_negatives) / len(y_test)
            
            metrics.update({
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'true_positives': int(true_positives),
                'false_positives': int(false_positives),
                'true_negatives': int(true_negatives),
                'false_negatives': int(false_negatives)
            })
            
            logger.info(f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        
        return metrics
        
    def save_model(self, model_path: str) -> None:
        """
        Save trained model to disk.
        
        Args:
            model_path: Path to save model file
        """
        if self.isolation_forest is None or self.scaler is None:
            raise RuntimeError("No trained model to save")
            
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'isolation_forest': self.isolation_forest,
            'scaler': self.scaler,
            'contamination': self.contamination,
            'training_stats': self.training_stats,
            'version': '1.0.0'
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
            
        logger.info(f"Model saved to {model_path}")
        
    def load_model(self, model_path: str) -> None:
        """
        Load trained model from disk.
        
        Args:
            model_path: Path to model file
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            
        self.isolation_forest = model_data['isolation_forest']
        self.scaler = model_data['scaler']
        self.contamination = model_data.get('contamination', self.contamination)
        self.training_stats = model_data.get('training_stats', {})
        
        logger.info(f"Model loaded from {model_path}")
        
    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """
        Estimate feature importance (not directly available in Isolation Forest).
        
        Args:
            feature_names: List of feature names
            
        Returns:
            Dictionary of feature names and importance scores
        """
        if self.isolation_forest is None:
            raise RuntimeError("Model not trained")
            
        # Isolation Forest doesn't have direct feature importance
        # Return placeholder - could implement permutation importance
        logger.warning("Feature importance not directly available for Isolation Forest")
        return {name: 1.0 / len(feature_names) for name in feature_names}
