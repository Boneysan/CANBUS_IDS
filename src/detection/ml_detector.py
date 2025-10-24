"""
Machine learning-based anomaly detection for CAN bus intrusion detection.

Uses Isolation Forest algorithm to detect anomalous CAN traffic patterns
that may indicate novel attacks or system compromise.
"""

import logging
import pickle
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from collections import deque, defaultdict
import statistics
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not available. ML detection will be disabled.")
    SKLEARN_AVAILABLE = False


@dataclass
class MLAlert:
    """ML-based anomaly alert."""
    timestamp: float
    can_id: int
    anomaly_score: float
    confidence: float
    features: Dict[str, float]
    message_data: Dict[str, Any]
    reason: str = "ML anomaly detection"


class MLDetector:
    """
    Machine learning-based anomaly detector for CAN bus traffic.
    
    Uses Isolation Forest to identify anomalous message patterns
    based on extracted features from CAN traffic.
    """
    
    def __init__(self, model_path: Optional[str] = None, 
                 contamination: float = 0.02,
                 feature_window: int = 100):
        """
        Initialize ML detector.
        
        Args:
            model_path: Path to trained model file
            contamination: Expected proportion of anomalies (0.01-0.1)
            feature_window: Number of messages to use for feature extraction
        """
        self.model_path = Path(model_path) if model_path else None
        self.contamination = contamination
        self.feature_window = feature_window
        
        # ML components
        self.isolation_forest: Optional['IsolationForest'] = None
        self.scaler: Optional['StandardScaler'] = None
        self.is_trained = False
        
        # Feature extraction state
        self._message_history = defaultdict(lambda: deque(maxlen=feature_window))
        self._frequency_trackers = defaultdict(lambda: deque(maxlen=1000))
        self._timing_trackers = defaultdict(list)
        
        # Statistics
        self._stats = {
            'messages_analyzed': 0,
            'anomalies_detected': 0,
            'model_loaded': False,
            'last_training_time': None,
            'feature_extraction_time': 0.0,
            'prediction_time': 0.0
        }
        
        # Initialize ML components
        if SKLEARN_AVAILABLE:
            self._initialize_ml_components()
            if self.model_path and self.model_path.exists():
                self.load_model()
        else:
            logger.warning("ML detection disabled - scikit-learn not available")
            
    def _initialize_ml_components(self) -> None:
        """Initialize ML components."""
        if not SKLEARN_AVAILABLE:
            return
            
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100,
            max_samples='auto',
            bootstrap=True
        )
        
        self.scaler = StandardScaler()
        
    def train(self, training_data: List[Dict[str, Any]], 
              save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Train the anomaly detection model.
        
        Args:
            training_data: List of normal CAN messages for training
            save_path: Path to save trained model
            
        Returns:
            Training statistics
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn not available for ML training")
            
        start_time = time.time()
        logger.info(f"Training ML model with {len(training_data)} samples")
        
        try:
            # Extract features from training data
            features = self._extract_batch_features(training_data)
            
            if len(features) == 0:
                raise ValueError("No features extracted from training data")
                
            # Convert to numpy array
            X = np.array(features)
            
            logger.info(f"Extracted features shape: {X.shape}")
            
            # Normalize features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train isolation forest
            self.isolation_forest.fit(X_scaled)
            self.is_trained = True
            
            # Calculate training statistics
            training_time = time.time() - start_time
            self._stats['last_training_time'] = training_time
            
            # Test on training data to get baseline scores
            scores = self.isolation_forest.decision_function(X_scaled)
            outliers = self.isolation_forest.predict(X_scaled)
            
            training_stats = {
                'training_samples': len(training_data),
                'feature_dimensions': X.shape[1],
                'training_time_seconds': training_time,
                'baseline_score_mean': float(np.mean(scores)),
                'baseline_score_std': float(np.std(scores)),
                'contamination_rate': self.contamination,
                'outliers_in_training': int(np.sum(outliers == -1))
            }
            
            # Save model if path provided
            if save_path:
                self.save_model(save_path)
                training_stats['model_saved'] = save_path
                
            logger.info(f"ML model trained in {training_time:.2f} seconds")
            return training_stats
            
        except Exception as e:
            logger.error(f"Error training ML model: {e}")
            raise
            
    def analyze_message(self, message: Dict[str, Any]) -> Optional[MLAlert]:
        """
        Analyze a single CAN message for anomalies.
        
        Args:
            message: CAN message to analyze
            
        Returns:
            MLAlert if anomaly detected, None otherwise
        """
        if not self.is_trained or not SKLEARN_AVAILABLE:
            return None
            
        start_time = time.time()
        
        try:
            self._stats['messages_analyzed'] += 1
            
            # Update message history
            self._update_message_state(message)
            
            # Extract features for this message
            features = self._extract_message_features(message)
            
            if not features:
                return None
                
            feature_time = time.time()
            self._stats['feature_extraction_time'] += feature_time - start_time
            
            # Convert to numpy array and normalize
            X = np.array([features])
            X_scaled = self.scaler.transform(X)
            
            # Get anomaly score and prediction
            anomaly_score = self.isolation_forest.decision_function(X_scaled)[0]
            is_anomaly = self.isolation_forest.predict(X_scaled)[0] == -1
            
            prediction_time = time.time()
            self._stats['prediction_time'] += prediction_time - feature_time
            
            if is_anomaly:
                self._stats['anomalies_detected'] += 1
                
                # Calculate confidence (higher magnitude = higher confidence)
                confidence = min(abs(anomaly_score), 1.0)
                
                alert = MLAlert(
                    timestamp=message['timestamp'],
                    can_id=message['can_id'],
                    anomaly_score=float(anomaly_score),
                    confidence=confidence,
                    features=self._format_features(features),
                    message_data=message.copy()
                )
                
                return alert
                
            return None
            
        except Exception as e:
            logger.warning(f"Error in ML analysis: {e}")
            return None
            
    def _extract_batch_features(self, messages: List[Dict[str, Any]]) -> List[List[float]]:
        """Extract features from a batch of messages."""
        features = []
        
        # Process messages in order to build state
        for message in messages:
            self._update_message_state(message)
            
            # Extract features for messages with sufficient history
            can_id = message['can_id']
            if len(self._message_history[can_id]) >= 5:  # Need some history
                msg_features = self._extract_message_features(message)
                if msg_features:
                    features.append(msg_features)
                    
        return features
        
    def _update_message_state(self, message: Dict[str, Any]) -> None:
        """Update internal state with new message."""
        can_id = message['can_id']
        timestamp = message['timestamp']
        
        # Update message history
        self._message_history[can_id].append(message)
        
        # Update frequency tracking
        freq_tracker = self._frequency_trackers[can_id]
        freq_tracker.append(timestamp)
        
        # Clean old frequency data (keep last 60 seconds)
        cutoff_time = timestamp - 60
        while freq_tracker and freq_tracker[0] < cutoff_time:
            freq_tracker.popleft()
            
        # Update timing analysis
        history = self._message_history[can_id]
        if len(history) >= 2:
            prev_timestamp = history[-2]['timestamp']
            interval = (timestamp - prev_timestamp) * 1000  # milliseconds
            
            timing_list = self._timing_trackers[can_id]
            timing_list.append(interval)
            
            # Keep last 50 intervals
            if len(timing_list) > 50:
                timing_list.pop(0)
                
    def _extract_message_features(self, message: Dict[str, Any]) -> Optional[List[float]]:
        """
        Extract features from a single CAN message.
        
        Returns:
            List of feature values or None if insufficient data
        """
        can_id = message['can_id']
        features = []
        
        # Basic message features
        features.append(float(can_id))  # CAN ID
        features.append(float(message['dlc']))  # Data length
        features.append(float(message.get('is_extended', False)))  # Extended frame
        features.append(float(message.get('is_remote', False)))  # Remote frame
        
        # Data byte features (pad to 8 bytes)
        data_bytes = message['data'][:8]  # Limit to 8 bytes
        while len(data_bytes) < 8:
            data_bytes.append(0)
            
        features.extend([float(b) for b in data_bytes])
        
        # Data statistics
        if data_bytes:
            features.append(float(np.mean(data_bytes)))  # Mean
            features.append(float(np.std(data_bytes)))   # Standard deviation
            features.append(float(min(data_bytes)))      # Min
            features.append(float(max(data_bytes)))      # Max
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
            
        # Frequency features
        freq_tracker = self._frequency_trackers[can_id]
        if len(freq_tracker) >= 2:
            # Messages per second (last 10 seconds)
            recent_time = message['timestamp'] - 10
            recent_count = sum(1 for t in freq_tracker if t >= recent_time)
            features.append(float(recent_count / 10))  # Messages per second
        else:
            features.append(0.0)
            
        # Timing features
        timing_list = self._timing_trackers[can_id]
        if len(timing_list) >= 3:
            features.append(float(np.mean(timing_list)))    # Mean interval
            features.append(float(np.std(timing_list)))     # Interval jitter
            features.append(float(min(timing_list)))        # Min interval
            features.append(float(max(timing_list)))        # Max interval
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
            
        # Message history features
        history = self._message_history[can_id]
        if len(history) >= 5:
            # Data entropy over recent messages
            all_bytes = []
            for msg in list(history)[-5:]:
                all_bytes.extend(msg['data'])
                
            if all_bytes:
                entropy = self._calculate_entropy(all_bytes)
                features.append(float(entropy))
            else:
                features.append(0.0)
                
            # DLC variance
            dlcs = [msg['dlc'] for msg in list(history)[-10:]]
            features.append(float(np.std(dlcs)))
        else:
            features.extend([0.0, 0.0])
            
        return features
        
    def _calculate_entropy(self, data: List[int]) -> float:
        """Calculate Shannon entropy of data."""
        if not data:
            return 0.0
            
        # Count byte frequencies
        byte_counts = defaultdict(int)
        for byte_val in data:
            byte_counts[byte_val] += 1
            
        # Calculate entropy
        entropy = 0.0
        data_len = len(data)
        
        for count in byte_counts.values():
            probability = count / data_len
            if probability > 0:
                entropy -= probability * np.log2(probability)
                
        return entropy
        
    def _format_features(self, features: List[float]) -> Dict[str, float]:
        """Format features into a readable dictionary."""
        feature_names = [
            'can_id', 'dlc', 'is_extended', 'is_remote',
            'data_0', 'data_1', 'data_2', 'data_3',
            'data_4', 'data_5', 'data_6', 'data_7',
            'data_mean', 'data_std', 'data_min', 'data_max',
            'frequency_per_sec', 'timing_mean', 'timing_std', 
            'timing_min', 'timing_max', 'data_entropy', 'dlc_variance'
        ]
        
        return {
            name: features[i] if i < len(features) else 0.0
            for i, name in enumerate(feature_names)
        }
        
    def save_model(self, filepath: str) -> None:
        """
        Save trained model to file.
        
        Args:
            filepath: Path to save model
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")
            
        model_data = {
            'isolation_forest': self.isolation_forest,
            'scaler': self.scaler,
            'contamination': self.contamination,
            'feature_window': self.feature_window,
            'training_time': self._stats.get('last_training_time'),
            'version': '1.0'
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
                
            self.model_path = Path(filepath)
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
            
    def load_model(self, filepath: Optional[str] = None) -> None:
        """
        Load trained model from file.
        
        Args:
            filepath: Path to model file (uses self.model_path if None)
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn not available")
            
        model_path = Path(filepath) if filepath else self.model_path
        
        if not model_path or not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                
            self.isolation_forest = model_data['isolation_forest']
            self.scaler = model_data['scaler']
            self.contamination = model_data.get('contamination', 0.02)
            self.feature_window = model_data.get('feature_window', 100)
            
            self.is_trained = True
            self._stats['model_loaded'] = True
            self.model_path = model_path
            
            logger.info(f"Model loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get ML detector statistics."""
        stats = self._stats.copy()
        
        # Calculate additional metrics
        if stats['messages_analyzed'] > 0:
            stats['anomaly_rate'] = stats['anomalies_detected'] / stats['messages_analyzed']
            
            if stats['feature_extraction_time'] > 0:
                stats['avg_feature_time_ms'] = (stats['feature_extraction_time'] * 1000) / stats['messages_analyzed']
                
            if stats['prediction_time'] > 0:
                stats['avg_prediction_time_ms'] = (stats['prediction_time'] * 1000) / stats['messages_analyzed']
        else:
            stats['anomaly_rate'] = 0.0
            stats['avg_feature_time_ms'] = 0.0
            stats['avg_prediction_time_ms'] = 0.0
            
        stats['is_trained'] = self.is_trained
        stats['sklearn_available'] = SKLEARN_AVAILABLE
        stats['model_path'] = str(self.model_path) if self.model_path else None
        
        return stats
        
    def update_threshold(self, contamination: float) -> None:
        """
        Update anomaly detection threshold.
        
        Args:
            contamination: New contamination rate (0.01-0.1)
        """
        if not SKLEARN_AVAILABLE or not self.is_trained:
            return
            
        self.contamination = contamination
        
        # Reinitialize with new contamination rate
        # Note: This requires retraining in scikit-learn
        logger.warning("Threshold update requires model retraining in current implementation")
        
    def reset_state(self) -> None:
        """Reset internal state (message history, timing trackers, etc.)."""
        self._message_history.clear()
        self._frequency_trackers.clear()  
        self._timing_trackers.clear()
        
        logger.info("ML detector state reset")