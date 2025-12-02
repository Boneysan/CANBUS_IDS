"""
Compatibility module for loading Vehicle_Models trained pipelines.

Provides placeholder classes that Vehicle_Models models expect.
These allow models trained in Vehicle_Models to be loaded in CANBUS_IDS.
"""

import numpy as np


class SimpleRuleDetector:
    """
    Simple rule-based detector for multi-stage pipeline.
    
    This is a lightweight rule detector used in Stage 2 of the multi-stage pipeline.
    It applies fast heuristic rules to filter obvious anomalies.
    """
    
    def __init__(self, rules=None):
        """
        Initialize rule detector.
        
        Args:
            rules: Optional dictionary of rule thresholds
        """
        self.rules = rules or {}
        
    def predict(self, X):
        """
        Predict anomalies based on simple rules.
        
        Args:
            X: Feature array (n_samples, n_features)
            
        Returns:
            Predictions: 1 for anomaly, 0 for normal
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        predictions = np.zeros(len(X), dtype=int)
        
        # Rule 1: Abnormal time delta (if available)
        if X.shape[1] >= 4:  # Has time_delta feature at index 3
            time_delta = X[:, 3]
            # Flag messages with very short (<1ms) or very long (>1s) intervals
            anomaly_mask = (time_delta < 0.001) | (time_delta > 1.0)
            predictions[anomaly_mask] = 1
        
        # Rule 2: High frequency (if id_frequency available)
        if X.shape[1] >= 3:  # Has id_frequency feature at index 2
            frequency = X[:, 2]
            # Flag messages with abnormally high frequency (>100 Hz)
            anomaly_mask = frequency > 100
            predictions[anomaly_mask] = 1
        
        # Rule 3: Unusual DLC (if data_length available)
        if X.shape[1] >= 2:  # Has data_length at index 1
            dlc = X[:, 1]
            # Flag messages with invalid DLC (not in range 0-8)
            anomaly_mask = (dlc < 0) | (dlc > 8)
            predictions[anomaly_mask] = 1
            
        return predictions
    
    def fit(self, X, y=None):
        """Compatibility method for sklearn pipeline."""
        return self
    
    def score(self, X, y):
        """Calculate accuracy for compatibility."""
        predictions = self.predict(X)
        return np.mean(predictions == y)


# Make available for unpickling
__all__ = ['SimpleRuleDetector']
