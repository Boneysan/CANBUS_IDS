"""
Detection classes for multi-stage pipeline.

These classes are in a proper module (not __main__) so pickle
can save/load them with proper module paths.
"""

import numpy as np


class SimpleRuleDetector:
    """
    Simple rule-based detector for Stage 2 of multi-stage pipeline.
    
    Applies fast heuristic rules to filter obvious anomalies:
    - Abnormal time deltas (< 1ms or > 1s)
    - High message frequency (> 100 Hz)
    - Invalid DLC values (not 0-8)
    """
    
    def __init__(self, rules=None):
        """
        Initialize rule detector.
        
        Args:
            rules: Optional dictionary of rule thresholds
        """
        self.rules = rules or {
            'min_time_delta': 0.001,  # 1ms
            'max_time_delta': 1.0,     # 1s
            'max_frequency': 100,       # 100 Hz
            'min_dlc': 0,
            'max_dlc': 8
        }
        
    def predict(self, X):
        """
        Predict anomalies based on simple rules.
        
        Args:
            X: Feature array (n_samples, n_features)
                Expected features at indices:
                - 0: arb_id_numeric
                - 1: data_length (DLC)
                - 2: id_frequency
                - 3: time_delta
            
        Returns:
            Predictions: 1 for anomaly, 0 for normal
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        predictions = np.zeros(len(X), dtype=int)
        
        # Rule 1: Abnormal time delta
        if X.shape[1] >= 4:
            time_delta = X[:, 3]
            anomaly_mask = (
                (time_delta < self.rules['min_time_delta']) | 
                (time_delta > self.rules['max_time_delta'])
            )
            predictions[anomaly_mask] = 1
        
        # Rule 2: High frequency
        if X.shape[1] >= 3:
            frequency = X[:, 2]
            anomaly_mask = frequency > self.rules['max_frequency']
            predictions[anomaly_mask] = 1
        
        # Rule 3: Invalid DLC
        if X.shape[1] >= 2:
            dlc = X[:, 1]
            anomaly_mask = (
                (dlc < self.rules['min_dlc']) | 
                (dlc > self.rules['max_dlc'])
            )
            predictions[anomaly_mask] = 1
            
        return predictions
    
    def fit(self, X, y=None):
        """Compatibility method for sklearn pipeline."""
        return self
    
    def score(self, X, y):
        """Calculate accuracy for compatibility."""
        predictions = self.predict(X)
        return np.mean(predictions == y)


# Export for easy import
__all__ = ['SimpleRuleDetector']
