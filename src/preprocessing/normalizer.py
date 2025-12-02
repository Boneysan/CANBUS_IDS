"""
Data normalization and preprocessing utilities for CAN-IDS.

Handles feature scaling, normalization, and data preparation
for machine learning algorithms.
"""

import logging
import pickle
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import statistics

logger = logging.getLogger(__name__)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    logger.warning("NumPy not available. Some normalization features will be limited.")
    NUMPY_AVAILABLE = False


class Normalizer:
    """
    Data normalization and preprocessing for CAN message features.
    
    Supports various normalization methods:
    - Min-Max scaling (0-1 range)
    - Z-score standardization (mean=0, std=1) 
    - Robust scaling (median-based)
    - Custom range scaling
    """
    
    def __init__(self, method: str = 'minmax'):
        """
        Initialize normalizer.
        
        Args:
            method: Normalization method ('minmax', 'zscore', 'robust', 'none')
        """
        self.method = method.lower()
        self.is_fitted = False
        
        # Normalization parameters
        self._feature_stats = {}
        self._feature_names = []
        
        # Statistics for different methods
        self._min_values = {}
        self._max_values = {}
        self._mean_values = {}
        self._std_values = {}
        self._median_values = {}
        self._iqr_values = {}  # Interquartile range
        
        # Validation
        valid_methods = ['minmax', 'zscore', 'robust', 'none']
        if self.method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Must be one of: {valid_methods}")
            
    def fit(self, features_list: List[Dict[str, float]]) -> None:
        """
        Fit normalizer to training data.
        
        Args:
            features_list: List of feature dictionaries from training data
        """
        if not features_list:
            raise ValueError("Cannot fit normalizer on empty data")
            
        logger.info(f"Fitting normalizer with {len(features_list)} samples using method '{self.method}'")
        
        # Get all unique feature names
        all_features = set()
        for features in features_list:
            all_features.update(features.keys())
            
        self._feature_names = sorted(list(all_features))
        
        # Calculate statistics for each feature
        for feature_name in self._feature_names:
            values = []
            
            # Collect all values for this feature
            for features in features_list:
                value = features.get(feature_name, 0.0)
                values.append(value)
                
            # Calculate statistics based on method
            if self.method == 'minmax':
                self._min_values[feature_name] = min(values)
                self._max_values[feature_name] = max(values)
                
            elif self.method == 'zscore':
                self._mean_values[feature_name] = statistics.mean(values)
                if len(values) > 1:
                    self._std_values[feature_name] = statistics.stdev(values)
                else:
                    self._std_values[feature_name] = 1.0  # Avoid division by zero
                    
            elif self.method == 'robust':
                self._median_values[feature_name] = statistics.median(values)
                
                # Calculate IQR (Interquartile Range)
                if NUMPY_AVAILABLE:
                    q75 = np.percentile(values, 75)
                    q25 = np.percentile(values, 25)
                    self._iqr_values[feature_name] = q75 - q25
                else:
                    # Simple IQR approximation without numpy
                    sorted_values = sorted(values)
                    n = len(sorted_values)
                    q1_idx = n // 4
                    q3_idx = 3 * n // 4
                    
                    if n > 4:
                        q1 = sorted_values[q1_idx]
                        q3 = sorted_values[q3_idx]
                        self._iqr_values[feature_name] = q3 - q1
                    else:
                        self._iqr_values[feature_name] = max(values) - min(values)
                        
                # Avoid division by zero
                if self._iqr_values[feature_name] == 0:
                    self._iqr_values[feature_name] = 1.0
                    
        self.is_fitted = True
        logger.info(f"Normalizer fitted on {len(self._feature_names)} features")
        
    def transform(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Transform features using fitted normalization.
        
        Args:
            features: Feature dictionary to normalize
            
        Returns:
            Normalized feature dictionary
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
            
        if self.method == 'none':
            return features.copy()
            
        normalized = {}
        
        for feature_name in self._feature_names:
            value = features.get(feature_name, 0.0)
            
            if self.method == 'minmax':
                min_val = self._min_values[feature_name]
                max_val = self._max_values[feature_name]
                
                if max_val == min_val:
                    normalized[feature_name] = 0.0  # Constant feature
                else:
                    normalized[feature_name] = (value - min_val) / (max_val - min_val)
                    
            elif self.method == 'zscore':
                mean_val = self._mean_values[feature_name]
                std_val = self._std_values[feature_name]
                
                normalized[feature_name] = (value - mean_val) / std_val
                
            elif self.method == 'robust':
                median_val = self._median_values[feature_name]
                iqr_val = self._iqr_values[feature_name]
                
                normalized[feature_name] = (value - median_val) / iqr_val
                
        # Include any features not in training set (with value 0.0)
        for feature_name in features:
            if feature_name not in normalized:
                normalized[feature_name] = 0.0
                
        return normalized
        
    def fit_transform(self, features_list: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """
        Fit normalizer and transform features in one step.
        
        Args:
            features_list: List of feature dictionaries
            
        Returns:
            List of normalized feature dictionaries
        """
        self.fit(features_list)
        
        return [self.transform(features) for features in features_list]
        
    def inverse_transform(self, normalized_features: Dict[str, float]) -> Dict[str, float]:
        """
        Reverse normalization to get original feature values.
        
        Args:
            normalized_features: Normalized feature dictionary
            
        Returns:
            Original feature dictionary (approximate)
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
            
        if self.method == 'none':
            return normalized_features.copy()
            
        original = {}
        
        for feature_name, norm_value in normalized_features.items():
            if feature_name in self._feature_names:
                
                if self.method == 'minmax':
                    min_val = self._min_values[feature_name]
                    max_val = self._max_values[feature_name]
                    original[feature_name] = norm_value * (max_val - min_val) + min_val
                    
                elif self.method == 'zscore':
                    mean_val = self._mean_values[feature_name]
                    std_val = self._std_values[feature_name]
                    original[feature_name] = norm_value * std_val + mean_val
                    
                elif self.method == 'robust':
                    median_val = self._median_values[feature_name]
                    iqr_val = self._iqr_values[feature_name]
                    original[feature_name] = norm_value * iqr_val + median_val
                    
            else:
                original[feature_name] = norm_value  # Unknown feature, keep as-is
                
        return original
        
    def save(self, filepath: str) -> None:
        """
        Save fitted normalizer to file.
        
        Args:
            filepath: Path to save normalizer
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted normalizer")
            
        normalizer_data = {
            'method': self.method,
            'is_fitted': self.is_fitted,
            'feature_names': self._feature_names,
            'min_values': self._min_values,
            'max_values': self._max_values,
            'mean_values': self._mean_values,
            'std_values': self._std_values,
            'median_values': self._median_values,
            'iqr_values': self._iqr_values,
            'version': '1.0'
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(normalizer_data, f)
                
            logger.info(f"Normalizer saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving normalizer: {e}")
            raise
            
    def load(self, filepath: str) -> None:
        """
        Load fitted normalizer from file.
        
        Args:
            filepath: Path to saved normalizer
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Normalizer file not found: {filepath}")
            
        try:
            with open(filepath, 'rb') as f:
                normalizer_data = pickle.load(f)
                
            self.method = normalizer_data['method']
            self.is_fitted = normalizer_data['is_fitted']
            self._feature_names = normalizer_data['feature_names']
            self._min_values = normalizer_data.get('min_values', {})
            self._max_values = normalizer_data.get('max_values', {})
            self._mean_values = normalizer_data.get('mean_values', {})
            self._std_values = normalizer_data.get('std_values', {})
            self._median_values = normalizer_data.get('median_values', {})
            self._iqr_values = normalizer_data.get('iqr_values', {})
            
            logger.info(f"Normalizer loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading normalizer: {e}")
            raise
            
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get normalizer statistics and parameters.
        
        Returns:
            Dictionary with normalizer information
        """
        stats = {
            'method': self.method,
            'is_fitted': self.is_fitted,
            'feature_count': len(self._feature_names),
            'feature_names': self._feature_names.copy()
        }
        
        if self.is_fitted:
            # Add method-specific statistics
            if self.method == 'minmax':
                stats['min_values'] = self._min_values.copy()
                stats['max_values'] = self._max_values.copy()
                
                # Calculate ranges
                ranges = {}
                for name in self._feature_names:
                    ranges[name] = self._max_values[name] - self._min_values[name]
                stats['feature_ranges'] = ranges
                
            elif self.method == 'zscore':
                stats['mean_values'] = self._mean_values.copy()
                stats['std_values'] = self._std_values.copy()
                
            elif self.method == 'robust':
                stats['median_values'] = self._median_values.copy()
                stats['iqr_values'] = self._iqr_values.copy()
                
        return stats
        
    def validate_features(self, features: Dict[str, float]) -> Tuple[bool, List[str]]:
        """
        Validate that features are suitable for normalization.
        
        Args:
            features: Feature dictionary to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if not self.is_fitted:
            issues.append("Normalizer not fitted")
            return False, issues
            
        # Check for missing features
        missing_features = set(self._feature_names) - set(features.keys())
        if missing_features:
            issues.append(f"Missing features: {list(missing_features)}")
            
        # Check for unexpected features
        extra_features = set(features.keys()) - set(self._feature_names)
        if extra_features:
            issues.append(f"Unexpected features: {list(extra_features)}")
            
        # Check for invalid values
        for name, value in features.items():
            if not isinstance(value, (int, float)):
                issues.append(f"Feature '{name}' has invalid type: {type(value)}")
            elif not (-1e10 < value < 1e10):  # Reasonable bounds
                issues.append(f"Feature '{name}' has extreme value: {value}")
                
        return len(issues) == 0, issues


class FeatureSelector:
    """
    Feature selection utilities for reducing dimensionality.
    
    Provides methods for selecting the most informative features
    for anomaly detection.
    """
    
    def __init__(self):
        """Initialize feature selector."""
        self.selected_features = []
        self.feature_importance = {}
        
    def select_by_variance(self, features_list: List[Dict[str, float]], 
                          min_variance: float = 0.01) -> List[str]:
        """
        Select features based on variance threshold.
        
        Args:
            features_list: List of feature dictionaries
            min_variance: Minimum variance threshold
            
        Returns:
            List of selected feature names
        """
        if not features_list:
            return []
            
        # Get all feature names
        all_features = set()
        for features in features_list:
            all_features.update(features.keys())
            
        selected = []
        
        # Calculate variance for each feature
        for feature_name in all_features:
            values = [features.get(feature_name, 0.0) for features in features_list]
            
            if len(values) > 1:
                variance = statistics.variance(values)
                
                if variance >= min_variance:
                    selected.append(feature_name)
                    self.feature_importance[feature_name] = variance
                    
        self.selected_features = sorted(selected)
        logger.info(f"Selected {len(selected)} features based on variance >= {min_variance}")
        
        return self.selected_features
        
    def select_by_correlation(self, features_list: List[Dict[str, float]], 
                            max_correlation: float = 0.95) -> List[str]:
        """
        Remove highly correlated features.
        
        Args:
            features_list: List of feature dictionaries
            max_correlation: Maximum allowed correlation
            
        Returns:
            List of selected feature names
        """
        # This is a simplified version - full implementation would
        # require correlation matrix calculation
        
        if not features_list:
            return []
            
        # For now, just return all features
        # TODO: Implement proper correlation analysis
        all_features = set()
        for features in features_list:
            all_features.update(features.keys())
            
        self.selected_features = sorted(list(all_features))
        
        logger.info(f"Correlation-based selection: {len(self.selected_features)} features")
        return self.selected_features
        
    def filter_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Filter features dictionary to only include selected features.
        
        Args:
            features: Original feature dictionary
            
        Returns:
            Filtered feature dictionary
        """
        if not self.selected_features:
            return features
            
        return {name: features.get(name, 0.0) for name in self.selected_features}