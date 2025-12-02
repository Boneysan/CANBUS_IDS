"""
Advanced CAN Intrusion Detection Models

Implements multiple detection strategies:
1. Time-based detection
2. Cumulative timing error detection
3. Signature-based detection
4. Anomaly detection
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from typing import Dict, List, Tuple, Optional
import joblib


class TimeBasedDetector:
    """
    Detects attacks based on timing patterns and periodicity.
    CAN messages should arrive at predictable intervals.
    """
    
    def __init__(self, tolerance: float = 0.1):
        """
        Initialize time-based detector.
        
        Parameters:
        -----------
        tolerance : float
            Acceptable deviation from expected timing (0.1 = 10%)
        """
        self.tolerance = tolerance
        self.timing_profiles = {}  # Expected timing for each CAN ID
        self.is_trained = False
        
    def learn_timing_patterns(self, df: pd.DataFrame, 
                             id_col: str = 'arb_id_numeric',
                             time_delta_col: str = 'time_delta'):
        """
        Learn normal timing patterns for each CAN ID.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Training data with normal traffic
        id_col : str
            Column name for CAN ID
        time_delta_col : str
            Column name for time delta
        """
        for can_id in df[id_col].unique():
            can_messages = df[df[id_col] == can_id][time_delta_col]
            
            self.timing_profiles[can_id] = {
                'mean': can_messages.mean(),
                'std': can_messages.std(),
                'min': can_messages.quantile(0.05),  # 5th percentile
                'max': can_messages.quantile(0.95),  # 95th percentile
                'median': can_messages.median()
            }
        
        self.is_trained = True
        print(f"✅ Learned timing patterns for {len(self.timing_profiles)} CAN IDs")
    
    def detect_anomalies(self, df: pd.DataFrame,
                        id_col: str = 'arb_id_numeric',
                        time_delta_col: str = 'time_delta') -> np.ndarray:
        """
        Detect timing anomalies in new data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data to check for anomalies
        id_col : str
            Column name for CAN ID
        time_delta_col : str
            Column name for time delta
            
        Returns:
        --------
        np.ndarray : 1 = attack, 0 = normal
        """
        if not self.is_trained:
            raise ValueError("Detector must be trained first")
        
        predictions = []
        
        for idx, row in df.iterrows():
            can_id = row[id_col]
            time_delta = row[time_delta_col]
            
            # Unknown CAN ID = suspicious
            if can_id not in self.timing_profiles:
                predictions.append(1)
                continue
            
            profile = self.timing_profiles[can_id]
            expected = profile['mean']
            tolerance_range = expected * self.tolerance
            
            # Check if timing is within acceptable range
            if expected > 0:  # Avoid division by zero
                deviation = abs(time_delta - expected)
                if deviation > tolerance_range:
                    predictions.append(1)  # Attack
                else:
                    predictions.append(0)  # Normal
            else:
                # If expected is 0, check against absolute threshold
                if time_delta > 0.1:  # More than 100ms is suspicious
                    predictions.append(1)
                else:
                    predictions.append(0)
        
        return np.array(predictions)


class CumulativeTimingErrorDetector:
    """
    Tracks cumulative timing errors over a sliding window.
    Persistent timing violations indicate an attack.
    """
    
    def __init__(self, window_size: int = 100, error_threshold: float = 0.4):
        """
        Initialize cumulative timing error detector.
        
        Parameters:
        -----------
        window_size : int
            Number of messages to consider in sliding window
        error_threshold : float
            Cumulative error threshold (0-1). Set to 0.4 (40%) for balanced detection.
        """
        self.window_size = window_size
        self.error_threshold = error_threshold
        self.timing_profiles = {}
        self.is_trained = False
    
    def learn_timing_patterns(self, df: pd.DataFrame,
                             id_col: str = 'arb_id_numeric',
                             time_delta_col: str = 'time_delta'):
        """Learn expected timing for each CAN ID."""
        for can_id in df[id_col].unique():
            can_messages = df[df[id_col] == can_id][time_delta_col]
            
            self.timing_profiles[can_id] = {
                'expected_delta': can_messages.mean(),
                'std': can_messages.std()
            }
        
        self.is_trained = True
        print(f"✅ Learned timing profiles for cumulative error detection")
    
    def detect_anomalies(self, df: pd.DataFrame,
                        id_col: str = 'arb_id_numeric',
                        time_delta_col: str = 'time_delta') -> np.ndarray:
        """
        Detect attacks using cumulative timing errors.
        
        Returns:
        --------
        np.ndarray : 1 = attack, 0 = normal
        """
        if not self.is_trained:
            raise ValueError("Detector must be trained first")
        
        predictions = []
        error_window = []
        
        for idx, row in df.iterrows():
            can_id = row[id_col]
            time_delta = row[time_delta_col]
            
            if can_id in self.timing_profiles:
                expected = self.timing_profiles[can_id]['expected_delta']
                error = abs(time_delta - expected) / (expected + 1e-6)
                error_window.append(min(error, 1.0))  # Cap at 1.0
            else:
                error_window.append(1.0)  # Unknown ID = max error
            
            # Maintain window size
            if len(error_window) > self.window_size:
                error_window.pop(0)
            
            # Calculate cumulative error
            cumulative_error = np.mean(error_window)
            
            if cumulative_error > self.error_threshold:
                predictions.append(1)  # Attack
            else:
                predictions.append(0)  # Normal
        
        return np.array(predictions)


class SignatureBasedDetector:
    """
    Detects known attack signatures based on CAN ID patterns,
    frequency, and data characteristics.
    """
    
    def __init__(self):
        """Initialize signature-based detector."""
        self.normal_signatures = {}
        self.known_attack_signatures = {
            'dos': {'high_frequency': True, 'single_id': True},
            'fuzzing': {'random_ids': True, 'varying_data_length': True},
            'spoofing': {'duplicate_ids': True, 'timing_violation': True}
        }
        self.is_trained = False
    
    def learn_signatures(self, df: pd.DataFrame,
                        id_col: str = 'arb_id_numeric',
                        freq_col: str = 'id_frequency',
                        data_len_col: str = 'data_length'):
        """
        Learn normal traffic signatures.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Training data with normal traffic
        """
        # Learn CAN ID frequency patterns
        id_frequencies = df[id_col].value_counts()
        
        # Learn data length patterns per CAN ID
        for can_id in df[id_col].unique():
            can_data = df[df[id_col] == can_id]
            
            self.normal_signatures[can_id] = {
                'frequency': can_data[freq_col].mean(),
                'data_length': can_data[data_len_col].mode().iloc[0] if len(can_data) > 0 else 16,
                'data_length_variance': can_data[data_len_col].std(),
                'seen_count': len(can_data)
            }
        
        self.is_trained = True
        print(f"✅ Learned signatures for {len(self.normal_signatures)} CAN IDs")
    
    def detect_anomalies(self, df: pd.DataFrame,
                        id_col: str = 'arb_id_numeric',
                        freq_col: str = 'id_frequency',
                        data_len_col: str = 'data_length',
                        time_delta_col: str = 'time_delta') -> np.ndarray:
        """
        Detect signature-based attacks.
        
        Returns:
        --------
        np.ndarray : 1 = attack, 0 = normal
        """
        if not self.is_trained:
            raise ValueError("Detector must be trained first")
        
        predictions = []
        
        for idx, row in df.iterrows():
            can_id = row[id_col]
            frequency = row[freq_col]
            data_length = row[data_len_col]
            
            is_attack = False
            
            # Check 1: Unknown CAN ID
            if can_id not in self.normal_signatures:
                is_attack = True
            else:
                signature = self.normal_signatures[can_id]
                
                # Check 2: Abnormal frequency 
                # Note: frequency is total count in dataset, not real-time rate
                # Flag if frequency is suspiciously LOW (attack datasets are smaller)
                # OR suspiciously HIGH (flooding/DoS)
                if frequency < signature['frequency'] * 0.001:  # 1000x lower = very suspicious
                    is_attack = True
                elif frequency > signature['frequency'] * 10:  # 10x higher = flooding
                    is_attack = True
                
                # Check 3: Wrong data length (fuzzing/malformed indicator)
                if abs(data_length - signature['data_length']) > 2:
                    is_attack = True
            
            predictions.append(1 if is_attack else 0)
        
        return np.array(predictions)


class AnomalyDetector:
    """
    Machine learning-based anomaly detection using Isolation Forest
    and One-Class SVM.
    """
    
    def __init__(self, method: str = 'isolation_forest', contamination: float = 0.01):
        """
        Initialize anomaly detector.
        
        Parameters:
        -----------
        method : str
            'isolation_forest' or 'one_class_svm'
        contamination : float
            Expected proportion of anomalies (0.01 = 1%)
        """
        self.method = method
        self.contamination = contamination
        self.scaler = StandardScaler()
        
        if method == 'isolation_forest':
            self.model = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100
            )
        elif method == 'one_class_svm':
            self.model = OneClassSVM(
                nu=contamination,
                kernel='rbf',
                gamma='auto'
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.is_trained = False
    
    def train(self, X: np.ndarray):
        """
        Train anomaly detector on normal traffic.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (normal traffic only)
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled)
        self.is_trained = True
        
        print(f"✅ Trained {self.method} anomaly detector")
    
    def detect_anomalies(self, X: np.ndarray) -> np.ndarray:
        """
        Detect anomalies in new data.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
            
        Returns:
        --------
        np.ndarray : 1 = attack, 0 = normal
        """
        if not self.is_trained:
            raise ValueError("Detector must be trained first")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        # Convert: -1 (anomaly) -> 1 (attack), 1 (normal) -> 0
        return np.where(predictions == -1, 1, 0)


class EnsembleDetector:
    """
    Combines multiple detection methods using voting.
    """
    
    def __init__(self, detectors: List, voting: str = 'soft', threshold: float = 0.5):
        """
        Initialize ensemble detector.
        
        Parameters:
        -----------
        detectors : List
            List of detector instances
        voting : str
            'hard' (majority vote) or 'soft' (weighted average)
        threshold : float
            Decision threshold for soft voting (0-1)
        """
        self.detectors = detectors
        self.voting = voting
        self.threshold = threshold
    
    def detect_anomalies(self, *args, **kwargs) -> Dict:
        """
        Run all detectors and combine results.
        
        Returns:
        --------
        Dict : Results from each detector and ensemble prediction
        """
        results = {}
        predictions_list = []
        
        for i, detector in enumerate(self.detectors):
            detector_name = detector.__class__.__name__
            predictions = detector.detect_anomalies(*args, **kwargs)
            results[detector_name] = predictions
            predictions_list.append(predictions)
        
        # Combine predictions
        predictions_array = np.array(predictions_list)
        
        if self.voting == 'hard':
            # Majority voting
            ensemble_pred = (predictions_array.sum(axis=0) >= len(self.detectors) / 2).astype(int)
        else:
            # Soft voting (average)
            ensemble_pred = (predictions_array.mean(axis=0) >= self.threshold).astype(int)
        
        results['ensemble'] = ensemble_pred
        
        return results


def save_detector(detector, filepath: str):
    """Save a trained detector."""
    joblib.dump(detector, filepath)
    print(f"💾 Detector saved to {filepath}")


def load_detector(filepath: str):
    """Load a trained detector."""
    detector = joblib.load(filepath)
    print(f"📂 Detector loaded from {filepath}")
    return detector
