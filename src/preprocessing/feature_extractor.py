"""
Feature extraction from CAN messages for machine learning.

Extracts statistical, temporal, and structural features from
CAN bus traffic for anomaly detection and classification.
"""

import logging
import time
import math
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extract features from CAN messages for machine learning analysis.
    
    Generates features including:
    - Message-level features (ID, DLC, data patterns)
    - Statistical features (frequency, timing, entropy) 
    - Behavioral features (communication patterns)
    """
    
    def __init__(self, window_size: int = 100, history_size: int = 1000):
        """
        Initialize feature extractor.
        
        Args:
            window_size: Number of messages for windowed statistics
            history_size: Maximum history to keep per CAN ID
        """
        self.window_size = window_size
        self.history_size = history_size
        
        # Message history per CAN ID
        self._message_history = defaultdict(lambda: deque(maxlen=history_size))
        self._timing_history = defaultdict(lambda: deque(maxlen=history_size))
        self._frequency_trackers = defaultdict(lambda: deque(maxlen=history_size))
        
        # Global statistics
        self._global_stats = {
            'total_messages': 0,
            'unique_can_ids': set(),
            'start_time': None,
            'last_update': None
        }
        
        # Feature cache for performance
        self._feature_cache = {}
        self._cache_timeout = 1.0  # seconds
        
    def extract_features(self, message: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract comprehensive features from a CAN message.
        
        Args:
            message: CAN message dictionary
            
        Returns:
            Dictionary of feature names and values
        """
        start_time = time.time()
        
        # Update internal state
        self._update_state(message)
        
        # Extract different feature categories
        features = {}
        
        # Basic message features
        features.update(self._extract_basic_features(message))
        
        # Statistical features
        features.update(self._extract_statistical_features(message))
        
        # Temporal features
        features.update(self._extract_temporal_features(message))
        
        # Data pattern features
        features.update(self._extract_data_features(message))
        
        # Behavioral features
        features.update(self._extract_behavioral_features(message))
        
        # Communication pattern features
        features.update(self._extract_communication_features(message))
        
        extraction_time = time.time() - start_time
        logger.debug(f"Feature extraction took {extraction_time*1000:.2f}ms")
        
        return features
        
    def _update_state(self, message: Dict[str, Any]) -> None:
        """Update internal state with new message."""
        can_id = message['can_id']
        timestamp = message['timestamp']
        
        # Update global statistics
        self._global_stats['total_messages'] += 1
        self._global_stats['unique_can_ids'].add(can_id)
        self._global_stats['last_update'] = timestamp
        
        if self._global_stats['start_time'] is None:
            self._global_stats['start_time'] = timestamp
            
        # Update message history
        self._message_history[can_id].append(message)
        
        # Update timing information
        timing_history = self._timing_history[can_id]
        if timing_history:
            interval = timestamp - timing_history[-1]
            timing_history.append(interval)
        else:
            timing_history.append(0.0)
            
        # Update frequency tracking
        freq_tracker = self._frequency_trackers[can_id]
        freq_tracker.append(timestamp)
        
        # Clean old frequency data (keep last 60 seconds)
        cutoff_time = timestamp - 60
        while freq_tracker and freq_tracker[0] < cutoff_time:
            freq_tracker.popleft()
            
    def _extract_basic_features(self, message: Dict[str, Any]) -> Dict[str, float]:
        """Extract basic message-level features."""
        features = {
            'can_id': float(message['can_id']),
            'dlc': float(message['dlc']),
            'is_extended': float(message.get('is_extended', False)),
            'is_remote': float(message.get('is_remote', False)),
            'is_error': float(message.get('is_error', False))
        }
        
        # Data bytes (pad to 8 bytes)
        data = message['data'][:8]
        for i in range(8):
            if i < len(data):
                features[f'data_byte_{i}'] = float(data[i])
            else:
                features[f'data_byte_{i}'] = 0.0
                
        return features
        
    def _extract_statistical_features(self, message: Dict[str, Any]) -> Dict[str, float]:
        """Extract statistical features from data payload."""
        data = message['data']
        
        features = {}
        
        if data:
            features['data_mean'] = float(statistics.mean(data))
            features['data_median'] = float(statistics.median(data))
            features['data_std'] = float(statistics.stdev(data) if len(data) > 1 else 0)
            features['data_min'] = float(min(data))
            features['data_max'] = float(max(data))
            features['data_range'] = float(max(data) - min(data))
            features['data_sum'] = float(sum(data))
            
            # Entropy
            features['data_entropy'] = self._calculate_entropy(data)
            
            # Zero bytes count
            features['zero_bytes'] = float(data.count(0))
            features['max_bytes'] = float(data.count(255))
            
        else:
            # No data payload
            for key in ['data_mean', 'data_median', 'data_std', 'data_min', 
                       'data_max', 'data_range', 'data_sum', 'data_entropy',
                       'zero_bytes', 'max_bytes']:
                features[key] = 0.0
                
        return features
        
    def _extract_temporal_features(self, message: Dict[str, Any]) -> Dict[str, float]:
        """Extract temporal/timing features."""
        can_id = message['can_id']
        features = {}
        
        # Frequency features
        freq_tracker = self._frequency_trackers[can_id]
        
        if len(freq_tracker) >= 2:
            # Messages in last N seconds
            current_time = message['timestamp']
            
            for window in [1, 5, 10, 30]:
                window_start = current_time - window
                count = sum(1 for t in freq_tracker if t >= window_start)
                features[f'freq_last_{window}s'] = float(count / window)
        else:
            for window in [1, 5, 10, 30]:
                features[f'freq_last_{window}s'] = 0.0
                
        # Timing interval features
        timing_history = self._timing_history[can_id]
        
        if len(timing_history) >= 3:
            recent_intervals = list(timing_history)[-10:]  # Last 10 intervals
            
            features['interval_mean'] = float(statistics.mean(recent_intervals))
            features['interval_std'] = float(statistics.stdev(recent_intervals) if len(recent_intervals) > 1 else 0)
            features['interval_min'] = float(min(recent_intervals))
            features['interval_max'] = float(max(recent_intervals))
            
            # Jitter (coefficient of variation)
            if features['interval_mean'] > 0:
                features['interval_jitter'] = features['interval_std'] / features['interval_mean']
            else:
                features['interval_jitter'] = 0.0
                
        else:
            for key in ['interval_mean', 'interval_std', 'interval_min', 
                       'interval_max', 'interval_jitter']:
                features[key] = 0.0
                
        return features
        
    def _extract_data_features(self, message: Dict[str, Any]) -> Dict[str, float]:
        """Extract data pattern and structure features."""
        data = message['data']
        features = {}
        
        if not data:
            # Empty data features
            for key in ['pattern_repetition', 'sequential_pattern', 
                       'alternating_pattern', 'ascending_bytes', 'descending_bytes']:
                features[key] = 0.0
            return features
            
        # Pattern repetition
        if len(data) >= 2:
            repetitions = 0
            for i in range(len(data) - 1):
                if data[i] == data[i + 1]:
                    repetitions += 1
            features['pattern_repetition'] = float(repetitions / (len(data) - 1))
        else:
            features['pattern_repetition'] = 0.0
            
        # Sequential patterns
        ascending = 0
        descending = 0
        
        for i in range(len(data) - 1):
            if data[i + 1] == data[i] + 1:
                ascending += 1
            elif data[i + 1] == data[i] - 1:
                descending += 1
                
        if len(data) > 1:
            features['ascending_bytes'] = float(ascending / (len(data) - 1))
            features['descending_bytes'] = float(descending / (len(data) - 1))
        else:
            features['ascending_bytes'] = 0.0
            features['descending_bytes'] = 0.0
            
        # Alternating pattern (0101 or ABAB)
        alternating = 0
        if len(data) >= 4:
            for i in range(len(data) - 3):
                if data[i] == data[i + 2] and data[i + 1] == data[i + 3]:
                    alternating += 1
            features['alternating_pattern'] = float(alternating / max(1, len(data) - 3))
        else:
            features['alternating_pattern'] = 0.0
            
        # Sequential pattern (counting up/down)
        sequential_up = all(data[i] <= data[i + 1] for i in range(len(data) - 1)) if len(data) > 1 else False
        sequential_down = all(data[i] >= data[i + 1] for i in range(len(data) - 1)) if len(data) > 1 else False
        
        features['sequential_pattern'] = float(sequential_up or sequential_down)
        
        return features
        
    def _extract_behavioral_features(self, message: Dict[str, Any]) -> Dict[str, float]:
        """Extract behavioral features based on message history."""
        can_id = message['can_id']
        features = {}
        
        # Message history for this CAN ID
        history = self._message_history[can_id]
        
        if len(history) >= 2:
            # DLC consistency
            recent_dlcs = [msg['dlc'] for msg in list(history)[-10:]]
            dlc_consistency = len(set(recent_dlcs)) == 1  # All same DLC
            features['dlc_consistency'] = float(dlc_consistency)
            
            # Data change rate
            current_data = message['data']
            prev_data = history[-2]['data']
            
            if len(current_data) == len(prev_data):
                changes = sum(1 for i in range(len(current_data)) 
                             if current_data[i] != prev_data[i])
                features['data_change_rate'] = float(changes / max(1, len(current_data)))
            else:
                features['data_change_rate'] = 1.0  # Different lengths = major change
                
            # Payload variance over recent messages
            if len(history) >= 5:
                payload_variances = []
                recent_messages = list(history)[-5:]
                
                max_dlc = max(msg['dlc'] for msg in recent_messages)
                
                for byte_pos in range(max_dlc):
                    byte_values = []
                    for msg in recent_messages:
                        if byte_pos < len(msg['data']):
                            byte_values.append(msg['data'][byte_pos])
                        else:
                            byte_values.append(0)
                    
                    if len(set(byte_values)) > 1:
                        payload_variances.append(statistics.stdev(byte_values))
                    else:
                        payload_variances.append(0.0)
                        
                features['payload_variance'] = float(statistics.mean(payload_variances) if payload_variances else 0.0)
            else:
                features['payload_variance'] = 0.0
                
        else:
            features['dlc_consistency'] = 1.0
            features['data_change_rate'] = 0.0
            features['payload_variance'] = 0.0
            
        return features
        
    def _extract_communication_features(self, message: Dict[str, Any]) -> Dict[str, float]:
        """Extract communication pattern features."""
        features = {}
        
        # Global communication patterns
        total_messages = self._global_stats['total_messages']
        unique_ids = len(self._global_stats['unique_can_ids'])
        
        features['total_messages'] = float(total_messages)
        features['unique_can_ids'] = float(unique_ids)
        
        if total_messages > 0:
            features['id_diversity'] = float(unique_ids / total_messages)
        else:
            features['id_diversity'] = 0.0
            
        # Time since start
        if self._global_stats['start_time']:
            runtime = message['timestamp'] - self._global_stats['start_time']
            features['runtime_seconds'] = float(runtime)
            
            if runtime > 0:
                features['global_message_rate'] = float(total_messages / runtime)
            else:
                features['global_message_rate'] = 0.0
        else:
            features['runtime_seconds'] = 0.0
            features['global_message_rate'] = 0.0
            
        # CAN ID characteristics
        can_id = message['can_id']
        features['can_id_normalized'] = float(can_id / 0x7FF)  # Normalize to 0-1
        
        # Priority estimation (lower CAN ID = higher priority)
        features['estimated_priority'] = float(1.0 - (can_id / 0x7FF))
        
        return features
        
    def _calculate_entropy(self, data: List[int]) -> float:
        """Calculate Shannon entropy of byte sequence."""
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
                entropy -= probability * math.log2(probability)
                
        return entropy
        
    def extract_batch_features(self, messages: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """
        Extract features from a batch of messages.
        
        Args:
            messages: List of CAN messages
            
        Returns:
            List of feature dictionaries
        """
        features_list = []
        
        for message in messages:
            features = self.extract_features(message)
            features_list.append(features)
            
        return features_list
        
    def get_feature_names(self) -> List[str]:
        """Get list of all possible feature names."""
        # This is a comprehensive list of all features that can be extracted
        return [
            # Basic features
            'can_id', 'dlc', 'is_extended', 'is_remote', 'is_error',
            
            # Data bytes
            'data_byte_0', 'data_byte_1', 'data_byte_2', 'data_byte_3',
            'data_byte_4', 'data_byte_5', 'data_byte_6', 'data_byte_7',
            
            # Statistical features
            'data_mean', 'data_median', 'data_std', 'data_min', 'data_max',
            'data_range', 'data_sum', 'data_entropy', 'zero_bytes', 'max_bytes',
            
            # Temporal features
            'freq_last_1s', 'freq_last_5s', 'freq_last_10s', 'freq_last_30s',
            'interval_mean', 'interval_std', 'interval_min', 'interval_max', 'interval_jitter',
            
            # Pattern features
            'pattern_repetition', 'sequential_pattern', 'alternating_pattern',
            'ascending_bytes', 'descending_bytes',
            
            # Behavioral features
            'dlc_consistency', 'data_change_rate', 'payload_variance',
            
            # Communication features
            'total_messages', 'unique_can_ids', 'id_diversity',
            'runtime_seconds', 'global_message_rate', 'can_id_normalized', 'estimated_priority'
        ]
        
    def reset_state(self) -> None:
        """Reset all internal state."""
        self._message_history.clear()
        self._timing_history.clear()
        self._frequency_trackers.clear()
        self._global_stats = {
            'total_messages': 0,
            'unique_can_ids': set(),
            'start_time': None,
            'last_update': None
        }
        self._feature_cache.clear()
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get feature extraction statistics."""
        return {
            'total_messages_processed': self._global_stats['total_messages'],
            'unique_can_ids': len(self._global_stats['unique_can_ids']),
            'window_size': self.window_size,
            'history_size': self.history_size,
            'feature_count': len(self.get_feature_names()),
            'cache_size': len(self._feature_cache)
        }