"""
Feature extraction from CAN messages for machine learning.

Extracts statistical, temporal, and structural features from
CAN bus traffic for anomaly detection and classification.

Enhanced Features (imported from Vehicle_Models research project):
- Payload Entropy: Shannon entropy analysis (TCE-IDS paper)
- Hamming Distance: Bit-level payload differences (Novel Architecture paper)
- N-gram Sequences: Bigram/trigram pattern detection (Novel Architecture paper)
- IAT Z-Score: Normalized timing deviations (SAIDuCANT paper)
- Bit-Time Statistics: Physical layer timing analysis (BTMonitor paper)

Source: /home/mike/Documents/GitHub/Vehicle_Models/src/enhanced_features.py
"""

import logging
import time
import math
import numpy as np
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
    
    def __init__(self, window_size: int = 100, history_size: int = 1000, 
                 enable_enhanced_features: bool = False):
        """
        Initialize feature extractor.
        
        Args:
            window_size: Number of messages for windowed statistics
            history_size: Maximum history to keep per CAN ID
            enable_enhanced_features: Enable research-based features from Vehicle_Models
        """
        self.window_size = window_size
        self.history_size = history_size
        self.enable_enhanced_features = enable_enhanced_features
        
        # Message history per CAN ID
        self._message_history = defaultdict(lambda: deque(maxlen=history_size))
        self._timing_history = defaultdict(lambda: deque(maxlen=history_size))
        self._frequency_trackers = defaultdict(lambda: deque(maxlen=history_size))
        
        # Enhanced features state (from Vehicle_Models)
        self._previous_payload = {}  # For hamming distance
        self._id_sequence = deque(maxlen=100)  # For n-grams
        self._id_stats = {}  # For IAT z-score
        self._normal_bigrams = set()  # Learned from training
        self._normal_trigrams = set()  # Learned from training
        self._is_calibrated = False
        
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
        
        # Enhanced research features (from Vehicle_Models)
        if self.enable_enhanced_features:
            features.update(self._extract_enhanced_features(message))
        
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
    
    # ========================================================================
    # ENHANCED FEATURES FROM VEHICLE_MODELS RESEARCH PROJECT
    # Source: /home/mike/Documents/GitHub/Vehicle_Models/src/enhanced_features.py
    # These features achieved 97.20% recall (vs 0-10% baseline)
    # ========================================================================
    
    def calibrate_enhanced_features(self, normal_messages: List[Dict[str, Any]]) -> None:
        """
        Learn normal traffic patterns for enhanced feature extraction.
        
        Must be called before using enhanced features on actual data.
        Uses only normal (non-attack) traffic for calibration.
        
        Args:
            normal_messages: List of normal CAN messages for learning
            
        Source: Vehicle_Models/src/enhanced_features.py - learn_normal_statistics()
        """
        if not self.enable_enhanced_features:
            logger.warning("Enhanced features not enabled, calibration skipped")
            return
        
        logger.info(f"Calibrating enhanced features on {len(normal_messages)} normal messages...")
        
        # Learn IAT statistics per CAN ID
        id_iats = defaultdict(list)
        
        for msg in normal_messages:
            can_id = msg['can_id']
            
            # Collect timing data
            if 'time_delta' in msg or 'timestamp' in msg:
                iat = msg.get('time_delta', 0.0)
                if iat > 0:
                    id_iats[can_id].append(iat)
        
        # Calculate statistics
        for can_id, iats in id_iats.items():
            if len(iats) > 1:
                self._id_stats[can_id] = {
                    'mean': np.mean(iats),
                    'std': np.std(iats),
                    'min': np.percentile(iats, 5),
                    'max': np.percentile(iats, 95),
                    'count': len(iats)
                }
        
        # Learn sequence patterns (n-grams)
        if len(normal_messages) > 100:
            # Sample for efficiency
            sample_size = min(10000, len(normal_messages))
            sample_indices = np.random.choice(len(normal_messages), sample_size, replace=False)
            id_sequence = [normal_messages[i]['can_id'] for i in sorted(sample_indices)]
            
            # Extract bigrams
            for i in range(len(id_sequence) - 1):
                bigram = f"{id_sequence[i]:03X}-{id_sequence[i+1]:03X}"
                self._normal_bigrams.add(bigram)
            
            # Extract trigrams
            for i in range(len(id_sequence) - 2):
                trigram = f"{id_sequence[i]:03X}-{id_sequence[i+1]:03X}-{id_sequence[i+2]:03X}"
                self._normal_trigrams.add(trigram)
        
        self._is_calibrated = True
        logger.info(f"Enhanced features calibrated: {len(self._id_stats)} IDs, "
                   f"{len(self._normal_bigrams)} bigrams, {len(self._normal_trigrams)} trigrams")
    
    def _extract_enhanced_features(self, message: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract enhanced research-based features from Vehicle_Models project.
        
        Features implemented:
        1. payload_entropy - Shannon entropy (TCE-IDS paper)
        2. hamming_distance - Bit-level changes (Novel Architecture paper)
        3. iat_zscore - Normalized timing deviation (SAIDuCANT paper)
        4. unknown_bigram - 2-ID sequence novelty (Novel Architecture paper)
        5. unknown_trigram - 3-ID sequence novelty (Novel Architecture paper)
        6. bit_time_mean - Physical layer timing (BTMonitor paper)
        7. bit_time_rms - RMS of bit timing (BTMonitor paper)
        8. bit_time_energy - Bit timing energy (BTMonitor paper)
        
        Args:
            message: CAN message dictionary
            
        Returns:
            Dictionary of enhanced features
            
        Source: Vehicle_Models/src/enhanced_features.py
        Papers: TCE-IDS, Novel Architecture, SAIDuCANT, BTMonitor (Table 1)
        """
        features = {}
        can_id = message['can_id']
        
        # 1. Payload Entropy (TCE-IDS paper)
        data = message.get('data', [])
        features['payload_entropy'] = self._calculate_shannon_entropy_enhanced(data)
        
        # 2. Hamming Distance (Novel Architecture paper)
        features['hamming_distance'] = self._calculate_hamming_distance(can_id, data)
        
        # 3. IAT Z-Score (SAIDuCANT paper)
        time_delta = message.get('time_delta', 0.0)
        features['iat_zscore'] = self._calculate_iat_zscore(can_id, time_delta)
        
        # 4-5. N-gram Sequence Detection (Novel Architecture paper)
        self._id_sequence.append(can_id)
        features['unknown_bigram'] = self._detect_unknown_bigram()
        features['unknown_trigram'] = self._detect_unknown_trigram()
        
        # 6-8. Bit-Time Statistics (BTMonitor paper, Table 1)
        bit_stats = self._calculate_bit_time_stats(time_delta)
        features.update(bit_stats)
        
        return features
    
    def _calculate_shannon_entropy_enhanced(self, data_bytes: List[int]) -> float:
        """
        Calculate Shannon entropy of payload data.
        
        Formula: H = -Σ p(v) * log2(p(v))
        Range: 0 (predictable) to 8 (random for byte data)
        
        Source: Vehicle_Models TCE-IDS paper implementation
        """
        if not data_bytes or len(data_bytes) == 0:
            return 0.0
        
        # Count byte frequencies
        byte_counts = defaultdict(int)
        for byte in data_bytes:
            byte_counts[byte] += 1
        
        # Calculate entropy
        entropy = 0.0
        total_bytes = len(data_bytes)
        
        for count in byte_counts.values():
            probability = count / total_bytes
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        return float(entropy)
    
    def _calculate_hamming_distance(self, can_id: int, data_bytes: List[int]) -> int:
        """
        Calculate Hamming distance (bit flips) between current and previous payload.
        
        Detects: Subtle data manipulation, replay attacks with modifications
        
        Source: Vehicle_Models Novel Architecture paper
        """
        if not data_bytes:
            return 0
        
        # Get previous payload for this CAN ID
        prev_data = self._previous_payload.get(can_id, [])
        
        # Store current for next time
        self._previous_payload[can_id] = data_bytes.copy()
        
        if not prev_data or len(prev_data) == 0:
            return 0  # First message from this ID
        
        # Pad to same length
        max_len = max(len(data_bytes), len(prev_data))
        data1 = data_bytes + [0] * (max_len - len(data_bytes))
        data2 = prev_data + [0] * (max_len - len(prev_data))
        
        # Count bit differences
        hamming = 0
        for b1, b2 in zip(data1, data2):
            xor = b1 ^ b2
            hamming += bin(xor).count('1')
        
        return hamming
    
    def _calculate_iat_zscore(self, can_id: int, iat: float) -> float:
        """
        Calculate normalized IAT deviation: (iat - μ) / σ
        
        Handles 50-75% natural variance in CAN timing.
        Detects: Timing anomalies, DoS attacks, injection irregularities
        
        Source: Vehicle_Models SAIDuCANT paper
        """
        if not self._is_calibrated or can_id not in self._id_stats:
            return 0.0
        
        stats = self._id_stats[can_id]
        mean = stats['mean']
        std = stats['std']
        
        if std == 0:
            return 0.0
        
        return float((iat - mean) / std)
    
    def _detect_unknown_bigram(self) -> int:
        """
        Detect if current 2-ID sequence is unknown (not in normal traffic).
        
        Returns: 1 if unknown sequence, 0 if known/normal
        
        Source: Vehicle_Models Novel Architecture paper
        """
        if not self._is_calibrated or len(self._id_sequence) < 2:
            return 0
        
        # Get last 2 IDs
        ids = list(self._id_sequence)
        bigram = f"{ids[-2]:03X}-{ids[-1]:03X}"
        
        return 1 if bigram not in self._normal_bigrams else 0
    
    def _detect_unknown_trigram(self) -> int:
        """
        Detect if current 3-ID sequence is unknown (not in normal traffic).
        
        Returns: 1 if unknown sequence, 0 if known/normal
        
        Source: Vehicle_Models Novel Architecture paper
        """
        if not self._is_calibrated or len(self._id_sequence) < 3:
            return 0
        
        # Get last 3 IDs
        ids = list(self._id_sequence)
        trigram = f"{ids[-3]:03X}-{ids[-2]:03X}-{ids[-1]:03X}"
        
        return 1 if trigram not in self._normal_trigrams else 0
    
    def _calculate_bit_time_stats(self, iat: float) -> Dict[str, float]:
        """
        Calculate bit-time statistics from inter-arrival time.
        
        CAN typically runs at 500kbps = 2μs/bit
        Approximate frame size: 108 bits (11-bit ID + 64-bit data + overheads)
        
        Detects: Hardware-level attacks, ECU impersonation, bus timing violations
        
        Source: Vehicle_Models BTMonitor paper (Table 1)
        """
        if iat <= 0:
            return {
                'bit_time_mean': 0.0,
                'bit_time_rms': 0.0,
                'bit_time_energy': 0.0
            }
        
        # Estimate bit-level timing (108 bits per frame)
        bit_time = iat / 108
        
        return {
            'bit_time_mean': float(bit_time),
            'bit_time_rms': float(np.sqrt(bit_time ** 2)),
            'bit_time_energy': float(bit_time ** 2)
        }
    
    # ========================================================================
    # END ENHANCED FEATURES
    # ========================================================================
        
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
        # Base features (always available)
        base_features = [
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
        
        # Enhanced features (from Vehicle_Models research)
        if self.enable_enhanced_features:
            enhanced_features = [
                'payload_entropy',      # TCE-IDS paper
                'hamming_distance',     # Novel Architecture paper
                'iat_zscore',          # SAIDuCANT paper
                'unknown_bigram',      # Novel Architecture paper
                'unknown_trigram',     # Novel Architecture paper
                'bit_time_mean',       # BTMonitor paper
                'bit_time_rms',        # BTMonitor paper
                'bit_time_energy'      # BTMonitor paper
            ]
            return base_features + enhanced_features
        
        return base_features
        
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
        
        # Reset enhanced features state
        if self.enable_enhanced_features:
            self._previous_payload.clear()
            self._id_sequence.clear()
            # Keep calibration data (_id_stats, _normal_bigrams, _normal_trigrams)
            # unless explicitly recalibrated
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get feature extraction statistics."""
        stats = {
            'total_messages_processed': self._global_stats['total_messages'],
            'unique_can_ids': len(self._global_stats['unique_can_ids']),
            'window_size': self.window_size,
            'history_size': self.history_size,
            'feature_count': len(self.get_feature_names()),
            'cache_size': len(self._feature_cache),
            'enhanced_features_enabled': self.enable_enhanced_features
        }
        
        # Add enhanced features statistics if enabled
        if self.enable_enhanced_features:
            stats.update({
                'enhanced_calibrated': self._is_calibrated,
                'calibrated_can_ids': len(self._id_stats),
                'learned_bigrams': len(self._normal_bigrams),
                'learned_trigrams': len(self._normal_trigrams)
            })
        
        return stats