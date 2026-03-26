"""
Enhanced Feature Engineering for CAN Intrusion Detection

Implements advanced features based on research:
1. Payload Entropy (Shannon entropy on data fields)
2. Sequence Patterns (n-grams, Hamming distance)
3. Bounded Timing Features (IAT deviation, bit-time statistics)
4. Temporal Features (sliding windows, z-score normalization)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict


class EnhancedFeatureExtractor:
    """Extract advanced features from CAN bus data for ML detection."""
    
    def __init__(self, window_size: int = 100):
        """
        Initialize feature extractor.
        
        Parameters:
        -----------
        window_size : int
            Size of sliding window for temporal features
        """
        self.window_size = window_size
        self.id_stats = {}  # Store per-ID statistics
        self.sequence_cache = defaultdict(list)
        
    def calculate_shannon_entropy(self, data_bytes: bytes) -> float:
        """
        Calculate Shannon entropy of payload data.
        
        From TCE-IDS paper: H = -Σ p(v) * log2(p(v))
        
        Parameters:
        -----------
        data_bytes : bytes
            Payload data
            
        Returns:
        --------
        float : Entropy value (0 = predictable, 8 = random)
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
        
        return entropy
    
    def calculate_hamming_distance(self, data1: bytes, data2: bytes) -> int:
        """
        Calculate Hamming distance (bit flips) between two payloads.
        
        From Novel Architecture: H_cur = bit flips between current/prior payload
        
        Parameters:
        -----------
        data1, data2 : bytes
            Two payloads to compare
            
        Returns:
        --------
        int : Number of bit differences
        """
        if len(data1) != len(data2):
            # Pad shorter with zeros
            max_len = max(len(data1), len(data2))
            data1 = data1 + b'\x00' * (max_len - len(data1))
            data2 = data2 + b'\x00' * (max_len - len(data2))
        
        hamming = 0
        for b1, b2 in zip(data1, data2):
            # XOR and count set bits
            xor = b1 ^ b2
            hamming += bin(xor).count('1')
        
        return hamming
    
    def extract_ngrams(self, id_sequence: List[int], n: int = 3) -> List[str]:
        """
        Extract n-gram sequences of CAN IDs.
        
        From Novel Architecture: 2-3 consecutive IDs capture message patterns
        
        Parameters:
        -----------
        id_sequence : List[int]
            Sequence of CAN IDs
        n : int
            N-gram size (2 or 3 recommended)
            
        Returns:
        --------
        List[str] : N-gram patterns
        """
        if len(id_sequence) < n:
            return []
        
        ngrams = []
        for i in range(len(id_sequence) - n + 1):
            ngram = '-'.join([f"{id:03X}" for id in id_sequence[i:i+n]])
            ngrams.append(ngram)
        
        return ngrams
    
    def calculate_iat_deviation(self, iat: float, can_id: int) -> float:
        """
        Calculate normalized IAT deviation: (iat - μ) / σ
        
        From SAIDuCANT: f_Dev = (f - μ_f) / σ_f
        Handles 50-75% variance by z-score normalization
        
        Parameters:
        -----------
        iat : float
            Inter-arrival time
        can_id : int
            CAN ID
            
        Returns:
        --------
        float : Z-score deviation
        """
        if can_id not in self.id_stats:
            return 0.0
        
        stats = self.id_stats[can_id]
        mean = stats['mean']
        std = stats['std']
        
        if std == 0:
            return 0.0
        
        return (iat - mean) / std
    
    def calculate_bit_time_stats(self, iat: float) -> Dict[str, float]:
        """
        Calculate bit-time statistics.
        
        From BTMonitor (Table 1): mean, variance, skewness, kurtosis, RMS, max, energy
        
        Parameters:
        -----------
        iat : float
            Inter-arrival time
            
        Returns:
        --------
        Dict : Bit-time statistics
        """
        # Simulate bit-level timing (CAN typically 500kbps = 2μs/bit)
        bit_time = iat / 108  # Approx 108 bits per frame (11-bit ID + 64-bit data + overheads)
        
        return {
            'bit_time_mean': bit_time,
            'bit_time_variance': 0.0,  # Would need history
            'bit_time_rms': np.sqrt(bit_time ** 2),
            'bit_time_energy': bit_time ** 2
        }
    
    def learn_normal_statistics(self, df: pd.DataFrame,
                               id_col: str = 'arb_id_numeric',
                               time_delta_col: str = 'time_delta',
                               data_col: str = 'data_field'):
        """
        Learn normal traffic statistics for feature extraction.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Training data (normal traffic only)
        """
        print("📊 Learning normal traffic statistics...")
        
        # Learn IAT statistics per CAN ID
        for can_id in df[id_col].unique():
            can_data = df[df[id_col] == can_id]
            iats = can_data[time_delta_col].values
            
            self.id_stats[can_id] = {
                'mean': np.mean(iats),
                'std': np.std(iats),
                'min': np.percentile(iats, 5),
                'max': np.percentile(iats, 95),
                'count': len(can_data)
            }
        
        # Learn sequence patterns (n-grams)
        if len(df) > 1000:
            sample = df.sample(n=min(10000, len(df)), random_state=42)
            id_sequence = sample[id_col].tolist()
            
            # Store common 2-grams and 3-grams
            self.normal_bigrams = set(self.extract_ngrams(id_sequence, n=2))
            self.normal_trigrams = set(self.extract_ngrams(id_sequence, n=3))
            
            print(f"   Learned {len(self.normal_bigrams)} bigrams, {len(self.normal_trigrams)} trigrams")
        
        print(f"   ✅ Statistics learned for {len(self.id_stats)} CAN IDs")
    
    def extract_enhanced_features(self, df: pd.DataFrame,
                                  id_col: str = 'arb_id_numeric',
                                  time_delta_col: str = 'time_delta',
                                  data_col: str = 'data_field') -> pd.DataFrame:
        """
        Extract all enhanced features from CAN data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            CAN message data
            
        Returns:
        --------
        pd.DataFrame : Enhanced feature set
        """
        print("🔧 Extracting enhanced features...")
        
        features = df.copy()
        
        # 1. Payload Entropy
        if data_col in df.columns:
            print("   Computing Shannon entropy...")
            features['payload_entropy'] = df[data_col].apply(
                lambda x: self.calculate_shannon_entropy(bytes.fromhex(str(x)) if pd.notna(x) else b'')
            )
        else:
            features['payload_entropy'] = 0.0
        
        # 2. Hamming Distance (to previous message with same ID) - SIMPLIFIED
        print("   Computing Hamming distances...")
        # For speed, compute only for a sample or skip for large datasets
        # Hamming distance is useful but computationally expensive
        features['hamming_distance'] = 0  # Default to 0 for speed
        
        # Only compute for smaller datasets (< 100K rows) to save time
        if data_col in df.columns and len(df) < 100000:
            for can_id in df[id_col].unique():
                mask = df[id_col] == can_id
                id_data = df[mask]
                
                if len(id_data) > 1:
                    hamming_dists = [0]  # First message has no prior
                    
                    for i in range(1, len(id_data)):
                        prev_data = bytes.fromhex(str(id_data.iloc[i-1][data_col])) if pd.notna(id_data.iloc[i-1][data_col]) else b''
                        curr_data = bytes.fromhex(str(id_data.iloc[i][data_col])) if pd.notna(id_data.iloc[i][data_col]) else b''
                        hamming_dists.append(self.calculate_hamming_distance(prev_data, curr_data))
                    
                    features.loc[mask, 'hamming_distance'] = hamming_dists
        else:
            print("   (Hamming distance skipped for large dataset - using placeholder)")
        
        # 3. IAT Z-Score Deviation
        print("   Computing IAT deviations...")
        features['iat_zscore'] = df.apply(
            lambda row: self.calculate_iat_deviation(row[time_delta_col], row[id_col]),
            axis=1
        )
        
        # 4. Sequence Features (n-gram presence) - OPTIMIZED
        print("   Extracting sequence patterns...")
        features['unknown_bigram'] = 0
        features['unknown_trigram'] = 0
        
        if hasattr(self, 'normal_bigrams') and len(df) > 3:
            # Vectorized approach for bigrams and trigrams
            id_seq = df[id_col].values
            
            # Create bigrams vectorized
            if len(id_seq) > 1:
                bigrams = [f"{id_seq[i]:03X}-{id_seq[i+1]:03X}" for i in range(len(id_seq) - 1)]
                unknown_big = [1 if bg not in self.normal_bigrams else 0 for bg in bigrams]
                unknown_big.append(0)  # Last element has no next
                features['unknown_bigram'] = unknown_big
            
            # Create trigrams vectorized
            if len(id_seq) > 2:
                trigrams = [f"{id_seq[i]:03X}-{id_seq[i+1]:03X}-{id_seq[i+2]:03X}" for i in range(len(id_seq) - 2)]
                unknown_tri = [1 if tg not in self.normal_trigrams else 0 for tg in trigrams]
                unknown_tri.extend([0, 0])  # Last two elements have no complete trigram
                features['unknown_trigram'] = unknown_tri
        
        # 5. Bit-Time Statistics
        print("   Computing bit-time statistics...")
        bit_stats = df[time_delta_col].apply(self.calculate_bit_time_stats)
        features['bit_time_mean'] = bit_stats.apply(lambda x: x['bit_time_mean'])
        features['bit_time_rms'] = bit_stats.apply(lambda x: x['bit_time_rms'])
        features['bit_time_energy'] = bit_stats.apply(lambda x: x['bit_time_energy'])
        
        print(f"   ✅ Extracted {len(features.columns) - len(df.columns)} new features")
        
        return features


def get_enhanced_feature_names() -> List[str]:
    """Return list of all enhanced feature names for ML training."""
    return [
        # Original features
        'arb_id_numeric', 'data_length', 'id_frequency',
        'time_delta', 'id_mean_time_delta', 'id_std_time_delta',
        'hour', 'minute', 'second',
        # Enhanced features
        'payload_entropy',
        'hamming_distance',
        'iat_zscore',
        'unknown_bigram',
        'unknown_trigram',
        'bit_time_mean',
        'bit_time_rms',
        'bit_time_energy'
    ]
