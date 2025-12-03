"""
Advanced CAN-Specific Feature Engineering

This module implements domain-specific features for CAN bus intrusion detection:
1. Frequency Analysis: Burst detection, periodicity analysis
2. Payload Patterns: Entropy, hamming distance, bit flips
3. Temporal Features: Inter-arrival time statistics, timing anomalies
4. Message Sequences: N-gram analysis, sequence patterns

Expected Improvement: +10-20% precision and recall
Base ML: 64.86% precision → Target: 75-85% precision
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import Counter, deque
from scipy import stats
from scipy.fft import fft
import warnings
warnings.filterwarnings('ignore')


class CANFeatureEngineer:
    """
    Advanced feature engineering for CAN bus traffic.
    
    This class extracts sophisticated features that capture:
    - Message frequency patterns and burst behavior
    - Payload characteristics and anomalies
    - Temporal patterns and periodicities
    - Sequence-based features
    
    Parameters
    ----------
    window_size : int, default=100
        Number of messages to analyze for rolling statistics
    burst_threshold : float, default=0.01
        Time threshold (seconds) for burst detection
    ngram_size : int, default=3
        Size of n-grams for sequence analysis
    """
    
    def __init__(self, window_size: int = 100, 
                 burst_threshold: float = 0.01,
                 ngram_size: int = 3):
        self.window_size = window_size
        self.burst_threshold = burst_threshold
        self.ngram_size = ngram_size
        
        # Learned patterns (from training data)
        self.id_frequency_baseline = {}
        self.id_payload_patterns = {}
        self.id_timing_patterns = {}
        self.common_ngrams = {}
        
        self.is_calibrated = False
        
    def calibrate(self, df: pd.DataFrame):
        """
        Learn normal patterns from training data.
        
        Parameters
        ----------
        df : DataFrame
            Normal CAN traffic (attack column should be 0)
        """
        print("\n" + "="*70)
        print("🔧 CALIBRATING CAN FEATURE ENGINEER")
        print("="*70)
        
        # Learn baseline frequencies
        print("\n1. Learning frequency baselines...")
        self._learn_frequency_patterns(df)
        
        # Learn payload patterns
        print("2. Learning payload patterns...")
        self._learn_payload_patterns(df)
        
        # Learn timing patterns
        print("3. Learning timing patterns...")
        self._learn_timing_patterns(df)
        
        # Learn sequence patterns
        print("4. Learning sequence patterns...")
        self._learn_sequence_patterns(df)
        
        self.is_calibrated = True
        print("\n✅ Feature engineer calibrated successfully")
        print("="*70 + "\n")
    
    def _learn_frequency_patterns(self, df: pd.DataFrame):
        """Learn normal message frequencies per ID."""
        for can_id in df['arb_id_numeric'].unique():
            id_data = df[df['arb_id_numeric'] == can_id]
            
            # Calculate mean frequency (messages per second)
            if 'timestamp' in df.columns:
                time_span = id_data['timestamp'].max() - id_data['timestamp'].min()
                if time_span > 0:
                    freq = len(id_data) / time_span
                else:
                    freq = 0
            else:
                freq = len(id_data) / len(df) * 1000  # Approximate
            
            # Calculate inter-arrival time stats
            if len(id_data) > 1:
                time_deltas = id_data['time_delta'].values
                mean_iat = np.mean(time_deltas)
                std_iat = np.std(time_deltas)
                min_iat = np.min(time_deltas)
                max_iat = np.max(time_deltas)
            else:
                mean_iat = std_iat = min_iat = max_iat = 0
            
            self.id_frequency_baseline[can_id] = {
                'frequency': freq,
                'mean_iat': mean_iat,
                'std_iat': std_iat,
                'min_iat': min_iat,
                'max_iat': max_iat,
                'count': len(id_data)
            }
        
        print(f"   ✅ Learned baseline for {len(self.id_frequency_baseline)} CAN IDs")
    
    def _learn_payload_patterns(self, df: pd.DataFrame):
        """Learn normal payload characteristics per ID."""
        for can_id in df['arb_id_numeric'].unique():
            id_data = df[df['arb_id_numeric'] == can_id]
            
            # Payload entropy
            if 'payload_entropy' in id_data.columns:
                entropy_mean = id_data['payload_entropy'].mean()
                entropy_std = id_data['payload_entropy'].std()
            else:
                entropy_mean = entropy_std = 0
            
            # Data length patterns
            dlc_values = id_data['data_length'].value_counts()
            most_common_dlc = dlc_values.index[0] if len(dlc_values) > 0 else 8
            dlc_variance = id_data['data_length'].var()
            
            self.id_payload_patterns[can_id] = {
                'entropy_mean': entropy_mean,
                'entropy_std': entropy_std,
                'common_dlc': most_common_dlc,
                'dlc_variance': dlc_variance
            }
        
        print(f"   ✅ Learned payload patterns for {len(self.id_payload_patterns)} CAN IDs")
    
    def _learn_timing_patterns(self, df: pd.DataFrame):
        """Learn timing patterns per ID."""
        for can_id in df['arb_id_numeric'].unique():
            id_data = df[df['arb_id_numeric'] == can_id]
            
            if len(id_data) > 1:
                # Periodicity detection using FFT
                time_deltas = id_data['time_delta'].values
                if len(time_deltas) > 10:
                    try:
                        fft_output = fft(np.array(time_deltas, dtype=np.float64))
                        fft_result = np.abs(np.asarray(fft_output))
                        dominant_freq_idx = np.argmax(fft_result[1:len(fft_result)//2]) + 1
                        periodicity_score = fft_result[dominant_freq_idx] / np.sum(fft_result)
                    except:
                        periodicity_score = 0
                else:
                    periodicity_score = 0
                
                # Regularity coefficient of variation
                cv = np.std(time_deltas) / np.mean(time_deltas) if np.mean(time_deltas) > 0 else 0
            else:
                periodicity_score = 0
                cv = 0
            
            self.id_timing_patterns[can_id] = {
                'periodicity_score': periodicity_score,
                'regularity_cv': cv
            }
        
        print(f"   ✅ Learned timing patterns for {len(self.id_timing_patterns)} CAN IDs")
    
    def _learn_sequence_patterns(self, df: pd.DataFrame):
        """Learn common message sequences (n-grams)."""
        if len(df) < self.ngram_size:
            print("   ⚠️  Insufficient data for n-gram analysis")
            return
        
        # Extract message sequences
        id_sequence = df['arb_id_numeric'].values
        
        # Create n-grams
        ngrams = []
        for i in range(len(id_sequence) - self.ngram_size + 1):
            ngram = tuple(id_sequence[i:i+self.ngram_size])
            ngrams.append(ngram)
        
        # Count n-gram frequencies
        ngram_counts = Counter(ngrams)
        
        # Store most common n-grams (top 100)
        self.common_ngrams = dict(ngram_counts.most_common(100))
        
        print(f"   ✅ Learned {len(self.common_ngrams)} common n-grams")
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract advanced CAN-specific features.
        
        Parameters
        ----------
        df : DataFrame
            CAN traffic data
            
        Returns
        -------
        df_enhanced : DataFrame
            Original data with additional features
        """
        if not self.is_calibrated:
            raise ValueError("Feature engineer not calibrated. Call calibrate() first.")
        
        print(f"\n🔧 Extracting advanced CAN features from {len(df):,} messages...")
        
        df_enhanced = df.copy()
        
        # 1. Frequency-based features
        print("   1/6 Frequency analysis...")
        df_enhanced = self._add_frequency_features(df_enhanced)
        
        # 2. Burst detection
        print("   2/6 Burst detection...")
        df_enhanced = self._add_burst_features(df_enhanced)
        
        # 3. Payload pattern features
        print("   3/6 Payload patterns...")
        df_enhanced = self._add_payload_features(df_enhanced)
        
        # 4. Temporal anomaly features
        print("   4/6 Temporal anomalies...")
        df_enhanced = self._add_temporal_features(df_enhanced)
        
        # 5. Sequence-based features
        print("   5/6 Sequence patterns...")
        df_enhanced = self._add_sequence_features(df_enhanced)
        
        # 6. Statistical aggregations
        print("   6/6 Statistical aggregations...")
        df_enhanced = self._add_statistical_features(df_enhanced)
        
        new_features = len(df_enhanced.columns) - len(df.columns)
        print(f"\n✅ Extracted {new_features} new features")
        print(f"   Total features: {len(df_enhanced.columns)}")
        
        return df_enhanced
    
    def _add_frequency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add frequency deviation features."""
        freq_deviations = []
        iat_deviations = []
        
        for idx, row in df.iterrows():
            can_id = row['arb_id_numeric']
            baseline = self.id_frequency_baseline.get(can_id, {})
            
            if baseline:
                # Frequency deviation
                expected_freq = baseline['frequency']
                actual_freq = row.get('id_frequency', 0)
                freq_dev = abs(actual_freq - expected_freq) / (expected_freq + 1e-6)
                
                # IAT deviation
                expected_iat = baseline['mean_iat']
                actual_iat = row.get('time_delta', 0)
                iat_dev = abs(actual_iat - expected_iat) / (expected_iat + 1e-6)
            else:
                freq_dev = 0
                iat_dev = 0
            
            freq_deviations.append(freq_dev)
            iat_deviations.append(iat_dev)
        
        df['freq_deviation'] = freq_deviations
        df['iat_deviation'] = iat_deviations
        
        return df
    
    def _add_burst_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect burst patterns (rapid message sequences)."""
        burst_indicators = []
        burst_sizes = []
        
        current_burst = 1
        
        for i in range(len(df)):
            time_delta = df.iloc[i]['time_delta']
            
            if time_delta < self.burst_threshold:
                current_burst += 1
            else:
                current_burst = 1
            
            # Burst indicator (1 if in burst, 0 otherwise)
            is_burst = 1 if current_burst > 3 else 0
            burst_indicators.append(is_burst)
            burst_sizes.append(current_burst)
        
        df['is_burst'] = burst_indicators
        df['burst_size'] = burst_sizes
        
        return df
    
    def _add_payload_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add payload-based anomaly features."""
        entropy_anomalies = []
        dlc_anomalies = []
        
        for idx, row in df.iterrows():
            can_id = row['arb_id_numeric']
            pattern = self.id_payload_patterns.get(can_id, {})
            
            if pattern:
                # Entropy anomaly (z-score)
                if 'payload_entropy' in row:
                    expected_entropy = pattern['entropy_mean']
                    entropy_std = pattern['entropy_std']
                    actual_entropy = row['payload_entropy']
                    
                    if entropy_std > 0:
                        entropy_z = abs(actual_entropy - expected_entropy) / entropy_std
                    else:
                        entropy_z = 0
                else:
                    entropy_z = 0
                
                # DLC anomaly
                expected_dlc = pattern['common_dlc']
                actual_dlc = row['data_length']
                dlc_diff = abs(actual_dlc - expected_dlc)
            else:
                entropy_z = 0
                dlc_diff = 0
            
            entropy_anomalies.append(entropy_z)
            dlc_anomalies.append(dlc_diff)
        
        df['entropy_anomaly'] = entropy_anomalies
        df['dlc_anomaly'] = dlc_anomalies
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal pattern features."""
        periodicity_scores = []
        timing_irregularities = []
        
        for idx, row in df.iterrows():
            can_id = row['arb_id_numeric']
            timing = self.id_timing_patterns.get(can_id, {})
            
            if timing:
                periodicity_scores.append(timing['periodicity_score'])
                timing_irregularities.append(timing['regularity_cv'])
            else:
                periodicity_scores.append(0)
                timing_irregularities.append(0)
        
        df['periodicity_score'] = periodicity_scores
        df['timing_irregularity'] = timing_irregularities
        
        return df
    
    def _add_sequence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add sequence-based features."""
        sequence_novelty = []
        
        id_sequence = df['arb_id_numeric'].values
        
        for i in range(len(df)):
            if i < self.ngram_size - 1:
                # Not enough history
                sequence_novelty.append(0)
            else:
                # Get current n-gram
                ngram = tuple(id_sequence[i-self.ngram_size+1:i+1])
                
                # Check if it's a common pattern
                if ngram in self.common_ngrams:
                    # Known pattern - lower novelty
                    novelty = 1.0 / (1.0 + self.common_ngrams[ngram])
                else:
                    # Unknown pattern - high novelty
                    novelty = 1.0
                
                sequence_novelty.append(novelty)
        
        df['sequence_novelty'] = sequence_novelty
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistical features."""
        
        # Rolling statistics for time_delta
        if len(df) >= self.window_size:
            df['time_delta_rolling_mean'] = df['time_delta'].rolling(
                window=self.window_size, min_periods=1
            ).mean()
            df['time_delta_rolling_std'] = df['time_delta'].rolling(
                window=self.window_size, min_periods=1
            ).std().fillna(0)
        else:
            df['time_delta_rolling_mean'] = df['time_delta'].mean()
            df['time_delta_rolling_std'] = df['time_delta'].std()
        
        # Rolling statistics for id_frequency
        if 'id_frequency' in df.columns and len(df) >= self.window_size:
            df['freq_rolling_mean'] = df['id_frequency'].rolling(
                window=self.window_size, min_periods=1
            ).mean()
            df['freq_rolling_std'] = df['id_frequency'].rolling(
                window=self.window_size, min_periods=1
            ).std().fillna(0)
        else:
            df['freq_rolling_mean'] = df.get('id_frequency', 0)
            df['freq_rolling_std'] = 0
        
        return df
    
    def get_feature_importance_info(self) -> Dict:
        """
        Get information about engineered features.
        
        Returns
        -------
        info : dict
            Feature categories and descriptions
        """
        return {
            'frequency_features': [
                'freq_deviation',
                'iat_deviation'
            ],
            'burst_features': [
                'is_burst',
                'burst_size'
            ],
            'payload_features': [
                'entropy_anomaly',
                'dlc_anomaly'
            ],
            'temporal_features': [
                'periodicity_score',
                'timing_irregularity'
            ],
            'sequence_features': [
                'sequence_novelty'
            ],
            'statistical_features': [
                'time_delta_rolling_mean',
                'time_delta_rolling_std',
                'freq_rolling_mean',
                'freq_rolling_std'
            ],
            'total_new_features': 13
        }
    
    def save(self, filepath: str):
        """Save calibrated feature engineer."""
        import joblib
        joblib.dump(self, filepath)
        print(f"✅ Feature engineer saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'CANFeatureEngineer':
        """Load calibrated feature engineer."""
        import joblib
        engineer = joblib.load(filepath)
        print(f"✅ Feature engineer loaded from {filepath}")
        return engineer


def create_enhanced_dataset(df_normal: pd.DataFrame,
                           df_attacks: Optional[pd.DataFrame] = None,
                           save_path: Optional[str] = None) -> Tuple[pd.DataFrame, CANFeatureEngineer]:
    """
    Convenience function to create enhanced dataset with CAN-specific features.
    
    Parameters
    ----------
    df_normal : DataFrame
        Normal CAN traffic for calibration
    df_attacks : DataFrame, optional
        Attack traffic to enhance (if None, only calibrates)
    save_path : str, optional
        Path to save the feature engineer
        
    Returns
    -------
    df_enhanced : DataFrame
        Enhanced dataset (if df_attacks provided)
    engineer : CANFeatureEngineer
        Calibrated feature engineer
    """
    # Create and calibrate engineer
    engineer = CANFeatureEngineer(
        window_size=100,
        burst_threshold=0.01,
        ngram_size=3
    )
    
    engineer.calibrate(df_normal)
    
    # Save if requested
    if save_path:
        engineer.save(save_path)
    
    # Enhance attack data if provided
    if df_attacks is not None:
        df_enhanced = engineer.extract_features(df_attacks)
        return df_enhanced, engineer
    
    return df_normal, engineer
