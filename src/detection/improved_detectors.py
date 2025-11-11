"""
Improved Detectors Based on Research Papers

Implements:
1. Enhanced Isolation Forest (PCB-iForest-inspired with sub-sampling)
2. Hybrid Detection (Rules + ML post-processing)
3. Cross-Check Algorithms (from Novel Architecture paper)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import IsolationForest
from collections import defaultdict, Counter
import joblib


class ImprovedIsolationForestDetector:
    """
    Enhanced Isolation Forest with sub-sampling and tuning.
    
    Based on PCB-iForest and research recommendations:
    - Sub-sampled datasets (256-512 samples/tree)
    - Tuned contamination (0.01 for rare attacks)
    - High n_estimators (200+)
    - Time-series lag features
    """
    
    def __init__(self, 
                 n_estimators: int = 300,
                 max_samples: float = 0.5,
                 contamination: float = 0.01,
                 n_jobs: int = -1,
                 random_state: int = 42):
        """
        Initialize improved Isolation Forest.
        
        Parameters from research:
        - n_estimators=200+ : More trees = better anomaly separation
        - max_samples=0.5-0.8 : Sub-sampling reduces noise from variance
        - contamination=0.01 : Assumes 1% attacks (rare event)
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            n_jobs=n_jobs,
            random_state=random_state,
            bootstrap=True  # Enable sub-sampling
        )
        
        self.feature_names = None
        self.scaler_params = None
    
    def add_lag_features(self, df: pd.DataFrame, 
                         lag_cols: List[str] = ['time_delta', 'payload_entropy', 'hamming_distance'],
                         lags: List[int] = [1, 2, 3]) -> pd.DataFrame:
        """
        Add time-series lag features.
        
        From TGMARL: Lag features capture temporal dependencies.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input features
        lag_cols : List[str]
            Columns to create lags for
        lags : List[int]
            Lag offsets (e.g., [1, 2, 3] for t-1, t-2, t-3)
        """
        features = df.copy()
        
        for col in lag_cols:
            if col in df.columns:
                for lag in lags:
                    features[f'{col}_lag{lag}'] = df[col].shift(lag).fillna(0)
        
        return features
    
    def normalize_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Z-score normalization to handle variance.
        
        From research: Reduces 50-75% timing variance impact.
        """
        if fit:
            self.scaler_params = {
                'mean': np.mean(X, axis=0),
                'std': np.std(X, axis=0)
            }
        
        if self.scaler_params is None:
            return X
        
        mean = self.scaler_params['mean']
        std = self.scaler_params['std']
        std[std == 0] = 1  # Avoid division by zero
        
        return (X - mean) / std
    
    def fit(self, X: pd.DataFrame):
        """
        Train improved Isolation Forest.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Training features (normal traffic only)
        """
        print(f"🌲 Training Improved Isolation Forest...")
        print(f"   Config: {self.n_estimators} trees, {self.max_samples} max_samples, {self.contamination} contamination")
        
        # FIX: Remove lag features to match prediction input
        # Lag features cause feature mismatch (26 vs 17 features)
        # X_enhanced = self.add_lag_features(X)  # DISABLED
        X_enhanced = X  # Use base features only
        
        # Store base features for validation
        self.base_features = X.columns.tolist()
        self.feature_names = X_enhanced.columns.tolist()
        
        # Normalize
        X_array = X_enhanced.values
        X_normalized = self.normalize_features(X_array, fit=True)
        
        # Train
        self.model.fit(X_normalized)
        
        print(f"   ✅ Trained on {len(X)} samples with {len(self.feature_names)} features")
        print(f"   📋 Base features: {len(self.base_features)}")
        
        return self
    
    def predict(self, X) -> np.ndarray:
        """
        Predict anomalies.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Features for prediction
        
        Returns:
        --------
        np.ndarray : 1 = anomaly (attack), 0 = normal
        """
        # Convert numpy array to DataFrame if needed
        if isinstance(X, np.ndarray):
            # Use base features (no lag features)
            if hasattr(self, 'base_features'):
                X = pd.DataFrame(X, columns=self.base_features[:X.shape[1]])
            else:
                # Fallback for old models
                base_features = [f for f in self.feature_names if '_lag' not in f]  # type: ignore[union-attr]
                X = pd.DataFrame(X, columns=base_features[:X.shape[1]])
        
        # FIX: No lag features in prediction (matching training)
        # X_enhanced = self.add_lag_features(X)  # DISABLED
        X_enhanced = X  # Use base features directly
        
        # Validate feature alignment and reorder if needed
        if hasattr(self, 'feature_names'):
            # Check if we have the right features (ignore order for now)
            if set(X_enhanced.columns) == set(self.feature_names):  # type: ignore[arg-type]
                # Same features, just reorder to match training
                X_enhanced = X_enhanced[self.feature_names]  # type: ignore[index]
            elif list(X_enhanced.columns) != self.feature_names:
                # Different features - this is a real problem
                print(f"⚠️  Feature mismatch detected!")
                print(f"   Expected: {self.feature_names}")
                print(f"   Got: {list(X_enhanced.columns)}")
                # Try to reorder anyway
                try:
                    X_enhanced = X_enhanced[self.feature_names]  # type: ignore[index]
                except KeyError:
                    # Can't fix - features are truly different
                    raise ValueError(f"Cannot align features: expected {self.feature_names}, got {list(X_enhanced.columns)}")
        
        # Normalize
        X_array = X_enhanced.values
        X_normalized = self.normalize_features(X_array, fit=False)  # type: ignore[arg-type]
        
        # Predict (-1 = anomaly, 1 = normal) -> convert to (1, 0)
        predictions = self.model.predict(X_normalized)
        return (predictions == -1).astype(int)


class HybridRuleBasedDetector:
    """
    Hybrid detector: ML + Rule-based cross-checks.
    
    From Novel Architecture (Algorithms 1-3):
    1. DoS Detection: Count low-ID floods (AC[ID] > FPmax)
    2. Spoofing Detection: Hamming distance > Hmax + bit-count BC >= DLC/2
    3. Fuzzy Detection: Unseen IDs + random entropy
    """
    
    def __init__(self):
        self.normal_ids = set()
        self.id_periods = {}
        self.id_max_hamming = {}
        
        # Thresholds from Novel Architecture
        self.dos_threshold = 100  # Max messages per ID per 0.1s window
        self.hamming_threshold = 32  # Max bit flips for same ID
        self.entropy_threshold = 6.5  # High entropy = randomness
    
    def learn_normal_patterns(self, df: pd.DataFrame):
        """
        Learn normal traffic patterns for rule-based detection.
        """
        print("📚 Learning normal patterns for hybrid detection...")
        
        # Learn normal CAN IDs
        self.normal_ids = set(df['arb_id_numeric'].unique())
        
        # Learn max Hamming distances per ID
        if 'hamming_distance' in df.columns:
            for can_id in self.normal_ids:
                id_data = df[df['arb_id_numeric'] == can_id]
                self.id_max_hamming[can_id] = id_data['hamming_distance'].quantile(0.99)
        
        print(f"   ✅ Learned patterns for {len(self.normal_ids)} normal IDs")
    
    def detect_dos(self, df: pd.DataFrame, window_size: float = 0.1) -> np.ndarray:
        """
        Algorithm 1: DoS Detection via message flooding.
        
        Detects if AC[ID] > FPmax (flood protection max)
        """
        dos_flags = np.zeros(len(df), dtype=int)
        
        if 'timestamp' not in df.columns:
            return dos_flags
        
        # Create time windows
        df = df.copy()
        df['time_window'] = (df['timestamp'] // window_size).astype(int)
        
        # Count messages per ID per window
        window_counts = df.groupby(['time_window', 'arb_id_numeric']).size()
        
        # Flag windows exceeding threshold
        for (window, can_id), count in window_counts.items():  # type: ignore[misc]
            if count > self.dos_threshold:
                mask = (df['time_window'] == window) & (df['arb_id_numeric'] == can_id)
                dos_flags[mask] = 1
        
        return dos_flags
    
    def detect_spoofing(self, df: pd.DataFrame) -> np.ndarray:
        """
        Algorithm 2: Spoofing Detection via Hamming distance.
        
        Checks: Hamming > Hmax AND bit-count BC >= DLC/2
        """
        spoof_flags = np.zeros(len(df), dtype=int)
        
        if 'hamming_distance' not in df.columns:
            return spoof_flags
        
        for i, row in df.iterrows():
            can_id = row['arb_id_numeric']
            hamming = row['hamming_distance']
            
            # Check if Hamming exceeds normal max
            if can_id in self.id_max_hamming:
                if hamming > self.id_max_hamming[can_id] + self.hamming_threshold:
                    spoof_flags[i] = 1  # type: ignore[call-overload]
            elif hamming > self.hamming_threshold:
                # Unknown ID with high Hamming
                spoof_flags[i] = 1  # type: ignore[call-overload]
        
        return spoof_flags
    
    def detect_fuzzy(self, df: pd.DataFrame) -> np.ndarray:
        """
        Algorithm 3: Fuzzy Attack Detection via unknown IDs + high entropy.
        
        Checks: ID not in normal_ids OR entropy > threshold
        """
        fuzzy_flags = np.zeros(len(df), dtype=int)
        
        # Check for unknown IDs
        unknown_id_mask = ~df['arb_id_numeric'].isin(self.normal_ids)
        fuzzy_flags[unknown_id_mask] = 1
        
        # Check for high entropy (randomness)
        if 'payload_entropy' in df.columns:
            high_entropy_mask = df['payload_entropy'] > self.entropy_threshold
            fuzzy_flags[high_entropy_mask] = 1
        
        return fuzzy_flags
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Hybrid prediction: Combine all three rule checks.
        
        Returns 1 if ANY rule triggers (OR logic).
        """
        dos = self.detect_dos(df)
        spoof = self.detect_spoofing(df)
        fuzzy = self.detect_fuzzy(df)
        
        # OR logic: Flag if any rule triggers
        combined = (dos | spoof | fuzzy).astype(int)
        
        return combined


class EnsembleHybridDetector:
    """
    Ensemble: ML (Isolation Forest) + Rule-Based + Timing-Based + Top Performers.
    
    Voting: 
    - Weight ML predictions (IF, SVM, etc.)
    - Add rule-based cross-checks
    - Add timing-based WCRT analysis
    - Threshold: Flag if weighted vote > 0.5
    """
    
    def __init__(self):
        self.ml_detectors = []
        self.rule_detector = None
        self.timing_detector = None  # WCRT-based timing detector
        self.weights = {}
        self.expected_features = None  # Store expected feature columns
    
    def add_ml_detector(self, name: str, detector, weight: float = 1.0):
        """Add ML detector to ensemble."""
        self.ml_detectors.append({
            'name': name,
            'detector': detector,
            'weight': weight
        })
    
    def add_rule_detector(self, detector):
        """Add rule-based detector."""
        self.rule_detector = detector
    
    def add_timing_detector(self, detector, weight: float = 1.0):
        """Add timing-based WCRT detector."""
        self.timing_detector = {
            'detector': detector,
            'weight': weight
        }
    
    def set_expected_features(self, feature_names: list):
        """Set expected feature names for consistent predictions."""
        self.expected_features = feature_names
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Ensemble prediction with weighted voting.
        
        Returns:
        --------
        np.ndarray : 1 = attack, 0 = normal
        """
        votes = np.zeros(len(df))
        total_weight = 0.0
        
        # Determine which features to use
        available_features = []
        feature_cols = []
        
        if self.expected_features is not None:
            # Use only the expected features (e.g., 17 features used during training)
            available_features = [f for f in self.expected_features if f in df.columns]
            if len(available_features) != len(self.expected_features):
                missing = set(self.expected_features) - set(available_features)
                print(f"⚠️  Warning: Missing features: {missing}")
            X_features = df[available_features].values
        else:
            # Fallback: use all numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude non-feature columns
            exclude_cols = {'timestamp', 'attack', 'dlc'}
            feature_cols = [c for c in numeric_cols if c not in exclude_cols]
            X_features = df[feature_cols].values if feature_cols else df.select_dtypes(include=[np.number]).values
        
        # ML predictions - handle different method signatures
        for det in self.ml_detectors:
            detector = det['detector']
            
            try:
                # Try different prediction methods
                if hasattr(detector, 'predict'):
                    # Check if detector needs DataFrame or numpy array
                    if hasattr(detector, 'feature_names'):  # ImprovedIsolationForest
                        # Create DataFrame with only the features IF expects
                        if self.expected_features and available_features:
                            df_subset = df[available_features]
                        elif feature_cols:
                            df_subset = df[feature_cols]
                        else:
                            df_subset = df
                        pred = detector.predict(df_subset)
                    else:  # OneClassSVM and others - use feature matrix
                        pred = detector.predict(X_features)
                elif hasattr(detector, 'detect'):
                    pred = detector.detect(X_features)
                elif hasattr(detector, 'detect_anomalies'):
                    pred = detector.detect_anomalies(X_features)
                else:
                    print(f"⚠️  Warning: {det['name']} has no compatible prediction method, skipping")
                    continue
                
                votes += pred * det['weight']
                total_weight += det['weight']
                
            except Exception as e:
                print(f"⚠️  Warning: {det['name']} failed with error: {e}, skipping")
                continue
        
        # Rule-based predictions (uses full DataFrame with all columns)
        if self.rule_detector is not None:
            try:
                rule_pred = self.rule_detector.predict(df)
                votes += rule_pred * 1.0  # Equal weight to rules
                total_weight += 1.0
            except Exception as e:
                print(f"⚠️  Warning: Rule detector failed with error: {e}, skipping")
        
        # Timing-based predictions (uses timestamp and arb_id_numeric)
        if self.timing_detector is not None:
            try:
                timing_pred = self.timing_detector['detector'].predict(df)
                votes += timing_pred * self.timing_detector['weight']
                total_weight += self.timing_detector['weight']
            except Exception as e:
                print(f"⚠️  Warning: Timing detector failed with error: {e}, skipping")
        
        # Threshold voting
        if total_weight > 0:
            final = (votes / total_weight) > 0.5
        else:
            # No detectors available, predict all normal
            final = np.zeros(len(df), dtype=bool)
        
        return final.astype(int)


class DoSCrossCheckFilter:
    """
    Algorithm 1: DoS Attack Cross-Check Filter
    
    From "A Novel Architecture for an Intrusion Detection System 
    Utilizing Cross-Check Filters for In-Vehicle Networks"
    (Im, Lee, Lee, 2024)
    
    Post-prediction validation for DoS attacks using:
    - Rule 1: Frequency-based validation (AC[ID] > FPmax)
    - Rule 2: ID threshold validation (high IDs can't dominate bus)
    
    This filter wraps an ML model and validates/corrects its predictions.
    """
    
    def __init__(self, 
                 ml_model,
                 id_threshold: int = 0x100,
                 reset_period: float = 10.0,
                 fp_max: Optional[int] = None):
        """
        Initialize DoS cross-check filter.
        
        Parameters:
        -----------
        ml_model : object
            Trained ML model (IsolationForest, OneClassSVM, etc.)
            Must have predict() method returning 0/1 (0=normal, 1=attack)
        id_threshold : int, default=0x100 (256)
            CAN IDs above this can't effectively perform DoS
            Based on CAN arbitration (lower ID = higher priority)
        reset_period : float, default=10.0
            Time window (seconds) for resetting attack counts
            Prevents indefinite accumulation
        fp_max : int, optional
            Maximum false positives per ID from validation
            If None, will be computed during calibration
        """
        self.ml_model = ml_model
        self.id_threshold = id_threshold
        self.reset_period = reset_period
        self.fp_max = fp_max
        
        # Attack tracking
        self.attack_counts: Dict[int, int] = defaultdict(int)
        self.last_reset_time: float = 0.0
        
        # Calibration data
        self.fp_per_id: Dict[int, int] = defaultdict(int)
        self.calibrated = False
    
    def calibrate(self, df_normal: pd.DataFrame):
        """
        Calibrate filter on normal traffic to measure FPmax.
        
        This establishes the IDS error margin by testing on normal data
        and counting false positives per message ID.
        
        Parameters:
        -----------
        df_normal : pd.DataFrame
            Normal traffic data (no attacks) for validation
            Must contain 'arb_id_numeric' column
        """
        print("🔧 Calibrating DoS cross-check filter...")
        
        # Get ML predictions on normal data
        ml_predictions = self._get_ml_predictions(df_normal)
        
        # Count false positives per ID (should all be 0 for normal data)
        # Use enumerate to get sequential index for ml_predictions
        for idx, (_, row) in enumerate(df_normal.iterrows()):
            can_id = int(row['arb_id_numeric'])
            if ml_predictions[idx] == 1:  # type: ignore[call-overload]
                # False positive - ML flagged normal as attack
                self.fp_per_id[can_id] += 1
        
        # Set FPmax to maximum FP count across all IDs
        if self.fp_per_id:
            self.fp_max = max(self.fp_per_id.values())
        else:
            self.fp_max = 0  # No false positives observed
        
        self.calibrated = True
        
        print(f"   ✅ Calibrated: FPmax = {self.fp_max}")
        print(f"   📊 IDs with false positives: {len(self.fp_per_id)}")
        if self.fp_per_id:
            top_fp = sorted(self.fp_per_id.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"   🔝 Top FP IDs: {top_fp}")
    
    def _get_ml_predictions(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get predictions from the ML model.
        
        Handles different model interfaces (predict, decision_function, etc.)
        """
        # Extract features for ML model (exclude non-numeric and metadata columns)
        exclude_cols = ['timestamp', 'label', 'attack', 'attack_type', 'arb_id', 'source_file',
                       'data_0', 'data_1', 'data_2', 'data_3', 
                       'data_4', 'data_5', 'data_6', 'data_7']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].values
        
        # Get predictions (handling different model types)
        if hasattr(self.ml_model, 'predict'):
            predictions = self.ml_model.predict(X)
            
            # Convert from -1/1 to 0/1 if needed (IsolationForest format)
            if set(np.unique(predictions)).issubset({-1, 1}):
                predictions = (predictions == -1).astype(int)
            
            return predictions
        else:
            raise ValueError(f"ML model {type(self.ml_model)} has no predict() method")
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict with cross-check validation.
        
        Algorithm 1 from Novel Architecture paper:
        1. Get ML predictions
        2. Reset attack counts if needed (periodic reset)
        3. Apply Rule 1: Frequency-based validation
        4. Apply Rule 2: ID threshold validation
        
        Parameters:
        -----------
        df : pd.DataFrame
            Test data with 'arb_id_numeric' and 'timestamp' columns
        
        Returns:
        --------
        np.ndarray
            Final predictions after cross-check (0=normal, 1=attack)
        """
        if not self.calibrated:
            print("⚠️  Warning: Filter not calibrated. Run calibrate() first.")
            print("   Using ML predictions without cross-check.")
            return self._get_ml_predictions(df)
        
        # Get ML predictions
        ml_predictions = self._get_ml_predictions(df)
        final_predictions = ml_predictions.copy()
        
        # Process each frame - use enumerate to get sequential index
        for idx, (_, row) in enumerate(df.iterrows()):
            can_id = int(row['arb_id_numeric'])
            timestamp = float(row['timestamp']) if 'timestamp' in df.columns else 0.0
            ml_pred = ml_predictions[idx]  # type: ignore[call-overload]
            
            # Step 1: Reset attack counts periodically
            if timestamp - self.last_reset_time > self.reset_period:
                self.attack_counts.clear()
                self.last_reset_time = timestamp
            
            # Step 2: Apply cross-check rules only if ML predicts attack
            if ml_pred == 0:
                # ML says normal → trust it
                final_predictions[idx] = 0  # type: ignore[call-overload]
                continue
            
            # ML says attack (1) → validate with cross-check rules
            
            # Step 3: RULE 2 - ID threshold validation (check this first)
            if can_id > self.id_threshold:
                # ID too high to dominate bus → likely false positive
                final_predictions[idx] = 0  # type: ignore[call-overload]
                continue
            
            # Step 4: Update attack count
            self.attack_counts[can_id] += 1
            
            # Step 5: RULE 1 - Frequency-based validation
            if self.attack_counts[can_id] <= self.fp_max:  # type: ignore[operator]
                # Within error margin → might be false positive
                final_predictions[idx] = 0  # type: ignore[call-overload]
            else:
                # Exceeds error margin → likely real attack
                final_predictions[idx] = 1  # type: ignore[call-overload]
        
        return final_predictions
    
    def get_statistics(self) -> Dict:
        """
        Get current filter statistics.
        
        Returns:
        --------
        dict
            Statistics including attack counts, FPmax, etc.
        """
        return {
            'calibrated': self.calibrated,
            'fp_max': self.fp_max,
            'id_threshold': self.id_threshold,
            'reset_period': self.reset_period,
            'current_attack_counts': dict(self.attack_counts),
            'ids_with_fp': len(self.fp_per_id),
            'total_fp': sum(self.fp_per_id.values())
        }


class SpoofingCrossCheckFilter:
    """
    Spoofing Attack Cross-Check Filter (Algorithm 2).
    
    Based on "A Novel Architecture for an Intrusion Detection System Utilizing
    Cross-Check Filters for In-Vehicle Networks" (Im, Lee, Lee, 2024).
    
    Spoofing attacks send messages with legitimate IDs but abnormal payloads.
    This filter validates ML predictions using:
    - Rule 1: Payload pattern validation (entropy, byte distributions)
    - Rule 2: Sequence consistency (message patterns for each ID)
    
    The filter learns normal payload patterns per message ID during calibration,
    then validates whether ML-flagged messages deviate from expected patterns.
    """
    
    def __init__(self, ml_model, entropy_threshold_factor=1.5, 
                 pattern_match_threshold=0.7, window_size=10):
        """
        Initialize Spoofing Cross-Check Filter.
        
        Parameters:
        -----------
        ml_model : object
            Trained ML model with predict() method
        entropy_threshold_factor : float
            Multiplier for max entropy threshold (default: 1.5)
            Higher = more tolerant of entropy variations
        pattern_match_threshold : float
            Minimum pattern match score (0-1) to accept as normal (default: 0.7)
            Lower = more strict pattern matching
        window_size : int
            Number of recent messages to track per ID (default: 10)
        """
        self.ml_model = ml_model
        self.entropy_threshold_factor = entropy_threshold_factor
        self.pattern_match_threshold = pattern_match_threshold
        self.window_size = window_size
        
        # Calibration results
        self.calibrated = False
        self.normal_ids = set()  # Set of legitimate message IDs
        self.entropy_stats = {}  # {ID: {'mean': float, 'std': float, 'max': float}}
        self.payload_patterns = {}  # {ID: [common_payloads]}
        self.byte_distributions = {}  # {ID: {byte_pos: {value: count}}}
        
        # Runtime tracking
        self.recent_payloads = {}  # {ID: deque of recent payloads}
        
    def calibrate(self, df_normal: pd.DataFrame):
        """
        Learn normal patterns from clean traffic.
        
        Extracts per-ID statistics:
        - Feature value statistics (mean, std, ranges)
        - Time delta patterns
        - ID frequency patterns
        
        Note: This implementation works with extracted features rather than
        raw payload bytes, since processed data doesn't include data_0...data_7.
        
        Parameters:
        -----------
        df_normal : pd.DataFrame
            Normal traffic data (no attacks)
            Must contain 'arb_id_numeric' column
        """
        print("🔧 Calibrating Spoofing cross-check filter...")
        
        from collections import defaultdict, Counter
        from collections import deque
        
        # Learn legitimate IDs
        self.normal_ids = set(df_normal['arb_id_numeric'].unique())
        
        # Feature columns (excluding metadata)
        exclude_cols = ['timestamp', 'label', 'attack', 'attack_type', 'arb_id', 
                       'source_file', 'arb_id_numeric']
        feature_cols = [col for col in df_normal.columns if col not in exclude_cols]
        
        # Group by CAN ID and learn patterns
        for can_id in self.normal_ids:
            df_id = df_normal[df_normal['arb_id_numeric'] == can_id]
            
            if len(df_id) == 0:
                continue
            
            # Calculate feature statistics for this ID
            feature_stats = {}
            for col in feature_cols:
                if col in df_id.columns and pd.api.types.is_numeric_dtype(df_id[col]):
                    feature_stats[col] = {
                        'mean': df_id[col].mean(),
                        'std': df_id[col].std(),
                        'min': df_id[col].min(),
                        'max': df_id[col].max()
                    }
            
            self.entropy_stats[can_id] = feature_stats
            
            # Store feature value patterns (for time_delta, id_frequency, etc.)
            # Use quantiles to define normal ranges
            if 'time_delta' in df_id.columns:
                td_values = df_id['time_delta'].values
                self.payload_patterns[can_id] = {
                    'time_delta_q10': np.percentile(td_values, 10),
                    'time_delta_q90': np.percentile(td_values, 90),
                    'time_delta_median': np.median(td_values)
                }
            
            # Initialize recent tracking
            self.recent_payloads[can_id] = deque(maxlen=self.window_size)
        
        self.calibrated = True
        
        print(f"   ✅ Calibrated: {len(self.normal_ids)} legitimate IDs")
        print(f"   📊 Feature stats: {len(self.entropy_stats)} IDs")
        
        # Count total feature stats
        total_stats = sum(len(stats) for stats in self.entropy_stats.values())
        print(f"   📋 Feature patterns: {total_stats} feature stats across all IDs")
    
    def _calculate_payload_entropy(self, payload: np.ndarray) -> float:
        """Calculate Shannon entropy of payload bytes."""
        from collections import Counter
        
        byte_counts = Counter(payload)
        total = len(payload)
        entropy = 0.0
        for count in byte_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        return entropy
    
    def _get_ml_predictions(self, df: pd.DataFrame) -> np.ndarray:
        """Get ML model predictions on data."""
        # Exclude non-feature columns
        exclude_cols = ['timestamp', 'label', 'attack', 'attack_type', 'arb_id', 
                       'source_file', 'data_0', 'data_1', 'data_2', 'data_3',
                       'data_4', 'data_5', 'data_6', 'data_7']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols].values
        
        # Get predictions
        if hasattr(self.ml_model, 'predict'):
            predictions = self.ml_model.predict(X)
            
            # Convert from -1/1 to 0/1 if needed
            if set(np.unique(predictions)).issubset({-1, 1}):
                predictions = (predictions == -1).astype(int)
            
            return predictions
        else:
            raise ValueError(f"ML model {type(self.ml_model)} has no predict() method")
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict with spoofing cross-check validation.
        
        Algorithm 2 from Novel Architecture paper (adapted for extracted features):
        1. Get ML predictions
        2. Apply Rule 1: Unknown ID check
        3. Apply Rule 2: Feature value validation  
        4. Apply Rule 3: Timing pattern consistency
        
        Parameters:
        -----------
        df : pd.DataFrame
            Test data with 'arb_id_numeric' and feature columns
        
        Returns:
        --------
        np.ndarray
            Final predictions after cross-check (0=normal, 1=attack)
        """
        if not self.calibrated:
            print("⚠️  Warning: Filter not calibrated. Run calibrate() first.")
            print("   Using ML predictions without cross-check.")
            return self._get_ml_predictions(df)
        
        # Get ML predictions
        ml_predictions = self._get_ml_predictions(df)
        final_predictions = ml_predictions.copy()
        
        # Feature columns
        exclude_cols = ['timestamp', 'label', 'attack', 'attack_type', 'arb_id', 
                       'source_file', 'arb_id_numeric']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Process each message
        for idx, (_, row) in enumerate(df.iterrows()):
            can_id = int(row['arb_id_numeric'])
            ml_pred = ml_predictions[idx]
            
            # Only validate if ML says attack
            if ml_pred == 0:
                final_predictions[idx] = 0
                continue
            
            # Rule 1: Unknown ID check
            if can_id not in self.normal_ids:
                # Unknown ID → likely spoofing (or new ECU)
                # Keep ML prediction (likely attack)
                final_predictions[idx] = 1
                continue
            
            # Rule 2: Feature value validation
            # Check if features are within normal ranges for this ID
            if can_id in self.entropy_stats:
                feature_stats = self.entropy_stats[can_id]
                outlier_features = 0
                total_features = 0
                
                for col in feature_cols:
                    if col in feature_stats and col in row.index:
                        stats = feature_stats[col]
                        value = row[col]
                        
                        # Check if value is within mean ± threshold*std
                        threshold = self.entropy_threshold_factor
                        lower_bound = stats['mean'] - threshold * stats['std']
                        upper_bound = stats['mean'] + threshold * stats['std']
                        
                        total_features += 1
                        if value < lower_bound or value > upper_bound:
                            outlier_features += 1
                
                # If most features are within normal range → likely false positive
                if total_features > 0:
                    outlier_ratio = outlier_features / total_features
                    if outlier_ratio < (1 - self.pattern_match_threshold):
                        # Most features normal → false positive
                        final_predictions[idx] = 0
                        continue
            
            # Rule 3: Timing pattern consistency
            # Check time_delta against learned patterns
            if can_id in self.payload_patterns and 'time_delta' in row.index:
                patterns = self.payload_patterns[can_id]
                if 'time_delta_q10' in patterns and 'time_delta_q90' in patterns:
                    td = row['time_delta']
                    # Allow some margin beyond the 10th-90th percentile range
                    margin = (patterns['time_delta_q90'] - patterns['time_delta_q10']) * 0.5
                    lower = patterns['time_delta_q10'] - margin
                    upper = patterns['time_delta_q90'] + margin
                    
                    if lower <= td <= upper:
                        # Time delta within expected range → likely false positive
                        final_predictions[idx] = 0
                        continue
            
            # If we get here, ML prediction is likely correct
            final_predictions[idx] = 1
        
        return final_predictions
    
    def get_statistics(self) -> Dict:
        """
        Get current filter statistics.
        
        Returns:
        --------
        dict
            Statistics including learned IDs, patterns, etc.
        """
        total_feature_stats = sum(len(stats) for stats in self.entropy_stats.values())
        
        return {
            'calibrated': self.calibrated,
            'normal_ids_count': len(self.normal_ids),
            'entropy_threshold_factor': self.entropy_threshold_factor,
            'pattern_match_threshold': self.pattern_match_threshold,
            'window_size': self.window_size,
            'total_feature_stats': total_feature_stats,
            'ids_with_stats': len(self.entropy_stats),
            'ids_with_patterns': len(self.payload_patterns)
        }


class FuzzingCrossCheckFilter:
    """
    Cross-check filter for fuzzing attack detection (Algorithm 3 from Im et al., 2024).
    
    Validates ML fuzzing predictions using entropy-based and feature randomness analysis.
    Fuzzing attacks inject random/unusual data to test system vulnerabilities.
    
    Detection Logic:
    ----------------
    Rule 1: Data Length Validation - Check if data_length is within normal range for ID
    Rule 2: Feature Variance Analysis - Detect excessive randomness in feature values
    Rule 3: Temporal Consistency - Validate timing patterns (fuzzing often disrupts timing)
    
    Parameters:
    -----------
    ml_model : object
        The base ML anomaly detector (e.g., One-Class SVM)
    randomness_threshold : float
        Z-score threshold for detecting excessive feature variance (default: 2.5)
    data_length_strict : bool
        If True, strictly enforce learned data_length per ID (default: True)
    window_size : int
        Number of recent messages to track per ID (default: 20)
    """
    
    def __init__(self, ml_model, randomness_threshold=2.5, 
                 data_length_strict=True, window_size=20):
        self.ml_model = ml_model
        self.randomness_threshold = randomness_threshold
        self.data_length_strict = data_length_strict
        self.window_size = window_size
        
        # Learned patterns from normal traffic
        self.normal_ids = set()
        self.data_length_map = {}  # ID → set of valid data lengths
        self.feature_variance_stats = {}  # ID → feature variance statistics
        self.timing_patterns = {}  # ID → timing distribution
        
        self.calibrated = False
        
        # Statistics
        self.stats = {
            'total_filtered': 0,
            'rule1_filtered': 0,  # Invalid data length
            'rule2_filtered': 0,  # Excessive randomness
            'rule3_filtered': 0,  # Timing anomaly
        }
    
    def calibrate(self, df_normal: pd.DataFrame) -> None:
        """
        Learn normal traffic patterns from clean data.
        
        Parameters:
        -----------
        df_normal : pd.DataFrame
            DataFrame with normal (non-attack) traffic
        """
        print("\n" + "="*60)
        print("🔧 CALIBRATING FUZZING CROSS-CHECK FILTER")
        print("="*60)
        
        # Learn legitimate IDs
        self.normal_ids = set(df_normal['arb_id_numeric'].unique())
        print(f"📋 Learned {len(self.normal_ids)} legitimate CAN IDs")
        
        # Learn data_length patterns per ID
        for can_id in self.normal_ids:
            df_id = df_normal[df_normal['arb_id_numeric'] == can_id]
            
            # Rule 1: Valid data lengths for this ID
            self.data_length_map[can_id] = set(df_id['data_length'].unique())
            
            # Rule 2: Feature variance statistics (for randomness detection)
            # Calculate variance of time_delta - fuzzing disrupts timing
            td_values = df_id['time_delta'].values
            if len(td_values) > 1:
                # Calculate rolling variance to detect sudden randomness
                self.feature_variance_stats[can_id] = {
                    'time_delta_mean': np.mean(td_values),
                    'time_delta_std': np.std(td_values),
                    'time_delta_variance': np.var(td_values),
                    'time_delta_cv': np.std(td_values) / (np.mean(td_values) + 1e-9),  # Coefficient of variation
                }
            
            # Rule 3: Timing patterns (similar to spoofing filter)
            if len(td_values) > 0:
                self.timing_patterns[can_id] = {
                    'time_delta_q01': np.percentile(td_values, 1),
                    'time_delta_q05': np.percentile(td_values, 5),
                    'time_delta_q95': np.percentile(td_values, 95),
                    'time_delta_q99': np.percentile(td_values, 99),
                    'time_delta_median': np.median(td_values),
                    'time_delta_iqr': np.percentile(td_values, 75) - np.percentile(td_values, 25),
                }
        
        self.calibrated = True
        
        print(f"✅ Calibrated data_length patterns for {len(self.data_length_map)} IDs")
        print(f"✅ Calibrated variance statistics for {len(self.feature_variance_stats)} IDs")
        print(f"✅ Calibrated timing patterns for {len(self.timing_patterns)} IDs")
        print("="*60 + "\n")
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Apply cross-check validation to ML predictions.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with messages to validate
            
        Returns:
        --------
        np.ndarray
            Binary predictions (1 = attack, 0 = normal) after cross-check
        """
        if not self.calibrated:
            raise RuntimeError("Filter must be calibrated before prediction. Call calibrate() first.")
        
        # Get ML model predictions
        ml_predictions = self._get_ml_predictions(df)
        
        # Start with ML predictions
        final_predictions = ml_predictions.copy()
        
        # Track recent messages per ID for variance calculation
        recent_messages = {}
        
        # Apply cross-check rules
        for idx, (_, row) in enumerate(df.iterrows()):
            # Only validate messages ML flagged as attacks
            if ml_predictions[idx] == 0:
                continue
            
            can_id = int(row['arb_id_numeric'])
            
            # Initialize tracking for this ID
            if can_id not in recent_messages:
                recent_messages[can_id] = []
            
            # Rule 1: Data Length Validation
            # Fuzzing often uses unusual payload sizes
            if can_id in self.data_length_map:
                valid_lengths = self.data_length_map[can_id]
                current_length = int(row['data_length'])
                
                if self.data_length_strict:
                    # Strict mode: unusual length → confirm as attack
                    if current_length not in valid_lengths:
                        # Unusual data length for this ID → likely fuzzing
                        final_predictions[idx] = 1
                        self.stats['rule1_filtered'] += 1
                        continue
                else:
                    # Lenient mode: normal length → likely false positive
                    if current_length in valid_lengths:
                        # Data length matches normal pattern → could be FP
                        # But don't immediately mark as FP, check other rules
                        pass
            
            # Rule 2: Feature Variance Analysis
            # Detect if time_delta shows excessive randomness compared to normal
            if can_id in self.feature_variance_stats:
                stats = self.feature_variance_stats[can_id]
                td = row['time_delta']
                
                # Calculate z-score
                if stats['time_delta_std'] > 0:
                    z_score = abs((td - stats['time_delta_mean']) / stats['time_delta_std'])
                    
                    # If time_delta is within normal range → likely false positive
                    if z_score < self.randomness_threshold:
                        final_predictions[idx] = 0
                        self.stats['rule2_filtered'] += 1
                        continue
            
            # Rule 3: Timing Consistency
            # Detect timing anomalies using tighter bounds and consistency checks
            if can_id in self.timing_patterns:
                patterns = self.timing_patterns[can_id]
                td = row['time_delta']
                
                # Use Q01-Q99 (tighter bounds) instead of Q05-Q95
                # This catches more extreme timing deviations
                lower = patterns['time_delta_q01']
                upper = patterns['time_delta_q99']
                median = patterns['time_delta_median']
                iqr = patterns['time_delta_iqr']
                
                # Check 1: Is timing within very tight bounds?
                timing_in_bounds = (lower <= td <= upper)
                
                # Check 2: Is timing consistent with median (within 2*IQR)?
                # IQR is more robust to outliers than std
                timing_consistent = (median - 2*iqr <= td <= median + 2*iqr)
                
                # If BOTH checks pass, likely false positive
                # If either fails, keep ML prediction (actual attack)
                if timing_in_bounds and timing_consistent:
                    # Timing within normal range AND consistent → likely false positive
                    final_predictions[idx] = 0
                    self.stats['rule3_filtered'] += 1
                    continue
            
            # If we get here, ML prediction is likely correct (actual fuzzing)
            final_predictions[idx] = 1
        
        # Update total filtered count
        self.stats['total_filtered'] = int(np.sum(ml_predictions) - np.sum(final_predictions))
        
        return final_predictions
    
    def _get_ml_predictions(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get predictions from base ML model.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with features
            
        Returns:
        --------
        np.ndarray
            Binary predictions (1 = attack, 0 = normal)
        """
        # Select numeric features (exclude attack and source_file columns)
        feature_cols = [col for col in df.columns 
                       if col not in ['attack', 'source_file', 'arb_id'] 
                       and pd.api.types.is_numeric_dtype(df[col])]
        
        X = df[feature_cols].values
        
        # ML model returns: 1 = normal, -1 = anomaly
        # Convert to binary: 0 = normal, 1 = attack
        raw_predictions = self.ml_model.predict(X)
        binary_predictions = np.where(raw_predictions == -1, 1, 0)
        
        return binary_predictions
    
    def get_statistics(self) -> Dict:
        """
        Get current filter statistics.
        
        Returns:
        --------
        dict
            Statistics including learned patterns and filtering results
        """
        return {
            'calibrated': self.calibrated,
            'normal_ids_count': len(self.normal_ids),
            'randomness_threshold': self.randomness_threshold,
            'data_length_strict': self.data_length_strict,
            'window_size': self.window_size,
            'ids_with_data_length_map': len(self.data_length_map),
            'ids_with_variance_stats': len(self.feature_variance_stats),
            'ids_with_timing_patterns': len(self.timing_patterns),
            'total_filtered': self.stats['total_filtered'],
            'rule1_filtered': self.stats['rule1_filtered'],
            'rule2_filtered': self.stats['rule2_filtered'],
            'rule3_filtered': self.stats['rule3_filtered'],
        }


def save_improved_detector(detector, filepath: str):
    """Save detector to disk."""
    joblib.dump(detector, filepath)
    print(f"💾 Saved improved detector to {filepath}")


def load_improved_detector(filepath: str):
    """Load detector from disk."""
    detector = joblib.load(filepath)
    print(f"📂 Loaded improved detector from {filepath}")
    return detector
