#!/usr/bin/env python3
"""
Multi-Attack Ensemble Cross-Check Detector

Integrates all three cross-check filters (DoS, Spoofing, Fuzzing) into a unified
detection system that can intelligently route attacks to the appropriate filter
or use voting mechanisms for unknown attack types.

Based on Im et al. (2024) - Novel Architecture for IDS with Cross-Check Filters

Author: Research Team
Date: October 27, 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import joblib

from improved_detectors import (
    DoSCrossCheckFilter,
    SpoofingCrossCheckFilter,
    FuzzingCrossCheckFilter
)


class MultiAttackCrossCheckEnsemble:
    """
    Unified ensemble detector that integrates DoS, Spoofing, and Fuzzing cross-check filters.
    
    Features:
    ---------
    1. Auto-detection of attack type based on traffic characteristics
    2. Intelligent routing to appropriate cross-check filter
    3. Voting mechanism for ambiguous cases
    4. Fallback to baseline ML for unknown attack types
    5. Comprehensive statistics and monitoring
    
    Parameters:
    -----------
    ml_model : object
        Base ML anomaly detector (e.g., One-Class SVM)
    mode : str
        Detection mode: 'auto', 'routing', 'voting', 'cascade'
        - 'auto': Auto-detect attack type and route (default)
        - 'routing': User specifies attack type
        - 'voting': All filters vote, majority wins
        - 'cascade': Apply filters in sequence (DoS → Spoofing → Fuzzing)
    dos_params : dict
        Parameters for DoS filter
    spoofing_params : dict
        Parameters for Spoofing filter
    fuzzing_params : dict
        Parameters for Fuzzing filter
    """
    
    def __init__(self, 
                 ml_model,
                 mode='auto',
                 dos_params=None,
                 spoofing_params=None,
                 fuzzing_params=None):
        
        self.ml_model = ml_model
        self.mode = mode
        
        # Initialize cross-check filters
        dos_params = dos_params or {}
        spoofing_params = spoofing_params or {}
        fuzzing_params = fuzzing_params or {}
        
        self.dos_filter = DoSCrossCheckFilter(ml_model, **dos_params)
        self.spoofing_filter = SpoofingCrossCheckFilter(ml_model, **spoofing_params)
        self.fuzzing_filter = FuzzingCrossCheckFilter(ml_model, **fuzzing_params)
        
        self.calibrated = False
        
        # Statistics
        self.stats = {
            'total_predictions': 0,
            'routed_to_dos': 0,
            'routed_to_spoofing': 0,
            'routed_to_fuzzing': 0,
            'routed_to_ml': 0,
            'voting_used': 0,
            'attack_type_distribution': {},
        }
    
    def calibrate(self, df_normal: pd.DataFrame) -> None:
        """
        Calibrate all three cross-check filters on normal traffic.
        
        Parameters:
        -----------
        df_normal : pd.DataFrame
            DataFrame with normal (non-attack) traffic
        """
        print("\n" + "="*70)
        print("🔧 CALIBRATING MULTI-ATTACK ENSEMBLE DETECTOR")
        print("="*70)
        print(f"Mode: {self.mode}")
        print(f"Calibration samples: {len(df_normal):,}")
        
        # Calibrate each filter
        print("\n1️⃣  Calibrating DoS Cross-Check Filter...")
        self.dos_filter.calibrate(df_normal)
        
        print("\n2️⃣  Calibrating Spoofing Cross-Check Filter...")
        self.spoofing_filter.calibrate(df_normal)
        
        print("\n3️⃣  Calibrating Fuzzing Cross-Check Filter...")
        self.fuzzing_filter.calibrate(df_normal)
        
        self.calibrated = True
        
        print("\n" + "="*70)
        print("✅ ALL FILTERS CALIBRATED SUCCESSFULLY")
        print("="*70 + "\n")
    
    def predict(self, 
                df: pd.DataFrame, 
                attack_type: Optional[str] = None) -> np.ndarray:
        """
        Predict attacks using ensemble of cross-check filters.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with messages to classify
        attack_type : str, optional
            Specify attack type ('dos', 'spoofing', 'fuzzing', 'unknown')
            If None and mode='auto', will auto-detect
            
        Returns:
        --------
        np.ndarray
            Binary predictions (1 = attack, 0 = normal)
        """
        if not self.calibrated:
            raise RuntimeError("Ensemble must be calibrated before prediction. Call calibrate() first.")
        
        self.stats['total_predictions'] += len(df)
        
        # Mode selection
        if self.mode == 'auto' and attack_type is None:
            return self._predict_auto(df)
        elif self.mode == 'routing' or attack_type is not None:
            # Ensure attack_type is not None for routing
            attack_type = attack_type if attack_type is not None else 'unknown'
            return self._predict_routing(df, attack_type)
        elif self.mode == 'voting':
            return self._predict_voting(df)
        elif self.mode == 'cascade':
            return self._predict_cascade(df)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _predict_auto(self, df: pd.DataFrame) -> np.ndarray:
        """
        Auto-detect attack type and route to appropriate filter.
        
        Attack Type Detection Rules:
        - DoS: High frequency on single ID (id_frequency > threshold)
        - Spoofing: Unknown or rare IDs
        - Fuzzing: Unusual data_length or high time_delta variance
        - Unknown: Use voting mechanism
        """
        # Classify attack type based on characteristics
        attack_types = self._classify_attack_type(df)
        
        # Initialize predictions array
        predictions = np.zeros(len(df), dtype=int)
        
        # Route each sample to appropriate filter
        for attack_type in set(attack_types):
            mask = attack_types == attack_type
            df_subset = df[mask].copy()
            
            if attack_type == 'dos':
                pred = self.dos_filter.predict(df_subset)
                self.stats['routed_to_dos'] += len(df_subset)
            elif attack_type == 'spoofing':
                pred = self.spoofing_filter.predict(df_subset)
                self.stats['routed_to_spoofing'] += len(df_subset)
            elif attack_type == 'fuzzing':
                pred = self.fuzzing_filter.predict(df_subset)
                self.stats['routed_to_fuzzing'] += len(df_subset)
            else:  # unknown
                # Use voting for unknown types
                pred = self._predict_voting(df_subset)
                self.stats['voting_used'] += len(df_subset)
            
            predictions[mask] = pred
            
            # Update statistics
            if attack_type not in self.stats['attack_type_distribution']:
                self.stats['attack_type_distribution'][attack_type] = 0
            self.stats['attack_type_distribution'][attack_type] += len(df_subset)
        
        return predictions
    
    def _predict_routing(self, df: pd.DataFrame, attack_type: str) -> np.ndarray:
        """
        Route to specific filter based on attack type.
        """
        attack_type = attack_type.lower() if attack_type else 'unknown'
        
        if attack_type == 'dos':
            self.stats['routed_to_dos'] += len(df)
            return self.dos_filter.predict(df)
        elif attack_type in ['spoofing', 'rpm', 'gear']:
            self.stats['routed_to_spoofing'] += len(df)
            return self.spoofing_filter.predict(df)
        elif attack_type == 'fuzzing':
            self.stats['routed_to_fuzzing'] += len(df)
            return self.fuzzing_filter.predict(df)
        else:
            # Unknown type - use baseline ML
            self.stats['routed_to_ml'] += len(df)
            return self._get_ml_predictions(df)
    
    def _predict_voting(self, df: pd.DataFrame) -> np.ndarray:
        """
        Use majority voting from all three filters.
        
        Each filter votes (0 or 1), final prediction is majority (≥2 votes).
        """
        self.stats['voting_used'] += len(df)
        
        # Get predictions from all filters
        dos_pred = self.dos_filter.predict(df)
        spoofing_pred = self.spoofing_filter.predict(df)
        fuzzing_pred = self.fuzzing_filter.predict(df)
        
        # Stack predictions
        votes = np.stack([dos_pred, spoofing_pred, fuzzing_pred], axis=1)
        
        # Majority vote (≥2 votes → attack)
        final_pred = (votes.sum(axis=1) >= 2).astype(int)
        
        return final_pred
    
    def _predict_cascade(self, df: pd.DataFrame) -> np.ndarray:
        """
        Apply filters in cascade: DoS → Spoofing → Fuzzing.
        
        Each filter refines the predictions from the previous.
        """
        # Start with DoS filter
        predictions = self.dos_filter.predict(df)
        self.stats['routed_to_dos'] += len(df)
        
        # Refine with Spoofing filter (only on flagged attacks)
        attack_mask = predictions == 1
        if attack_mask.sum() > 0:
            df_attacks = df[attack_mask].copy()
            spoofing_pred = self.spoofing_filter.predict(df_attacks)
            predictions[attack_mask] = spoofing_pred
            self.stats['routed_to_spoofing'] += attack_mask.sum()
        
        # Further refine with Fuzzing filter
        attack_mask = predictions == 1
        if attack_mask.sum() > 0:
            df_attacks = df[attack_mask].copy()
            fuzzing_pred = self.fuzzing_filter.predict(df_attacks)
            predictions[attack_mask] = fuzzing_pred
            self.stats['routed_to_fuzzing'] += attack_mask.sum()
        
        return predictions
    
    def _classify_attack_type(self, df: pd.DataFrame) -> np.ndarray:
        """
        Classify attack type based on traffic characteristics.
        
        Returns:
        --------
        np.ndarray
            Array of attack type labels ('dos', 'spoofing', 'fuzzing', 'unknown')
        """
        attack_types = np.full(len(df), 'unknown', dtype=object)
        
        # Rule 1: DoS detection - high frequency on specific IDs
        # Count occurrences of each ID
        id_counts = df['arb_id_numeric'].value_counts()
        threshold = float(len(df) * 0.1)
        high_freq_ids = id_counts[id_counts > threshold].index  # type: ignore  # >10% of traffic
        dos_mask = df['arb_id_numeric'].isin(high_freq_ids)
        attack_types[dos_mask] = 'dos'
        
        # Rule 2: Spoofing detection - unknown IDs (not in calibration)
        # Use spoofing filter's normal_ids instead
        normal_ids = self.spoofing_filter.normal_ids
        spoofing_mask = ~df['arb_id_numeric'].isin(normal_ids) & ~dos_mask
        attack_types[spoofing_mask] = 'spoofing'
        
        # Rule 3: Fuzzing detection - unusual data_length or high variance
        if 'data_length' in df.columns:
            # Check if data_length deviates from normal
            for can_id in df['arb_id_numeric'].unique():
                if can_id in self.fuzzing_filter.data_length_map:
                    valid_lengths = self.fuzzing_filter.data_length_map[can_id]
                    id_mask = (df['arb_id_numeric'] == can_id) & ~dos_mask & ~spoofing_mask
                    # Get data_length for this ID - use full DataFrame to preserve index alignment
                    invalid_length_mask = (~df['data_length'].isin(valid_lengths)) & id_mask  # type: ignore
                    attack_types[invalid_length_mask] = 'fuzzing'
        
        return attack_types
    
    def _get_ml_predictions(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get predictions from base ML model.
        """
        feature_cols = [col for col in df.columns 
                       if col not in ['attack', 'source_file', 'arb_id', 'attack_type'] 
                       and pd.api.types.is_numeric_dtype(df[col])]
        
        X = df[feature_cols].values
        raw_predictions = self.ml_model.predict(X)
        binary_predictions = np.where(raw_predictions == -1, 1, 0)
        
        return binary_predictions
    
    def get_statistics(self) -> Dict:
        """
        Get comprehensive statistics from ensemble and all filters.
        
        Returns:
        --------
        dict
            Statistics including routing info and filter-specific stats
        """
        return {
            'ensemble': {
                'mode': self.mode,
                'calibrated': self.calibrated,
                'total_predictions': self.stats['total_predictions'],
                'routed_to_dos': self.stats['routed_to_dos'],
                'routed_to_spoofing': self.stats['routed_to_spoofing'],
                'routed_to_fuzzing': self.stats['routed_to_fuzzing'],
                'routed_to_ml': self.stats['routed_to_ml'],
                'voting_used': self.stats['voting_used'],
                'attack_type_distribution': self.stats['attack_type_distribution'],
            },
            'dos_filter': self.dos_filter.get_statistics(),
            'spoofing_filter': self.spoofing_filter.get_statistics(),
            'fuzzing_filter': self.fuzzing_filter.get_statistics(),
        }
    
    def print_summary(self) -> None:
        """
        Print comprehensive summary of ensemble performance.
        """
        stats = self.get_statistics()
        
        print("\n" + "="*70)
        print("📊 MULTI-ATTACK ENSEMBLE DETECTOR SUMMARY")
        print("="*70)
        
        print(f"\n🔧 Configuration:")
        print(f"   Mode: {stats['ensemble']['mode']}")
        print(f"   Calibrated: {stats['ensemble']['calibrated']}")
        print(f"   Total Predictions: {stats['ensemble']['total_predictions']:,}")
        
        print(f"\n📈 Routing Statistics:")
        total = stats['ensemble']['total_predictions']
        if total > 0:
            print(f"   DoS Filter:      {stats['ensemble']['routed_to_dos']:,} ({stats['ensemble']['routed_to_dos']/total*100:.1f}%)")
            print(f"   Spoofing Filter: {stats['ensemble']['routed_to_spoofing']:,} ({stats['ensemble']['routed_to_spoofing']/total*100:.1f}%)")
            print(f"   Fuzzing Filter:  {stats['ensemble']['routed_to_fuzzing']:,} ({stats['ensemble']['routed_to_fuzzing']/total*100:.1f}%)")
            print(f"   Baseline ML:     {stats['ensemble']['routed_to_ml']:,} ({stats['ensemble']['routed_to_ml']/total*100:.1f}%)")
            print(f"   Voting Used:     {stats['ensemble']['voting_used']:,} ({stats['ensemble']['voting_used']/total*100:.1f}%)")
        
        if stats['ensemble']['attack_type_distribution']:
            print(f"\n🎯 Attack Type Distribution:")
            for attack_type, count in stats['ensemble']['attack_type_distribution'].items():
                print(f"   {attack_type.capitalize()}: {count:,} ({count/total*100:.1f}%)")
        
        print(f"\n🔍 Filter Details:")
        print(f"   DoS Filter FPmax: {stats['dos_filter'].get('fp_max', 'N/A')}")
        print(f"   Spoofing IDs: {stats['spoofing_filter']['normal_ids_count']}")
        print(f"   Fuzzing IDs: {stats['fuzzing_filter']['normal_ids_count']}")
        
        print("="*70 + "\n")


def save_ensemble_detector(detector, filepath: str):
    """
    Save ensemble detector to disk.
    
    Parameters:
    -----------
    detector : MultiAttackCrossCheckEnsemble
        The ensemble detector to save
    filepath : str
        Path to save the detector
    """
    joblib.dump(detector, filepath)
    print(f"💾 Saved ensemble detector to {filepath}")


def load_ensemble_detector(filepath: str):
    """
    Load ensemble detector from disk.
    
    Parameters:
    -----------
    filepath : str
        Path to load the detector from
        
    Returns:
    --------
    MultiAttackCrossCheckEnsemble
        Loaded ensemble detector
    """
    detector = joblib.load(filepath)
    print(f"📂 Loaded ensemble detector from {filepath}")
    return detector
