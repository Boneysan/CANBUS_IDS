"""
Weighted Ensemble Detector with Attack-Type Specific Weighting

This module implements intelligent ensemble weighting based on validated
per-attack-type performance. Instead of equal voting, filters receive
weights proportional to their proven effectiveness on each attack type.

Performance-based weights (from Section 3.9 analysis):
- Fuzzing attacks: Fuzzing filter (98.71% recall) gets highest weight
- DoS attacks: DoS filter (100% recall) gets highest weight
- Gear attacks: Lower weights due to poor performance (0.67% recall)
- RPM attacks: Balanced weights across filters

Expected Improvement: +5-10pp precision vs equal-weight ensemble
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class WeightedEnsembleDetector:
    """
    Ensemble detector with performance-based weighting.
    
    This detector combines predictions from multiple filters (ML, DoS, Spoofing, 
    Fuzzing) using weights learned from their validated performance on each 
    attack type.
    
    Parameters
    ----------
    ml_weight : float, default=0.4
        Base weight for ML predictions
    dos_weight : float, default=0.2
        Base weight for DoS filter
    spoofing_weight : float, default=0.2
        Base weight for Spoofing filter
    fuzzing_weight : float, default=0.2
        Base weight for Fuzzing filter
    use_adaptive_weights : bool, default=True
        If True, adjust weights based on attack type classification
    confidence_threshold : float, default=0.5
        Minimum weighted confidence for positive detection
    """
    
    def __init__(self,
                 ml_weight: float = 0.4,
                 dos_weight: float = 0.2,
                 spoofing_weight: float = 0.2,
                 fuzzing_weight: float = 0.2,
                 use_adaptive_weights: bool = True,
                 confidence_threshold: float = 0.5):
        
        # Base weights (sum to 1.0)
        self.base_weights = {
            'ml': ml_weight,
            'dos': dos_weight,
            'spoofing': spoofing_weight,
            'fuzzing': fuzzing_weight
        }
        
        # Validate weights sum to 1.0
        weight_sum = sum(self.base_weights.values())
        if not np.isclose(weight_sum, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")
        
        self.use_adaptive_weights = use_adaptive_weights
        self.confidence_threshold = confidence_threshold
        
        # Performance-based weights per attack type (from Section 3.9)
        # These reflect validated recall performance
        self.attack_type_weights = {
            'fuzzing': {
                'ml': 0.15,        # ML: 100% recall (baseline)
                'dos': 0.05,       # DoS filter: 20.34% recall on fuzzing
                'spoofing': 0.05,  # Spoofing filter: 12.82% recall
                'fuzzing': 0.75    # Fuzzing filter: 98.71% recall ⭐
            },
            'dos': {
                'ml': 0.30,        # ML: 100% recall (baseline)
                'dos': 0.60,       # DoS filter: 100% recall ⭐
                'spoofing': 0.05,  # Spoofing filter: 12.82% recall
                'fuzzing': 0.05    # Fuzzing filter: 20.34% recall
            },
            'rpm': {
                'ml': 0.50,        # ML: 100% recall (baseline)
                'dos': 0.20,       # DoS filter: moderate performance
                'spoofing': 0.20,  # Spoofing filter: moderate performance
                'fuzzing': 0.10    # Fuzzing filter: lower on spoofing
            },
            'gear': {
                'ml': 0.70,        # ML: 100% recall (primary detector)
                'dos': 0.10,       # DoS filter: 0.67% recall (very poor)
                'spoofing': 0.10,  # Spoofing filter: 0.67% recall
                'fuzzing': 0.10    # Fuzzing filter: 0.67% recall
            },
            'unknown': {
                'ml': 0.40,        # Default: equal base weights
                'dos': 0.20,
                'spoofing': 0.20,
                'fuzzing': 0.20
            }
        }
        
        self.is_calibrated = False
        self.detection_stats = {
            'total_predictions': 0,
            'positive_detections': 0,
            'by_attack_type': {}
        }
    
    def classify_attack_type(self, features: Dict) -> str:
        """
        Classify likely attack type based on traffic features.
        
        Uses heuristics from Section 3.9 traffic classification:
        - High frequency (>3x normal) → DoS
        - DLC changes (>50%) → Spoofing
        - High entropy + randomness → Fuzzing
        
        Parameters
        ----------
        features : dict
            Dictionary containing traffic features
            
        Returns
        -------
        attack_type : str
            One of: 'fuzzing', 'dos', 'rpm', 'gear', 'unknown'
        """
        # DoS: High frequency (>3x baseline)
        freq_ratio = features.get('freq_deviation', 0)
        if freq_ratio > 3.0:
            return 'dos'
        
        # Fuzzing: High entropy + randomness
        entropy = features.get('payload_entropy', 0)
        randomness = features.get('entropy_anomaly', 0)
        if entropy > 3.0 and randomness > 2.0:
            return 'fuzzing'
        
        # Spoofing (RPM/Gear): DLC changes
        dlc_changes = features.get('dlc_anomaly', 0)
        if dlc_changes > 0.5:
            # Distinguish RPM vs Gear by frequency
            if freq_ratio > 1.0:
                return 'rpm'
            else:
                return 'gear'
        
        return 'unknown'
    
    def get_adaptive_weights(self, attack_type: str) -> Dict[str, float]:
        """
        Get performance-optimized weights for detected attack type.
        
        Parameters
        ----------
        attack_type : str
            Classified attack type
            
        Returns
        -------
        weights : dict
            Filter weights optimized for this attack type
        """
        if attack_type in self.attack_type_weights:
            return self.attack_type_weights[attack_type]
        else:
            return self.attack_type_weights['unknown']
    
    def predict_weighted(self, 
                        ml_pred: float,
                        dos_pred: float,
                        spoofing_pred: float,
                        fuzzing_pred: float,
                        features: Optional[Dict] = None) -> Tuple[int, float, str]:
        """
        Make weighted ensemble prediction.
        
        Parameters
        ----------
        ml_pred : float
            ML model prediction probability (0-1)
        dos_pred : float
            DoS filter prediction (0-1)
        spoofing_pred : float
            Spoofing filter prediction (0-1)
        fuzzing_pred : float
            Fuzzing filter prediction (0-1)
        features : dict, optional
            Traffic features for attack type classification
            
        Returns
        -------
        prediction : int
            Binary prediction (0=normal, 1=attack)
        confidence : float
            Weighted confidence score (0-1)
        attack_type : str
            Classified attack type used for weighting
        """
        # Classify attack type if adaptive weighting enabled
        if self.use_adaptive_weights and features is not None:
            attack_type = self.classify_attack_type(features)
            weights = self.get_adaptive_weights(attack_type)
        else:
            attack_type = 'unknown'
            weights = self.base_weights
        
        # Calculate weighted confidence
        confidence = (
            weights['ml'] * ml_pred +
            weights['dos'] * dos_pred +
            weights['spoofing'] * spoofing_pred +
            weights['fuzzing'] * fuzzing_pred
        )
        
        # Make binary decision
        prediction = 1 if confidence >= self.confidence_threshold else 0
        
        # Update statistics
        self.detection_stats['total_predictions'] += 1
        if prediction == 1:
            self.detection_stats['positive_detections'] += 1
        
        if attack_type not in self.detection_stats['by_attack_type']:
            self.detection_stats['by_attack_type'][attack_type] = {
                'count': 0,
                'detections': 0
            }
        self.detection_stats['by_attack_type'][attack_type]['count'] += 1
        if prediction == 1:
            self.detection_stats['by_attack_type'][attack_type]['detections'] += 1
        
        return prediction, confidence, attack_type
    
    def predict_batch(self,
                     ml_preds: np.ndarray,
                     dos_preds: np.ndarray,
                     spoofing_preds: np.ndarray,
                     fuzzing_preds: np.ndarray,
                     features_df: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Make weighted predictions for batch of samples.
        
        Parameters
        ----------
        ml_preds : array-like
            ML model predictions
        dos_preds : array-like
            DoS filter predictions
        spoofing_preds : array-like
            Spoofing filter predictions
        fuzzing_preds : array-like
            Fuzzing filter predictions
        features_df : DataFrame, optional
            Features for attack type classification
            
        Returns
        -------
        predictions : ndarray
            Binary predictions
        confidences : ndarray
            Confidence scores
        attack_types : list
            Classified attack types
        """
        n_samples = len(ml_preds)
        predictions = np.zeros(n_samples, dtype=int)
        confidences = np.zeros(n_samples, dtype=float)
        attack_types = []
        
        for i in range(n_samples):
            # Extract features for this sample
            if features_df is not None:
                features = {
                    'freq_deviation': features_df.iloc[i].get('freq_deviation', 0),
                    'payload_entropy': features_df.iloc[i].get('payload_entropy', 0),
                    'entropy_anomaly': features_df.iloc[i].get('entropy_anomaly', 0),
                    'dlc_anomaly': features_df.iloc[i].get('dlc_anomaly', 0)
                }
            else:
                features = None
            
            # Make prediction
            pred, conf, attack_type = self.predict_weighted(
                ml_preds[i],
                dos_preds[i],
                spoofing_preds[i],
                fuzzing_preds[i],
                features
            )
            
            predictions[i] = pred
            confidences[i] = conf
            attack_types.append(attack_type)
        
        return predictions, confidences, attack_types
    
    def optimize_weights(self,
                        ml_preds: np.ndarray,
                        dos_preds: np.ndarray,
                        spoofing_preds: np.ndarray,
                        fuzzing_preds: np.ndarray,
                        y_true: np.ndarray,
                        features_df: Optional[pd.DataFrame] = None) -> Dict[str, Dict[str, float]]:
        """
        Optimize weights based on validation data.
        
        Uses grid search to find optimal weights that maximize F1-score
        for each attack type.
        
        Parameters
        ----------
        ml_preds, dos_preds, spoofing_preds, fuzzing_preds : array-like
            Predictions from each filter
        y_true : array-like
            True labels
        features_df : DataFrame, optional
            Features for attack type classification
            
        Returns
        -------
        optimized_weights : dict
            Best weights per attack type
        """
        from sklearn.metrics import f1_score
        
        print("\n" + "="*70)
        print("🎯 OPTIMIZING ENSEMBLE WEIGHTS")
        print("="*70)
        
        # Classify all samples by attack type
        if features_df is not None:
            attack_types = []
            for i in range(len(features_df)):
                features = {
                    'freq_deviation': features_df.iloc[i].get('freq_deviation', 0),
                    'payload_entropy': features_df.iloc[i].get('payload_entropy', 0),
                    'entropy_anomaly': features_df.iloc[i].get('entropy_anomaly', 0),
                    'dlc_anomaly': features_df.iloc[i].get('dlc_anomaly', 0)
                }
                attack_types.append(self.classify_attack_type(features))
        else:
            attack_types = ['unknown'] * len(y_true)
        
        attack_types = np.array(attack_types)
        unique_types = np.unique(attack_types)
        
        optimized_weights = {}
        
        # Optimize for each attack type separately
        for attack_type in unique_types:
            print(f"\n📊 Optimizing weights for {attack_type.upper()} attacks...")
            
            # Get samples of this attack type
            mask = attack_types == attack_type
            if not np.any(mask):
                continue
            
            ml_subset = ml_preds[mask]
            dos_subset = dos_preds[mask]
            spoofing_subset = spoofing_preds[mask]
            fuzzing_subset = fuzzing_preds[mask]
            y_subset = y_true[mask]
            
            # Grid search over weights
            best_f1 = 0
            best_weights = self.base_weights.copy()
            
            # Try different weight combinations
            for ml_w in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
                remaining = 1.0 - ml_w
                for dos_w in np.linspace(0, remaining, 5):
                    remaining2 = remaining - dos_w
                    for spoofing_w in np.linspace(0, remaining2, 5):
                        fuzzing_w = remaining2 - spoofing_w
                        
                        # Calculate weighted predictions
                        weighted_conf = (
                            ml_w * ml_subset +
                            dos_w * dos_subset +
                            spoofing_w * spoofing_subset +
                            fuzzing_w * fuzzing_subset
                        )
                        y_pred = (weighted_conf >= self.confidence_threshold).astype(int)
                        
                        # Calculate F1
                        f1 = f1_score(y_subset, y_pred, zero_division=0)
                        
                        if f1 > best_f1:
                            best_f1 = f1
                            best_weights = {
                                'ml': ml_w,
                                'dos': dos_w,
                                'spoofing': spoofing_w,
                                'fuzzing': fuzzing_w
                            }
            
            optimized_weights[attack_type] = best_weights
            
            print(f"   ✅ Best F1: {best_f1:.4f}")
            print(f"   Optimal weights:")
            print(f"      ML:       {best_weights['ml']:.3f}")
            print(f"      DoS:      {best_weights['dos']:.3f}")
            print(f"      Spoofing: {best_weights['spoofing']:.3f}")
            print(f"      Fuzzing:  {best_weights['fuzzing']:.3f}")
        
        print("\n" + "="*70)
        
        # Update internal weights
        self.attack_type_weights.update(optimized_weights)
        self.is_calibrated = True
        
        return optimized_weights
    
    def get_statistics(self) -> Dict:
        """Get detection statistics."""
        stats = self.detection_stats.copy()
        
        # Calculate overall detection rate
        if stats['total_predictions'] > 0:
            stats['detection_rate'] = stats['positive_detections'] / stats['total_predictions']
        else:
            stats['detection_rate'] = 0.0
        
        # Calculate per-attack-type rates
        for attack_type, type_stats in stats['by_attack_type'].items():
            if type_stats['count'] > 0:
                type_stats['detection_rate'] = type_stats['detections'] / type_stats['count']
            else:
                type_stats['detection_rate'] = 0.0
        
        return stats
    
    def reset_statistics(self):
        """Reset detection statistics."""
        self.detection_stats = {
            'total_predictions': 0,
            'positive_detections': 0,
            'by_attack_type': {}
        }
    
    def save(self, filepath: str):
        """Save detector configuration."""
        import joblib
        joblib.dump(self, filepath)
        print(f"✅ Weighted ensemble detector saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'WeightedEnsembleDetector':
        """Load detector configuration."""
        import joblib
        detector = joblib.load(filepath)
        print(f"✅ Weighted ensemble detector loaded from {filepath}")
        return detector


def create_weighted_ensemble(base_weights: Optional[Dict[str, float]] = None,
                             adaptive: bool = True,
                             threshold: float = 0.5) -> WeightedEnsembleDetector:
    """
    Convenience function to create weighted ensemble detector.
    
    Parameters
    ----------
    base_weights : dict, optional
        Base weights for filters {'ml', 'dos', 'spoofing', 'fuzzing'}
    adaptive : bool, default=True
        Use adaptive weighting based on attack type
    threshold : float, default=0.5
        Detection confidence threshold
        
    Returns
    -------
    detector : WeightedEnsembleDetector
        Configured detector instance
    """
    if base_weights is None:
        base_weights = {
            'ml': 0.4,
            'dos': 0.2,
            'spoofing': 0.2,
            'fuzzing': 0.2
        }
    
    return WeightedEnsembleDetector(
        ml_weight=base_weights['ml'],
        dos_weight=base_weights['dos'],
        spoofing_weight=base_weights['spoofing'],
        fuzzing_weight=base_weights['fuzzing'],
        use_adaptive_weights=adaptive,
        confidence_threshold=threshold
    )
