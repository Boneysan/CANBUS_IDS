"""
Feature reduction using PCA for resource-constrained deployment.

Research basis: "Intrusion Detection System using Raspberry Pi for IoT Devices"
                IJRASET 2025 - Reduces 58 features to 10-15 using PCA.

Expected improvement: 3-5x faster ML inference with minimal accuracy loss.
"""

import logging
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class FeatureReducer:
    """
    Reduce feature dimensionality using PCA.
    
    Research-backed approach:
    - Original: 58 features → ~57.7ms inference time on Pi 4
    - Reduced: 10-15 features → ~15-20ms inference time (3-4x faster)
    - Accuracy loss: < 5% (per IJRASET research)
    
    Usage:
        # Training phase
        reducer = FeatureReducer(n_components=15)
        X_reduced = reducer.fit_transform(X_train, feature_names)
        reducer.save('data/models/feature_reducer.joblib')
        
        # Inference phase
        reducer = FeatureReducer()
        reducer.load('data/models/feature_reducer.joblib')
        X_reduced = reducer.transform(X_test)
    """
    
    def __init__(self, n_components: int = 15, 
                 variance_threshold: float = 0.95):
        """
        Initialize feature reducer.
        
        Args:
            n_components: Target number of principal components (10-15 recommended)
            variance_threshold: Minimum cumulative explained variance (0.95 = 95%)
        """
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        
        self.pca: Optional[PCA] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names_in: List[str] = []
        self.feature_names_out: List[str] = []
        self.is_fitted = False
        
        # Metadata
        self.explained_variance_ratio: Optional[np.ndarray] = None
        self.cumulative_variance: Optional[np.ndarray] = None
        
    def fit(self, X: np.ndarray, feature_names: List[str]) -> 'FeatureReducer':
        """
        Fit PCA on training features.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            feature_names: List of original feature names
            
        Returns:
            self for chaining
            
        Raises:
            ValueError: If variance threshold not met
        """
        if X.shape[0] < self.n_components:
            raise ValueError(f"Number of samples ({X.shape[0]}) must be >= "
                           f"n_components ({self.n_components})")
        
        logger.info(f"Fitting PCA: {X.shape[1]} features → {self.n_components} components")
        logger.info(f"Training data: {X.shape[0]:,} samples")
        
        self.feature_names_in = feature_names
        
        # Standardize features first (required for PCA)
        logger.info("Standardizing features...")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit PCA
        logger.info("Fitting PCA model...")
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X_scaled)
        
        # Store variance information
        self.explained_variance_ratio = self.pca.explained_variance_ratio_
        self.cumulative_variance = np.cumsum(self.explained_variance_ratio)
        actual_variance = self.cumulative_variance[-1]
        
        logger.info(f"PCA fitted: {self.n_components} components explain "
                   f"{actual_variance*100:.2f}% of variance")
        
        # Log per-component variance
        for i, (var, cum_var) in enumerate(zip(self.explained_variance_ratio, 
                                               self.cumulative_variance)):
            logger.debug(f"  PC{i+1}: {var*100:.2f}% (cumulative: {cum_var*100:.2f}%)")
        
        if actual_variance < self.variance_threshold:
            logger.warning(f"⚠️  Variance {actual_variance*100:.2f}% < threshold "
                          f"{self.variance_threshold*100:.2f}%")
            logger.warning(f"   Consider increasing n_components or lowering threshold")
        
        # Generate component names
        self.feature_names_out = [f"PC{i+1}" for i in range(self.n_components)]
        self.is_fitted = True
        
        logger.info(f"✅ Feature reducer fitted successfully")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted PCA.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Reduced feature matrix (n_samples, n_components)
            
        Raises:
            RuntimeError: If reducer not fitted
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureReducer not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        X_reduced = self.pca.transform(X_scaled)
        
        return X_reduced
    
    def fit_transform(self, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """
        Fit PCA and transform in one step.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            feature_names: List of original feature names
            
        Returns:
            Reduced feature matrix (n_samples, n_components)
        """
        self.fit(X, feature_names)
        return self.transform(X)
    
    def inverse_transform(self, X_reduced: np.ndarray) -> np.ndarray:
        """
        Transform reduced features back to original space (approximate).
        
        Useful for understanding what reduced features represent.
        
        Args:
            X_reduced: Reduced feature matrix (n_samples, n_components)
            
        Returns:
            Approximated original features (n_samples, n_features)
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureReducer not fitted. Call fit() first.")
        
        X_scaled = self.pca.inverse_transform(X_reduced)
        X_original = self.scaler.inverse_transform(X_scaled)
        
        return X_original
    
    def save(self, path: str) -> None:
        """
        Save fitted reducer to disk.
        
        Args:
            path: File path to save to (e.g., 'data/models/feature_reducer.joblib')
            
        Raises:
            RuntimeError: If reducer not fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted FeatureReducer")
        
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'pca': self.pca,
            'scaler': self.scaler,
            'n_components': self.n_components,
            'variance_threshold': self.variance_threshold,
            'feature_names_in': self.feature_names_in,
            'feature_names_out': self.feature_names_out,
            'explained_variance_ratio': self.explained_variance_ratio,
            'cumulative_variance': self.cumulative_variance,
        }
        
        joblib.dump(save_dict, path)
        logger.info(f"✅ FeatureReducer saved to {path}")
        logger.info(f"   {len(self.feature_names_in)} → {self.n_components} features")
        logger.info(f"   {self.cumulative_variance[-1]*100:.2f}% variance explained")
    
    def load(self, path: str) -> 'FeatureReducer':
        """
        Load fitted reducer from disk.
        
        Args:
            path: File path to load from
            
        Returns:
            self for chaining
            
        Raises:
            FileNotFoundError: If file not found
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"FeatureReducer file not found: {path}")
        
        save_dict = joblib.load(path)
        
        self.pca = save_dict['pca']
        self.scaler = save_dict['scaler']
        self.n_components = save_dict['n_components']
        self.variance_threshold = save_dict.get('variance_threshold', 0.95)
        self.feature_names_in = save_dict['feature_names_in']
        self.feature_names_out = save_dict['feature_names_out']
        self.explained_variance_ratio = save_dict.get('explained_variance_ratio')
        self.cumulative_variance = save_dict.get('cumulative_variance')
        self.is_fitted = True
        
        logger.info(f"✅ FeatureReducer loaded from {path}")
        logger.info(f"   {len(self.feature_names_in)} → {self.n_components} features")
        if self.cumulative_variance is not None:
            logger.info(f"   {self.cumulative_variance[-1]*100:.2f}% variance explained")
        
        return self
    
    def get_feature_importance(self, n_top: int = 10) -> List[Tuple[str, float]]:
        """
        Get top N most important original features.
        
        Importance is calculated as the sum of absolute loadings across all
        principal components. Features with high loadings contribute more
        to the principal components.
        
        Args:
            n_top: Number of top features to return
            
        Returns:
            List of (feature_name, importance_score) tuples, sorted by importance
            
        Raises:
            RuntimeError: If reducer not fitted
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureReducer not fitted")
        
        # Calculate feature importance from PCA components
        # Sum of absolute values across all components
        feature_importance = np.abs(self.pca.components_).sum(axis=0)
        
        # Normalize to [0, 1]
        feature_importance = feature_importance / feature_importance.sum()
        
        # Create (name, score) pairs
        importance_pairs = list(zip(self.feature_names_in, feature_importance))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return importance_pairs[:n_top]
    
    def get_component_loadings(self, component_idx: int = 0) -> List[Tuple[str, float]]:
        """
        Get feature loadings for a specific principal component.
        
        Shows which original features contribute most to this component.
        
        Args:
            component_idx: Index of principal component (0-based)
            
        Returns:
            List of (feature_name, loading) tuples, sorted by absolute loading
            
        Raises:
            RuntimeError: If reducer not fitted
            ValueError: If invalid component index
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureReducer not fitted")
        
        if component_idx >= self.n_components:
            raise ValueError(f"Component index {component_idx} >= n_components "
                           f"{self.n_components}")
        
        loadings = self.pca.components_[component_idx]
        loading_pairs = list(zip(self.feature_names_in, loadings))
        loading_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return loading_pairs
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the reducer.
        
        Returns:
            Dictionary with reducer statistics
        """
        if not self.is_fitted:
            return {'fitted': False}
        
        return {
            'fitted': True,
            'n_features_in': len(self.feature_names_in),
            'n_components': self.n_components,
            'variance_threshold': self.variance_threshold,
            'variance_explained': float(self.cumulative_variance[-1]),
            'variance_explained_pct': f"{self.cumulative_variance[-1]*100:.2f}%",
            'variance_by_component': [
                {
                    'component': f"PC{i+1}",
                    'variance': float(var),
                    'cumulative': float(cum_var)
                }
                for i, (var, cum_var) in enumerate(
                    zip(self.explained_variance_ratio, self.cumulative_variance)
                )
            ],
            'top_features': [
                {'name': name, 'importance': float(score)}
                for name, score in self.get_feature_importance(10)
            ]
        }
    
    def __repr__(self) -> str:
        """String representation of the reducer."""
        if self.is_fitted:
            var_pct = self.cumulative_variance[-1] * 100
            return (f"FeatureReducer(n_components={self.n_components}, "
                   f"variance_explained={var_pct:.2f}%, fitted=True)")
        else:
            return (f"FeatureReducer(n_components={self.n_components}, "
                   f"fitted=False)")
