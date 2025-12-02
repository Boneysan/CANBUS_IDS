"""
Test suite for preprocessing modules.

Tests for FeatureExtractor and Normalizer.
"""

import pytest
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.feature_extractor import FeatureExtractor
from src.preprocessing.normalizer import Normalizer


class TestFeatureExtractor:
    """Test cases for FeatureExtractor class."""
    
    def test_initialization(self):
        """Test FeatureExtractor initialization."""
        extractor = FeatureExtractor(window_size=50, history_size=500)
        
        assert extractor.window_size == 50
        assert extractor.history_size == 500
        
    def test_extract_basic_features(self):
        """Test extraction of basic features."""
        extractor = FeatureExtractor()
        
        message = {
            'timestamp': time.time(),
            'can_id': 0x123,
            'dlc': 8,
            'data': [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08],
            'is_extended': False,
            'is_remote': False,
            'is_error': False
        }
        
        features = extractor.extract_features(message)
        
        # Check that features were extracted
        assert isinstance(features, dict)
        assert len(features) > 0
        
        # Check for expected feature categories
        assert 'can_id' in features
        assert 'dlc' in features
        
    def test_statistical_features(self):
        """Test statistical feature extraction."""
        extractor = FeatureExtractor()
        
        # Process multiple messages to build history
        for i in range(10):
            message = {
                'timestamp': time.time() + i * 0.1,
                'can_id': 0x100,
                'dlc': 8,
                'data': [i] * 8,
                'is_extended': False,
                'is_remote': False,
                'is_error': False
            }
            features = extractor.extract_features(message)
        
        # After processing multiple messages, should have frequency features
        assert 'data_mean' in features or 'mean' in str(features)
        
    def test_temporal_features(self):
        """Test temporal feature extraction."""
        extractor = FeatureExtractor()
        
        base_time = time.time()
        
        # Process messages with varying intervals
        for i in range(5):
            message = {
                'timestamp': base_time + i * 0.1,
                'can_id': 0x100,
                'dlc': 8,
                'data': [0x00] * 8,
                'is_extended': False,
                'is_remote': False,
                'is_error': False
            }
            features = extractor.extract_features(message)
        
        # Should have temporal features
        assert isinstance(features, dict)
        
    def test_empty_data_handling(self):
        """Test handling of messages with empty data."""
        extractor = FeatureExtractor()
        
        message = {
            'timestamp': time.time(),
            'can_id': 0x100,
            'dlc': 0,
            'data': [],
            'is_extended': False,
            'is_remote': False,
            'is_error': False
        }
        
        features = extractor.extract_features(message)
        
        # Should handle empty data gracefully
        assert isinstance(features, dict)
        assert features['dlc'] == 0


class TestNormalizer:
    """Test cases for Normalizer class."""
    
    def test_initialization(self):
        """Test Normalizer initialization."""
        normalizer = Normalizer(method='minmax')
        
        assert normalizer.method == 'minmax'
        assert normalizer.is_fitted == False
        
    def test_fit_minmax(self):
        """Test fitting with min-max normalization."""
        normalizer = Normalizer(method='minmax')
        
        # Sample feature data
        features = [
            {'feat1': 0.0, 'feat2': 100.0},
            {'feat1': 5.0, 'feat2': 150.0},
            {'feat1': 10.0, 'feat2': 200.0}
        ]
        
        normalizer.fit(features)
        
        assert normalizer.is_fitted == True
        assert 'feat1' in normalizer.feature_stats
        assert 'feat2' in normalizer.feature_stats
        
    def test_transform_minmax(self):
        """Test transforming features with min-max normalization."""
        normalizer = Normalizer(method='minmax')
        
        # Fit on training data
        train_features = [
            {'feat1': 0.0, 'feat2': 100.0},
            {'feat1': 10.0, 'feat2': 200.0}
        ]
        normalizer.fit(train_features)
        
        # Transform test data
        test_feature = {'feat1': 5.0, 'feat2': 150.0}
        normalized = normalizer.transform(test_feature)
        
        assert isinstance(normalized, dict)
        # Min-max should be between 0 and 1
        assert 0.0 <= normalized['feat1'] <= 1.0
        assert 0.0 <= normalized['feat2'] <= 1.0
        
    def test_fit_zscore(self):
        """Test fitting with z-score normalization."""
        normalizer = Normalizer(method='zscore')
        
        features = [
            {'feat1': 1.0, 'feat2': 10.0},
            {'feat1': 2.0, 'feat2': 20.0},
            {'feat1': 3.0, 'feat2': 30.0}
        ]
        
        normalizer.fit(features)
        
        assert normalizer.is_fitted == True
        assert 'feat1' in normalizer.feature_stats
        
    def test_transform_unfitted(self):
        """Test that transform fails on unfitted normalizer."""
        normalizer = Normalizer()
        
        with pytest.raises(RuntimeError):
            normalizer.transform({'feat1': 1.0})
            
    def test_save_load(self, tmp_path):
        """Test saving and loading normalizer."""
        normalizer = Normalizer(method='minmax')
        
        # Fit normalizer
        features = [
            {'feat1': 0.0, 'feat2': 100.0},
            {'feat1': 10.0, 'feat2': 200.0}
        ]
        normalizer.fit(features)
        
        # Save
        save_path = tmp_path / "normalizer.pkl"
        normalizer.save(str(save_path))
        
        # Load into new normalizer
        loaded_normalizer = Normalizer()
        loaded_normalizer.load(str(save_path))
        
        assert loaded_normalizer.is_fitted == True
        assert loaded_normalizer.method == 'minmax'
        
        # Test that loaded normalizer works
        test_feature = {'feat1': 5.0, 'feat2': 150.0}
        normalized = loaded_normalizer.transform(test_feature)
        assert isinstance(normalized, dict)
        
    def test_handle_missing_features(self):
        """Test handling of missing features during transform."""
        normalizer = Normalizer()
        
        # Fit with certain features
        train_features = [
            {'feat1': 0.0, 'feat2': 100.0},
            {'feat1': 10.0, 'feat2': 200.0}
        ]
        normalizer.fit(train_features)
        
        # Transform with missing feature
        test_feature = {'feat1': 5.0}  # Missing feat2
        normalized = normalizer.transform(test_feature)
        
        # Should handle gracefully
        assert isinstance(normalized, dict)
        assert 'feat1' in normalized


class TestFeatureExtractionIntegration:
    """Integration tests for feature extraction pipeline."""
    
    def test_extract_and_normalize(self):
        """Test complete feature extraction and normalization pipeline."""
        extractor = FeatureExtractor()
        normalizer = Normalizer(method='minmax')
        
        # Generate training data
        train_messages = []
        for i in range(20):
            message = {
                'timestamp': time.time() + i * 0.1,
                'can_id': 0x100,
                'dlc': 8,
                'data': [i % 256] * 8,
                'is_extended': False,
                'is_remote': False,
                'is_error': False
            }
            train_messages.append(message)
        
        # Extract features
        train_features = [extractor.extract_features(msg) for msg in train_messages]
        
        # Fit normalizer
        normalizer.fit(train_features)
        
        # Process new message
        test_message = {
            'timestamp': time.time(),
            'can_id': 0x100,
            'dlc': 8,
            'data': [0x55] * 8,
            'is_extended': False,
            'is_remote': False,
            'is_error': False
        }
        
        features = extractor.extract_features(test_message)
        normalized = normalizer.transform(features)
        
        # Verify pipeline works
        assert isinstance(normalized, dict)
        assert len(normalized) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
