"""
Test enhanced features integration from Vehicle_Models research project.

Validates that enhanced features (payload entropy, hamming distance, etc.)
are correctly integrated into the FeatureExtractor class.
"""

import sys
import os
import unittest
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing.feature_extractor import FeatureExtractor


class TestEnhancedFeatures(unittest.TestCase):
    """Test enhanced features from Vehicle_Models research."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create extractors with and without enhanced features
        self.extractor_basic = FeatureExtractor(enable_enhanced_features=False)
        self.extractor_enhanced = FeatureExtractor(enable_enhanced_features=True)
        
        # Sample CAN messages for testing
        self.sample_messages = [
            {
                'can_id': 0x123,
                'dlc': 8,
                'data': [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08],
                'timestamp': 1.0,
                'time_delta': 0.01,
                'is_extended': False,
                'is_remote': False,
                'is_error': False
            },
            {
                'can_id': 0x456,
                'dlc': 8,
                'data': [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00, 0x11],
                'timestamp': 1.01,
                'time_delta': 0.01,
                'is_extended': False,
                'is_remote': False,
                'is_error': False
            },
            {
                'can_id': 0x123,
                'dlc': 8,
                'data': [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x09],  # 1 byte changed
                'timestamp': 1.02,
                'time_delta': 0.01,
                'is_extended': False,
                'is_remote': False,
                'is_error': False
            },
            {
                'can_id': 0x789,
                'dlc': 8,
                'data': [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],  # High entropy
                'timestamp': 1.03,
                'time_delta': 0.01,
                'is_extended': False,
                'is_remote': False,
                'is_error': False
            }
        ]
    
    def test_enhanced_features_disabled_by_default(self):
        """Test that enhanced features are not extracted when disabled."""
        features = self.extractor_basic.extract_features(self.sample_messages[0])
        
        # Should NOT have enhanced features
        self.assertNotIn('payload_entropy', features)
        self.assertNotIn('hamming_distance', features)
        self.assertNotIn('iat_zscore', features)
        self.assertNotIn('unknown_bigram', features)
        self.assertNotIn('unknown_trigram', features)
        self.assertNotIn('bit_time_mean', features)
        
        # Should have basic features
        self.assertIn('can_id', features)
        self.assertIn('dlc', features)
        self.assertIn('data_entropy', features)
    
    def test_enhanced_features_when_enabled(self):
        """Test that enhanced features are extracted when enabled."""
        features = self.extractor_enhanced.extract_features(self.sample_messages[0])
        
        # Should have ALL enhanced features
        self.assertIn('payload_entropy', features)
        self.assertIn('hamming_distance', features)
        self.assertIn('iat_zscore', features)
        self.assertIn('unknown_bigram', features)
        self.assertIn('unknown_trigram', features)
        self.assertIn('bit_time_mean', features)
        self.assertIn('bit_time_rms', features)
        self.assertIn('bit_time_energy', features)
        
        # Should also have basic features
        self.assertIn('can_id', features)
        self.assertIn('dlc', features)
    
    def test_payload_entropy_calculation(self):
        """Test Shannon entropy calculation (TCE-IDS paper)."""
        # Test with uniform data (low entropy)
        msg_uniform = self.sample_messages[0].copy()
        msg_uniform['data'] = [0x00] * 8
        features = self.extractor_enhanced.extract_features(msg_uniform)
        self.assertEqual(features['payload_entropy'], 0.0)  # All same = no entropy
        
        # Test with random data (high entropy)
        msg_random = self.sample_messages[3]
        features = self.extractor_enhanced.extract_features(msg_random)
        self.assertEqual(features['payload_entropy'], 0.0)  # All 0xFF = no entropy
        
        # Test with varied data (medium entropy)
        msg_varied = self.sample_messages[0]
        features = self.extractor_enhanced.extract_features(msg_varied)
        self.assertGreater(features['payload_entropy'], 2.5)  # Sequential data has entropy
    
    def test_hamming_distance_calculation(self):
        """Test bit-level distance calculation (Novel Architecture paper)."""
        # First message - no previous, should be 0
        features1 = self.extractor_enhanced.extract_features(self.sample_messages[0])
        self.assertEqual(features1['hamming_distance'], 0)
        
        # Second message different ID - no previous for this ID
        features2 = self.extractor_enhanced.extract_features(self.sample_messages[1])
        self.assertEqual(features2['hamming_distance'], 0)
        
        # Third message same as first ID, 1 byte changed (0x08 -> 0x09)
        # Bit difference: 0x08 (00001000) vs 0x09 (00001001) = 1 bit flip
        features3 = self.extractor_enhanced.extract_features(self.sample_messages[2])
        self.assertEqual(features3['hamming_distance'], 1)
    
    def test_bit_time_statistics(self):
        """Test bit-time calculations (BTMonitor paper)."""
        features = self.extractor_enhanced.extract_features(self.sample_messages[0])
        
        # Should have bit time features
        self.assertIn('bit_time_mean', features)
        self.assertIn('bit_time_rms', features)
        self.assertIn('bit_time_energy', features)
        
        # Values should be positive and reasonable
        self.assertGreater(features['bit_time_mean'], 0)
        self.assertGreater(features['bit_time_rms'], 0)
        self.assertGreater(features['bit_time_energy'], 0)
        
        # RMS should be close to mean for single sample
        self.assertAlmostEqual(features['bit_time_rms'], features['bit_time_mean'], places=10)
    
    def test_calibration_required_for_advanced_features(self):
        """Test that calibration is needed for IAT z-score and n-grams."""
        # Without calibration
        features = self.extractor_enhanced.extract_features(self.sample_messages[0])
        self.assertEqual(features['iat_zscore'], 0.0)  # No calibration = default 0
        self.assertEqual(features['unknown_bigram'], 0)
        self.assertEqual(features['unknown_trigram'], 0)
        
        # With calibration
        self.extractor_enhanced.calibrate_enhanced_features(self.sample_messages)
        
        # Process messages to build sequence
        for msg in self.sample_messages:
            features = self.extractor_enhanced.extract_features(msg)
        
        # Now should detect patterns
        # (actual values depend on calibration, just check they compute)
        self.assertIsInstance(features['iat_zscore'], float)
        self.assertIn(features['unknown_bigram'], [0, 1])
        self.assertIn(features['unknown_trigram'], [0, 1])
    
    def test_feature_count_difference(self):
        """Test that enhanced mode has more features."""
        basic_features = self.extractor_basic.get_feature_names()
        enhanced_features = self.extractor_enhanced.get_feature_names()
        
        # Enhanced should have 8 more features
        self.assertEqual(len(enhanced_features), len(basic_features) + 8)
        
        # Verify the 8 new features
        new_features = set(enhanced_features) - set(basic_features)
        expected_new = {
            'payload_entropy', 'hamming_distance', 'iat_zscore',
            'unknown_bigram', 'unknown_trigram',
            'bit_time_mean', 'bit_time_rms', 'bit_time_energy'
        }
        self.assertEqual(new_features, expected_new)
    
    def test_statistics_reporting(self):
        """Test that statistics report calibration status."""
        stats_basic = self.extractor_basic.get_statistics()
        stats_enhanced = self.extractor_enhanced.get_statistics()
        
        # Basic shouldn't have enhanced stats
        self.assertFalse(stats_basic['enhanced_features_enabled'])
        self.assertNotIn('enhanced_calibrated', stats_basic)
        
        # Enhanced should have calibration info
        self.assertTrue(stats_enhanced['enhanced_features_enabled'])
        self.assertIn('enhanced_calibrated', stats_enhanced)
        self.assertFalse(stats_enhanced['enhanced_calibrated'])  # Not calibrated yet
        
        # After calibration
        self.extractor_enhanced.calibrate_enhanced_features(self.sample_messages)
        stats_after = self.extractor_enhanced.get_statistics()
        self.assertTrue(stats_after['enhanced_calibrated'])
        self.assertGreater(stats_after['calibrated_can_ids'], 0)
    
    def test_batch_feature_extraction(self):
        """Test batch extraction with enhanced features."""
        features_list = self.extractor_enhanced.extract_batch_features(self.sample_messages)
        
        # Should get features for all messages
        self.assertEqual(len(features_list), len(self.sample_messages))
        
        # Each should have enhanced features
        for features in features_list:
            self.assertIn('payload_entropy', features)
            self.assertIn('hamming_distance', features)
            self.assertIn('bit_time_mean', features)


def run_tests():
    """Run all enhanced feature tests."""
    print("\n" + "="*70)
    print("TESTING ENHANCED FEATURES INTEGRATION")
    print("Source: Vehicle_Models/src/enhanced_features.py")
    print("="*70)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEnhancedFeatures)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ All enhanced feature tests passed!")
        print("\nFeatures successfully integrated from Vehicle_Models:")
        print("  • payload_entropy (TCE-IDS paper)")
        print("  • hamming_distance (Novel Architecture paper)")
        print("  • iat_zscore (SAIDuCANT paper)")
        print("  • unknown_bigram/trigram (Novel Architecture paper)")
        print("  • bit_time_* (BTMonitor paper)")
    else:
        print("\n❌ Some tests failed - check output above")
    
    print("="*70 + "\n")
    
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
