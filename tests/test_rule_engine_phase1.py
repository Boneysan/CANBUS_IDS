"""
Test suite for Rule Engine Phase 1 Critical Parameters.

Tests the 3 critical rule parameters implemented in Phase 1:
1. validate_dlc - Strict DLC validation
2. check_frame_format - Frame format checking
3. global_message_rate - Global rate monitoring

Author: CANBUS_IDS Project
Date: December 2, 2025
"""

import unittest
import tempfile
import os
import time
from pathlib import Path
import yaml

from src.detection.rule_engine import RuleEngine, DetectionRule, Alert


class TestPhase1DLCValidation(unittest.TestCase):
    """Test strict DLC validation (Parameter 1)."""
    
    def setUp(self):
        """Create temporary rules file with DLC validation rule."""
        self.temp_dir = tempfile.mkdtemp()
        self.rules_file = os.path.join(self.temp_dir, 'test_rules.yaml')
        
        rules_config = {
            'rules': [
                {
                    'name': 'strict_dlc_check',
                    'description': 'Test strict DLC validation',
                    'severity': 'CRITICAL',
                    'action': 'alert',
                    'can_id': 0x100,  # Only check this specific CAN ID
                    'validate_dlc': True,
                    'dlc_min': 4,
                    'dlc_max': 8
                }
            ]
        }
        
        with open(self.rules_file, 'w') as f:
            yaml.dump(rules_config, f)
        
        self.engine = RuleEngine(self.rules_file)
    
    def tearDown(self):
        """Clean up temporary files."""
        os.remove(self.rules_file)
        os.rmdir(self.temp_dir)
    
    def test_valid_dlc_passes(self):
        """Test that valid DLC passes validation."""
        message = {
            'can_id': 0x100,
            'timestamp': time.time(),
            'dlc': 8,
            'data': [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07]
        }
        
        alerts = self.engine.analyze_message(message)
        self.assertEqual(len(alerts), 0, "Valid DLC should not trigger alert")
    
    def test_dlc_out_of_range_fails(self):
        """Test that DLC > 8 fails validation."""
        message = {
            'can_id': 0x100,
            'timestamp': time.time(),
            'dlc': 9,  # Invalid: CAN 2.0 max is 8
            'data': [0x00] * 9
        }
        
        alerts = self.engine.analyze_message(message)
        self.assertEqual(len(alerts), 1, "DLC > 8 should trigger alert")
        self.assertEqual(alerts[0].rule_name, 'strict_dlc_check')
    
    def test_dlc_data_mismatch_fails(self):
        """Test that DLC not matching data length fails."""
        message = {
            'can_id': 0x100,
            'timestamp': time.time(),
            'dlc': 8,
            'data': [0x00, 0x01, 0x02]  # Only 3 bytes, but DLC says 8
        }
        
        alerts = self.engine.analyze_message(message)
        self.assertEqual(len(alerts), 1, "DLC mismatch should trigger alert")
    
    def test_dlc_below_minimum_fails(self):
        """Test that DLC below rule minimum fails."""
        message = {
            'can_id': 0x100,
            'timestamp': time.time(),
            'dlc': 2,  # Below minimum of 4
            'data': [0x00, 0x01]
        }
        
        alerts = self.engine.analyze_message(message)
        self.assertEqual(len(alerts), 1, "DLC below minimum should trigger alert")
    
    def test_dlc_negative_fails(self):
        """Test that negative DLC fails validation."""
        message = {
            'can_id': 0x100,
            'timestamp': time.time(),
            'dlc': -1,
            'data': []
        }
        
        alerts = self.engine.analyze_message(message)
        self.assertEqual(len(alerts), 1, "Negative DLC should trigger alert")


class TestPhase1FrameFormat(unittest.TestCase):
    """Test frame format checking (Parameter 2)."""
    
    def setUp(self):
        """Create temporary rules file with frame format rule."""
        self.temp_dir = tempfile.mkdtemp()
        self.rules_file = os.path.join(self.temp_dir, 'test_rules.yaml')
        
        rules_config = {
            'rules': [
                {
                    'name': 'frame_format_check',
                    'description': 'Test frame format validation',
                    'severity': 'HIGH',
                    'action': 'alert',
                    # No CAN ID restriction - frame format applies to ALL messages
                    'check_frame_format': True
                }
            ]
        }
        
        with open(self.rules_file, 'w') as f:
            yaml.dump(rules_config, f)
        
        self.engine = RuleEngine(self.rules_file)
    
    def tearDown(self):
        """Clean up temporary files."""
        os.remove(self.rules_file)
        os.rmdir(self.temp_dir)
    
    def test_valid_standard_frame_passes(self):
        """Test that valid standard frame passes."""
        message = {
            'can_id': 0x7FF,  # Max standard ID
            'timestamp': time.time(),
            'dlc': 8,
            'data': [0x00] * 8,
            'is_extended': False
        }
        
        alerts = self.engine.analyze_message(message)
        self.assertEqual(len(alerts), 0, "Valid standard frame should pass")
    
    def test_standard_frame_id_overflow_fails(self):
        """Test that standard frame with ID > 0x7FF fails."""
        message = {
            'can_id': 0x800,  # Exceeds standard 11-bit range
            'timestamp': time.time(),
            'dlc': 8,
            'data': [0x00] * 8,
            'is_extended': False
        }
        
        alerts = self.engine.analyze_message(message)
        self.assertEqual(len(alerts), 1, "Standard frame ID overflow should fail")
    
    def test_valid_extended_frame_passes(self):
        """Test that valid extended frame passes."""
        message = {
            'can_id': 0x1FFFFFFF,  # Max extended ID
            'timestamp': time.time(),
            'dlc': 8,
            'data': [0x00] * 8,
            'is_extended': True
        }
        
        alerts = self.engine.analyze_message(message)
        self.assertEqual(len(alerts), 0, "Valid extended frame should pass")
    
    def test_extended_frame_id_overflow_fails(self):
        """Test that extended frame with ID > 0x1FFFFFFF fails."""
        message = {
            'can_id': 0x20000000,  # Exceeds extended 29-bit range
            'timestamp': time.time(),
            'dlc': 8,
            'data': [0x00] * 8,
            'is_extended': True
        }
        
        alerts = self.engine.analyze_message(message)
        self.assertEqual(len(alerts), 1, "Extended frame ID overflow should fail")
    
    def test_error_frame_fails(self):
        """Test that error frame is detected."""
        message = {
            'can_id': 0x100,
            'timestamp': time.time(),
            'dlc': 8,
            'data': [0x00] * 8,
            'is_error': True
        }
        
        alerts = self.engine.analyze_message(message)
        self.assertEqual(len(alerts), 1, "Error frame should be detected")
    
    def test_remote_frame_with_data_fails(self):
        """Test that remote frame with data fails."""
        message = {
            'can_id': 0x100,
            'timestamp': time.time(),
            'dlc': 4,
            'data': [0x00, 0x01, 0x02, 0x03],  # Remote frames shouldn't have data
            'is_remote': True
        }
        
        alerts = self.engine.analyze_message(message)
        self.assertEqual(len(alerts), 1, "Remote frame with data should fail")
    
    def test_invalid_dlc_in_frame_fails(self):
        """Test that invalid DLC in frame format fails."""
        message = {
            'can_id': 0x100,
            'timestamp': time.time(),
            'dlc': 10,  # Invalid
            'data': [0x00] * 10
        }
        
        alerts = self.engine.analyze_message(message)
        self.assertEqual(len(alerts), 1, "Invalid DLC should fail frame format check")


class TestPhase1GlobalMessageRate(unittest.TestCase):
    """Test global message rate monitoring (Parameter 3)."""
    
    def setUp(self):
        """Create temporary rules file with global rate rule."""
        self.temp_dir = tempfile.mkdtemp()
        self.rules_file = os.path.join(self.temp_dir, 'test_rules.yaml')
        
        rules_config = {
            'rules': [
                {
                    'name': 'bus_flooding_detection',
                    'description': 'Detect bus flooding DoS attack',
                    'severity': 'CRITICAL',
                    'action': 'alert',
                    'can_id_range': [0x100, 0x7FF],  # Monitor all standard CAN IDs
                    'global_message_rate': 100,  # Max 100 messages
                    'time_window': 1  # Per 1 second
                }
            ]
        }
        
        with open(self.rules_file, 'w') as f:
            yaml.dump(rules_config, f)
        
        self.engine = RuleEngine(self.rules_file)
    
    def tearDown(self):
        """Clean up temporary files."""
        os.remove(self.rules_file)
        os.rmdir(self.temp_dir)
    
    def test_normal_rate_passes(self):
        """Test that normal message rate passes."""
        base_time = time.time()
        
        # Send 50 messages (below threshold of 100)
        for i in range(50):
            message = {
                'can_id': 0x100 + (i % 10),
                'timestamp': base_time + (i * 0.01),  # 10ms apart
                'dlc': 8,
                'data': [0x00] * 8
            }
            alerts = self.engine.analyze_message(message)
        
        # Last message should not trigger alert
        self.assertEqual(len(alerts), 0, "Normal rate should not trigger alert")
    
    def test_high_rate_triggers_alert(self):
        """Test that high message rate triggers alert."""
        base_time = time.time()
        
        # Send 150 messages (exceeds threshold of 100 in 1 second)
        alerts_generated = []
        for i in range(150):
            message = {
                'can_id': 0x100 + (i % 10),
                'timestamp': base_time + (i * 0.005),  # 5ms apart
                'dlc': 8,
                'data': [0x00] * 8
            }
            alerts = self.engine.analyze_message(message)
            alerts_generated.extend(alerts)
        
        # Should trigger alert after exceeding threshold
        self.assertGreater(len(alerts_generated), 0, "High rate should trigger alerts")
        self.assertTrue(any(a.rule_name == 'bus_flooding_detection' for a in alerts_generated),
                       "Bus flooding rule should trigger")
    
    def test_rate_window_sliding(self):
        """Test that rate window slides correctly over time."""
        base_time = time.time()
        
        # Send 100 messages at time 0
        for i in range(100):
            message = {
                'can_id': 0x100,
                'timestamp': base_time + (i * 0.001),
                'dlc': 8,
                'data': [0x00] * 8
            }
            self.engine.analyze_message(message)
        
        # Wait simulated 2 seconds (outside window)
        # Send 50 more messages - should not trigger
        for i in range(50):
            message = {
                'can_id': 0x100,
                'timestamp': base_time + 2.0 + (i * 0.01),
                'dlc': 8,
                'data': [0x00] * 8
            }
            alerts = self.engine.analyze_message(message)
        
        # Should not trigger because old messages outside window
        self.assertEqual(len(alerts), 0, "Rate window should slide correctly")
    
    def test_global_rate_tracks_all_ids(self):
        """Test that global rate monitors all CAN IDs together."""
        base_time = time.time()
        
        # Send 150 messages across different CAN IDs
        alerts_generated = []
        for i in range(150):
            message = {
                'can_id': 0x100 + (i % 30),  # 30 different IDs
                'timestamp': base_time + (i * 0.005),
                'dlc': 8,
                'data': [0x00] * 8
            }
            alerts = self.engine.analyze_message(message)
            alerts_generated.extend(alerts)
        
        # Should trigger because TOTAL rate exceeds threshold
        self.assertGreater(len(alerts_generated), 0, 
                          "Global rate should track all CAN IDs together")


class TestPhase1Integration(unittest.TestCase):
    """Integration tests for all Phase 1 parameters together."""
    
    def setUp(self):
        """Create rules file with all Phase 1 parameters."""
        self.temp_dir = tempfile.mkdtemp()
        self.rules_file = os.path.join(self.temp_dir, 'test_rules.yaml')
        
        rules_config = {
            'rules': [
                {
                    'name': 'comprehensive_check',
                    'description': 'All Phase 1 checks enabled',
                    'severity': 'CRITICAL',
                    'action': 'alert',
                    # No CAN ID restriction - frame format checks apply globally
                    'validate_dlc': True,
                    'check_frame_format': True,
                    'global_message_rate': 1000,
                    'time_window': 1,
                    'dlc_min': 0,
                    'dlc_max': 8
                }
            ]
        }
        
        with open(self.rules_file, 'w') as f:
            yaml.dump(rules_config, f)
        
        self.engine = RuleEngine(self.rules_file)
    
    def tearDown(self):
        """Clean up temporary files."""
        os.remove(self.rules_file)
        os.rmdir(self.temp_dir)
    
    def test_valid_message_passes_all_checks(self):
        """Test that valid message passes all Phase 1 checks."""
        message = {
            'can_id': 0x123,
            'timestamp': time.time(),
            'dlc': 8,
            'data': [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07],
            'is_extended': False,
            'is_error': False,
            'is_remote': False
        }
        
        alerts = self.engine.analyze_message(message)
        self.assertEqual(len(alerts), 0, "Valid message should pass all checks")
    
    def test_any_violation_triggers_alert(self):
        """Test that any Phase 1 violation triggers alert."""
        # DLC violation
        message1 = {
            'can_id': 0x123,
            'timestamp': time.time(),
            'dlc': 9,  # Invalid
            'data': [0x00] * 9,
            'is_extended': False
        }
        alerts1 = self.engine.analyze_message(message1)
        self.assertGreater(len(alerts1), 0, "DLC violation should trigger")
        
        # Frame format violation
        message2 = {
            'can_id': 0x1000,  # Exceeds standard 11-bit
            'timestamp': time.time() + 0.1,
            'dlc': 8,
            'data': [0x00] * 8,
            'is_extended': False
        }
        alerts2 = self.engine.analyze_message(message2)
        self.assertGreater(len(alerts2), 0, "Frame format violation should trigger")
    
    def test_statistics_tracking(self):
        """Test that statistics are properly tracked."""
        base_time = time.time()
        
        # Send mix of valid and invalid messages
        for i in range(10):
            # Valid message
            message = {
                'can_id': 0x100,
                'timestamp': base_time + (i * 0.1),
                'dlc': 8,
                'data': [0x00] * 8,
                'is_extended': False
            }
            self.engine.analyze_message(message)
        
        # Invalid message
        invalid_message = {
            'can_id': 0x100,
            'timestamp': base_time + 1.0,
            'dlc': 9,  # Invalid
            'data': [0x00] * 9
        }
        self.engine.analyze_message(invalid_message)
        
        stats = self.engine.get_statistics()
        self.assertEqual(stats['messages_processed'], 11)
        self.assertGreater(stats['alerts_generated'], 0)


def run_phase1_tests():
    """Run all Phase 1 tests and report results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPhase1DLCValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase1FrameFormat))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase1GlobalMessageRate))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase1Integration))
    
    # Run with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("PHASE 1 TEST SUMMARY")
    print("="*70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_phase1_tests()
    exit(0 if success else 1)
