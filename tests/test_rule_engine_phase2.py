"""
Test suite for Rule Engine Phase 2 Important Parameters.

Tests the 3 important rule parameters implemented in Phase 2:
1. check_source - Source validation for diagnostics
2. check_replay - Replay attack detection
3. data_byte_0-7 - Byte-level validation

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


class TestPhase2SourceValidation(unittest.TestCase):
    """Test source validation for diagnostic messages (Parameter 1)."""
    
    def setUp(self):
        """Create temporary rules file with source validation rule."""
        self.temp_dir = tempfile.mkdtemp()
        self.rules_file = os.path.join(self.temp_dir, 'test_rules.yaml')
        
        rules_config = {
            'rules': [
                {
                    'name': 'diagnostic_source_check',
                    'description': 'Test source validation for OBD-II',
                    'severity': 'HIGH',
                    'action': 'alert',
                    'can_id_range': [0x7E0, 0x7E7],  # Diagnostic request range
                    'check_source': True,
                    'allowed_sources': [0, 1, 2]  # Only these ECUs allowed
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
    
    def test_authorized_source_passes(self):
        """Test that authorized diagnostic source passes."""
        message = {
            'can_id': 0x7E0,
            'timestamp': time.time(),
            'dlc': 8,
            'data': [0x02, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            'source_id': 0  # Authorized source
        }
        
        alerts = self.engine.analyze_message(message)
        self.assertEqual(len(alerts), 0, "Authorized source should not trigger alert")
    
    def test_unauthorized_source_fails(self):
        """Test that unauthorized diagnostic source triggers alert."""
        message = {
            'can_id': 0x7E0,
            'timestamp': time.time(),
            'dlc': 8,
            'data': [0x02, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            'source_id': 99  # Unauthorized source
        }
        
        alerts = self.engine.analyze_message(message)
        self.assertEqual(len(alerts), 1, "Unauthorized source should trigger alert")
        self.assertEqual(alerts[0].rule_name, 'diagnostic_source_check')
    
    def test_non_diagnostic_message_passes(self):
        """Test that non-diagnostic messages are not source-checked."""
        message = {
            'can_id': 0x123,  # Non-diagnostic CAN ID
            'timestamp': time.time(),
            'dlc': 8,
            'data': [0x00] * 8,
            'source_id': 99  # Would be unauthorized, but not a diagnostic message
        }
        
        alerts = self.engine.analyze_message(message)
        self.assertEqual(len(alerts), 0, "Non-diagnostic messages should skip source check")
    
    def test_too_many_sources_triggers(self):
        """Test that too many different sources for one CAN ID triggers alert."""
        base_time = time.time()
        
        # Send messages from 4 different sources (exceeds threshold of 3)
        for source in [0, 1, 2, 3]:
            message = {
                'can_id': 0x7E1,
                'timestamp': base_time + source * 0.1,
                'dlc': 8,
                'data': [0x02, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                'source_id': source
            }
            alerts = self.engine.analyze_message(message)
        
        # Should trigger on 4th source
        self.assertGreater(len(alerts), 0, "Too many sources should trigger alert")
    
    def test_broadcast_diagnostic_recognized(self):
        """Test that broadcast diagnostic (0x7DF) is recognized."""
        message = {
            'can_id': 0x7DF,  # Broadcast diagnostic
            'timestamp': time.time(),
            'dlc': 8,
            'data': [0x02, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            'source_id': 0
        }
        
        alerts = self.engine.analyze_message(message)
        # Should be recognized as diagnostic but pass if source is authorized
        self.assertEqual(len(alerts), 0, "Broadcast diagnostic with authorized source should pass")


class TestPhase2ReplayDetection(unittest.TestCase):
    """Test replay attack detection (Parameter 2)."""
    
    def setUp(self):
        """Create temporary rules file with replay detection rule."""
        self.temp_dir = tempfile.mkdtemp()
        self.rules_file = os.path.join(self.temp_dir, 'test_rules.yaml')
        
        rules_config = {
            'rules': [
                {
                    'name': 'replay_attack_detection',
                    'description': 'Test replay attack detection',
                    'severity': 'CRITICAL',
                    'action': 'alert',
                    'check_replay': True,
                    'replay_time_threshold': 1.0  # 1 second threshold
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
    
    def test_unique_messages_pass(self):
        """Test that unique messages don't trigger replay detection."""
        base_time = time.time()
        
        for i in range(5):
            message = {
                'can_id': 0x100,
                'timestamp': base_time + (i * 0.2),
                'dlc': 8,
                'data': [i, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07]  # Different each time
            }
            alerts = self.engine.analyze_message(message)
            self.assertEqual(len(alerts), 0, f"Unique message {i} should not trigger replay alert")
    
    def test_rapid_replay_triggers(self):
        """Test that rapid message replay (< 100ms) triggers alert."""
        base_time = time.time()
        
        # Send same message twice rapidly
        message1 = {
            'can_id': 0x100,
            'timestamp': base_time,
            'dlc': 8,
            'data': [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x11, 0x22]
        }
        self.engine.analyze_message(message1)
        
        # Replay same message 50ms later
        message2 = {
            'can_id': 0x100,
            'timestamp': base_time + 0.05,  # 50ms later
            'dlc': 8,
            'data': [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x11, 0x22]  # Identical data
        }
        alerts = self.engine.analyze_message(message2)
        
        self.assertEqual(len(alerts), 1, "Rapid replay should trigger alert")
        self.assertEqual(alerts[0].rule_name, 'replay_attack_detection')
    
    def test_multiple_replays_triggers(self):
        """Test that multiple replays within threshold trigger alert."""
        base_time = time.time()
        
        # Send same message 4 times within 1 second
        for i in range(4):
            message = {
                'can_id': 0x200,
                'timestamp': base_time + (i * 0.3),  # 300ms apart
                'dlc': 4,
                'data': [0x12, 0x34, 0x56, 0x78]  # Identical data
            }
            alerts = self.engine.analyze_message(message)
        
        # Should trigger on 3rd or 4th replay
        self.assertGreater(len(alerts), 0, "Multiple replays should trigger alert")
    
    def test_replay_after_threshold_passes(self):
        """Test that message repeated after threshold doesn't trigger."""
        base_time = time.time()
        
        # Send message
        message1 = {
            'can_id': 0x300,
            'timestamp': base_time,
            'dlc': 8,
            'data': [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]
        }
        self.engine.analyze_message(message1)
        
        # Send same message 2 seconds later (outside threshold)
        message2 = {
            'can_id': 0x300,
            'timestamp': base_time + 2.0,  # 2 seconds later
            'dlc': 8,
            'data': [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]
        }
        alerts = self.engine.analyze_message(message2)
        
        self.assertEqual(len(alerts), 0, "Replay after threshold should not trigger")
    
    def test_different_data_not_replay(self):
        """Test that similar but different messages aren't detected as replay."""
        base_time = time.time()
        
        message1 = {
            'can_id': 0x400,
            'timestamp': base_time,
            'dlc': 8,
            'data': [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01]
        }
        self.engine.analyze_message(message1)
        
        # Very similar but last byte different
        message2 = {
            'can_id': 0x400,
            'timestamp': base_time + 0.05,
            'dlc': 8,
            'data': [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02]  # Different last byte
        }
        alerts = self.engine.analyze_message(message2)
        
        self.assertEqual(len(alerts), 0, "Different data should not be detected as replay")


class TestPhase2DataByteValidation(unittest.TestCase):
    """Test data byte validation (Parameter 3)."""
    
    def setUp(self):
        """Create temporary rules file with data byte validation rule."""
        self.temp_dir = tempfile.mkdtemp()
        self.rules_file = os.path.join(self.temp_dir, 'test_rules.yaml')
        
        rules_config = {
            'rules': [
                {
                    'name': 'brake_command_validation',
                    'description': 'Test data byte validation',
                    'severity': 'CRITICAL',
                    'action': 'alert',
                    'can_id': 0x200,
                    'data_byte_0': 0x10,  # Expected: 0x10
                    'data_byte_1': 0x00,  # Expected: 0x00
                    'data_byte_7': 0xFF   # Expected: 0xFF (checksum)
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
    
    def test_correct_bytes_pass(self):
        """Test that correct data bytes pass validation."""
        message = {
            'can_id': 0x200,
            'timestamp': time.time(),
            'dlc': 8,
            'data': [0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF]
        }
        
        alerts = self.engine.analyze_message(message)
        self.assertEqual(len(alerts), 0, "Correct data bytes should not trigger alert")
    
    def test_incorrect_byte_0_fails(self):
        """Test that incorrect byte 0 triggers alert."""
        message = {
            'can_id': 0x200,
            'timestamp': time.time(),
            'dlc': 8,
            'data': [0x11, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF]  # Wrong byte 0
        }
        
        alerts = self.engine.analyze_message(message)
        self.assertEqual(len(alerts), 1, "Incorrect byte 0 should trigger alert")
    
    def test_incorrect_byte_1_fails(self):
        """Test that incorrect byte 1 triggers alert."""
        message = {
            'can_id': 0x200,
            'timestamp': time.time(),
            'dlc': 8,
            'data': [0x10, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF]  # Wrong byte 1
        }
        
        alerts = self.engine.analyze_message(message)
        self.assertEqual(len(alerts), 1, "Incorrect byte 1 should trigger alert")
    
    def test_incorrect_byte_7_fails(self):
        """Test that incorrect byte 7 (checksum) triggers alert."""
        message = {
            'can_id': 0x200,
            'timestamp': time.time(),
            'dlc': 8,
            'data': [0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFE]  # Wrong byte 7
        }
        
        alerts = self.engine.analyze_message(message)
        self.assertEqual(len(alerts), 1, "Incorrect checksum byte should trigger alert")
    
    def test_insufficient_data_length_fails(self):
        """Test that insufficient data length triggers alert."""
        message = {
            'can_id': 0x200,
            'timestamp': time.time(),
            'dlc': 4,
            'data': [0x10, 0x00, 0x00, 0x00]  # Only 4 bytes, missing byte 7
        }
        
        alerts = self.engine.analyze_message(message)
        self.assertEqual(len(alerts), 1, "Insufficient data length should trigger alert")
    
    def test_non_matching_can_id_passes(self):
        """Test that rules don't apply to non-matching CAN IDs."""
        message = {
            'can_id': 0x201,  # Different CAN ID
            'timestamp': time.time(),
            'dlc': 8,
            'data': [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00]  # All wrong bytes
        }
        
        alerts = self.engine.analyze_message(message)
        self.assertEqual(len(alerts), 0, "Non-matching CAN ID should not be validated")


class TestPhase2MultiByteValidation(unittest.TestCase):
    """Test validation with multiple byte checks."""
    
    def setUp(self):
        """Create rule with multiple byte validations."""
        self.temp_dir = tempfile.mkdtemp()
        self.rules_file = os.path.join(self.temp_dir, 'test_rules.yaml')
        
        rules_config = {
            'rules': [
                {
                    'name': 'emergency_brake_check',
                    'description': 'Validate emergency brake command format',
                    'severity': 'CRITICAL',
                    'action': 'alert',
                    'can_id': 0x777,
                    'data_byte_0': 0xEB,  # Emergency brake command
                    'data_byte_1': 0x00,  # Reserved
                    'data_byte_2': 0x01,  # Activation flag
                    'data_byte_7': 0xA5   # Safety checksum
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
    
    def test_all_bytes_correct_passes(self):
        """Test that all correct bytes pass validation."""
        message = {
            'can_id': 0x777,
            'timestamp': time.time(),
            'dlc': 8,
            'data': [0xEB, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0xA5]
        }
        
        alerts = self.engine.analyze_message(message)
        self.assertEqual(len(alerts), 0, "All correct bytes should pass")
    
    def test_any_byte_wrong_fails(self):
        """Test that any single incorrect byte triggers alert."""
        # Test each byte position
        test_cases = [
            [0xEA, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0xA5],  # Wrong byte 0
            [0xEB, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0xA5],  # Wrong byte 1
            [0xEB, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xA5],  # Wrong byte 2
            [0xEB, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0xA4],  # Wrong byte 7
        ]
        
        for data in test_cases:
            message = {
                'can_id': 0x777,
                'timestamp': time.time(),
                'dlc': 8,
                'data': data
            }
            alerts = self.engine.analyze_message(message)
            self.assertEqual(len(alerts), 1, f"Incorrect data {data} should trigger alert")


class TestPhase2Integration(unittest.TestCase):
    """Integration tests for all Phase 2 parameters together."""
    
    def setUp(self):
        """Create rules file with all Phase 2 parameters."""
        self.temp_dir = tempfile.mkdtemp()
        self.rules_file = os.path.join(self.temp_dir, 'test_rules.yaml')
        
        rules_config = {
            'rules': [
                {
                    'name': 'comprehensive_phase2',
                    'description': 'All Phase 2 checks enabled',
                    'severity': 'HIGH',
                    'action': 'alert',
                    'can_id_range': [0x7E0, 0x7E7],
                    'check_source': True,
                    'check_replay': True,
                    'allowed_sources': [0, 1],
                    'replay_time_threshold': 0.5,
                    'data_byte_0': 0x02  # UDS service request
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
    
    def test_valid_diagnostic_passes_all_checks(self):
        """Test that valid diagnostic message passes all Phase 2 checks."""
        message = {
            'can_id': 0x7E0,
            'timestamp': time.time(),
            'dlc': 8,
            'data': [0x02, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            'source_id': 0
        }
        
        alerts = self.engine.analyze_message(message)
        self.assertEqual(len(alerts), 0, "Valid diagnostic should pass all checks")
    
    def test_any_phase2_violation_triggers(self):
        """Test that any Phase 2 violation triggers alert."""
        base_time = time.time()
        
        # Test 1: Source violation
        msg1 = {
            'can_id': 0x7E0,
            'timestamp': base_time,
            'dlc': 8,
            'data': [0x02, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            'source_id': 99  # Unauthorized
        }
        alerts1 = self.engine.analyze_message(msg1)
        self.assertGreater(len(alerts1), 0, "Source violation should trigger")
        
        # Test 2: Data byte violation
        msg2 = {
            'can_id': 0x7E1,
            'timestamp': base_time + 0.1,
            'dlc': 8,
            'data': [0x03, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],  # Wrong byte 0
            'source_id': 0
        }
        alerts2 = self.engine.analyze_message(msg2)
        self.assertGreater(len(alerts2), 0, "Data byte violation should trigger")
    
    def test_statistics_tracking(self):
        """Test that Phase 2 statistics are tracked correctly."""
        base_time = time.time()
        
        # Send mix of valid and invalid messages
        for i in range(5):
            message = {
                'can_id': 0x7E0,
                'timestamp': base_time + (i * 0.2),
                'dlc': 8,
                'data': [0x02, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, i],
                'source_id': 0
            }
            self.engine.analyze_message(message)
        
        # Invalid message
        invalid = {
            'can_id': 0x7E0,
            'timestamp': base_time + 1.0,
            'dlc': 8,
            'data': [0x02, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            'source_id': 99  # Unauthorized
        }
        self.engine.analyze_message(invalid)
        
        stats = self.engine.get_statistics()
        self.assertEqual(stats['messages_processed'], 6)
        self.assertGreater(stats['alerts_generated'], 0)


def run_phase2_tests():
    """Run all Phase 2 tests and report results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPhase2SourceValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase2ReplayDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase2DataByteValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase2MultiByteValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase2Integration))
    
    # Run with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("PHASE 2 TEST SUMMARY")
    print("="*70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_phase2_tests()
    exit(0 if success else 1)
