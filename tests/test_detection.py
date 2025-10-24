"""
Test suite for detection engines.

Tests for RuleEngine and MLDetector.
"""

import pytest
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.rule_engine import RuleEngine, DetectionRule, Alert
from src.detection.ml_detector import MLDetector


class TestDetectionRule:
    """Test cases for DetectionRule dataclass."""
    
    def test_rule_creation(self):
        """Test creating a detection rule."""
        rule = DetectionRule(
            name="Test Rule",
            severity="HIGH",
            description="Test description",
            action="alert",
            can_id=0x123
        )
        
        assert rule.name == "Test Rule"
        assert rule.severity == "HIGH"
        assert rule.can_id == 0x123
        
    def test_can_id_matching_exact(self):
        """Test exact CAN ID matching."""
        rule = DetectionRule(
            name="Test",
            severity="HIGH",
            description="Test",
            action="alert",
            can_id=0x123
        )
        
        assert rule.matches_can_id(0x123) == True
        assert rule.matches_can_id(0x124) == False
        
    def test_can_id_matching_range(self):
        """Test CAN ID range matching."""
        rule = DetectionRule(
            name="Test",
            severity="HIGH",
            description="Test",
            action="alert",
            can_id_range=[0x100, 0x200]
        )
        
        assert rule.matches_can_id(0x100) == True
        assert rule.matches_can_id(0x150) == True
        assert rule.matches_can_id(0x200) == True
        assert rule.matches_can_id(0x99) == False
        assert rule.matches_can_id(0x201) == False


class TestRuleEngine:
    """Test cases for RuleEngine class."""
    
    @pytest.fixture
    def sample_rules_file(self, tmp_path):
        """Create a temporary rules file for testing."""
        rules_content = """
rules:
  - name: "Test High Frequency"
    can_id: 0x100
    severity: HIGH
    description: "High frequency test"
    action: alert
    max_frequency: 100
    time_window: 1
    
  - name: "Test Data Pattern"
    can_id: 0x200
    severity: MEDIUM
    description: "Pattern test"
    action: alert
    data_pattern: "DE AD"
"""
        rules_file = tmp_path / "test_rules.yaml"
        rules_file.write_text(rules_content)
        return str(rules_file)
    
    def test_initialization(self, sample_rules_file):
        """Test RuleEngine initialization."""
        engine = RuleEngine(sample_rules_file)
        
        assert len(engine.rules) == 2
        assert engine.rules[0].name == "Test High Frequency"
        assert engine.rules[1].name == "Test Data Pattern"
        
    def test_load_nonexistent_file(self):
        """Test loading non-existent rules file."""
        with pytest.raises(FileNotFoundError):
            RuleEngine("nonexistent_rules.yaml")
            
    def test_analyze_message(self, sample_rules_file):
        """Test analyzing a CAN message."""
        engine = RuleEngine(sample_rules_file)
        
        message = {
            'timestamp': time.time(),
            'can_id': 0x100,
            'dlc': 8,
            'data': [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07]
        }
        
        alerts = engine.analyze_message(message)
        
        # Should return a list (may be empty if no rules matched)
        assert isinstance(alerts, list)
        
    def test_data_pattern_matching(self, sample_rules_file):
        """Test data pattern matching."""
        engine = RuleEngine(sample_rules_file)
        
        # Message matching pattern
        message = {
            'timestamp': time.time(),
            'can_id': 0x200,
            'dlc': 8,
            'data': [0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x00, 0x00, 0x00]
        }
        
        alerts = engine.analyze_message(message)
        
        # Should trigger the pattern rule
        assert len(alerts) >= 1
        assert any(alert.rule_name == "Test Data Pattern" for alert in alerts)
        
    def test_statistics(self, sample_rules_file):
        """Test statistics tracking."""
        engine = RuleEngine(sample_rules_file)
        
        # Process some messages
        for i in range(10):
            message = {
                'timestamp': time.time(),
                'can_id': 0x100,
                'dlc': 8,
                'data': [i] * 8
            }
            engine.analyze_message(message)
        
        stats = engine.get_statistics()
        
        assert stats['messages_processed'] == 10
        assert 'alert_rate' in stats
        assert 'match_rate' in stats


class TestMLDetector:
    """Test cases for MLDetector class."""
    
    def test_initialization(self):
        """Test MLDetector initialization."""
        detector = MLDetector(contamination=0.05)
        
        assert detector.contamination == 0.05
        assert detector.is_trained == False
        
    def test_training_without_sklearn(self):
        """Test that training fails gracefully without sklearn."""
        detector = MLDetector()
        
        # Mock feature data
        features = [
            {'feat1': 1.0, 'feat2': 2.0} for _ in range(10)
        ]
        
        # Should handle missing sklearn gracefully
        try:
            detector.train(features)
        except ImportError:
            pytest.skip("sklearn not installed")
            
    def test_analyze_message_untrained(self):
        """Test analyzing message with untrained model."""
        detector = MLDetector()
        
        message = {
            'timestamp': time.time(),
            'can_id': 0x123,
            'dlc': 8,
            'data': [0x00] * 8
        }
        
        # Should return None or handle gracefully when untrained
        result = detector.analyze_message(message)
        assert result is None or isinstance(result, type(None))
        
    def test_statistics(self):
        """Test statistics tracking."""
        detector = MLDetector()
        
        stats = detector.get_statistics()
        
        assert 'messages_analyzed' in stats
        assert 'anomalies_detected' in stats
        assert stats['model_loaded'] == False


class TestAlert:
    """Test cases for Alert dataclass."""
    
    def test_alert_creation(self):
        """Test creating an alert."""
        alert = Alert(
            rule_name="Test Rule",
            severity="HIGH",
            description="Test alert",
            timestamp=time.time(),
            can_id=0x123,
            message_data={'test': 'data'},
            confidence=0.95
        )
        
        assert alert.rule_name == "Test Rule"
        assert alert.severity == "HIGH"
        assert alert.confidence == 0.95


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
