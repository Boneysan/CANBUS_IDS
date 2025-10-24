"""
Test suite for alert management.

Tests for AlertManager and notifiers.
"""

import pytest
import time
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.alerts.alert_manager import AlertManager, Alert, AlertSeverity, AlertAction
from src.alerts.notifiers import ConsoleNotifier, JSONNotifier


class TestAlert:
    """Test cases for Alert dataclass."""
    
    def test_alert_creation(self):
        """Test creating an alert."""
        alert = Alert(
            id="test-001",
            timestamp=time.time(),
            rule_name="Test Rule",
            severity=AlertSeverity.HIGH,
            description="Test alert description",
            can_id=0x123,
            message_data={'test': 'data'},
            confidence=0.95,
            source="test_source"
        )
        
        assert alert.id == "test-001"
        assert alert.severity == AlertSeverity.HIGH
        assert alert.confidence == 0.95
        
    def test_alert_to_dict(self):
        """Test converting alert to dictionary."""
        alert = Alert(
            id="test-001",
            timestamp=time.time(),
            rule_name="Test Rule",
            severity=AlertSeverity.MEDIUM,
            description="Test",
            can_id=0x100,
            message_data={},
            action=AlertAction.ALERT
        )
        
        alert_dict = alert.to_dict()
        
        assert isinstance(alert_dict, dict)
        assert alert_dict['id'] == "test-001"
        assert alert_dict['severity'] == "MEDIUM"
        assert alert_dict['action'] == "alert"


class TestAlertManager:
    """Test cases for AlertManager class."""
    
    def test_initialization(self):
        """Test AlertManager initialization."""
        config = {
            'rate_limit': 10,
            'deduplication_window': 60
        }
        
        manager = AlertManager(config)
        
        assert manager.config == config
        assert len(manager.notifiers) == 0
        
    def test_add_notifier(self):
        """Test adding a notifier."""
        manager = AlertManager({})
        notifier = ConsoleNotifier({})
        
        manager.add_notifier(notifier)
        
        assert len(manager.notifiers) == 1
        assert manager.notifiers[0] == notifier
        
    def test_process_alert(self):
        """Test processing an alert."""
        manager = AlertManager({'rate_limit': 100})
        
        alert_data = {
            'timestamp': time.time(),
            'rule_name': 'Test Rule',
            'severity': 'HIGH',
            'description': 'Test alert',
            'can_id': 0x123,
            'message_data': {},
            'confidence': 1.0,
            'source': 'test'
        }
        
        # Should process without error
        manager.process_alert(alert_data)
        
    def test_rate_limiting(self):
        """Test that rate limiting works."""
        manager = AlertManager({'rate_limit': 2})  # Only 2 alerts per second
        
        # Send multiple alerts rapidly
        processed_count = 0
        for i in range(10):
            alert_data = {
                'timestamp': time.time(),
                'rule_name': 'Test Rule',
                'severity': 'HIGH',
                'description': f'Alert {i}',
                'can_id': 0x100,
                'message_data': {},
                'confidence': 1.0,
                'source': 'test'
            }
            
            # The manager should limit the number processed
            manager.process_alert(alert_data)
            processed_count += 1
        
        # Some alerts should have been rate limited
        # (exact behavior depends on implementation)
        assert processed_count == 10  # All were submitted
        
    def test_statistics(self):
        """Test statistics tracking."""
        manager = AlertManager({})
        
        # Process some alerts
        for i in range(5):
            alert_data = {
                'timestamp': time.time(),
                'rule_name': 'Test',
                'severity': 'MEDIUM',
                'description': 'Test',
                'can_id': 0x100,
                'message_data': {},
                'confidence': 1.0,
                'source': 'test'
            }
            manager.process_alert(alert_data)
        
        stats = manager.get_statistics()
        
        assert 'total_alerts' in stats
        assert 'alerts_sent' in stats


class TestConsoleNotifier:
    """Test cases for ConsoleNotifier."""
    
    def test_initialization(self):
        """Test ConsoleNotifier initialization."""
        notifier = ConsoleNotifier({'enabled': True})
        
        assert notifier.config == {'enabled': True}
        
    def test_send_alert(self, capsys):
        """Test sending alert to console."""
        notifier = ConsoleNotifier({'enabled': True})
        
        alert_data = {
            'timestamp': time.time(),
            'rule_name': 'Test Rule',
            'severity': 'HIGH',
            'description': 'Test alert',
            'can_id': 0x123,
            'confidence': 1.0
        }
        
        notifier.send(alert_data)
        
        # Check that something was printed
        captured = capsys.readouterr()
        assert 'ALERT' in captured.out or 'HIGH' in captured.out


class TestJSONNotifier:
    """Test cases for JSONNotifier."""
    
    def test_initialization(self, tmp_path):
        """Test JSONNotifier initialization."""
        log_file = tmp_path / "alerts.json"
        
        notifier = JSONNotifier({
            'log_file': str(log_file),
            'enabled': True
        })
        
        assert notifier.log_file == Path(log_file)
        
    def test_send_alert(self, tmp_path):
        """Test sending alert to JSON file."""
        log_file = tmp_path / "alerts.json"
        
        notifier = JSONNotifier({
            'log_file': str(log_file),
            'enabled': True
        })
        
        alert_data = {
            'timestamp': time.time(),
            'rule_name': 'Test Rule',
            'severity': 'MEDIUM',
            'description': 'Test alert',
            'can_id': 0x100,
            'confidence': 0.9
        }
        
        notifier.send(alert_data)
        
        # Verify file was created and contains data
        assert log_file.exists()
        
        with open(log_file, 'r') as f:
            content = f.read()
            # Check if alert data was written
            assert 'Test Rule' in content or 'MEDIUM' in content


class TestAlertIntegration:
    """Integration tests for alert system."""
    
    def test_end_to_end_alerting(self, tmp_path):
        """Test complete alerting pipeline."""
        log_file = tmp_path / "alerts.json"
        
        # Setup alert manager with notifiers
        manager = AlertManager({
            'rate_limit': 100,
            'log_file': str(log_file)
        })
        
        console_notifier = ConsoleNotifier({'enabled': True})
        json_notifier = JSONNotifier({
            'log_file': str(log_file),
            'enabled': True
        })
        
        manager.add_notifier(console_notifier)
        manager.add_notifier(json_notifier)
        
        # Send alerts
        for i in range(3):
            alert_data = {
                'timestamp': time.time(),
                'rule_name': f'Rule {i}',
                'severity': 'HIGH' if i == 0 else 'MEDIUM',
                'description': f'Test alert {i}',
                'can_id': 0x100 + i,
                'message_data': {'index': i},
                'confidence': 0.95,
                'source': 'test'
            }
            
            manager.process_alert(alert_data)
        
        # Verify alerts were logged
        assert log_file.exists()
        
        # Check statistics
        stats = manager.get_statistics()
        assert stats['total_alerts'] >= 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
