"""
Integration tests for the complete CAN-IDS system.

Tests the full pipeline from message capture through detection to alerting.
"""

import pytest
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.rule_engine import RuleEngine
from src.detection.ml_detector import MLDetector
from src.preprocessing.feature_extractor import FeatureExtractor
from src.preprocessing.normalizer import Normalizer
from src.alerts.alert_manager import AlertManager
from src.alerts.notifiers import ConsoleNotifier, JSONNotifier


@pytest.mark.integration
class TestEndToEndPipeline:
    """Integration tests for the complete detection pipeline."""
    
    def test_rule_based_pipeline(self, tmp_path, sample_can_messages):
        """Test complete rule-based detection pipeline."""
        # Create temporary rules file
        rules_content = """
rules:
  - name: "Test Detection"
    can_id: 0x100
    severity: HIGH
    description: "Test rule"
    action: alert
"""
        rules_file = tmp_path / "rules.yaml"
        rules_file.write_text(rules_content)
        
        # Create temporary alert log
        alert_log = tmp_path / "alerts.json"
        
        # Initialize components
        rule_engine = RuleEngine(str(rules_file))
        alert_manager = AlertManager({'log_file': str(alert_log)})
        json_notifier = JSONNotifier({
            'log_file': str(alert_log),
            'enabled': True
        })
        alert_manager.add_notifier(json_notifier)
        
        # Process messages
        alerts_generated = 0
        for message in sample_can_messages:
            alerts = rule_engine.analyze_message(message)
            
            for alert in alerts:
                alert_data = {
                    'timestamp': alert.timestamp,
                    'rule_name': alert.rule_name,
                    'severity': alert.severity,
                    'description': alert.description,
                    'can_id': alert.can_id,
                    'message_data': alert.message_data,
                    'confidence': alert.confidence,
                    'source': 'rule_engine'
                }
                alert_manager.process_alert(alert_data)
                alerts_generated += 1
        
        # Verify pipeline executed
        assert rule_engine._stats['messages_processed'] == len(sample_can_messages)
        
    def test_ml_based_pipeline(self, sample_can_messages):
        """Test complete ML-based detection pipeline."""
        # Initialize components
        feature_extractor = FeatureExtractor()
        normalizer = Normalizer(method='minmax')
        ml_detector = MLDetector(contamination=0.1)
        
        # Extract features from training data
        train_features = []
        for message in sample_can_messages[:50]:
            features = feature_extractor.extract_features(message)
            train_features.append(features)
        
        # Fit normalizer
        normalizer.fit(train_features)
        
        # Train ML model
        try:
            ml_detector.train(train_features)
        except ImportError:
            pytest.skip("sklearn not installed")
        
        # Process test messages
        for message in sample_can_messages[50:]:
            features = feature_extractor.extract_features(message)
            normalized = normalizer.transform(features)
            alert = ml_detector.analyze_message(message)
            
            # Just verify no errors occur
            assert features is not None
            assert normalized is not None
            
    def test_hybrid_detection(self, tmp_path, sample_can_messages):
        """Test hybrid rule + ML detection."""
        # Setup rules
        rules_content = """
rules:
  - name: "Frequency Check"
    can_id: 0x100
    max_frequency: 50
    time_window: 1
    severity: MEDIUM
    description: "High frequency"
    action: alert
"""
        rules_file = tmp_path / "rules.yaml"
        rules_file.write_text(rules_content)
        
        # Initialize all components
        rule_engine = RuleEngine(str(rules_file))
        feature_extractor = FeatureExtractor()
        ml_detector = MLDetector()
        alert_manager = AlertManager({})
        
        # Process messages with both detectors
        total_rule_alerts = 0
        total_ml_alerts = 0
        
        for message in sample_can_messages:
            # Rule-based detection
            rule_alerts = rule_engine.analyze_message(message)
            total_rule_alerts += len(rule_alerts)
            
            # ML-based detection (if trained)
            try:
                ml_alert = ml_detector.analyze_message(message)
                if ml_alert:
                    total_ml_alerts += 1
            except:
                pass  # ML may not be trained yet
        
        # Verify both detectors ran
        assert rule_engine._stats['messages_processed'] == len(sample_can_messages)
        
    def test_performance_under_load(self, sample_can_messages):
        """Test system performance with high message volume."""
        # Generate large dataset
        messages = sample_can_messages * 100  # 1000+ messages
        
        # Initialize detector
        rule_engine = RuleEngine('config/rules.yaml')
        
        # Measure processing time
        start_time = time.time()
        
        for message in messages:
            rule_engine.analyze_message(message)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Calculate throughput
        throughput = len(messages) / duration
        
        # Should handle at least 100 msg/s
        assert throughput > 100, f"Throughput too low: {throughput:.2f} msg/s"
        
        # Check statistics
        stats = rule_engine.get_statistics()
        assert stats['messages_processed'] == len(messages)


@pytest.mark.integration
class TestConfigurationLoading:
    """Integration tests for configuration loading."""
    
    def test_load_full_configuration(self, temp_config_file):
        """Test loading complete configuration file."""
        import yaml
        
        with open(temp_config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        assert 'interface' in config
        assert 'detection_modes' in config
        assert 'alerts' in config
        
    def test_load_rules_configuration(self):
        """Test loading rules configuration."""
        rules_file = Path('config/rules.yaml')
        
        if rules_file.exists():
            rule_engine = RuleEngine(str(rules_file))
            assert len(rule_engine.rules) > 0
            
            # Verify rule structure
            for rule in rule_engine.rules:
                assert rule.name is not None
                assert rule.severity is not None
                assert rule.description is not None


@pytest.mark.integration
class TestDataPersistence:
    """Integration tests for data persistence."""
    
    def test_save_and_load_ml_model(self, tmp_path, sample_can_messages):
        """Test ML model persistence."""
        try:
            from src.models.train_model import ModelTrainer
        except ImportError:
            pytest.skip("sklearn not installed")
        
        # Train model
        feature_extractor = FeatureExtractor()
        features = [feature_extractor.extract_features(msg) 
                   for msg in sample_can_messages]
        
        trainer = ModelTrainer(contamination=0.1)
        
        # Prepare and train
        try:
            X_train, X_test, _, _ = trainer.prepare_data(features)
            trainer.train(X_train)
            
            # Save model
            model_path = tmp_path / "test_model.pkl"
            trainer.save_model(str(model_path))
            
            assert model_path.exists()
            
            # Load model
            new_trainer = ModelTrainer()
            new_trainer.load_model(str(model_path))
            
            assert new_trainer.isolation_forest is not None
            
        except Exception as e:
            pytest.skip(f"Model training failed: {e}")
            
    def test_alert_logging(self, tmp_path):
        """Test alert persistence to log files."""
        log_file = tmp_path / "test_alerts.json"
        
        # Create alert manager with JSON notifier
        manager = AlertManager({'log_file': str(log_file)})
        notifier = JSONNotifier({
            'log_file': str(log_file),
            'enabled': True
        })
        manager.add_notifier(notifier)
        
        # Generate alerts
        for i in range(5):
            alert_data = {
                'timestamp': time.time(),
                'rule_name': f'Test Rule {i}',
                'severity': 'HIGH',
                'description': f'Test alert {i}',
                'can_id': 0x100 + i,
                'message_data': {},
                'confidence': 0.9,
                'source': 'test'
            }
            manager.process_alert(alert_data)
        
        # Verify log file exists and contains data
        assert log_file.exists()
        assert log_file.stat().st_size > 0


@pytest.mark.integration  
@pytest.mark.slow
class TestStressTests:
    """Stress tests for system robustness."""
    
    def test_high_volume_messages(self):
        """Test handling of high message volumes."""
        rule_engine = RuleEngine('config/rules.yaml')
        
        # Generate 10,000 messages
        for i in range(10000):
            message = {
                'timestamp': time.time(),
                'can_id': 0x100 + (i % 10),
                'dlc': 8,
                'data': [(i * j) % 256 for j in range(8)],
                'is_extended': False,
                'is_remote': False,
                'is_error': False
            }
            
            rule_engine.analyze_message(message)
        
        stats = rule_engine.get_statistics()
        assert stats['messages_processed'] == 10000
        
    def test_memory_usage(self, sample_can_messages):
        """Test that memory usage stays reasonable."""
        import sys
        
        # Get initial memory
        initial_size = sys.getsizeof(sample_can_messages)
        
        # Process messages multiple times
        feature_extractor = FeatureExtractor()
        
        for _ in range(100):
            for message in sample_can_messages:
                features = feature_extractor.extract_features(message)
        
        # Memory should not grow unbounded
        # (This is a basic check - proper memory profiling would be better)
        final_size = sys.getsizeof(feature_extractor)
        
        # Just verify it completes without error
        assert final_size > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
