"""
Test configuration file for pytest.

Contains common fixtures and configuration for all tests.
"""

import pytest
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_can_message():
    """Fixture providing a sample CAN message."""
    import time
    return {
        'timestamp': time.time(),
        'can_id': 0x123,
        'dlc': 8,
        'data': [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08],
        'data_hex': '01 02 03 04 05 06 07 08',
        'is_extended': False,
        'is_remote': False,
        'is_error': False
    }


@pytest.fixture
def sample_can_messages():
    """Fixture providing multiple sample CAN messages."""
    import time
    base_time = time.time()
    
    messages = []
    for i in range(10):
        messages.append({
            'timestamp': base_time + i * 0.1,
            'can_id': 0x100 + (i % 3),
            'dlc': 8,
            'data': [(i * j) % 256 for j in range(8)],
            'is_extended': False,
            'is_remote': False,
            'is_error': False
        })
    
    return messages


@pytest.fixture
def temp_rules_file(tmp_path):
    """Fixture providing a temporary rules file."""
    rules_content = """
rules:
  - name: "Test Rule 1"
    can_id: 0x100
    severity: HIGH
    description: "Test rule"
    action: alert
    
  - name: "Test Rule 2"
    can_id_range: [0x200, 0x2FF]
    severity: MEDIUM
    description: "Range test"
    action: alert
"""
    
    rules_file = tmp_path / "test_rules.yaml"
    rules_file.write_text(rules_content)
    return str(rules_file)


@pytest.fixture
def temp_config_file(tmp_path):
    """Fixture providing a temporary configuration file."""
    config_content = """
log_level: INFO
interface: vcan0
bustype: socketcan

rules_file: config/rules.yaml
detection_modes:
  - rule_based

alerts:
  console_output: true
  rate_limit: 10

capture:
  buffer_size: 100
"""
    
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)
    return str(config_file)


# Configure pytest
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_hardware: marks tests that require CAN hardware"
    )
