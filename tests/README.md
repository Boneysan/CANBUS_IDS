# CAN-IDS Test Suite

This directory contains a comprehensive test suite for the CAN-IDS system.

## Test Structure

Tests are organized by module:

- **`test_capture.py`** - CAN capture and PCAP reading tests
  - CANSniffer initialization and operation
  - PCAP/candump log parsing
  - Message format handling
  
- **`test_detection.py`** - Detection engine tests
  - Rule engine functionality
  - ML detector operation
  - Alert generation
  
- **`test_preprocessing.py`** - Data preprocessing tests
  - Feature extraction (40+ features)
  - Data normalization (min-max, z-score, robust)
  - Pipeline integration
  
- **`test_alerts.py`** - Alert management tests
  - Alert manager coordination
  - Rate limiting and deduplication
  - Multiple notifiers (console, JSON, email, webhook)
  
- **`test_integration.py`** - End-to-end integration tests
  - Complete detection pipeline
  - Configuration loading
  - Data persistence
  - Performance and stress tests
  
- **`conftest.py`** - Pytest configuration and shared fixtures

## Running Tests

### 1. Install Test Dependencies

```bash
pip install -r requirements-dev.txt
```

This installs:
- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting
- Additional test utilities

### 2. Run All Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run with coverage report
pytest --cov=src tests/

# Generate HTML coverage report
pytest --cov=src --cov-report=html tests/
```

### 3. Run Specific Tests

```bash
# Run specific test file
pytest tests/test_detection.py

# Run specific test class
pytest tests/test_detection.py::TestRuleEngine

# Run specific test method
pytest tests/test_detection.py::TestRuleEngine::test_initialization

# Run tests matching pattern
pytest tests/ -k "detection"
```

### 4. Run Tests by Marker

```bash
# Run only integration tests
pytest tests/ -m integration

# Skip slow tests
pytest tests/ -m "not slow"

# Skip tests requiring hardware
pytest tests/ -m "not requires_hardware"
```

### 5. Advanced Options

```bash
# Stop on first failure
pytest tests/ -x

# Show local variables on failure
pytest tests/ -l

# Run in parallel (requires pytest-xdist)
pytest tests/ -n auto

# Generate JUnit XML report (for CI/CD)
pytest tests/ --junit-xml=test-results.xml
```

## Test Coverage

Current test coverage includes:

- ✅ **Capture Module** (~80% coverage)
  - CAN sniffer operations
  - PCAP reading
  - Message parsing

- ✅ **Detection Module** (~85% coverage)
  - Rule engine with 20+ rules
  - ML anomaly detection
  - Alert generation

- ✅ **Preprocessing Module** (~90% coverage)
  - Feature extraction
  - Data normalization
  - Pipeline integration

- ✅ **Alert Module** (~85% coverage)
  - Alert management
  - Rate limiting
  - Multiple notifiers

- ✅ **Integration Tests**
  - End-to-end pipeline
  - Performance benchmarks
  - Configuration loading

## Writing New Tests

### Test File Template

```python
"""Test suite for [module name]."""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.module import ClassToTest


class TestClassName:
    """Test cases for ClassName."""
    
    def test_feature(self):
        """Test specific feature."""
        # Arrange
        obj = ClassToTest()
        
        # Act
        result = obj.method()
        
        # Assert
        assert result == expected_value


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

### Using Fixtures

Fixtures are defined in `conftest.py`:

```python
def test_with_fixture(sample_can_message):
    """Use predefined CAN message fixture."""
    assert sample_can_message['can_id'] == 0x123
```

## Continuous Integration

Tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Run tests
  run: |
    pip install -r requirements-dev.txt
    pytest tests/ --cov=src --junit-xml=test-results.xml
```

## Troubleshooting

### Import Errors

If you see import errors:
```bash
# Make sure you're in the project root
cd /path/to/CANBUS_IDS

# Run with Python path set
PYTHONPATH=. pytest tests/
```

### Dependency Errors

If sklearn/numpy tests fail:
```bash
# Install optional dependencies
pip install numpy scikit-learn
```