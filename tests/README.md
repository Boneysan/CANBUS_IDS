# Tests Directory

Unit and integration tests for CAN-IDS.

## Test Structure

- `test_capture.py` - CAN capture module tests
- `test_detection.py` - Detection engine tests
- `test_ml_detector.py` - ML detector tests
- `test_features.py` - Feature extraction tests
- `test_alerts.py` - Alert manager tests
- `test_integration.py` - End-to-end integration tests

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_detection.py -v

# Run tests in parallel
pytest tests/ -n auto
```

## Test Data

Test fixtures and sample data are stored in `tests/fixtures/`.

## TODO: Add Tests

Tests need to be implemented for:
- [ ] CAN message capture
- [ ] Rule engine evaluation
- [ ] ML anomaly detection
- [ ] Feature extraction
- [ ] Alert generation
- [ ] Notifiers
- [ ] Configuration loading
- [ ] Integration scenarios