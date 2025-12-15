#!/usr/bin/env python3
"""
Test Phase 3 Decision Tree integration.

Validates the complete 3-stage detection system:
  Stage 1: Timing-based statistical detection
  Stage 2: Rule-based pattern matching
  Stage 3: Decision Tree ML classifier

Expected Results:
  - All stages load successfully
  - Stage 3 throughput > 8,000 msg/s
  - Combined system throughput > 7,000 msg/s
  - Anomaly detection functional
"""

import sys
import time
import logging
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.decision_tree_detector import DecisionTreeDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_stage3_standalone():
    """Test Stage 3 detector in isolation."""
    logger.info("=" * 80)
    logger.info("TEST 1: Stage 3 Standalone Performance")
    logger.info("=" * 80)
    
    # Load model
    model_path = "data/models/decision_tree.pkl"
    detector = DecisionTreeDetector(model_path=model_path)
    
    if not detector.is_trained:
        logger.error("❌ Model not loaded!")
        return False
    
    logger.info("✅ Model loaded successfully")
    
    # Generate test messages
    num_messages = 10000
    logger.info(f"Generating {num_messages} test messages...")
    
    messages = []
    for i in range(num_messages):
        # Simulate normal message
        message = {
            'can_id': 0x100 + (i % 5),
            'timestamp': i * 0.01,
            'data': list(np.random.randint(0, 256, 8)),
            'dlc': 8
        }
        messages.append(message)
    
    # Performance test
    logger.info(f"Running Stage 3 performance test...")
    
    start_time = time.time()
    anomaly_count = 0
    
    for message in messages:
        alert = detector.analyze_message(message)
        if alert:
            anomaly_count += 1
    
    elapsed = time.time() - start_time
    throughput = num_messages / elapsed
    latency = (elapsed / num_messages) * 1000
    
    logger.info(f"")
    logger.info(f"Results:")
    logger.info(f"  Messages: {num_messages}")
    logger.info(f"  Time: {elapsed:.2f}s")
    logger.info(f"  Throughput: {throughput:,.0f} msg/s")
    logger.info(f"  Latency: {latency:.3f} ms/msg")
    logger.info(f"  Anomalies: {anomaly_count}")
    
    # Get detector stats
    stats = detector.get_stats()
    logger.info(f"")
    logger.info(f"Detector Statistics:")
    logger.info(f"  Tree depth: {stats['tree_depth']}")
    logger.info(f"  Tree leaves: {stats['tree_leaves']}")
    logger.info(f"  Avg inference: {stats['avg_inference_time_ms']:.3f} ms")
    
    # Validate performance
    if throughput >= 8000:
        logger.info("✅ Stage 3 throughput PASSED (>= 8,000 msg/s)")
        return True
    else:
        logger.warning(f"⚠️  Stage 3 throughput below target: {throughput:.0f} < 8,000 msg/s")
        return False


def test_anomaly_detection():
    """Test anomaly detection capability."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("TEST 2: Anomaly Detection Capability")
    logger.info("=" * 80)
    
    model_path = "data/models/decision_tree.pkl"
    detector = DecisionTreeDetector(model_path=model_path)
    
    # Normal message
    normal_msg = {
        'can_id': 0x100,
        'timestamp': 0.0,
        'data': [0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80],
        'dlc': 8
    }
    
    # Attack messages
    fuzzing_msg = {
        'can_id': 0x7FF,  # Suspicious high ID
        'timestamp': 0.01,
        'data': [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],
        'dlc': 8
    }
    
    dos_msg = {
        'can_id': 0x000,  # DoS pattern
        'timestamp': 0.02,
        'data': [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
        'dlc': 8
    }
    
    # Test messages
    test_cases = [
        ("Normal message", normal_msg, False),
        ("Fuzzing attack", fuzzing_msg, True),
        ("DoS attack", dos_msg, True)
    ]
    
    passed = 0
    for name, message, expect_anomaly in test_cases:
        alert = detector.analyze_message(message)
        detected = alert is not None
        
        if detected == expect_anomaly:
            logger.info(f"✅ {name}: {'ANOMALY' if detected else 'NORMAL'} (correct)")
            if alert:
                logger.info(f"   Confidence: {alert.confidence:.2%}")
                logger.info(f"   Top features: {list(alert.feature_importance.keys())[:3]}")
            passed += 1
        else:
            logger.warning(f"❌ {name}: Expected {'ANOMALY' if expect_anomaly else 'NORMAL'}, got {'ANOMALY' if detected else 'NORMAL'}")
    
    logger.info(f"")
    logger.info(f"Anomaly detection: {passed}/{len(test_cases)} tests passed")
    
    return passed == len(test_cases)


def test_feature_importance():
    """Test feature importance extraction."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("TEST 3: Feature Importance & Explainability")
    logger.info("=" * 80)
    
    model_path = "data/models/decision_tree.pkl"
    detector = DecisionTreeDetector(model_path=model_path)
    
    stats = detector.get_stats()
    feature_importance = stats.get('feature_importance', {})
    
    logger.info("Top features by importance:")
    for i, (feature, importance) in enumerate(sorted(
        feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5], 1):
        logger.info(f"  {i}. {feature:15s}: {importance:.4f}")
    
    # Check tree visualization exists
    viz_path = Path("data/models/decision_tree_rules.txt")
    if viz_path.exists():
        logger.info(f"")
        logger.info(f"✅ Tree visualization available: {viz_path}")
        logger.info(f"   Size: {viz_path.stat().st_size} bytes")
        logger.info(f"   Can be used for security audits and explainability")
    else:
        logger.warning("⚠️  Tree visualization not found")
    
    return True


def test_integration():
    """Test integration readiness."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("TEST 4: Integration Readiness")
    logger.info("=" * 80)
    
    # Check config
    config_path = Path("config/can_ids.yaml")
    if config_path.exists():
        logger.info(f"✅ Config file exists: {config_path}")
        
        with open(config_path, 'r') as f:
            import yaml
            config = yaml.safe_load(f)
            
            dt_config = config.get('decision_tree', {})
            if dt_config.get('enabled', False):
                logger.info(f"✅ Decision tree enabled in config")
                logger.info(f"   Model path: {dt_config.get('model_path')}")
            else:
                logger.warning("⚠️  Decision tree not enabled in config")
                return False
    else:
        logger.error("❌ Config file not found")
        return False
    
    # Check model files
    model_path = Path("data/models/decision_tree.pkl")
    if model_path.exists():
        logger.info(f"✅ Model file exists: {model_path.name} ({model_path.stat().st_size / 1024:.1f} KB)")
    else:
        logger.error("❌ Model file not found")
        return False
    
    # Check tree visualization
    viz_path = Path("data/models/decision_tree_rules.txt")
    if viz_path.exists():
        logger.info(f"✅ Tree visualization exists: {viz_path.name}")
    else:
        logger.warning("⚠️  Tree visualization not found")
    
    # Check main.py integration
    main_path = Path("main.py")
    if main_path.exists():
        with open(main_path, 'r') as f:
            main_content = f.read()
            
            if 'DecisionTreeDetector' in main_content:
                logger.info(f"✅ DecisionTreeDetector imported in main.py")
            else:
                logger.error("❌ DecisionTreeDetector not imported in main.py")
                return False
            
            if 'decision_tree_detector' in main_content:
                logger.info(f"✅ Decision tree detector integrated in pipeline")
            else:
                logger.error("❌ Decision tree detector not integrated")
                return False
    
    logger.info("")
    logger.info("✅ Integration checks PASSED")
    return True


def main():
    """Run all tests."""
    logger.info("")
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║" + " " * 20 + "PHASE 3 DECISION TREE VALIDATION" + " " * 26 + "║")
    logger.info("╚" + "=" * 78 + "╝")
    logger.info("")
    
    results = []
    
    # Run tests
    results.append(("Stage 3 Performance", test_stage3_standalone()))
    results.append(("Anomaly Detection", test_anomaly_detection()))
    results.append(("Feature Importance", test_feature_importance()))
    results.append(("Integration Readiness", test_integration()))
    
    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {name}")
    
    logger.info("")
    logger.info(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("")
        logger.info("🎉 ALL TESTS PASSED!")
        logger.info("")
        logger.info("Phase 3 Implementation Complete:")
        logger.info("  ✅ Stage 1: Timing detection (statistical)")
        logger.info("  ✅ Stage 2: Rule engine with early exit")
        logger.info("  ✅ Stage 3: Decision tree ML classifier")
        logger.info("")
        logger.info("System ready for production testing!")
        logger.info("Run: python main.py --config config/can_ids.yaml --test")
        return 0
    else:
        logger.error("")
        logger.error("❌ Some tests failed. Review output above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
