#!/usr/bin/env python3
"""
Test script to verify ML detection initialization.
Validates that ML detector properly initializes and can analyze messages.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_ml_initialization():
    """Test ML detector initialization."""
    from src.detection.ml_detector import MLDetector
    
    print("\n" + "="*70)
    print("ML DETECTOR INITIALIZATION TEST")
    print("="*70)
    
    # Test 1: Check if model file exists
    model_path = project_root / "data/models/aggressive_load_shedding.joblib"
    
    print(f"\n1️⃣ Checking model file...")
    print(f"   Path: {model_path}")
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"   ✅ Model file exists ({size_mb:.2f} MB)")
    else:
        print(f"   ❌ Model file NOT found!")
        return False
    
    # Test 2: Create ML detector
    print(f"\n2️⃣ Creating ML detector...")
    try:
        ml_detector = MLDetector(
            model_path=str(model_path),
            contamination=0.20
        )
        print(f"   ✅ ML detector created")
    except Exception as e:
        print(f"   ❌ Failed to create ML detector: {e}")
        return False
    
    # Test 3: Load model
    print(f"\n3️⃣ Loading model...")
    try:
        ml_detector.load_model()
        print(f"   ✅ Model loaded successfully")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        return False
    
    # Test 4: Check training status
    print(f"\n4️⃣ Checking training status...")
    if ml_detector.is_trained:
        print(f"   ✅ Model is trained and ready")
    else:
        print(f"   ❌ Model is NOT trained!")
        return False
    
    # Test 5: Test message analysis
    print(f"\n5️⃣ Testing message analysis...")
    test_message = {
        'timestamp': 1234567890.0,
        'can_id': 0x123,
        'dlc': 8,
        'data': [0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04],
        'is_extended': False,
        'is_remote': False,
        'is_error': False
    }
    
    try:
        # Analyze multiple messages to build history
        for i in range(10):
            test_message['timestamp'] += 0.01
            test_message['data'][0] = i * 10
            alert = ml_detector.analyze_message(test_message)
            
            if alert:
                print(f"   ℹ️  Anomaly detected on message {i+1}: score={alert.anomaly_score:.3f}")
        
        print(f"   ✅ Message analysis working")
        
    except RuntimeError as e:
        print(f"   ❌ RuntimeError: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Analysis error: {e}")
        return False
    
    # Test 6: Check statistics
    print(f"\n6️⃣ Checking statistics...")
    stats = ml_detector.get_statistics()
    print(f"   Messages analyzed: {stats['messages_analyzed']}")
    print(f"   Anomalies detected: {stats['anomalies_detected']}")
    print(f"   Model loaded: {stats['model_loaded']}")
    print(f"   Is trained: {stats['is_trained']}")
    
    if stats['messages_analyzed'] > 0:
        print(f"   ✅ Statistics working")
    else:
        print(f"   ❌ No messages were analyzed!")
        return False
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED - ML DETECTION IS WORKING!")
    print("="*70)
    print("\nML detector is properly initialized and ready to use.")
    print(f"You can now run: python main.py -i can0\n")
    
    return True


def test_with_application_config():
    """Test using actual application configuration."""
    import yaml
    from src.detection.ml_detector import MLDetector
    
    print("\n" + "="*70)
    print("APPLICATION CONFIGURATION TEST")
    print("="*70)
    
    # Load actual config
    config_file = project_root / "config/can_ids_rpi4.yaml"
    
    print(f"\n1️⃣ Loading configuration...")
    print(f"   Config: {config_file}")
    
    try:
        with open(config_file) as f:
            config = yaml.safe_load(f)
        print(f"   ✅ Configuration loaded")
    except Exception as e:
        print(f"   ❌ Failed to load config: {e}")
        return False
    
    # Check detection modes
    print(f"\n2️⃣ Checking detection modes...")
    detection_modes = config.get('detection_modes', [])
    print(f"   Detection modes: {detection_modes}")
    
    if 'ml_based' in detection_modes:
        print(f"   ✅ ML detection is enabled in config")
    else:
        print(f"   ⚠️  ML detection is NOT enabled in config!")
        print(f"   You need to add 'ml_based' to detection_modes in config file")
        return False
    
    # Check ML model config
    print(f"\n3️⃣ Checking ML model configuration...")
    ml_config = config.get('ml_model', {})
    model_path = ml_config.get('path')
    contamination = ml_config.get('contamination', 0.20)
    
    print(f"   Model path: {model_path}")
    print(f"   Contamination: {contamination}")
    
    if model_path:
        full_path = project_root / model_path
        if full_path.exists():
            print(f"   ✅ Model file exists")
        else:
            print(f"   ❌ Model file NOT found at: {full_path}")
            return False
    else:
        print(f"   ❌ Model path not configured!")
        return False
    
    print("\n" + "="*70)
    print("✅ CONFIGURATION TEST PASSED")
    print("="*70)
    print("\nYour configuration is correct for ML detection.\n")
    
    return True


if __name__ == '__main__':
    print("\n🔍 CAN-IDS ML Detection Test Suite")
    print("="*70)
    
    # Run both tests
    test1_passed = test_ml_initialization()
    test2_passed = test_with_application_config()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"ML Initialization Test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Configuration Test:     {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 SUCCESS! ML detection is ready to use on your Raspberry Pi!")
        print("\nNext steps:")
        print("  1. Copy this updated code to your Raspberry Pi")
        print("  2. Run: python main.py -i can0")
        print("  3. Watch for the ML DETECTION ENABLED message")
        print("  4. Monitor for ML anomaly alerts in the output")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
        sys.exit(1)
