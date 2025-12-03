#!/usr/bin/env python3
"""
Test script to verify new models from Vehicle_Models load correctly.

Tests all 12 models exported on Dec 1, 2025.
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

try:
    import joblib
    import numpy as np
except ImportError as e:
    logger.error(f"Missing dependencies: {e}")
    logger.error("Install with: pip install joblib numpy")
    sys.exit(1)


def test_model_loading(model_path: Path) -> dict:
    """
    Test if a model loads successfully.
    
    Returns:
        dict with test results
    """
    result = {
        'model': model_path.name,
        'size_mb': model_path.stat().st_size / 1024 / 1024,
        'loaded': False,
        'has_predict': False,
        'error': None,
        'model_type': None
    }
    
    try:
        # Load model
        model = joblib.load(model_path)
        result['loaded'] = True
        result['model_type'] = type(model).__name__
        
        # Check for prediction method
        if hasattr(model, 'predict'):
            result['has_predict'] = True
        elif hasattr(model, 'detect_anomalies'):
            result['has_predict'] = True
            result['predict_method'] = 'detect_anomalies'
        elif hasattr(model, 'decision_function'):
            result['has_predict'] = True
            result['predict_method'] = 'decision_function'
        elif isinstance(model, dict):
            result['is_dict'] = True
            result['dict_keys'] = list(model.keys())
            # Check if it's a multi-stage dict
            if 'stage1_model' in model:
                result['format'] = 'multi-stage'
                result['has_predict'] = True
        
        logger.info(f"✅ {model_path.name}: {result['model_type']} ({result['size_mb']:.1f} MB)")
        
    except Exception as e:
        result['error'] = str(e)
        logger.error(f"❌ {model_path.name}: {e}")
    
    return result


def test_model_prediction(model_path: Path) -> dict:
    """
    Test if a model can make predictions.
    
    Returns:
        dict with prediction test results
    """
    result = {
        'model': model_path.name,
        'can_predict': False,
        'error': None
    }
    
    try:
        model = joblib.load(model_path)
        
        # Create dummy test data (typical CAN message features)
        # Features: [can_id, dlc, data_bytes, frequency, timing, etc.]
        test_data = np.array([[0x123, 8, 0xDE, 0xAD, 0xBE, 0xEF, 0.5, 0.001, 50.0]])
        
        # Try different prediction methods
        if hasattr(model, 'predict'):
            prediction = model.predict(test_data)
            result['can_predict'] = True
            result['prediction'] = str(prediction)
        elif hasattr(model, 'detect_anomalies'):
            prediction = model.detect_anomalies(test_data)
            result['can_predict'] = True
            result['prediction'] = str(prediction)
        elif hasattr(model, 'decision_function'):
            prediction = model.decision_function(test_data)
            result['can_predict'] = True
            result['prediction'] = str(prediction)
        elif isinstance(model, dict) and 'stage1_model' in model:
            # Multi-stage format
            stage1 = model['stage1_model']
            if hasattr(stage1, 'predict'):
                prediction = stage1.predict(test_data)
                result['can_predict'] = True
                result['prediction'] = str(prediction)
                result['format'] = 'multi-stage'
        
        if result['can_predict']:
            logger.info(f"  ✅ Prediction test passed")
        else:
            logger.warning(f"  ⚠️ No prediction method found")
            
    except Exception as e:
        result['error'] = str(e)
        logger.error(f"  ❌ Prediction test failed: {e}")
    
    return result


def main():
    """Main test function."""
    print("\n" + "="*70)
    print("Testing New Models from Vehicle_Models")
    print("="*70 + "\n")
    
    models_dir = Path("data/models")
    
    if not models_dir.exists():
        logger.error(f"Models directory not found: {models_dir}")
        sys.exit(1)
    
    # Get all .joblib files
    model_files = sorted(models_dir.glob("*.joblib"))
    
    if not model_files:
        logger.error("No .joblib files found in data/models/")
        sys.exit(1)
    
    print(f"Found {len(model_files)} model files\n")
    
    # Test 1: Model Loading
    print("TEST 1: Model Loading")
    print("-" * 70)
    
    load_results = []
    for model_file in model_files:
        result = test_model_loading(model_file)
        load_results.append(result)
    
    # Test 2: Prediction Capability
    print("\n" + "TEST 2: Prediction Capability")
    print("-" * 70)
    
    predict_results = []
    for model_file in model_files:
        if model_file.name in ['can_feature_engineer.joblib']:
            # Skip feature engineer (not a predictor)
            logger.info(f"⊘ {model_file.name}: Skipped (feature extractor)")
            continue
            
        result = test_model_prediction(model_file)
        predict_results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    loaded_count = sum(1 for r in load_results if r['loaded'])
    predict_count = sum(1 for r in predict_results if r['can_predict'])
    error_count = sum(1 for r in load_results if r['error'])
    
    print(f"\nModels tested: {len(model_files)}")
    print(f"Successfully loaded: {loaded_count}/{len(model_files)}")
    print(f"Can make predictions: {predict_count}/{len(predict_results)}")
    print(f"Errors encountered: {error_count}")
    
    # Model details
    print("\nModel Details:")
    print("-" * 70)
    print(f"{'Model':<40} {'Size':<12} {'Type':<20} {'Status'}")
    print("-" * 70)
    
    for result in load_results:
        status = "✅ OK" if result['loaded'] else "❌ FAIL"
        size_str = f"{result['size_mb']:.1f} MB"
        model_type = result.get('model_type', 'Unknown')[:18]
        print(f"{result['model']:<40} {size_str:<12} {model_type:<20} {status}")
    
    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    # Find best models
    high_precision = [r for r in load_results if 'weighted' in r['model'] and r['loaded']]
    fast_models = [r for r in load_results if 'adaptive' in r['model'] and r['loaded']]
    
    if high_precision:
        print("\n✨ Best Accuracy (97.20% recall, 100% precision):")
        print("   - adaptive_weighted_detector.joblib")
    
    if fast_models:
        print("\n⚡ High Performance (40-48K msg/s):")
        print("   - aggressive_load_shedding.joblib (2% Stage 3 load)")
        print("   - adaptive_load_shedding.joblib (3% Stage 3 load)")
    
    print("\n📊 Full Feature Set:")
    print("   - ensemble_detector.joblib (680MB)")
    print("   - improved_isolation_forest.joblib (658MB)")
    print("   - improved_svm.joblib (23MB)")
    
    # Exit status
    if error_count > 0:
        print("\n⚠️  Some models failed to load. Check errors above.")
        return 1
    else:
        print("\n✅ All models loaded successfully!")
        print("   Ready for integration into CANBUS_IDS")
        return 0


if __name__ == '__main__':
    sys.exit(main())
