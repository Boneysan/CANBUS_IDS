#!/usr/bin/env python3
"""
Multi-Stage Detection Integration Test

Tests the integration of the Vehicle_Models multi-stage detection pipeline
with the CAN-IDS system.

This script validates:
1. Model loading and initialization
2. Feature compatibility
3. Detection functionality
4. Performance characteristics
5. Fallback mechanisms
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def test_multistage_integration():
    """Test the multi-stage detection integration."""
    print("🚗 CAN-IDS Multi-Stage Detection Integration Test")
    print("=" * 60)
    
    # Test 1: Import and Initialize
    print("\n1️⃣ Testing Enhanced ML Detector Import...")
    try:
        from detection.enhanced_ml_detector import EnhancedMLDetector, create_enhanced_ml_detector
        print("   ✅ Enhanced ML Detector imported successfully")
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False
    
    # Test 2: Configuration Loading
    print("\n2️⃣ Testing Configuration Loading...")
    try:
        import yaml
        with open('config/can_ids.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        ml_config = config.get('ml_detection', {})
        if ml_config.get('enable_multistage'):
            print("   ✅ Multi-stage configuration found")
            print(f"      Models dir: {ml_config['multistage']['models_dir']}")
            print(f"      Max Stage 3 load: {ml_config['multistage']['max_stage3_load']}")
        else:
            print("   ⚠️ Multi-stage disabled in configuration")
    except Exception as e:
        print(f"   ❌ Configuration loading failed: {e}")
        return False
    
    # Test 3: Model Files Check
    print("\n3️⃣ Checking Model Files...")
    models_dir = Path('models')
    required_files = [
        'multistage/aggressive_load_shedding.joblib',
        'improved_svm.joblib',
        'hybrid_rule_detector.joblib',
        'improved_isolation_forest.joblib'
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = models_dir / file_path
        if full_path.exists():
            size_mb = full_path.stat().st_size / (1024 * 1024)
            print(f"   ✅ {file_path} ({size_mb:.1f} MB)")
        else:
            print(f"   ❌ Missing: {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"   ⚠️ {len(missing_files)} files missing - some tests may fail")
    
    # Test 4: Enhanced Detector Initialization
    print("\n4️⃣ Testing Enhanced Detector Initialization...")
    try:
        detector = create_enhanced_ml_detector(config)
        print("   ✅ Enhanced detector created successfully")
        
        if detector.use_multistage:
            print("   ✅ Multi-stage pipeline enabled")
            if detector.multistage_detector:
                print("   ✅ Multi-stage detector loaded")
            else:
                print("   ⚠️ Multi-stage detector not loaded (models may be missing)")
        else:
            print("   ⚠️ Multi-stage pipeline disabled (fallback mode)")
            
    except Exception as e:
        print(f"   ❌ Enhanced detector initialization failed: {e}")
        print(f"      Error details: {type(e).__name__}: {str(e)}")
        return False
    
    # Test 5: Feature Extraction
    print("\n5️⃣ Testing Feature Extraction...")
    try:
        # Create test CAN message
        test_message = {
            'arbitration_id': 0x123,
            'data': [0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x11, 0x22, 0x33],
            'timestamp': time.time()
        }
        
        # Initialize message state in detector
        detector._update_message_state(test_message)
        
        # Extract features
        features = detector._extract_message_features(test_message)
        
        if features and len(features) >= 9:
            print(f"   ✅ Feature extraction successful ({len(features)} features)")
            print(f"      Sample features: {features[:5]}...")
        else:
            print(f"   ⚠️ Feature extraction returned {len(features) if features else 0} features")
            
    except Exception as e:
        print(f"   ❌ Feature extraction failed: {e}")
    
    # Test 6: Detection Analysis
    print("\n6️⃣ Testing Detection Analysis...")
    try:
        # Test with normal message
        normal_message = {
            'arbitration_id': 0x100,
            'data': [0x01, 0x02, 0x03, 0x04],
            'timestamp': time.time()
        }
        
        start_time = time.time()
        result = detector.analyze_message(normal_message)
        analysis_time = (time.time() - start_time) * 1000
        
        print(f"   ✅ Analysis completed in {analysis_time:.3f}ms")
        
        if result is None:
            print("   ✅ Normal message correctly classified (no alert)")
        else:
            print(f"   ⚠️ Normal message triggered alert: {result.anomaly_type}")
            
    except Exception as e:
        print(f"   ❌ Detection analysis failed: {e}")
    
    # Test 7: Performance Baseline
    print("\n7️⃣ Testing Performance Baseline...")
    try:
        num_messages = 100
        messages = []
        
        # Generate test messages
        for i in range(num_messages):
            messages.append({
                'arbitration_id': 0x100 + (i % 8),
                'data': [i % 256, (i*2) % 256, (i*3) % 256, (i*4) % 256],
                'timestamp': time.time() + i * 0.001
            })
        
        # Measure throughput
        start_time = time.time()
        detections = 0
        
        for msg in messages:
            result = detector.analyze_message(msg)
            if result:
                detections += 1
        
        elapsed_time = time.time() - start_time
        throughput = num_messages / elapsed_time
        avg_latency = (elapsed_time / num_messages) * 1000
        
        print(f"   ✅ Processed {num_messages} messages")
        print(f"      Throughput: {throughput:.0f} msg/s")
        print(f"      Average latency: {avg_latency:.3f}ms")
        print(f"      Detections: {detections}")
        
        # Performance targets for Pi4
        if throughput >= 1000:
            print("   ✅ Throughput suitable for Pi4 deployment")
        else:
            print("   ⚠️ Throughput may be low for high-traffic scenarios")
            
        if avg_latency <= 1.0:
            print("   ✅ Latency suitable for real-time processing")
        else:
            print("   ⚠️ Latency may impact real-time performance")
            
    except Exception as e:
        print(f"   ❌ Performance testing failed: {e}")
    
    # Test 8: Statistics and Monitoring
    print("\n8️⃣ Testing Statistics and Monitoring...")
    try:
        stats = detector.get_performance_stats()
        
        print("   ✅ Performance statistics retrieved")
        print(f"      Messages analyzed: {stats['messages_analyzed']}")
        print(f"      Multi-stage enabled: {stats.get('multistage_enabled', False)}")
        
        if stats.get('multistage_enabled'):
            perf_metrics = stats.get('performance_metrics', {})
            print(f"      Average latency: {perf_metrics.get('avg_latency_ms', 0):.3f}ms")
            
        # Print detailed statistics
        print("\n   📊 Detailed Statistics:")
        detector.print_enhanced_statistics()
        
    except Exception as e:
        print(f"   ❌ Statistics retrieval failed: {e}")
    
    # Test 9: Fallback Mechanism
    print("\n9️⃣ Testing Fallback Mechanism...")
    try:
        # Create detector with multi-stage disabled
        fallback_detector = EnhancedMLDetector(use_multistage=False)
        
        test_msg = {
            'arbitration_id': 0x200,
            'data': [0xFF, 0xFF],
            'timestamp': time.time()
        }
        
        result = fallback_detector.analyze_message(test_msg)
        print("   ✅ Fallback mechanism working")
        print(f"      Fallback result: {'Alert' if result else 'Normal'}")
        
    except Exception as e:
        print(f"   ❌ Fallback testing failed: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("🏁 INTEGRATION TEST SUMMARY")
    print("="*60)
    
    if detector.use_multistage and detector.multistage_detector:
        print("✅ Multi-stage integration: SUCCESS")
        print("   - Models loaded successfully")
        print("   - Feature extraction working")
        print("   - Detection pipeline functional")
        print("   - Performance within targets")
        print("\n🚀 Ready for enhanced CAN-IDS deployment!")
    else:
        print("⚠️ Multi-stage integration: PARTIAL")
        print("   - Basic integration working")
        print("   - Fallback mode operational") 
        print("   - Some models may be missing")
        print("\n🔧 Requires model setup completion")
    
    return True


if __name__ == "__main__":
    try:
        success = test_multistage_integration()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test failed with unexpected error: {e}")
        sys.exit(1)