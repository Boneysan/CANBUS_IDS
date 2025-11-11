"""
Enhanced ML Detector with Multi-Stage Pipeline Support

Integrates the advanced multi-stage detection pipeline from Vehicle_Models
while maintaining backward compatibility with the existing single-stage detector.

Key Features:
- Multi-stage progressive detection (Stage 1: IF → Stage 2: Rules → Stage 3: SVM)
- 100K+ msg/s throughput with adaptive load balancing
- Vehicle-specific model calibration
- Graceful fallback to single-stage detection
- Real-time performance monitoring
"""

import time
import logging
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path

# Import existing classes
from .ml_detector import MLDetector, MLAlert

logger = logging.getLogger(__name__)

try:
    from .multistage_detector import create_default_multistage_detector, MultiStageDetector
    from .vehicle_calibration import VehicleCalibrationManager
    MULTISTAGE_AVAILABLE = True
    logger.info("✅ Multi-stage detection components available")
except ImportError as e:
    MULTISTAGE_AVAILABLE = False
    logger.warning(f"⚠️ Multi-stage detection not available: {e}")
    logger.warning("Falling back to single-stage detection")


class EnhancedMLDetector(MLDetector):
    """Enhanced ML detector with multi-stage pipeline support."""
    
    def __init__(self, use_multistage: bool = True, 
                 models_dir: str = "models",
                 max_stage3_load: float = 0.15,
                 enable_vehicle_calibration: bool = False,
                 **kwargs):
        """
        Initialize enhanced ML detector.
        
        Parameters:
        -----------
        use_multistage : bool, default=True
            Whether to use multi-stage detection pipeline
        models_dir : str, default="models"
            Directory containing trained models
        max_stage3_load : float, default=0.15
            Maximum percentage of traffic to process in Stage 3 (for Pi4 optimization)
        enable_vehicle_calibration : bool, default=False
            Whether to enable vehicle-specific model calibration
        **kwargs : Additional arguments passed to base MLDetector
        """
        # Initialize base detector first
        super().__init__(**kwargs)
        
        # Multi-stage configuration
        self.use_multistage = use_multistage and MULTISTAGE_AVAILABLE
        self.models_dir = Path(models_dir)
        self.max_stage3_load = max_stage3_load
        self.enable_vehicle_calibration = enable_vehicle_calibration
        
        # Multi-stage components
        self.multistage_detector = None
        self.vehicle_calibration_mgr = None
        self.detected_vehicle_type = 'unknown'
        
        # Enhanced statistics (update existing stats dict)
        self._stats.update({
            'multistage_detections': 0,
            'stage1_filtered': 0,
            'stage2_filtered': 0,
            'stage3_processed': 0,
            'vehicle_detections': {},
            'multistage_performance': {
                'avg_latency_ms': 0.0,
                'stage_load_distribution': [0, 0, 0]
            }
        })
        
        # Initialize multi-stage components
        if self.use_multistage:
            self._initialize_multistage_detector()
            
        if self.enable_vehicle_calibration:
            self._initialize_vehicle_calibration()
    
    def _initialize_multistage_detector(self) -> None:
        """Initialize the multi-stage detection pipeline."""
        try:
            if not self.models_dir.exists():
                logger.warning(f"Models directory not found: {self.models_dir}")
                logger.warning("Multi-stage detection disabled")
                self.use_multistage = False
                return
            
            self.multistage_detector = create_default_multistage_detector(
                models_dir=str(self.models_dir),
                enable_adaptive_gating=True,
                enable_load_shedding=True,
                max_stage3_load=self.max_stage3_load,
                verbose=False
            )
            
            logger.info("✅ Multi-stage detector initialized successfully")
            logger.info(f"   Stage 3 load limit: {self.max_stage3_load:.1%}")
            logger.info(f"   Models directory: {self.models_dir}")
            
            # Update model loaded status
            self._stats['model_loaded'] = True
            
        except Exception as e:
            logger.error(f"❌ Multi-stage detector initialization failed: {e}")
            logger.warning("Falling back to single-stage detection")
            self.use_multistage = False
            self.multistage_detector = None
    
    def _initialize_vehicle_calibration(self) -> None:
        """Initialize vehicle-specific calibration manager."""
        try:
            self.vehicle_calibration_mgr = VehicleCalibrationManager()
            logger.info("✅ Vehicle calibration manager initialized")
        except Exception as e:
            logger.warning(f"Vehicle calibration initialization failed: {e}")
            self.enable_vehicle_calibration = False
    
    def analyze_message(self, message: Dict[str, Any]) -> Optional[MLAlert]:
        """
        Analyze message with enhanced multi-stage detection.
        
        Parameters:
        -----------
        message : Dict[str, Any]
            CAN message to analyze
            
        Returns:
        --------
        MLAlert or None
            Alert if attack detected, None otherwise
        """
        start_time = time.time()
        
        try:
            if self.use_multistage and self.multistage_detector is not None:
                result = self._analyze_multistage(message)
            else:
                # Fallback to single-stage detection
                result = super().analyze_message(message)
            
            # Update performance metrics
            elapsed_ms = (time.time() - start_time) * 1000
            self._update_performance_metrics(elapsed_ms)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in enhanced ML analysis: {e}")
            # Fallback to base detector
            return super().analyze_message(message)
    
    def _analyze_multistage(self, message: Dict[str, Any]) -> Optional[MLAlert]:
        """Perform multi-stage analysis with enhanced features."""
        # Extract features using existing feature extraction
        features = self._extract_message_features(message)
        
        if not features:
            return None
        
        # Update message state for feature extraction
        self._update_message_state(message)
        
        # Prepare features for multi-stage detector
        feature_array = np.array([features])
        
        # Multi-stage prediction with confidence
        try:
            prediction, confidence = self.multistage_detector.predict_with_confidence(feature_array)
            
            # Update stage statistics
            stage_metrics = self.multistage_detector.get_stage_metrics()
            self._update_stage_statistics(stage_metrics)
            
            if prediction[0] == 1:  # Attack detected
                # Vehicle type detection for enhanced context
                vehicle_context = self._get_vehicle_context(message)
                
                # Create enhanced alert
                alert = MLAlert(
                    timestamp=message.get('timestamp', time.time()),
                    can_id=message['arbitration_id'],
                    confidence_score=float(confidence[0]),
                    anomaly_type='multistage_detection',
                    features_used=len(features),
                    details={
                        'detection_stage': self._determine_detection_stage(confidence[0]),
                        'stage_metrics': {
                            'stage1_passed': stage_metrics.get('stage1_passed', 0),
                            'stage2_passed': stage_metrics.get('stage2_passed', 0),
                            'stage3_processed': stage_metrics.get('stage3_processed', 0)
                        },
                        'vehicle_context': vehicle_context,
                        'load_shedding_active': getattr(self.multistage_detector, 'is_load_shedding_active', lambda: False)(),
                        'model_path': str(self.models_dir)
                    }
                )
                
                # Update enhanced statistics
                self._stats['multistage_detections'] += 1
                if vehicle_context['type'] != 'unknown':
                    vehicle_type = vehicle_context['type']
                    self._stats['vehicle_detections'][vehicle_type] = (
                        self._stats['vehicle_detections'].get(vehicle_type, 0) + 1
                    )
                
                return alert
            
            return None
            
        except Exception as e:
            logger.error(f"Multi-stage detection error: {e}")
            # Fallback to single-stage
            return super().analyze_message({'arbitration_id': message['arbitration_id'], 
                                          'data': message.get('data', []),
                                          'timestamp': message.get('timestamp', time.time())})
    
    def _extract_message_features(self, message: Dict[str, Any]) -> List[float]:
        """Extract features compatible with multi-stage detector."""
        try:
            # Use existing feature extraction from base class
            features = []
            
            # Basic message features
            can_id = message['arbitration_id']
            data = message.get('data', [])
            timestamp = message.get('timestamp', time.time())
            
            # Core features expected by multi-stage detector
            features.extend([
                float(can_id),  # arb_id_numeric
                float(len(data)),  # data_length
                self._calculate_id_frequency(can_id),  # id_frequency
                self._calculate_time_delta(timestamp),  # time_delta
                self._calculate_id_mean_time_delta(can_id),  # id_mean_time_delta
                self._calculate_id_std_time_delta(can_id),  # id_std_time_delta
                float(time.localtime(timestamp).tm_hour),  # hour
                float(time.localtime(timestamp).tm_min),  # minute
                float(time.localtime(timestamp).tm_sec)  # second
            ])
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return []
    
    def _calculate_id_frequency(self, can_id: int) -> float:
        """Calculate frequency of CAN ID appearance."""
        # Use existing message history from base class
        if hasattr(self, '_message_history') and self._message_history:
            total_messages = sum(len(history) for history in self._message_history.values())
            id_count = len(self._message_history.get(can_id, []))
            return float(id_count / max(total_messages, 1))
        return 0.0
    
    def _calculate_time_delta(self, timestamp: float) -> float:
        """Calculate time delta from last message."""
        if hasattr(self, 'last_message_time') and self.last_message_time:
            return float(timestamp - self.last_message_time)
        return 0.0
    
    def _calculate_id_mean_time_delta(self, can_id: int) -> float:
        """Calculate mean time delta for specific CAN ID."""
        if hasattr(self, 'id_timings') and can_id in self.id_timings:
            timings = self.id_timings[can_id]
            if len(timings) > 1:
                deltas = [timings[i] - timings[i-1] for i in range(1, len(timings))]
                return float(np.mean(deltas))
        return 0.0
    
    def _calculate_id_std_time_delta(self, can_id: int) -> float:
        """Calculate standard deviation of time delta for CAN ID."""
        if hasattr(self, 'id_timings') and can_id in self.id_timings:
            timings = self.id_timings[can_id]
            if len(timings) > 2:
                deltas = [timings[i] - timings[i-1] for i in range(1, len(timings))]
                return float(np.std(deltas))
        return 0.0
    
    def _get_vehicle_context(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Get vehicle-specific context information."""
        context = {
            'type': 'unknown',
            'confidence': 0.0,
            'calibration_available': False
        }
        
        if self.enable_vehicle_calibration and self.vehicle_calibration_mgr:
            try:
                # Detect vehicle type from recent CAN IDs
                if hasattr(self, '_message_history') and self._message_history:
                    recent_ids = list(self._message_history.keys())[-50:]  # Last 50 unique IDs
                    vehicle_type = self.vehicle_calibration_mgr.detect_vehicle_type(recent_ids)
                    
                    context.update({
                        'type': vehicle_type,
                        'calibration_available': True,
                        'can_ids_analyzed': len(recent_ids)
                    })
                    
                    self.detected_vehicle_type = vehicle_type
                
            except Exception as e:
                logger.debug(f"Vehicle detection error: {e}")
        
        return context
    
    def _determine_detection_stage(self, confidence: float) -> str:
        """Determine which stage likely made the detection."""
        if confidence >= 0.8:
            return 'stage1_high_confidence'
        elif confidence >= 0.6:
            return 'stage2_rule_validation'
        else:
            return 'stage3_deep_analysis'
    
    def _update_stage_statistics(self, stage_metrics: Dict) -> None:
        """Update statistics based on stage metrics."""
        # Update stage filtering counts
        self._stats['stage1_filtered'] = stage_metrics.get('stage1_filtered', 0)
        self._stats['stage2_filtered'] = stage_metrics.get('stage2_filtered', 0) 
        self._stats['stage3_processed'] = stage_metrics.get('stage3_processed', 0)
        
        # Update load distribution
        total = sum([
            stage_metrics.get('stage1_processed', 0),
            stage_metrics.get('stage2_processed', 0), 
            stage_metrics.get('stage3_processed', 0)
        ])
        
        if total > 0:
            self._stats['multistage_performance']['stage_load_distribution'] = [
                stage_metrics.get('stage1_processed', 0) / total,
                stage_metrics.get('stage2_processed', 0) / total,
                stage_metrics.get('stage3_processed', 0) / total
            ]
    
    def _update_performance_metrics(self, elapsed_ms: float) -> None:
        """Update performance metrics."""
        # Update average latency (exponential moving average)
        alpha = 0.1  # Smoothing factor
        current_avg = self._stats['multistage_performance']['avg_latency_ms']
        self._stats['multistage_performance']['avg_latency_ms'] = (
            alpha * elapsed_ms + (1 - alpha) * current_avg
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        base_stats = super().get_statistics()
        
        if self.use_multistage:
            enhanced_stats = {
                'multistage_enabled': True,
                'multistage_detections': self._stats['multistage_detections'],
                'stage_statistics': {
                    'stage1_filtered': self._stats['stage1_filtered'],
                    'stage2_filtered': self._stats['stage2_filtered'], 
                    'stage3_processed': self._stats['stage3_processed']
                },
                'vehicle_context': {
                    'detected_type': self.detected_vehicle_type,
                    'vehicle_detections': self._stats['vehicle_detections'],
                    'calibration_enabled': self.enable_vehicle_calibration
                },
                'performance_metrics': self._stats['multistage_performance'],
                'model_info': {
                    'models_directory': str(self.models_dir),
                    'max_stage3_load': self.max_stage3_load,
                    'adaptive_gating': True,
                    'load_shedding': True
                }
            }
            
            base_stats.update(enhanced_stats)
        else:
            base_stats['multistage_enabled'] = False
            base_stats['fallback_reason'] = 'Multi-stage components not available'
        
        return base_stats
    
    def print_enhanced_statistics(self) -> None:
        """Print comprehensive statistics including multi-stage metrics."""
        stats = self.get_performance_stats()
        
        print("\n" + "="*60)
        print("ENHANCED ML DETECTOR STATISTICS")
        print("="*60)
        
        print(f"Messages Analyzed: {stats['messages_analyzed']}")
        print(f"Anomalies Detected: {stats['anomalies_detected']}")
        print(f"Model Loaded: {stats['model_loaded']}")
        print(f"Multi-stage Enabled: {stats.get('multistage_enabled', False)}")
        
        if stats.get('multistage_enabled'):
            print(f"\nMULTI-STAGE PERFORMANCE:")
            print(f"  Multi-stage Detections: {stats['multistage_detections']}")
            print(f"  Average Latency: {stats['performance_metrics']['avg_latency_ms']:.3f}ms")
            
            print(f"\nSTAGE PROCESSING:")
            print(f"  Stage 1 Filtered: {stats['stage_statistics']['stage1_filtered']}")
            print(f"  Stage 2 Filtered: {stats['stage_statistics']['stage2_filtered']}")
            print(f"  Stage 3 Processed: {stats['stage_statistics']['stage3_processed']}")
            
            load_dist = stats['performance_metrics']['stage_load_distribution']
            print(f"\nLOAD DISTRIBUTION:")
            print(f"  Stage 1: {load_dist[0]:.1%}")
            print(f"  Stage 2: {load_dist[1]:.1%}")
            print(f"  Stage 3: {load_dist[2]:.1%}")
            
            if stats['vehicle_context']['calibration_enabled']:
                print(f"\nVEHICLE CONTEXT:")
                print(f"  Detected Type: {stats['vehicle_context']['detected_type']}")
                print(f"  Vehicle Detections: {stats['vehicle_context']['vehicle_detections']}")
        
        print("="*60)


# Factory function for creating enhanced detector
def create_enhanced_ml_detector(config: Dict[str, Any]) -> EnhancedMLDetector:
    """
    Factory function to create enhanced ML detector from configuration.
    
    Parameters:
    -----------
    config : Dict[str, Any]
        Configuration dictionary
        
    Returns:
    --------
    EnhancedMLDetector
        Configured enhanced ML detector
    """
    ml_config = config.get('ml_detection', {})
    multistage_config = ml_config.get('multistage', {})
    
    return EnhancedMLDetector(
        use_multistage=ml_config.get('enable_multistage', True),
        models_dir=multistage_config.get('models_dir', 'models'),
        max_stage3_load=multistage_config.get('max_stage3_load', 0.15),
        enable_vehicle_calibration=multistage_config.get('enable_vehicle_calibration', False),
        contamination=ml_config.get('contamination', 0.02),
        feature_window=ml_config.get('feature_window', 100)
    )