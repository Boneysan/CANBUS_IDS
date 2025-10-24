#!/usr/bin/env python3
"""
CAN-IDS: Controller Area Network Intrusion Detection System

Main entry point for the CAN-IDS application.
Provides command-line interface for real-time monitoring and PCAP analysis.
"""

import argparse
import logging
import signal
import sys
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    logger.error("PyYAML required but not installed. Install with: pip install PyYAML")
    sys.exit(1)

# Import CAN-IDS modules
try:
    from src.capture.can_sniffer import CANSniffer
    from src.capture.pcap_reader import PCAPReader, CANDumpReader
    from src.detection.rule_engine import RuleEngine
    from src.detection.ml_detector import MLDetector
    from src.preprocessing.feature_extractor import FeatureExtractor
    from src.preprocessing.normalizer import Normalizer
    from src.alerts.alert_manager import AlertManager
    from src.alerts.notifiers import create_notifiers
except ImportError as e:
    logger.error(f"Error importing CAN-IDS modules: {e}")
    logger.error("Make sure you're running from the project root directory")
    sys.exit(1)


class CANIDSApplication:
    """
    Main CAN-IDS application class.
    
    Coordinates all components for real-time intrusion detection.
    """
    
    def __init__(self, config_file: str):
        """
        Initialize CAN-IDS application.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = Path(config_file)
        self.config = {}
        self.running = False
        
        # Core components
        self.can_sniffer: Optional[CANSniffer] = None
        self.pcap_reader: Optional[PCAPReader] = None
        self.rule_engine: Optional[RuleEngine] = None
        self.ml_detector: Optional[MLDetector] = None
        self.feature_extractor: Optional[FeatureExtractor] = None
        self.normalizer: Optional[Normalizer] = None
        self.alert_manager: Optional[AlertManager] = None
        
        # Statistics
        self.stats = {
            'messages_processed': 0,
            'alerts_generated': 0,
            'start_time': None,
            'processing_errors': 0
        }
        
        # Load configuration
        self.load_config()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            logger.info(f"Loading configuration from {self.config_file}")
            
            if not self.config_file.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
                
            with open(self.config_file, 'r') as f:
                self.config = yaml.safe_load(f)
                
            logger.info("Configuration loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
            
    def initialize_components(self) -> None:
        """Initialize all CAN-IDS components."""
        logger.info("Initializing CAN-IDS components...")
        
        try:
            # Initialize alert manager first
            alert_config = self.config.get('alerts', {})
            self.alert_manager = AlertManager(alert_config)
            
            # Create and add notifiers
            notifiers = create_notifiers(alert_config)
            for notifier in notifiers:
                self.alert_manager.add_notifier(notifier)
                
            # Initialize rule engine
            rules_file = self.config.get('rules_file', 'config/rules.yaml')
            if Path(rules_file).exists():
                self.rule_engine = RuleEngine(rules_file)
                logger.info(f"Rule engine initialized with {len(self.rule_engine.rules)} rules")
            else:
                logger.warning(f"Rules file not found: {rules_file}")
                
            # Initialize ML detector if enabled
            detection_modes = self.config.get('detection_modes', ['rule_based'])
            
            if 'ml_based' in detection_modes:
                ml_config = self.config.get('ml_model', {})
                model_path = ml_config.get('path')
                
                self.ml_detector = MLDetector(model_path)
                
                if model_path and Path(model_path).exists():
                    self.ml_detector.load_model()
                    logger.info("ML detector initialized with trained model")
                else:
                    logger.warning("ML detector initialized without trained model")
                    
            # Initialize feature extractor for ML
            if self.ml_detector:
                self.feature_extractor = FeatureExtractor()
                logger.info("Feature extractor initialized")
                
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
            
    def start_live_monitoring(self, interface: str) -> None:
        """
        Start live CAN traffic monitoring.
        
        Args:
            interface: CAN interface name (e.g., 'can0')
        """
        logger.info(f"Starting live monitoring on interface: {interface}")
        
        try:
            # Initialize CAN sniffer
            capture_config = self.config.get('capture', {})
            buffer_size = capture_config.get('buffer_size', 1000)
            
            self.can_sniffer = CANSniffer(
                interface=interface,
                buffer_size=buffer_size
            )
            
            # Start CAN sniffer
            self.can_sniffer.start()
            
            # Start monitoring loop
            self.running = True
            self.stats['start_time'] = time.time()
            
            logger.info("Live monitoring started - Press Ctrl+C to stop")
            
            # Process messages
            for message in self.can_sniffer.capture_messages():
                if not self.running:
                    break
                    
                self.process_message(message.__dict__ if hasattr(message, '__dict__') else {
                    'timestamp': time.time(),
                    'can_id': message.arbitration_id,
                    'dlc': message.dlc,
                    'data': list(message.data),
                    'data_hex': ' '.join(f"{b:02X}" for b in message.data),
                    'is_extended': message.is_extended_id,
                    'is_remote': message.is_remote_frame,
                    'is_error': message.is_error_frame
                })
                
        except Exception as e:
            logger.error(f"Error in live monitoring: {e}")
            raise
        finally:
            if self.can_sniffer:
                self.can_sniffer.stop()
                
    def analyze_pcap(self, pcap_file: str) -> None:
        """
        Analyze CAN traffic from PCAP file.
        
        Args:
            pcap_file: Path to PCAP file
        """
        logger.info(f"Analyzing PCAP file: {pcap_file}")
        
        try:
            # Initialize PCAP reader
            if pcap_file.endswith('.log'):
                self.pcap_reader = CANDumpReader(pcap_file)
            else:
                self.pcap_reader = PCAPReader(pcap_file)
                
            # Start analysis
            self.running = True
            self.stats['start_time'] = time.time()
            
            message_count = 0
            
            # Process messages from PCAP
            for message in self.pcap_reader.read_messages():
                if not self.running:
                    break
                    
                self.process_message(message)
                message_count += 1
                
                # Progress indicator for large files
                if message_count % 10000 == 0:
                    logger.info(f"Processed {message_count} messages...")
                    
            logger.info(f"PCAP analysis complete. Processed {message_count} messages")
            
        except Exception as e:
            logger.error(f"Error analyzing PCAP: {e}")
            raise
            
    def process_message(self, message: Dict[str, Any]) -> None:
        """
        Process a single CAN message through detection engines.
        
        Args:
            message: CAN message dictionary
        """
        try:
            self.stats['messages_processed'] += 1
            
            # Rule-based detection
            if self.rule_engine:
                rule_alerts = self.rule_engine.analyze_message(message)
                
                for alert in rule_alerts:
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
                    
                    self.alert_manager.process_alert(alert_data)
                    self.stats['alerts_generated'] += 1
                    
            # ML-based detection
            if self.ml_detector and self.feature_extractor:
                # Extract features
                features = self.feature_extractor.extract_features(message)
                
                # Normalize features if normalizer available
                if self.normalizer:
                    features = self.normalizer.transform(features)
                    
                # ML detection
                ml_alert = self.ml_detector.analyze_message(message)
                
                if ml_alert:
                    alert_data = {
                        'timestamp': ml_alert.timestamp,
                        'rule_name': 'ML Anomaly Detection',
                        'severity': 'MEDIUM',  # Default ML severity
                        'description': f"ML anomaly detected (score: {ml_alert.anomaly_score:.3f})",
                        'can_id': ml_alert.can_id,
                        'message_data': ml_alert.message_data,
                        'confidence': ml_alert.confidence,
                        'source': 'ml_detector',
                        'additional_info': {
                            'anomaly_score': ml_alert.anomaly_score,
                            'features': ml_alert.features
                        }
                    }
                    
                    self.alert_manager.process_alert(alert_data)
                    self.stats['alerts_generated'] += 1
                    
        except Exception as e:
            self.stats['processing_errors'] += 1
            logger.warning(f"Error processing message: {e}")
            
    def print_statistics(self) -> None:
        """Print system statistics."""
        if self.stats['start_time']:
            runtime = time.time() - self.stats['start_time']
        else:
            runtime = 0
            
        print("\n" + "="*60)
        print("CAN-IDS STATISTICS")
        print("="*60)
        print(f"Runtime: {runtime:.2f} seconds")
        print(f"Messages processed: {self.stats['messages_processed']}")
        print(f"Alerts generated: {self.stats['alerts_generated']}")
        print(f"Processing errors: {self.stats['processing_errors']}")
        
        if runtime > 0:
            msg_rate = self.stats['messages_processed'] / runtime
            print(f"Processing rate: {msg_rate:.2f} messages/second")
            
        # Component statistics
        if self.rule_engine:
            rule_stats = self.rule_engine.get_statistics()
            print(f"\nRule Engine:")
            print(f"  Rules loaded: {rule_stats['rules_loaded']}")
            print(f"  Rules matched: {rule_stats['rules_matched']}")
            
        if self.ml_detector:
            ml_stats = self.ml_detector.get_statistics()
            print(f"\nML Detector:")
            print(f"  Model loaded: {ml_stats['model_loaded']}")
            print(f"  Anomalies detected: {ml_stats['anomalies_detected']}")
            
        if self.alert_manager:
            alert_stats = self.alert_manager.get_statistics()
            print(f"\nAlert Manager:")
            print(f"  Total alerts: {alert_stats['total_alerts']}")
            print(f"  Alerts by severity: {dict(alert_stats['alerts_by_severity'])}")
            
        print("="*60)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("Shutdown signal received")
        self.shutdown()
        
    def shutdown(self) -> None:
        """Shutdown CAN-IDS application."""
        logger.info("Shutting down CAN-IDS...")
        
        self.running = False
        
        # Print final statistics
        self.print_statistics()
        
        # Shutdown components
        if self.can_sniffer:
            self.can_sniffer.stop()
            
        if self.alert_manager:
            self.alert_manager.shutdown()
            
        logger.info("Shutdown complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='CAN-IDS: Controller Area Network Intrusion Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Monitor live CAN traffic
  python main.py -i can0
  
  # Analyze PCAP file
  python main.py --mode replay --file traffic.pcap
  
  # Use custom configuration
  python main.py -i can0 --config config/can_ids_rpi4.conf
  
  # Enable debug logging
  python main.py -i can0 --log-level DEBUG
"""
    )
    
    # Required arguments group
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('-i', '--interface', 
                           help='CAN interface for live monitoring (e.g., can0)')
    mode_group.add_argument('--mode', choices=['replay'],
                           help='Operation mode')
    
    # Optional arguments
    parser.add_argument('--file', 
                       help='PCAP or candump log file for analysis')
    parser.add_argument('--config', default='config/can_ids.conf',
                       help='Configuration file path (default: config/can_ids.conf)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level (default: INFO)')
    parser.add_argument('--version', action='version', version='CAN-IDS 1.0.0')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == 'replay' and not args.file:
        parser.error("--file required when using --mode replay")
        
    # Set logging level
    log_level = getattr(logging, args.log_level.upper())
    logging.getLogger().setLevel(log_level)
    
    try:
        # Initialize application
        app = CANIDSApplication(args.config)
        app.initialize_components()
        
        # Run based on mode
        if args.interface:
            # Live monitoring mode
            app.start_live_monitoring(args.interface)
        elif args.mode == 'replay':
            # PCAP analysis mode
            app.analyze_pcap(args.file)
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)
    finally:
        if 'app' in locals():
            app.shutdown()


if __name__ == '__main__':
    main()