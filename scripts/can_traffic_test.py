#!/usr/bin/env python3
"""
CAN traffic monitoring and connectivity testing utility.

This script provides functions to test if the CAN-IDS can see traffic
and monitor live CAN bus activity, similar to being plugged into a real network.
"""

import argparse
import time
import sys
import signal
import subprocess
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import can
    CAN_AVAILABLE = True
except ImportError:
    print("Warning: python-can not available. Some features will be limited.")
    CAN_AVAILABLE = False

try:
    from src.capture.can_sniffer import CANSniffer
    from src.detection.rule_engine import RuleEngine
    from src.detection.ml_detector import MLDetector
    CANIDS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: CAN-IDS modules not available: {e}")
    CANIDS_AVAILABLE = False


class CANTrafficMonitor:
    """Monitor and test CAN traffic connectivity."""
    
    def __init__(self, interface: str = 'can0'):
        """
        Initialize CAN traffic monitor.
        
        Args:
            interface: CAN interface name (e.g., 'can0', 'vcan0')
        """
        self.interface = interface
        self.running = False
        self.message_count = 0
        self.unique_ids = set()
        self.start_time = None
        self.last_message_time = None
        
        # Statistics
        self.stats = {
            'total_messages': 0,
            'unique_can_ids': 0,
            'message_rate': 0.0,
            'runtime_seconds': 0.0,
            'last_activity': None,
            'interface_status': 'unknown'
        }
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def test_interface_connectivity(self) -> bool:
        """
        Test if CAN interface exists and is accessible.
        
        Returns:
            True if interface is accessible, False otherwise
        """
        print(f"Testing CAN interface connectivity: {self.interface}")
        
        # Test 1: Check if interface exists in system
        try:
            result = subprocess.run(['ip', 'link', 'show', self.interface], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                print(f"✗ Interface {self.interface} not found in system")
                self.stats['interface_status'] = 'not_found'
                return False
            else:
                print(f"✓ Interface {self.interface} exists in system")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print(f"✗ Cannot check interface status (ip command not available)")
            self.stats['interface_status'] = 'unknown'
            return False
        
        # Test 2: Try to open CAN interface with python-can
        if not CAN_AVAILABLE:
            print("✗ python-can not available, cannot test interface access")
            self.stats['interface_status'] = 'python_can_missing'
            return False
        
        try:
            bus = can.Bus(channel=self.interface, interface='socketcan')
            print(f"✓ Successfully opened {self.interface} with python-can")
            
            # Test 3: Try to receive a message (with timeout)
            print("  Testing message reception (5 second timeout)...")
            message = bus.recv(timeout=5.0)
            
            if message:
                print(f"✓ Received CAN message: ID=0x{message.arbitration_id:03X}, DLC={message.dlc}")
                self.stats['interface_status'] = 'active_traffic'
                bus.shutdown()
                return True
            else:
                print("⚠ No traffic detected (interface may be up but no messages)")
                self.stats['interface_status'] = 'no_traffic'
                bus.shutdown()
                return True  # Interface works, just no traffic
                
        except Exception as e:
            print(f"✗ Cannot open {self.interface}: {e}")
            self.stats['interface_status'] = 'access_denied'
            return False
    
    def monitor_traffic(self, duration: float = 30.0, show_messages: bool = True) -> Dict[str, Any]:
        """
        Monitor CAN traffic for a specified duration.
        
        Args:
            duration: Monitoring duration in seconds
            show_messages: Whether to display individual messages
            
        Returns:
            Dictionary with traffic statistics
        """
        print(f"\nMonitoring CAN traffic on {self.interface} for {duration} seconds...")
        print("Press Ctrl+C to stop early\n")
        
        if not CAN_AVAILABLE:
            print("Error: python-can not available")
            return self.stats
        
        try:
            bus = can.Bus(channel=self.interface, interface='socketcan')
            
            self.running = True
            self.start_time = time.time()
            self.message_count = 0
            self.unique_ids.clear()
            
            end_time = self.start_time + duration
            
            while self.running and time.time() < end_time:
                message = bus.recv(timeout=1.0)
                
                if message:
                    self.message_count += 1
                    self.unique_ids.add(message.arbitration_id)
                    self.last_message_time = time.time()
                    
                    if show_messages:
                        self._display_message(message)
                    
                    # Update stats periodically
                    if self.message_count % 100 == 0:
                        self._update_stats()
                        print(f"\rMessages: {self.message_count}, Rate: {self.stats['message_rate']:.1f} msg/s, IDs: {len(self.unique_ids)}", end='')
            
            self.running = False
            self._update_stats()
            bus.shutdown()
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
            self.running = False
        except Exception as e:
            print(f"Error monitoring traffic: {e}")
            self.running = False
        
        return self.stats
    
    def _display_message(self, message) -> None:
        """Display a CAN message in a readable format."""
        timestamp = time.time()
        data_hex = ' '.join(f'{b:02X}' for b in message.data)
        
        print(f"{timestamp:.6f}  {self.interface}  {message.arbitration_id:03X}  [{message.dlc}]  {data_hex}")
    
    def _update_stats(self) -> None:
        """Update traffic statistics."""
        if self.start_time:
            runtime = time.time() - self.start_time
            self.stats.update({
                'total_messages': self.message_count,
                'unique_can_ids': len(self.unique_ids),
                'message_rate': self.message_count / runtime if runtime > 0 else 0.0,
                'runtime_seconds': runtime,
                'last_activity': self.last_message_time,
                'interface_status': 'active_traffic' if self.message_count > 0 else 'no_traffic'
            })
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print("\nShutdown signal received")
        self.running = False
    
    def test_with_canids(self, duration: float = 10.0) -> Dict[str, Any]:
        """
        Test CAN-IDS detection capabilities with live traffic.
        
        Args:
            duration: Test duration in seconds
            
        Returns:
            Detection test results
        """
        print(f"\nTesting CAN-IDS detection on {self.interface} for {duration} seconds...")
        
        if not CANIDS_AVAILABLE:
            print("Error: CAN-IDS modules not available")
            return {}
        
        results = {
            'messages_processed': 0,
            'alerts_generated': 0,
            'detection_engines': [],
            'performance': {}
        }
        
        try:
            # Initialize CAN-IDS components
            sniffer = CANSniffer(interface=self.interface, buffer_size=500)
            
            # Initialize detection engines if available
            rule_engine = None
            ml_detector = None
            
            try:
                rule_engine = RuleEngine('config/rules.yaml')
                results['detection_engines'].append('rule_engine')
                print("✓ Rule engine loaded")
            except Exception as e:
                print(f"⚠ Rule engine not available: {e}")
            
            try:
                ml_detector = MLDetector()
                results['detection_engines'].append('ml_detector')
                print("✓ ML detector initialized")
            except Exception as e:
                print(f"⚠ ML detector not available: {e}")
            
            # Start monitoring
            sniffer.start()
            start_time = time.time()
            
            print("Processing messages through CAN-IDS detection engines...")
            
            for message in sniffer.capture_messages():
                if time.time() - start_time > duration:
                    break
                
                # Convert to CAN-IDS format
                canids_message = {
                    'timestamp': time.time(),
                    'can_id': message.arbitration_id,
                    'dlc': message.dlc,
                    'data': list(message.data),
                    'data_hex': ' '.join(f'{b:02X}' for b in message.data),
                    'is_extended': message.is_extended_id,
                    'is_remote': message.is_remote_frame,
                    'is_error': message.is_error_frame
                }
                
                results['messages_processed'] += 1
                
                # Test rule engine
                if rule_engine:
                    try:
                        alerts = rule_engine.analyze_message(canids_message)
                        results['alerts_generated'] += len(alerts)
                        
                        for alert in alerts:
                            print(f"🚨 ALERT: {alert.rule_name} - {alert.description}")
                    except Exception as e:
                        print(f"Rule engine error: {e}")
                
                # Test ML detector
                if ml_detector:
                    try:
                        ml_alert = ml_detector.analyze_message(canids_message)
                        if ml_alert:
                            results['alerts_generated'] += 1
                            print(f"🤖 ML ALERT: Anomaly score {ml_alert.anomaly_score:.3f}")
                    except Exception as e:
                        print(f"ML detector error: {e}")
                
                # Show progress
                if results['messages_processed'] % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = results['messages_processed'] / elapsed
                    print(f"Processed: {results['messages_processed']}, Rate: {rate:.1f} msg/s, Alerts: {results['alerts_generated']}")
            
            # Calculate performance metrics
            total_time = time.time() - start_time
            results['performance'] = {
                'processing_rate': results['messages_processed'] / total_time if total_time > 0 else 0,
                'alert_rate': results['alerts_generated'] / results['messages_processed'] if results['messages_processed'] > 0 else 0,
                'total_duration': total_time
            }
            
            sniffer.stop()
            
        except Exception as e:
            print(f"Error testing CAN-IDS: {e}")
        
        return results
    
    def generate_test_traffic(self, interface: str = None, count: int = 100) -> bool:
        """
        Generate test CAN traffic (requires cansend utility).
        
        Args:
            interface: CAN interface to send on (defaults to self.interface)
            count: Number of test messages to send
            
        Returns:
            True if successful, False otherwise
        """
        target_interface = interface or self.interface
        
        print(f"Generating {count} test CAN messages on {target_interface}...")
        
        test_messages = [
            "123#DEADBEEF",
            "456#12345678",
            "789#CAFEBABE",
            "ABC#11223344",
            "DEF#AABBCCDD"
        ]
        
        try:
            for i in range(count):
                message = test_messages[i % len(test_messages)]
                
                # Modify message to make it unique
                message_id = f"{0x100 + (i % 256):03X}"
                data = f"{i & 0xFF:02X}{(i >> 8) & 0xFF:02X}{(i >> 16) & 0xFF:02X}{(i >> 24) & 0xFF:02X}"
                unique_message = f"{message_id}#{data}"
                
                result = subprocess.run(['cansend', target_interface, unique_message], 
                                      capture_output=True, text=True, timeout=1)
                
                if result.returncode != 0:
                    print(f"Failed to send message {i+1}: {result.stderr}")
                    return False
                
                if i % 20 == 0:
                    print(f"Sent {i+1}/{count} messages...")
                
                time.sleep(0.01)  # 10ms between messages
            
            print(f"✓ Successfully sent {count} test messages")
            return True
            
        except FileNotFoundError:
            print("Error: cansend utility not found. Install can-utils package.")
            return False
        except Exception as e:
            print(f"Error generating test traffic: {e}")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='CAN traffic monitoring and connectivity testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test interface connectivity
  python scripts/can_traffic_test.py --test-connectivity
  
  # Monitor traffic for 30 seconds
  python scripts/can_traffic_test.py --monitor --duration 30
  
  # Test CAN-IDS detection
  python scripts/can_traffic_test.py --test-canids --duration 15
  
  # Generate test traffic and monitor
  python scripts/can_traffic_test.py --generate-traffic --monitor --duration 10
"""
    )
    
    parser.add_argument('-i', '--interface', default='can0',
                       help='CAN interface (default: can0)')
    parser.add_argument('--test-connectivity', action='store_true',
                       help='Test if interface is accessible')
    parser.add_argument('--monitor', action='store_true',
                       help='Monitor live CAN traffic')
    parser.add_argument('--test-canids', action='store_true',
                       help='Test CAN-IDS detection capabilities')
    parser.add_argument('--generate-traffic', action='store_true',
                       help='Generate test CAN traffic')
    parser.add_argument('--duration', type=float, default=30.0,
                       help='Monitoring/testing duration in seconds (default: 30)')
    parser.add_argument('--count', type=int, default=100,
                       help='Number of test messages to generate (default: 100)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress individual message display')
    parser.add_argument('--save-results', 
                       help='Save results to JSON file')
    
    args = parser.parse_args()
    
    print("CAN Traffic Monitor and Connectivity Test")
    print("=" * 60)
    
    monitor = CANTrafficMonitor(args.interface)
    results = {}
    
    # Test connectivity
    if args.test_connectivity or not any([args.monitor, args.test_canids, args.generate_traffic]):
        connectivity_ok = monitor.test_interface_connectivity()
        results['connectivity'] = {
            'interface': args.interface,
            'accessible': connectivity_ok,
            'status': monitor.stats['interface_status']
        }
        
        if not connectivity_ok:
            print(f"\n⚠ Interface {args.interface} is not accessible or has no traffic")
            print("Try:")
            print(f"  1. Check if interface exists: ip link show {args.interface}")
            print(f"  2. Setup virtual CAN: sudo python scripts/setup_vcan.py")
            print(f"  3. Generate test traffic: cansend {args.interface} 123#DEADBEEF")
    
    # Generate test traffic
    if args.generate_traffic:
        print(f"\n{'-'*60}")
        success = monitor.generate_test_traffic(count=args.count)
        results['traffic_generation'] = {
            'success': success,
            'message_count': args.count
        }
    
    # Monitor traffic
    if args.monitor:
        print(f"\n{'-'*60}")
        traffic_stats = monitor.monitor_traffic(
            duration=args.duration, 
            show_messages=not args.quiet
        )
        results['traffic_monitoring'] = traffic_stats
    
    # Test CAN-IDS
    if args.test_canids:
        print(f"\n{'-'*60}")
        canids_results = monitor.test_with_canids(duration=args.duration)
        results['canids_testing'] = canids_results
    
    # Display summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    if 'connectivity' in results:
        status = results['connectivity']['status']
        print(f"Interface Status: {status}")
    
    if 'traffic_monitoring' in results:
        stats = results['traffic_monitoring']
        print(f"Messages Received: {stats['total_messages']}")
        print(f"Unique CAN IDs: {stats['unique_can_ids']}")
        print(f"Message Rate: {stats['message_rate']:.1f} msg/s")
        print(f"Runtime: {stats['runtime_seconds']:.1f} seconds")
    
    if 'canids_testing' in results:
        canids = results['canids_testing']
        print(f"CAN-IDS Messages Processed: {canids.get('messages_processed', 0)}")
        print(f"Alerts Generated: {canids.get('alerts_generated', 0)}")
        print(f"Detection Engines: {', '.join(canids.get('detection_engines', []))}")
        if 'performance' in canids:
            perf = canids['performance']
            print(f"Processing Rate: {perf.get('processing_rate', 0):.1f} msg/s")
            print(f"Alert Rate: {perf.get('alert_rate', 0):.3f}")
    
    # Save results
    if args.save_results:
        results['timestamp'] = time.time()
        results['interface'] = args.interface
        results['duration'] = args.duration
        
        with open(args.save_results, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {args.save_results}")


if __name__ == '__main__':
    main()