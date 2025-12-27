#!/usr/bin/env python3
"""
Comprehensive CAN-IDS Testing Framework for Academic Research

Monitors and logs:
- CPU utilization (overall and per-core)
- Memory usage (RSS, VSZ)
- Temperature (Raspberry Pi specific)
- Processing throughput (messages/second)
- Dropped frames
- Detection latency
- Alert statistics
- System load

Outputs detailed CSV logs suitable for academic analysis.
"""

import argparse
import time
import psutil
import os
import sys
import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import subprocess
import threading
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.capture.pcap_reader import PCAPReader
    from src.detection.rule_engine import RuleEngine
    from src.detection.ml_detector import MLDetector
    from src.preprocessing.feature_extractor import FeatureExtractor
    from src.alerts.alert_manager import AlertManager
except ImportError as e:
    print(f"Error importing CAN-IDS modules: {e}")
    sys.exit(1)


class SystemMonitor:
    """Monitor system resources during testing."""
    
    def __init__(self, output_dir: str, sample_interval: float = 1.0):
        """
        Initialize system monitor.
        
        Args:
            output_dir: Directory to save monitoring data
            sample_interval: Seconds between samples
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_interval = sample_interval
        self.monitoring = False
        self.monitor_thread = None
        
        # Data storage
        self.samples = []
        self.process = psutil.Process()
        
        # Output files
        self.metrics_file = self.output_dir / "system_metrics.csv"
        self.summary_file = self.output_dir / "test_summary.json"
        
    def get_temperature(self) -> float:
        """Get CPU temperature (Raspberry Pi)."""
        try:
            result = subprocess.run(
                ['vcgencmd', 'measure_temp'],
                capture_output=True,
                text=True,
                timeout=1
            )
            if result.returncode == 0:
                # Output format: temp=42.8'C
                temp_str = result.stdout.strip().split('=')[1].split("'")[0]
                return float(temp_str)
        except Exception:
            pass
        return 0.0
    
    def get_throttle_status(self) -> str:
        """Get thermal throttling status (Raspberry Pi)."""
        try:
            result = subprocess.run(
                ['vcgencmd', 'get_throttled'],
                capture_output=True,
                text=True,
                timeout=1
            )
            if result.returncode == 0:
                # Output format: throttled=0x0
                return result.stdout.strip().split('=')[1]
        except Exception:
            pass
        return "unknown"
    
    def collect_sample(self) -> Dict[str, Any]:
        """Collect a single monitoring sample."""
        cpu_count = psutil.cpu_count()
        sample = {
            'timestamp': time.time(),
            'time_str': datetime.now().isoformat(),
            
            # CPU metrics (normalized to 0-100% across all cores)
            'cpu_percent': self.process.cpu_percent(interval=0.1) / cpu_count,
            'cpu_percent_system': psutil.cpu_percent(interval=0.1),
            'cpu_count': cpu_count,
            
            # Memory metrics
            'memory_rss_mb': self.process.memory_info().rss / 1024 / 1024,
            'memory_vms_mb': self.process.memory_info().vms / 1024 / 1024,
            'memory_percent': self.process.memory_percent(),
            'memory_available_mb': psutil.virtual_memory().available / 1024 / 1024,
            'memory_total_mb': psutil.virtual_memory().total / 1024 / 1024,
            
            # System load
            'load_1min': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0,
            'load_5min': psutil.getloadavg()[1] if hasattr(psutil, 'getloadavg') else 0,
            'load_15min': psutil.getloadavg()[2] if hasattr(psutil, 'getloadavg') else 0,
            
            # Temperature (RPi specific)
            'temperature_c': self.get_temperature(),
            'throttle_status': self.get_throttle_status(),
            
            # Thread/process info
            'num_threads': self.process.num_threads(),
            'num_fds': self.process.num_fds() if hasattr(self.process, 'num_fds') else 0,
        }
        
        # Per-core CPU usage
        cpu_percents = psutil.cpu_percent(interval=0.1, percpu=True)
        for i, cpu_pct in enumerate(cpu_percents):
            sample[f'cpu_core_{i}_percent'] = cpu_pct
        
        return sample
    
    def monitor_loop(self):
        """Background monitoring loop."""
        # Write CSV header
        with open(self.metrics_file, 'w', newline='') as f:
            writer = None
            
            while self.monitoring:
                sample = self.collect_sample()
                self.samples.append(sample)
                
                # Write to CSV
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=sample.keys())
                    writer.writeheader()
                
                writer.writerow(sample)
                f.flush()
                
                time.sleep(self.sample_interval)
    
    def start(self):
        """Start monitoring in background thread."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()
        print(f"System monitoring started (sampling every {self.sample_interval}s)")
    
    def stop(self) -> Dict[str, Any]:
        """Stop monitoring and return summary statistics."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        if not self.samples:
            return {}
        
        # Calculate statistics
        summary = {
            'total_samples': len(self.samples),
            'duration_seconds': self.samples[-1]['timestamp'] - self.samples[0]['timestamp'],
            'cpu_percent': {
                'mean': sum(s['cpu_percent'] for s in self.samples) / len(self.samples),
                'max': max(s['cpu_percent'] for s in self.samples),
                'min': min(s['cpu_percent'] for s in self.samples),
            },
            'cpu_percent_system': {
                'mean': sum(s['cpu_percent_system'] for s in self.samples) / len(self.samples),
                'max': max(s['cpu_percent_system'] for s in self.samples),
                'min': min(s['cpu_percent_system'] for s in self.samples),
            },
            'memory_rss_mb': {
                'mean': sum(s['memory_rss_mb'] for s in self.samples) / len(self.samples),
                'max': max(s['memory_rss_mb'] for s in self.samples),
                'min': min(s['memory_rss_mb'] for s in self.samples),
            },
            'temperature_c': {
                'mean': sum(s['temperature_c'] for s in self.samples) / len(self.samples),
                'max': max(s['temperature_c'] for s in self.samples),
                'min': min(s['temperature_c'] for s in self.samples),
            },
            'throttling_occurred': any(s['throttle_status'] != '0x0' for s in self.samples),
        }
        
        print(f"\nSystem monitoring stopped ({len(self.samples)} samples collected)")
        print(f"  CPU Usage: {summary['cpu_percent']['mean']:.1f}% avg, {summary['cpu_percent']['max']:.1f}% peak")
        print(f"  Memory: {summary['memory_rss_mb']['mean']:.1f} MB avg, {summary['memory_rss_mb']['max']:.1f} MB peak")
        print(f"  Temperature: {summary['temperature_c']['mean']:.1f}°C avg, {summary['temperature_c']['max']:.1f}°C peak")
        
        return summary


class PerformanceTracker:
    """Track CAN-IDS performance metrics."""
    
    def __init__(self, output_dir: str):
        """Initialize performance tracker."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics
        self.start_time = None
        self.end_time = None
        self.messages_processed = 0
        self.messages_dropped = 0
        self.alerts_generated = 0
        self.processing_times = []
        
        # Alert breakdown
        self.alerts_by_severity = defaultdict(int)
        self.alerts_by_type = defaultdict(int)
        
        # Detection accuracy metrics
        self.true_positives = 0   # Attack detected correctly
        self.false_positives = 0  # Normal traffic flagged as attack
        self.true_negatives = 0   # Normal traffic passed correctly
        self.false_negatives = 0  # Attack missed
        
        # Output file
        self.performance_file = self.output_dir / "performance_metrics.json"
    
    def start(self):
        """Start tracking."""
        self.start_time = time.time()
    
    def record_message(self, processing_time: float, is_attack: bool = False, alerts_triggered: int = 0):
        """Record a processed message."""
        self.messages_processed += 1
        self.processing_times.append(processing_time)
        
        # Update detection accuracy
        if is_attack:
            if alerts_triggered > 0:
                self.true_positives += 1  # Correctly detected attack
            else:
                self.false_negatives += 1  # Missed attack
        else:
            if alerts_triggered > 0:
                self.false_positives += 1  # False alarm on normal traffic
            else:
                self.true_negatives += 1  # Correctly identified as normal
    
    def record_dropped(self):
        """Record a dropped message."""
        self.messages_dropped += 1
    
    def record_alert(self, alert: Dict[str, Any]):
        """Record an alert."""
        self.alerts_generated += 1
        severity = alert.get('severity', 'UNKNOWN')
        alert_type = alert.get('rule_name', 'UNKNOWN')
        self.alerts_by_severity[severity] += 1
        self.alerts_by_type[alert_type] += 1
    
    def stop(self) -> Dict[str, Any]:
        """Stop tracking and return summary."""
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        if not self.processing_times:
            return {}
        
        # Calculate statistics
        self.processing_times.sort()
        n = len(self.processing_times)
        
        # Calculate detection accuracy metrics
        total_predictions = self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
        precision = self.true_positives / max(self.true_positives + self.false_positives, 1)
        recall = self.true_positives / max(self.true_positives + self.false_negatives, 1)
        f1_score = 2 * (precision * recall) / max(precision + recall, 0.001)
        accuracy = (self.true_positives + self.true_negatives) / max(total_predictions, 1)
        
        summary = {
            'duration_seconds': duration,
            'messages_processed': self.messages_processed,
            'messages_dropped': self.messages_dropped,
            'drop_rate_percent': (self.messages_dropped / max(self.messages_processed, 1)) * 100,
            'throughput_msg_per_sec': self.messages_processed / duration if duration > 0 else 0,
            'alerts_generated': self.alerts_generated,
            'alert_rate_percent': (self.alerts_generated / max(self.messages_processed, 1)) * 100,
            'alerts_by_severity': dict(self.alerts_by_severity),
            'alerts_by_type': dict(self.alerts_by_type),
            'detection_accuracy': {
                'true_positives': self.true_positives,
                'false_positives': self.false_positives,
                'true_negatives': self.true_negatives,
                'false_negatives': self.false_negatives,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'accuracy': accuracy,
            },
            'latency_ms': {
                'mean': sum(self.processing_times) / n * 1000,
                'median': self.processing_times[n // 2] * 1000,
                'min': min(self.processing_times) * 1000,
                'max': max(self.processing_times) * 1000,
                'p95': self.processing_times[int(n * 0.95)] * 1000 if n > 20 else 0,
                'p99': self.processing_times[int(n * 0.99)] * 1000 if n > 100 else 0,
            }
        }
        
        # Save to file
        with open(self.performance_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nPerformance Summary:")
        print(f"  Messages: {self.messages_processed:,} processed, {self.messages_dropped} dropped ({summary['drop_rate_percent']:.2f}%)")
        print(f"  Throughput: {summary['throughput_msg_per_sec']:.2f} msg/s")
        print(f"  Latency: {summary['latency_ms']['mean']:.3f} ms avg, {summary['latency_ms']['p95']:.3f} ms p95")
        print(f"  Alerts: {self.alerts_generated} generated ({summary['alert_rate_percent']:.2f}%)")
        print(f"\nDetection Accuracy:")
        print(f"  Precision: {precision*100:.2f}% | Recall: {recall*100:.2f}% | F1-Score: {f1_score:.3f}")
        print(f"  Accuracy: {accuracy*100:.2f}%")
        print(f"  TP: {self.true_positives:,} | FP: {self.false_positives:,} | TN: {self.true_negatives:,} | FN: {self.false_negatives:,}")
        
        return summary


def run_comprehensive_test(data_file: str, output_dir: str, config: Dict[str, Any]):
    """
    Run comprehensive test with full monitoring.
    
    Args:
        data_file: Path to CAN data file (CSV or PCAP)
        output_dir: Directory for output files
        config: Test configuration
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("CAN-IDS Comprehensive Performance Test")
    print("=" * 70)
    print(f"Data file: {data_file}")
    print(f"Output directory: {output_dir}")
    print(f"Configuration: {config}")
    print("=" * 70)
    
    # Initialize monitors
    system_monitor = SystemMonitor(output_dir, sample_interval=config.get('sample_interval', 1.0))
    performance_tracker = PerformanceTracker(output_dir)
    
    # Initialize IDS components
    print("\nInitializing CAN-IDS components...")
    rule_engine = RuleEngine(config.get('rules_file', 'config/rules.yaml'))
    ml_detector = None
    if config.get('enable_ml', False):
        # Load model path from config if available
        ml_config = config.get('ml_model', {})
        model_path = ml_config.get('path', 'data/models/aggressive_load_shedding.joblib')
        contamination = ml_config.get('contamination', 0.20)
        print(f"Loading ML model: {model_path} (contamination={contamination})")
        ml_detector = MLDetector(model_path=model_path, contamination=contamination)
    feature_extractor = FeatureExtractor()
    
    # Start monitoring
    system_monitor.start()
    performance_tracker.start()
    
    # Process data
    print(f"\nProcessing data from: {data_file}")
    
    try:
        # Read messages from file
        messages = []
        data_path = Path(data_file)
        
        if data_path.suffix.lower() == '.csv':
            # Read CSV file
            print("Reading CSV file...")
            with open(data_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert CSV row to message format
                    # Handle arbitration_id as hex string
                    arb_id_str = row.get('arbitration_id', row.get('can_id', '0'))
                    can_id = int(arb_id_str, 16) if isinstance(arb_id_str, str) else int(arb_id_str)
                    
                    # Handle data field (could be 'data', 'data_field', or 'Data')
                    data_str = row.get('data_field', row.get('data', row.get('Data', '00' * 8)))
                    data_str = data_str.replace(' ', '').replace('0x', '')
                    
                    # Pad or truncate to 8 bytes (16 hex chars)
                    if len(data_str) < 16:
                        data_str = data_str + '00' * (8 - len(data_str) // 2)
                    data_bytes = bytes.fromhex(data_str[:16])
                    
                    # Calculate DLC from actual data length
                    dlc = len(data_bytes)
                    
                    # Check if this is an attack (ground truth from CSV)
                    is_attack = int(row.get('attack', row.get('Attack', 0))) == 1
                    
                    msg = {
                        'timestamp': float(row.get('timestamp', time.time())),
                        'can_id': can_id,
                        'dlc': dlc,
                        'data': data_bytes,
                        'is_attack': is_attack,
                    }
                    messages.append(msg)
        
        print(f"Loaded {len(messages)} messages")
        
        # Process each message
        for i, msg in enumerate(messages):
            msg_start = time.time()
            
            try:
                # Get ground truth
                is_attack = msg.get('is_attack', False)
                alerts_count = 0
                
                # Rule-based detection
                rule_alerts = rule_engine.analyze_message(msg)
                alerts_count += len(rule_alerts)
                for alert in rule_alerts:
                    # Convert Alert object to dict if needed
                    if hasattr(alert, '__dict__'):
                        alert_dict = alert.__dict__ if not hasattr(alert, 'to_dict') else alert.to_dict()
                    else:
                        alert_dict = alert
                    performance_tracker.record_alert(alert_dict)
                
                # ML-based detection (if enabled)
                if ml_detector:
                    ml_alert = ml_detector.analyze_message(msg)
                    if ml_alert:
                        alerts_count += 1
                        # Convert Alert object to dict if needed
                        if hasattr(ml_alert, '__dict__'):
                            alert_dict = ml_alert.__dict__ if not hasattr(ml_alert, 'to_dict') else ml_alert.to_dict()
                        else:
                            alert_dict = ml_alert
                        performance_tracker.record_alert(alert_dict)
                
                msg_time = time.time() - msg_start
                performance_tracker.record_message(msg_time, is_attack=is_attack, alerts_triggered=alerts_count)
                
            except Exception as e:
                #print(f"Error processing message {i}: {e}")
                performance_tracker.record_dropped()
            
            # Progress indicator
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1:,} / {len(messages):,} messages...")
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Stop monitoring
        print("\nStopping monitors...")
        perf_summary = performance_tracker.stop()
        sys_summary = system_monitor.stop()
        
        # Save combined summary
        combined_summary = {
            'test_info': {
                'data_file': str(data_file),
                'test_date': datetime.now().isoformat(),
                'config': config,
            },
            'performance': perf_summary,
            'system': sys_summary,
        }
        
        summary_file = output_path / "comprehensive_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(combined_summary, f, indent=2)
        
        print(f"\nResults saved to: {output_dir}")
        print(f"  - system_metrics.csv: Detailed system metrics")
        print(f"  - performance_metrics.json: Performance statistics")
        print(f"  - comprehensive_summary.json: Combined summary")
        
        return combined_summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Comprehensive CAN-IDS testing framework for academic research'
    )
    
    parser.add_argument('data_file', help='CAN data file (CSV or PCAP)')
    parser.add_argument('--output', '-o', default='test_results',
                       help='Output directory for results (default: test_results)')
    parser.add_argument('--sample-interval', type=float, default=1.0,
                       help='System monitoring sample interval in seconds (default: 1.0)')
    parser.add_argument('--enable-ml', action='store_true',
                       help='Enable ML-based detection')
    parser.add_argument('--rules', default='config/rules.yaml',
                       help='Rules file path (default: config/rules.yaml)')
    
    args = parser.parse_args()
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output) / timestamp
    
    config = {
        'sample_interval': args.sample_interval,
        'enable_ml': args.enable_ml,
        'rules_file': args.rules,
    }
    
    run_comprehensive_test(args.data_file, str(output_dir), config)


if __name__ == '__main__':
    main()
