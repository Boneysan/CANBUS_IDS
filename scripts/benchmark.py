#!/usr/bin/env python3
"""
Benchmark CAN-IDS performance and resource usage.

Measures throughput, latency, CPU/memory usage, and detection accuracy.
"""

import argparse
import time
import psutil
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys
import statistics

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.detection.rule_engine import RuleEngine
    from src.detection.ml_detector import MLDetector
    from src.preprocessing.feature_extractor import FeatureExtractor
    from src.alerts.alert_manager import AlertManager
    from src.alerts.notifiers import create_notifiers
except ImportError as e:
    print(f"Error importing CAN-IDS modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


class CANIDSBenchmark:
    """Benchmark CAN-IDS system performance."""
    
    def __init__(self, config_file: str = 'config/can_ids.yaml'):
        """
        Initialize benchmark.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file
        self.results = {}
        
    def benchmark_rule_engine(self, messages: List[Dict[str, Any]], 
                             rules_file: str = 'config/rules.yaml') -> Dict[str, Any]:
        """
        Benchmark rule engine performance.
        
        Args:
            messages: List of CAN messages to process
            rules_file: Path to rules file
            
        Returns:
            Dictionary of benchmark results
        """
        print(f"\nBenchmarking Rule Engine...")
        print(f"  Messages: {len(messages)}")
        
        # Initialize rule engine
        rule_engine = RuleEngine(rules_file)
        
        # Warm-up
        for msg in messages[:100]:
            rule_engine.analyze_message(msg)
        
        # Benchmark
        process = psutil.Process()
        cpu_before = process.cpu_percent()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        alerts_generated = 0
        processing_times = []
        
        for msg in messages:
            msg_start = time.time()
            alerts = rule_engine.analyze_message(msg)
            msg_time = time.time() - msg_start
            processing_times.append(msg_time * 1000)  # Convert to ms
            alerts_generated += len(alerts)
        
        end_time = time.time()
        duration = end_time - start_time
        
        cpu_after = process.cpu_percent()
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        
        # Calculate metrics
        throughput = len(messages) / duration
        avg_latency = statistics.mean(processing_times)
        p95_latency = statistics.quantiles(processing_times, n=20)[18]  # 95th percentile
        p99_latency = statistics.quantiles(processing_times, n=100)[98]  # 99th percentile
        
        results = {
            'component': 'Rule Engine',
            'messages_processed': len(messages),
            'duration_seconds': duration,
            'throughput_msg_per_sec': throughput,
            'alerts_generated': alerts_generated,
            'alert_rate': alerts_generated / len(messages),
            'latency_ms': {
                'mean': avg_latency,
                'min': min(processing_times),
                'max': max(processing_times),
                'p95': p95_latency,
                'p99': p99_latency,
                'stddev': statistics.stdev(processing_times)
            },
            'resource_usage': {
                'cpu_percent': cpu_after - cpu_before,
                'memory_mb': mem_after - mem_before,
                'memory_total_mb': mem_after
            }
        }
        
        self._print_results(results)
        return results
        
    def benchmark_ml_detector(self, messages: List[Dict[str, Any]],
                             model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Benchmark ML detector performance.
        
        Args:
            messages: List of CAN messages to process
            model_path: Path to trained model (optional)
            
        Returns:
            Dictionary of benchmark results
        """
        print(f"\nBenchmarking ML Detector...")
        print(f"  Messages: {len(messages)}")
        
        # Initialize ML detector
        ml_detector = MLDetector(model_path=model_path)
        feature_extractor = FeatureExtractor()
        
        # Train if no model provided
        if not model_path:
            print("  Training model (no pre-trained model provided)...")
            features = [feature_extractor.extract_features(msg) for msg in messages[:1000]]
            ml_detector.train(features)
        
        # Warm-up
        for msg in messages[:100]:
            ml_detector.analyze_message(msg)
        
        # Benchmark
        process = psutil.Process()
        cpu_before = process.cpu_percent()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        anomalies_detected = 0
        processing_times = []
        
        for msg in messages:
            msg_start = time.time()
            alert = ml_detector.analyze_message(msg)
            msg_time = time.time() - msg_start
            processing_times.append(msg_time * 1000)  # Convert to ms
            if alert:
                anomalies_detected += 1
        
        end_time = time.time()
        duration = end_time - start_time
        
        cpu_after = process.cpu_percent()
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        
        # Calculate metrics
        throughput = len(messages) / duration
        avg_latency = statistics.mean(processing_times)
        p95_latency = statistics.quantiles(processing_times, n=20)[18]
        p99_latency = statistics.quantiles(processing_times, n=100)[98]
        
        results = {
            'component': 'ML Detector',
            'messages_processed': len(messages),
            'duration_seconds': duration,
            'throughput_msg_per_sec': throughput,
            'anomalies_detected': anomalies_detected,
            'anomaly_rate': anomalies_detected / len(messages),
            'latency_ms': {
                'mean': avg_latency,
                'min': min(processing_times),
                'max': max(processing_times),
                'p95': p95_latency,
                'p99': p99_latency,
                'stddev': statistics.stdev(processing_times)
            },
            'resource_usage': {
                'cpu_percent': cpu_after - cpu_before,
                'memory_mb': mem_after - mem_before,
                'memory_total_mb': mem_after
            }
        }
        
        self._print_results(results)
        return results
        
    def benchmark_end_to_end(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Benchmark complete end-to-end pipeline.
        
        Args:
            messages: List of CAN messages to process
            
        Returns:
            Dictionary of benchmark results
        """
        print(f"\nBenchmarking End-to-End Pipeline...")
        print(f"  Messages: {len(messages)}")
        
        # Initialize all components
        rule_engine = RuleEngine('config/rules.yaml')
        ml_detector = MLDetector()
        feature_extractor = FeatureExtractor()
        alert_manager = AlertManager({})
        
        # Train ML model
        print("  Training ML model...")
        features = [feature_extractor.extract_features(msg) for msg in messages[:1000]]
        ml_detector.train(features)
        
        # Warm-up
        for msg in messages[:100]:
            rule_engine.analyze_message(msg)
            ml_detector.analyze_message(msg)
        
        # Benchmark
        process = psutil.Process()
        cpu_samples = []
        mem_samples = []
        
        start_time = time.time()
        total_alerts = 0
        processing_times = []
        
        for i, msg in enumerate(messages):
            msg_start = time.time()
            
            # Rule engine
            rule_alerts = rule_engine.analyze_message(msg)
            total_alerts += len(rule_alerts)
            
            # ML detector
            ml_alert = ml_detector.analyze_message(msg)
            if ml_alert:
                total_alerts += 1
            
            msg_time = time.time() - msg_start
            processing_times.append(msg_time * 1000)
            
            # Sample resource usage periodically
            if i % 100 == 0:
                cpu_samples.append(process.cpu_percent())
                mem_samples.append(process.memory_info().rss / 1024 / 1024)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Calculate metrics
        throughput = len(messages) / duration
        avg_latency = statistics.mean(processing_times)
        p95_latency = statistics.quantiles(processing_times, n=20)[18]
        p99_latency = statistics.quantiles(processing_times, n=100)[98]
        
        results = {
            'component': 'End-to-End Pipeline',
            'messages_processed': len(messages),
            'duration_seconds': duration,
            'throughput_msg_per_sec': throughput,
            'total_alerts': total_alerts,
            'alert_rate': total_alerts / len(messages),
            'latency_ms': {
                'mean': avg_latency,
                'min': min(processing_times),
                'max': max(processing_times),
                'p95': p95_latency,
                'p99': p99_latency,
                'stddev': statistics.stdev(processing_times)
            },
            'resource_usage': {
                'cpu_percent_avg': statistics.mean(cpu_samples) if cpu_samples else 0,
                'cpu_percent_max': max(cpu_samples) if cpu_samples else 0,
                'memory_mb_avg': statistics.mean(mem_samples) if mem_samples else 0,
                'memory_mb_max': max(mem_samples) if mem_samples else 0
            }
        }
        
        self._print_results(results)
        return results
        
    def _print_results(self, results: Dict[str, Any]) -> None:
        """Print benchmark results in formatted output."""
        print(f"\n  Results:")
        print(f"    Throughput:  {results['throughput_msg_per_sec']:.2f} msg/s")
        print(f"    Mean Latency: {results['latency_ms']['mean']:.3f} ms")
        print(f"    P95 Latency:  {results['latency_ms']['p95']:.3f} ms")
        print(f"    P99 Latency:  {results['latency_ms']['p99']:.3f} ms")
        
        if 'cpu_percent' in results['resource_usage']:
            print(f"    CPU Usage:    {results['resource_usage']['cpu_percent']:.1f}%")
        elif 'cpu_percent_avg' in results['resource_usage']:
            print(f"    CPU Usage:    {results['resource_usage']['cpu_percent_avg']:.1f}% (avg)")
        
        if 'memory_mb' in results['resource_usage']:
            print(f"    Memory:       {results['resource_usage']['memory_mb']:.1f} MB")
        elif 'memory_mb_avg' in results['resource_usage']:
            print(f"    Memory:       {results['resource_usage']['memory_mb_avg']:.1f} MB (avg)")
        
    def load_test_data(self, data_file: str) -> List[Dict[str, Any]]:
        """Load test data from JSON file."""
        data_path = Path(data_file)
        
        if not data_path.exists():
            print(f"Error: Data file not found: {data_file}")
            print("Generate test data with: python scripts/generate_dataset.py")
            sys.exit(1)
        
        with open(data_path, 'r') as f:
            messages = json.load(f)
        
        return messages
        
    def save_results(self, output_file: str = 'benchmark_results.json') -> None:
        """Save benchmark results to file."""
        output_path = Path('data/benchmarks') / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Benchmark CAN-IDS performance'
    )
    
    parser.add_argument('--data', default='data/synthetic/normal_traffic.json',
                       help='Test data file (default: data/synthetic/normal_traffic.json)')
    parser.add_argument('--component', 
                       choices=['rule-engine', 'ml-detector', 'end-to-end', 'all'],
                       default='all',
                       help='Component to benchmark (default: all)')
    parser.add_argument('--output', default='benchmark_results.json',
                       help='Output file for results')
    parser.add_argument('--messages', type=int,
                       help='Limit number of messages to process')
    
    args = parser.parse_args()
    
    print("CAN-IDS Performance Benchmark")
    print("=" * 60)
    
    # Initialize benchmark
    benchmark = CANIDSBenchmark()
    
    # Load test data
    print(f"\nLoading test data from: {args.data}")
    messages = benchmark.load_test_data(args.data)
    
    if args.messages:
        messages = messages[:args.messages]
    
    print(f"Loaded {len(messages)} messages")
    
    # Run benchmarks
    if args.component == 'all' or args.component == 'rule-engine':
        benchmark.results['rule_engine'] = benchmark.benchmark_rule_engine(messages)
    
    if args.component == 'all' or args.component == 'ml-detector':
        benchmark.results['ml_detector'] = benchmark.benchmark_ml_detector(messages)
    
    if args.component == 'all' or args.component == 'end-to-end':
        benchmark.results['end_to_end'] = benchmark.benchmark_end_to_end(messages)
    
    # Save results
    benchmark.save_results(args.output)
    
    print("\n" + "=" * 60)
    print("Benchmark complete!")


if __name__ == '__main__':
    main()
