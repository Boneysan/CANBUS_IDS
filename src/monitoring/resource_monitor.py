"""
Lightweight resource monitoring for CAN-IDS.

Optimized for Raspberry Pi 4 with minimal performance overhead.
Monitors CPU, memory, temperature, and system health.
"""

import time
import logging
import threading
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available - resource monitoring disabled")

try:
    import subprocess
    SUBPROCESS_AVAILABLE = True
except ImportError:
    SUBPROCESS_AVAILABLE = False
    logger.warning("subprocess not available - temperature monitoring disabled")


class MetricsCollector:
    """
    Collects system utilization metrics with minimal overhead.
    
    Designed for Raspberry Pi with configurable sampling intervals.
    """
    
    def __init__(self, 
                 enable_cpu: bool = True,
                 enable_memory: bool = True,
                 enable_temperature: bool = True,
                 cache_process: bool = True):
        """
        Initialize metrics collector.
        
        Args:
            enable_cpu: Enable CPU usage monitoring
            enable_memory: Enable memory usage monitoring
            enable_temperature: Enable temperature monitoring (Pi-specific)
            cache_process: Cache process handle for performance
        """
        self.enable_cpu = enable_cpu and PSUTIL_AVAILABLE
        self.enable_memory = enable_memory and PSUTIL_AVAILABLE
        self.enable_temperature = enable_temperature and SUBPROCESS_AVAILABLE
        
        # Cache process handle to avoid repeated lookups
        self.process = psutil.Process() if (cache_process and PSUTIL_AVAILABLE) else None
        
        # Pre-compile temperature command
        self._temp_cmd = ['vcgencmd', 'measure_temp']
        self._last_temp = None
        self._last_temp_time = 0
        
        # Statistics
        self.collections = 0
        self.collection_errors = 0
        
    def collect(self) -> Dict[str, Any]:
        """
        Collect current system metrics.
        
        Returns:
            Dictionary with current metrics
        """
        metrics = {
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat()
        }
        
        try:
            # CPU usage (non-blocking, uses cached value)
            if self.enable_cpu:
                if self.process:
                    metrics['cpu_percent'] = self.process.cpu_percent(interval=0)
                else:
                    metrics['cpu_percent'] = psutil.cpu_percent(interval=0)
            
            # Memory usage
            if self.enable_memory:
                if self.process:
                    mem_info = self.process.memory_info()
                    metrics['memory_mb'] = mem_info.rss / 1024 / 1024
                    metrics['memory_percent'] = self.process.memory_percent()
                else:
                    mem = psutil.virtual_memory()
                    metrics['memory_mb'] = mem.used / 1024 / 1024
                    metrics['memory_percent'] = mem.percent
                
                # System-wide memory
                mem = psutil.virtual_memory()
                metrics['system_memory_percent'] = mem.percent
                metrics['system_memory_available_mb'] = mem.available / 1024 / 1024
            
            # CPU temperature (Pi-specific, expensive operation)
            if self.enable_temperature:
                metrics['cpu_temp_celsius'] = self.get_temperature()
            
            self.collections += 1
            
        except Exception as e:
            logger.debug(f"Error collecting metrics: {e}")
            self.collection_errors += 1
        
        return metrics
    
    def get_temperature(self, cache_seconds: int = 5) -> Optional[float]:
        """
        Get CPU temperature with caching to reduce overhead.
        
        Args:
            cache_seconds: Cache temperature reading for this many seconds
            
        Returns:
            Temperature in Celsius or None if unavailable
        """
        # Use cached value if recent
        now = time.time()
        if self._last_temp and (now - self._last_temp_time) < cache_seconds:
            return self._last_temp
        
        try:
            result = subprocess.run(
                self._temp_cmd,
                capture_output=True,
                text=True,
                timeout=1
            )
            
            if result.returncode == 0:
                # Parse: temp=52.7'C
                temp_str = result.stdout.strip()
                temp = float(temp_str.split('=')[1].split("'")[0])
                
                # Cache the result
                self._last_temp = temp
                self._last_temp_time = now
                
                return temp
                
        except (subprocess.TimeoutExpired, ValueError, IndexError, FileNotFoundError) as e:
            logger.debug(f"Temperature read failed: {e}")
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get collector statistics."""
        return {
            'collections': self.collections,
            'collection_errors': self.collection_errors,
            'success_rate': self.collections / max(1, self.collections + self.collection_errors)
        }


class ResourceMonitor:
    """
    Background resource monitoring with configurable intervals.
    
    Runs in separate thread to avoid blocking CAN message processing.
    """
    
    def __init__(self,
                 sample_interval: float = 10.0,
                 log_interval: float = 60.0,
                 log_file: Optional[str] = None,
                 console_output: bool = False,
                 metrics_window: int = 100,
                 enable_alerts: bool = True,
                 alert_thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize resource monitor.
        
        Args:
            sample_interval: Seconds between metric samples (default: 10s)
            log_interval: Seconds between writing to log file (default: 60s)
            log_file: Path to metrics log file
            console_output: Print metrics to console
            metrics_window: Number of samples to keep in memory
            enable_alerts: Enable threshold-based alerts
            alert_thresholds: Custom alert thresholds
        """
        self.sample_interval = sample_interval
        self.log_interval = log_interval
        self.log_file = Path(log_file) if log_file else None
        self.console_output = console_output
        self.enable_alerts = enable_alerts
        
        # Alert thresholds (defaults for Raspberry Pi 4)
        self.thresholds = alert_thresholds or {
            'cpu_percent': 80.0,          # CPU usage warning
            'memory_percent': 85.0,        # Memory usage warning
            'cpu_temp_celsius': 75.0,      # Temperature warning
            'cpu_temp_critical': 80.0      # Temperature critical
        }
        
        # Metrics collection
        self.collector = MetricsCollector()
        self.metrics_buffer = deque(maxlen=metrics_window)
        
        # Threading
        self._running = False
        self._monitor_thread = None
        self._lock = threading.Lock()
        
        # Statistics
        self.samples_collected = 0
        self.alerts_generated = 0
        self.log_writes = 0
        self.start_time = None
        
        # Alert state tracking (prevent spam)
        self._alert_state = {}
        self._alert_cooldown = 300  # 5 minutes between repeated alerts
        
    def start(self) -> None:
        """Start background monitoring thread."""
        if self._running:
            logger.warning("Resource monitor already running")
            return
        
        if not PSUTIL_AVAILABLE:
            logger.warning("psutil not available - resource monitoring disabled")
            return
        
        self._running = True
        self.start_time = time.time()
        
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="ResourceMonitor",
            daemon=True
        )
        self._monitor_thread.start()
        
        logger.info(f"Resource monitor started (sample: {self.sample_interval}s, log: {self.log_interval}s)")
    
    def stop(self) -> None:
        """Stop background monitoring thread."""
        if not self._running:
            return
        
        logger.info("Stopping resource monitor...")
        self._running = False
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        
        # Final log flush
        self._flush_to_log()
        
        logger.info("Resource monitor stopped")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop running in background thread."""
        last_log_time = time.time()
        
        while self._running:
            try:
                # Collect metrics
                metrics = self.collector.collect()
                
                with self._lock:
                    self.metrics_buffer.append(metrics)
                    self.samples_collected += 1
                
                # Check for alerts
                if self.enable_alerts:
                    self._check_alerts(metrics)
                
                # Console output
                if self.console_output:
                    self._print_metrics(metrics)
                
                # Periodic log flush
                now = time.time()
                if self.log_file and (now - last_log_time) >= self.log_interval:
                    self._flush_to_log()
                    last_log_time = now
                
                # Sleep until next sample
                time.sleep(self.sample_interval)
                
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                time.sleep(self.sample_interval)
    
    def _check_alerts(self, metrics: Dict[str, Any]) -> None:
        """Check metrics against thresholds and generate alerts."""
        for key, threshold in self.thresholds.items():
            if key not in metrics:
                continue
            
            value = metrics[key]
            
            # Skip if value is None
            if value is None:
                continue
            
            # Check if threshold exceeded
            if value > threshold:
                # Check cooldown to prevent alert spam
                last_alert = self._alert_state.get(key, 0)
                if time.time() - last_alert < self._alert_cooldown:
                    continue
                
                # Generate alert
                alert_level = 'CRITICAL' if 'critical' in key else 'WARNING'
                logger.warning(
                    f"{alert_level}: {key} = {value:.1f} exceeds threshold {threshold:.1f}"
                )
                
                self._alert_state[key] = time.time()
                self.alerts_generated += 1
    
    def _print_metrics(self, metrics: Dict[str, Any]) -> None:
        """Print metrics to console."""
        output = f"[Metrics] "
        
        if 'cpu_percent' in metrics:
            output += f"CPU: {metrics['cpu_percent']:.1f}% "
        
        if 'memory_mb' in metrics:
            output += f"Mem: {metrics['memory_mb']:.1f}MB ({metrics.get('memory_percent', 0):.1f}%) "
        
        if 'cpu_temp_celsius' in metrics and metrics['cpu_temp_celsius'] is not None:
            output += f"Temp: {metrics['cpu_temp_celsius']:.1f}°C"
        
        logger.info(output)
    
    def _flush_to_log(self) -> None:
        """Write buffered metrics to log file."""
        if not self.log_file or not self.metrics_buffer:
            return
        
        try:
            # Ensure log directory exists
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write metrics buffer to file
            with self._lock:
                metrics_to_write = list(self.metrics_buffer)
            
            # Append to log file
            with open(self.log_file, 'a') as f:
                for metrics in metrics_to_write:
                    json.dump(metrics, f)
                    f.write('\n')
            
            self.log_writes += 1
            logger.debug(f"Wrote {len(metrics_to_write)} metrics to {self.log_file}")
            
        except Exception as e:
            logger.error(f"Error writing metrics to log: {e}")
    
    def get_current_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Get most recent metrics sample.
        
        Returns:
            Latest metrics or None if no samples collected
        """
        with self._lock:
            if self.metrics_buffer:
                return self.metrics_buffer[-1].copy()
        return None
    
    def get_average_metrics(self, samples: int = 6) -> Dict[str, float]:
        """
        Get average metrics over recent samples.
        
        Args:
            samples: Number of recent samples to average (default: last 6)
            
        Returns:
            Dictionary with averaged metrics
        """
        with self._lock:
            if not self.metrics_buffer:
                return {}
            
            recent = list(self.metrics_buffer)[-samples:]
        
        # Calculate averages
        averages = {}
        
        numeric_keys = ['cpu_percent', 'memory_mb', 'memory_percent', 
                       'system_memory_percent', 'cpu_temp_celsius']
        
        for key in numeric_keys:
            values = [m[key] for m in recent if key in m and m[key] is not None]
            if values:
                averages[f'{key}_avg'] = sum(values) / len(values)
                averages[f'{key}_min'] = min(values)
                averages[f'{key}_max'] = max(values)
        
        return averages
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        uptime = time.time() - self.start_time if self.start_time else 0
        
        stats = {
            'running': self._running,
            'uptime_seconds': uptime,
            'samples_collected': self.samples_collected,
            'alerts_generated': self.alerts_generated,
            'log_writes': self.log_writes,
            'sample_interval': self.sample_interval,
            'log_interval': self.log_interval,
            'collector_stats': self.collector.get_statistics()
        }
        
        # Add current metrics
        current = self.get_current_metrics()
        if current:
            stats['current'] = current
        
        # Add averages
        averages = self.get_average_metrics()
        if averages:
            stats['averages'] = averages
        
        return stats
    
    def print_summary(self) -> None:
        """Print monitoring summary."""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("RESOURCE MONITORING SUMMARY")
        print("="*60)
        
        print(f"Uptime: {stats['uptime_seconds']:.1f} seconds")
        print(f"Samples collected: {stats['samples_collected']}")
        print(f"Alerts generated: {stats['alerts_generated']}")
        print(f"Log writes: {stats['log_writes']}")
        
        if 'current' in stats:
            current = stats['current']
            print(f"\nCurrent Metrics:")
            if 'cpu_percent' in current:
                print(f"  CPU Usage: {current['cpu_percent']:.1f}%")
            if 'memory_mb' in current:
                print(f"  Memory: {current['memory_mb']:.1f}MB ({current.get('memory_percent', 0):.1f}%)")
            if 'cpu_temp_celsius' in current and current['cpu_temp_celsius'] is not None:
                print(f"  Temperature: {current['cpu_temp_celsius']:.1f}°C")
        
        if 'averages' in stats:
            avg = stats['averages']
            print(f"\nAverages (last {self.sample_interval * 6:.0f}s):")
            if 'cpu_percent_avg' in avg:
                print(f"  CPU: {avg['cpu_percent_avg']:.1f}% (min: {avg['cpu_percent_min']:.1f}%, max: {avg['cpu_percent_max']:.1f}%)")
            if 'memory_mb_avg' in avg:
                print(f"  Memory: {avg['memory_mb_avg']:.1f}MB (min: {avg['memory_mb_min']:.1f}MB, max: {avg['memory_mb_max']:.1f}MB)")
            if 'cpu_temp_celsius_avg' in avg:
                print(f"  Temp: {avg['cpu_temp_celsius_avg']:.1f}°C (min: {avg['cpu_temp_celsius_min']:.1f}°C, max: {avg['cpu_temp_celsius_max']:.1f}°C)")
        
        print("="*60)
