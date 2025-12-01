"""
Alert management and coordination system.

Handles alert generation, deduplication, rate limiting,
and routing to various notification channels.
"""

import logging
import time
import threading
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class AlertAction(Enum):
    """Actions to take for alerts."""
    ALERT = "alert"
    LOG = "log"
    BLOCK = "block"  # Future implementation
    IGNORE = "ignore"


@dataclass
class Alert:
    """Represents a security alert."""
    id: str
    timestamp: float
    rule_name: str
    severity: AlertSeverity
    description: str
    can_id: int
    message_data: Dict[str, Any]
    confidence: float = 1.0
    source: str = "unknown"  # rule_engine, ml_detector, etc.
    action: AlertAction = AlertAction.ALERT
    additional_info: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary format."""
        data = asdict(self)
        data['severity'] = self.severity.value
        data['action'] = self.action.value
        return data


class AlertManager:
    """
    Central alert management system.
    
    Provides:
    - Alert deduplication and aggregation
    - Rate limiting to prevent spam
    - Severity-based filtering
    - Multiple notification channels
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize alert manager.
        
        Args:
            config: Alert configuration dictionary
        """
        self.config = config
        self.notifiers = []
        
        # Rate limiting
        self.rate_limit = config.get('rate_limit', 10)  # alerts per second
        self._rate_limiter = deque(maxlen=self.rate_limit * 2)  # 2 second window
        
        # Alert tracking
        self._alert_counter = 0
        self._alert_history = deque(maxlen=1000)  # Keep last 1000 alerts
        self._duplicate_tracker = {}  # Track duplicate alerts
        self._duplicate_window = 60  # seconds
        
        # Statistics
        self._stats = {
            'total_alerts': 0,
            'alerts_by_severity': defaultdict(int),
            'alerts_by_source': defaultdict(int),
            'alerts_dropped_rate_limit': 0,
            'alerts_deduplicated': 0,
            'start_time': time.time()
        }
        
        # Threading
        self._lock = threading.Lock()
        
        logger.info("Alert manager initialized")
        
    def add_notifier(self, notifier) -> None:
        """
        Add a notification handler.
        
        Args:
            notifier: Object with notify(alert) method
        """
        self.notifiers.append(notifier)
        logger.info(f"Added notifier: {type(notifier).__name__}")
        
    def process_alert(self, alert_data: Dict[str, Any]) -> Optional[Alert]:
        """
        Process an incoming alert.
        
        Args:
            alert_data: Raw alert data from detection engines
            
        Returns:
            Processed Alert object or None if filtered/dropped
        """
        with self._lock:
            # Generate unique alert ID
            self._alert_counter += 1
            alert_id = f"CANIDS-{int(time.time())}-{self._alert_counter:06d}"
            
            # Create Alert object
            alert = Alert(
                id=alert_id,
                timestamp=alert_data.get('timestamp', time.time()),
                rule_name=alert_data.get('rule_name', 'unknown'),
                severity=AlertSeverity(alert_data.get('severity', 'MEDIUM')),
                description=alert_data.get('description', ''),
                can_id=alert_data.get('can_id', 0),
                message_data=alert_data.get('message_data', {}),
                confidence=alert_data.get('confidence', 1.0),
                source=alert_data.get('source', 'unknown'),
                additional_info=alert_data.get('additional_info')
            )
            
            # Apply filters
            if not self._should_process_alert(alert):
                return None
                
            # Check rate limiting
            if not self._check_rate_limit():
                self._stats['alerts_dropped_rate_limit'] += 1
                logger.warning(f"Alert dropped due to rate limit: {alert.rule_name}")
                return None
                
            # Check for duplicates
            if self._is_duplicate(alert):
                self._stats['alerts_deduplicated'] += 1
                logger.debug(f"Duplicate alert suppressed: {alert.rule_name}")
                return None
                
            # Update statistics
            self._stats['total_alerts'] += 1
            self._stats['alerts_by_severity'][alert.severity.value] += 1
            self._stats['alerts_by_source'][alert.source] += 1
            
            # Store in history
            self._alert_history.append(alert)
            
            # Add to duplicate tracker
            self._add_to_duplicate_tracker(alert)
            
            # Send notifications
            self._send_notifications(alert)
            
            logger.info(f"Alert processed: {alert.severity.value} - {alert.rule_name}")
            
            return alert
            
    def _should_process_alert(self, alert: Alert) -> bool:
        """
        Check if alert should be processed based on configuration.
        
        Args:
            alert: Alert to check
            
        Returns:
            True if alert should be processed
        """
        # Check severity filtering
        min_severity = self.config.get('min_severity', 'LOW')
        severity_levels = {
            'LOW': 0,
            'MEDIUM': 1, 
            'HIGH': 2,
            'CRITICAL': 3
        }
        
        alert_level = severity_levels.get(alert.severity.value, 0)
        min_level = severity_levels.get(min_severity, 0)
        
        if alert_level < min_level:
            return False
            
        # Check confidence threshold
        min_confidence = self.config.get('min_confidence', 0.0)
        if alert.confidence < min_confidence:
            return False
            
        # Check source filtering
        allowed_sources = self.config.get('allowed_sources', [])
        if allowed_sources and alert.source not in allowed_sources:
            return False
            
        return True
        
    def _check_rate_limit(self) -> bool:
        """
        Check if we're within rate limits.
        
        Returns:
            True if alert can be processed
        """
        current_time = time.time()
        
        # Clean old entries from rate limiter
        while (self._rate_limiter and 
               current_time - self._rate_limiter[0] > 1.0):
            self._rate_limiter.popleft()
            
        # Check if we can add another alert
        if len(self._rate_limiter) >= self.rate_limit:
            return False
            
        # Add current time to rate limiter
        self._rate_limiter.append(current_time)
        return True
        
    def _is_duplicate(self, alert: Alert) -> bool:
        """
        Check if alert is a duplicate within the time window.
        
        Args:
            alert: Alert to check
            
        Returns:
            True if alert is a duplicate
        """
        # Create signature for duplicate detection
        signature = (
            alert.rule_name,
            alert.can_id,
            alert.severity.value
        )
        
        current_time = time.time()
        
        # Check if we've seen this signature recently
        if signature in self._duplicate_tracker:
            last_time = self._duplicate_tracker[signature]
            if current_time - last_time < self._duplicate_window:
                return True
                
        return False
        
    def _add_to_duplicate_tracker(self, alert: Alert) -> None:
        """Add alert to duplicate tracking."""
        signature = (
            alert.rule_name,
            alert.can_id,
            alert.severity.value
        )
        
        self._duplicate_tracker[signature] = alert.timestamp
        
        # Clean old entries
        current_time = time.time()
        expired_signatures = []
        
        for sig, timestamp in self._duplicate_tracker.items():
            if current_time - timestamp > self._duplicate_window:
                expired_signatures.append(sig)
                
        for sig in expired_signatures:
            del self._duplicate_tracker[sig]
            
    def _send_notifications(self, alert: Alert) -> None:
        """
        Send alert to all configured notifiers.
        
        Args:
            alert: Alert to send
        """
        for notifier in self.notifiers:
            try:
                notifier.notify(alert)
            except Exception as e:
                logger.error(f"Error sending notification via {type(notifier).__name__}: {e}")
                
    def get_recent_alerts(self, count: int = 100) -> List[Alert]:
        """
        Get recent alerts.
        
        Args:
            count: Number of recent alerts to return
            
        Returns:
            List of recent alerts
        """
        with self._lock:
            return list(self._alert_history)[-count:]
            
    def get_alerts_by_severity(self, severity: AlertSeverity, 
                             hours: int = 24) -> List[Alert]:
        """
        Get alerts by severity within time range.
        
        Args:
            severity: Alert severity to filter by
            hours: Time range in hours
            
        Returns:
            List of matching alerts
        """
        cutoff_time = time.time() - (hours * 3600)
        
        with self._lock:
            return [
                alert for alert in self._alert_history
                if alert.severity == severity and alert.timestamp >= cutoff_time
            ]
            
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get alert manager statistics.
        
        Returns:
            Statistics dictionary
        """
        with self._lock:
            stats = self._stats.copy()
            
            # Calculate additional metrics
            runtime = time.time() - stats['start_time']
            stats['runtime_hours'] = runtime / 3600
            
            if runtime > 0:
                stats['alerts_per_hour'] = stats['total_alerts'] / (runtime / 3600)
            else:
                stats['alerts_per_hour'] = 0.0
                
            stats['active_notifiers'] = len(self.notifiers)
            stats['duplicate_tracker_size'] = len(self._duplicate_tracker)
            stats['recent_alerts_count'] = len(self._alert_history)
            
            return stats
            
    def clear_history(self) -> None:
        """Clear alert history and reset statistics."""
        with self._lock:
            self._alert_history.clear()
            self._duplicate_tracker.clear()
            self._rate_limiter.clear()
            
            # Reset statistics
            self._stats = {
                'total_alerts': 0,
                'alerts_by_severity': defaultdict(int),
                'alerts_by_source': defaultdict(int),
                'alerts_dropped_rate_limit': 0,
                'alerts_deduplicated': 0,
                'start_time': time.time()
            }
            
            logger.info("Alert history cleared")
            
    def set_rate_limit(self, alerts_per_second: int) -> None:
        """
        Update rate limiting configuration.
        
        Args:
            alerts_per_second: New rate limit
        """
        self.rate_limit = alerts_per_second
        self._rate_limiter = deque(maxlen=alerts_per_second * 2)
        
        logger.info(f"Rate limit updated to {alerts_per_second} alerts/second")
        
    def export_alerts(self, filepath: str, format: str = 'json') -> None:
        """
        Export alert history to file.
        
        Args:
            filepath: Output file path
            format: Export format ('json', 'csv')
        """
        import json
        
        with self._lock:
            alerts_data = [alert.to_dict() for alert in self._alert_history]
            
        if format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(alerts_data, f, indent=2, default=str)
        elif format.lower() == 'csv':
            import csv
            
            if alerts_data:
                with open(filepath, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=alerts_data[0].keys())
                    writer.writeheader()
                    writer.writerows(alerts_data)
                    
        logger.info(f"Exported {len(alerts_data)} alerts to {filepath}")
        
    def shutdown(self) -> None:
        """Shutdown alert manager."""
        logger.info("Shutting down alert manager")
        
        # Notify all notifiers to cleanup
        for notifier in self.notifiers:
            if hasattr(notifier, 'shutdown'):
                try:
                    notifier.shutdown()
                except Exception as e:
                    logger.error(f"Error shutting down notifier: {e}")
                    
        logger.info("Alert manager shutdown complete")