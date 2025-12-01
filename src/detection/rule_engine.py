"""
Signature-based detection engine for CAN bus intrusion detection.

Implements rule-based pattern matching for known attack signatures
and policy violations.
"""

import logging
import yaml
import time
import math
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from pathlib import Path
import re
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)


@dataclass
class DetectionRule:
    """Represents a single detection rule."""
    name: str
    severity: str
    description: str
    action: str
    can_id: Optional[int] = None
    can_id_range: Optional[List[int]] = None
    dlc_min: Optional[int] = None
    dlc_max: Optional[int] = None
    data_pattern: Optional[str] = None
    data_contains: Optional[List[str]] = None
    max_frequency: Optional[int] = None
    time_window: Optional[int] = None
    check_timing: bool = False
    expected_interval: Optional[int] = None
    interval_variance: Optional[int] = None
    allowed_sources: Optional[List[int]] = None
    check_checksum: bool = False
    check_counter: bool = False
    entropy_threshold: Optional[float] = None
    whitelist_mode: bool = False
    allowed_can_ids: Optional[List[int]] = None
    
    def matches_can_id(self, can_id: int) -> bool:
        """Check if CAN ID matches rule criteria."""
        if self.can_id is not None:
            return can_id == self.can_id
        elif self.can_id_range is not None:
            return self.can_id_range[0] <= can_id <= self.can_id_range[1]
        elif self.whitelist_mode and self.allowed_can_ids:
            return can_id not in self.allowed_can_ids
        return True


@dataclass
class Alert:
    """Represents a detection alert."""
    rule_name: str
    severity: str
    description: str
    timestamp: float
    can_id: int
    message_data: Dict[str, Any]
    confidence: float = 1.0
    additional_info: Optional[Dict[str, Any]] = None


class RuleEngine:
    """
    Signature-based detection engine for CAN bus messages.
    
    Loads rules from YAML configuration and applies them to
    incoming CAN messages to detect known attack patterns.
    """
    
    def __init__(self, rules_file: str):
        """
        Initialize rule engine.
        
        Args:
            rules_file: Path to YAML rules configuration file
        """
        self.rules_file = Path(rules_file)
        self.rules: List[DetectionRule] = []
        
        # State tracking for stateful rules
        self._message_history = defaultdict(deque)  # CAN ID -> recent messages
        self._frequency_counters = defaultdict(deque)  # CAN ID -> timestamps
        self._timing_analysis = defaultdict(list)  # CAN ID -> intervals
        self._sequence_counters = defaultdict(int)  # CAN ID -> expected counter
        
        self._stats = {
            'rules_loaded': 0,
            'messages_processed': 0,
            'alerts_generated': 0,
            'rules_matched': 0,
            'load_time': None
        }
        
        self.load_rules()
        
    def load_rules(self) -> None:
        """Load detection rules from YAML file."""
        start_time = time.time()
        
        try:
            logger.info(f"Loading rules from {self.rules_file}")
            
            if not self.rules_file.exists():
                raise FileNotFoundError(f"Rules file not found: {self.rules_file}")
                
            with open(self.rules_file, 'r') as f:
                config = yaml.safe_load(f)
                
            rules_data = config.get('rules', [])
            self.rules = []
            
            for rule_data in rules_data:
                try:
                    rule = DetectionRule(**rule_data)
                    self.rules.append(rule)
                    self._stats['rules_loaded'] += 1
                except Exception as e:
                    logger.warning(f"Error loading rule '{rule_data.get('name', 'Unknown')}': {e}")
                    
            load_time = time.time() - start_time
            self._stats['load_time'] = load_time
            
            logger.info(f"Loaded {len(self.rules)} rules in {load_time:.3f} seconds")
            
        except Exception as e:
            logger.error(f"Error loading rules: {e}")
            raise
            
    def reload_rules(self) -> None:
        """Reload rules from file (for runtime updates)."""
        logger.info("Reloading detection rules")
        old_count = len(self.rules)
        
        try:
            self.load_rules()
            logger.info(f"Rules reloaded: {old_count} -> {len(self.rules)}")
        except Exception as e:
            logger.error(f"Failed to reload rules, keeping existing: {e}")
            
    def analyze_message(self, message: Dict[str, Any]) -> List[Alert]:
        """
        Analyze a CAN message against all loaded rules.
        
        Args:
            message: CAN message dictionary
            
        Returns:
            List of alerts generated for this message
        """
        alerts = []
        self._stats['messages_processed'] += 1
        
        # Update message history for stateful analysis
        self._update_message_history(message)
        
        # Check each rule
        for rule in self.rules:
            try:
                if self._evaluate_rule(rule, message):
                    alert = Alert(
                        rule_name=rule.name,
                        severity=rule.severity,
                        description=rule.description,
                        timestamp=message['timestamp'],
                        can_id=message['can_id'],
                        message_data=message.copy(),
                        confidence=self._calculate_confidence(rule, message)
                    )
                    alerts.append(alert)
                    self._stats['alerts_generated'] += 1
                    self._stats['rules_matched'] += 1
                    
            except Exception as e:
                logger.warning(f"Error evaluating rule '{rule.name}': {e}")
                
        return alerts
        
    def _evaluate_rule(self, rule: DetectionRule, message: Dict[str, Any]) -> bool:
        """
        Evaluate a single rule against a message.
        
        Args:
            rule: Detection rule to evaluate
            message: CAN message to check
            
        Returns:
            True if rule matches, False otherwise
        """
        can_id = message['can_id']
        
        # CAN ID matching
        if not rule.matches_can_id(can_id):
            return False
            
        # DLC validation
        if rule.dlc_min is not None and message['dlc'] < rule.dlc_min:
            return True  # Invalid DLC detected
        if rule.dlc_max is not None and message['dlc'] > rule.dlc_max:
            return True  # Invalid DLC detected
            
        # Data pattern matching
        if rule.data_pattern and not self._match_data_pattern(rule.data_pattern, message['data']):
            return False
            
        # Data contains check
        if rule.data_contains and not self._check_data_contains(rule.data_contains, message['data']):
            return False
            
        # Frequency analysis
        if rule.max_frequency and self._check_frequency_violation(rule, can_id):
            return True
            
        # Timing analysis
        if rule.check_timing and self._check_timing_violation(rule, can_id, message['timestamp']):
            return True
            
        # Source validation
        if rule.allowed_sources and not self._validate_source(rule, message):
            return True
            
        # Checksum validation
        if rule.check_checksum and not self._validate_checksum(message):
            return True
            
        # Counter validation
        if rule.check_counter and not self._validate_counter(rule, can_id, message):
            return True
            
        # Entropy analysis
        if rule.entropy_threshold and self._calculate_entropy(message['data']) > rule.entropy_threshold:
            return True
            
        return True  # Rule matches if we get here
        
    def _update_message_history(self, message: Dict[str, Any]) -> None:
        """Update message history for stateful analysis."""
        can_id = message['can_id']
        timestamp = message['timestamp']
        
        # Update message history (keep last 100 messages per ID)
        history = self._message_history[can_id]
        history.append(message)
        if len(history) > 100:
            history.popleft()
            
        # Update frequency tracking (keep last minute)
        freq_history = self._frequency_counters[can_id]
        freq_history.append(timestamp)
        
        # Remove old entries (older than 60 seconds)
        cutoff_time = timestamp - 60
        while freq_history and freq_history[0] < cutoff_time:
            freq_history.popleft()
            
        # Update timing analysis
        if len(history) >= 2:
            prev_msg = history[-2]
            interval = timestamp - prev_msg['timestamp']
            
            timing_history = self._timing_analysis[can_id]
            timing_history.append(interval * 1000)  # Convert to milliseconds
            
            # Keep last 50 intervals
            if len(timing_history) > 50:
                timing_history.pop(0)
                
    def _match_data_pattern(self, pattern: str, data: List[int]) -> bool:
        """
        Match data pattern against message data.
        
        Args:
            pattern: Hex pattern (e.g., "DE AD BE EF" or "10 *")
            data: Message data bytes
            
        Returns:
            True if pattern matches
        """
        pattern_bytes = pattern.split()
        
        if len(pattern_bytes) > len(data):
            return False
            
        for i, pattern_byte in enumerate(pattern_bytes):
            if pattern_byte == '*':
                continue  # Wildcard matches anything
                
            try:
                expected = int(pattern_byte, 16)
                if data[i] != expected:
                    return False
            except (ValueError, IndexError):
                return False
                
        return True
        
    def _check_data_contains(self, contains_list: List[str], data: List[int]) -> bool:
        """Check if data contains any of the specified patterns."""
        data_hex = ' '.join(f"{b:02X}" for b in data)
        
        for pattern in contains_list:
            if pattern.upper() in data_hex:
                return True
                
        return False
        
    def _check_frequency_violation(self, rule: DetectionRule, can_id: int) -> bool:
        """Check if message frequency exceeds threshold."""
        if not rule.max_frequency or not rule.time_window:
            return False
            
        freq_history = self._frequency_counters[can_id]
        
        if len(freq_history) < rule.max_frequency:
            return False
            
        # Check if we have max_frequency messages within time_window
        time_span = freq_history[-1] - freq_history[-rule.max_frequency]
        return time_span <= rule.time_window
        
    def _check_timing_violation(self, rule: DetectionRule, can_id: int, timestamp: float) -> bool:
        """Check for timing anomalies."""
        if not rule.expected_interval or not rule.interval_variance:
            return False
            
        timing_history = self._timing_analysis[can_id]
        
        if len(timing_history) < 5:  # Need some history
            return False
            
        # Calculate recent interval statistics
        recent_intervals = timing_history[-10:]
        avg_interval = statistics.mean(recent_intervals)
        
        expected = rule.expected_interval
        tolerance = rule.interval_variance
        
        # Check if average interval is outside expected range
        return not (expected - tolerance <= avg_interval <= expected + tolerance)
        
    def _validate_source(self, rule: DetectionRule, message: Dict[str, Any]) -> bool:
        """Validate message source (simplified - would need ECU identification)."""
        # This is a placeholder - real implementation would need
        # to track ECU sources based on network topology
        return True
        
    def _validate_checksum(self, message: Dict[str, Any]) -> bool:
        """Validate message checksum (if present)."""
        # This is a placeholder - real implementation would need
        # to know the specific checksum algorithm used
        return True
        
    def _validate_counter(self, rule: DetectionRule, can_id: int, message: Dict[str, Any]) -> bool:
        """Validate message counter/sequence number."""
        if len(message['data']) == 0:
            return True
            
        # Assume counter is in first byte (common convention)
        current_counter = message['data'][0] & 0x0F  # Lower 4 bits
        expected_counter = self._sequence_counters[can_id]
        
        # Check if counter increments correctly
        next_expected = (expected_counter + 1) % 16
        self._sequence_counters[can_id] = current_counter
        
        return current_counter == next_expected
        
    def _calculate_entropy(self, data: List[int]) -> float:
        """Calculate Shannon entropy of data."""
        if not data:
            return 0.0
            
        # Count byte frequencies
        byte_counts = defaultdict(int)
        for byte_val in data:
            byte_counts[byte_val] += 1
            
        # Calculate entropy
        entropy = 0.0
        data_len = len(data)
        
        for count in byte_counts.values():
            probability = count / data_len
            if probability > 0:
                entropy -= probability * math.log2(probability)
                
        return entropy
        
    def _calculate_confidence(self, rule: DetectionRule, message: Dict[str, Any]) -> float:
        """Calculate confidence score for alert (0.0 to 1.0)."""
        # Simple confidence calculation - could be enhanced
        confidence = 0.5
        
        # Higher confidence for specific patterns
        if rule.data_pattern:
            confidence += 0.3
            
        # Higher confidence for frequency violations
        if rule.max_frequency:
            confidence += 0.2
            
        # Higher confidence for critical severity
        if rule.severity == 'CRITICAL':
            confidence += 0.2
            
        return min(confidence, 1.0)
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get rule engine statistics."""
        stats = self._stats.copy()
        
        # Calculate additional metrics
        if stats['messages_processed'] > 0:
            stats['alert_rate'] = stats['alerts_generated'] / stats['messages_processed']
            stats['match_rate'] = stats['rules_matched'] / stats['messages_processed']
        else:
            stats['alert_rate'] = 0.0
            stats['match_rate'] = 0.0
            
        return stats
        
    def get_rule_info(self) -> List[Dict[str, Any]]:
        """Get information about loaded rules."""
        return [
            {
                'name': rule.name,
                'severity': rule.severity,
                'description': rule.description,
                'can_id': rule.can_id,
                'can_id_range': rule.can_id_range,
                'stateful': any([
                    rule.max_frequency,
                    rule.check_timing,
                    rule.check_counter
                ])
            }
            for rule in self.rules
        ]