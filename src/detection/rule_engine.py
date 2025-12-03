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
    
    # Phase 1 Critical Parameters (Dec 2, 2025)
    validate_dlc: bool = False                    # Strict DLC validation
    check_frame_format: bool = False              # Frame format checking
    global_message_rate: Optional[int] = None     # Global rate monitoring
    
    # Phase 2 Important Parameters (Dec 2, 2025)
    check_source: bool = False                    # Source validation for diagnostics
    check_replay: bool = False                    # Replay attack detection
    data_byte_0: Optional[int] = None             # Expected value for data byte 0
    data_byte_1: Optional[int] = None             # Expected value for data byte 1
    data_byte_2: Optional[int] = None             # Expected value for data byte 2
    data_byte_3: Optional[int] = None             # Expected value for data byte 3
    data_byte_4: Optional[int] = None             # Expected value for data byte 4
    data_byte_5: Optional[int] = None             # Expected value for data byte 5
    data_byte_6: Optional[int] = None             # Expected value for data byte 6
    data_byte_7: Optional[int] = None             # Expected value for data byte 7
    replay_time_threshold: Optional[float] = None # Max time between replays (seconds)
    
    # Phase 3 Specialized Parameters (Dec 2, 2025)
    check_data_integrity: bool = False            # Data integrity validation
    check_steering_range: bool = False            # Steering angle validation
    check_repetition: bool = False                # Repetitive pattern detection
    frame_type: Optional[str] = None              # Expected frame type ('standard' or 'extended')
    steering_min: Optional[float] = None          # Min steering angle (degrees)
    steering_max: Optional[float] = None          # Max steering angle (degrees)
    repetition_threshold: Optional[int] = None    # Max consecutive identical messages
    integrity_checksum_offset: Optional[int] = None # Byte offset for integrity checksum
    
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
        
        # Phase 1 state tracking (Dec 2, 2025)
        self._global_message_times = deque(maxlen=10000)  # Global rate tracking
        
        # Phase 2 state tracking (Dec 2, 2025)
        self._message_signatures = defaultdict(lambda: {'data': None, 'timestamp': None, 'count': 0})  # Replay detection
        self._source_tracking = defaultdict(set)  # CAN ID -> set of source IDs seen
        
        # Phase 3 state tracking (Dec 2, 2025)
        self._data_repetition_counts = defaultdict(lambda: {'data': None, 'count': 0})  # Repetition tracking
        
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
        
        # Phase 1 Critical Checks (Dec 2, 2025)
        
        # 1. Strict DLC validation
        if rule.validate_dlc and not self._validate_dlc_strict(rule, message):
            return True  # DLC violation detected
        
        # 2. Frame format checking
        if rule.check_frame_format and not self._check_frame_format(message):
            return True  # Malformed frame detected
        
        # 3. Global message rate monitoring
        if rule.global_message_rate and self._check_global_message_rate(rule, message['timestamp']):
            return True  # Bus flooding detected
        
        # Phase 2 Important Checks (Dec 2, 2025)
        
        # 4. Source validation for diagnostics
        if rule.check_source and not self._validate_source_enhanced(rule, message):
            return True  # Unauthorized diagnostic source detected
        
        # 5. Replay attack detection
        if rule.check_replay and self._check_replay_attack(rule, message):
            return True  # Replay attack detected
        
        # 6. Data byte validation
        if any([rule.data_byte_0, rule.data_byte_1, rule.data_byte_2, rule.data_byte_3,
                rule.data_byte_4, rule.data_byte_5, rule.data_byte_6, rule.data_byte_7]) is not None:
            if not self._validate_data_bytes(rule, message):
                return True  # Data byte mismatch detected
        
        # Phase 3 Specialized Checks (Dec 2, 2025)
        
        # 7. Data integrity validation
        if rule.check_data_integrity and not self._check_data_integrity(rule, message):
            return True  # Data integrity failure detected
        
        # 8. Steering range validation
        if rule.check_steering_range and not self._check_steering_range(rule, message):
            return True  # Steering angle out of range detected
        
        # 9. Repetition pattern detection
        if rule.check_repetition and self._check_repetition_pattern(rule, message):
            return True  # Repetition attack detected
        
        # 10. Frame type validation
        if rule.frame_type and not self._validate_frame_type(rule, message):
            return True  # Frame type violation detected
            
        # Legacy DLC validation (only if validate_dlc is not enabled)
        # Note: When validate_dlc is True, strict validation is used instead
        if not rule.validate_dlc:
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
        
        # Whitelist mode: if specified, alert on CAN IDs NOT in the whitelist
        if rule.whitelist_mode and rule.allowed_can_ids:
            if can_id not in rule.allowed_can_ids:
                return True
            
        # If we reach here, no violations detected
        return False
        
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
    
    # ========================================================================
    # PHASE 1 CRITICAL VALIDATION METHODS (Dec 2, 2025)
    # Implementing 3 critical rule parameters for basic dual-detection
    # ========================================================================
    
    def _validate_dlc_strict(self, rule: DetectionRule, message: Dict[str, Any]) -> bool:
        """
        Strict DLC (Data Length Code) validation.
        
        Validates that:
        - DLC is within valid CAN range (0-8)
        - DLC matches actual data length
        - DLC meets rule-specific requirements
        
        Args:
            rule: Detection rule with DLC requirements
            message: CAN message to validate
            
        Returns:
            True if DLC is valid, False if violation detected
            
        Implementation: Phase 1, Parameter 1
        """
        if not rule.validate_dlc:
            return True
            
        dlc = message['dlc']
        
        # CAN 2.0 specification: DLC must be 0-8
        if dlc < 0 or dlc > 8:
            logger.debug(f"DLC violation: DLC={dlc} out of valid range [0-8]")
            return False
            
        # Validate data length matches DLC
        actual_length = len(message['data'])
        if actual_length != dlc:
            logger.debug(f"DLC mismatch: DLC={dlc} but data length={actual_length}")
            return False
            
        # Check against rule-specific DLC range if specified
        if rule.dlc_min is not None and dlc < rule.dlc_min:
            logger.debug(f"DLC below minimum: DLC={dlc} < min={rule.dlc_min}")
            return False
            
        if rule.dlc_max is not None and dlc > rule.dlc_max:
            logger.debug(f"DLC above maximum: DLC={dlc} > max={rule.dlc_max}")
            return False
            
        return True
    
    def _check_frame_format(self, message: Dict[str, Any]) -> bool:
        """
        Validate CAN frame format and structure.
        
        Checks for:
        - Valid CAN ID range (standard: 11-bit, extended: 29-bit)
        - Proper DLC range
        - Data field consistency
        - Error frame detection
        
        Args:
            message: CAN message to validate
            
        Returns:
            True if frame format is valid, False if malformed
            
        Implementation: Phase 1, Parameter 2
        """
        can_id = message['can_id']
        is_extended = message.get('is_extended', False)
        dlc = message['dlc']
        data = message['data']
        
        # Validate CAN ID range based on frame type
        if not is_extended:
            # Standard CAN: 11-bit ID (0x000 - 0x7FF)
            if can_id > 0x7FF:
                logger.debug(f"Standard frame CAN ID out of range: 0x{can_id:X} > 0x7FF")
                return False
        else:
            # Extended CAN: 29-bit ID (0x00000000 - 0x1FFFFFFF)
            if can_id > 0x1FFFFFFF:
                logger.debug(f"Extended frame CAN ID out of range: 0x{can_id:X} > 0x1FFFFFFF")
                return False
        
        # Validate DLC range (0-8 for CAN 2.0)
        if dlc < 0 or dlc > 8:
            logger.debug(f"Invalid DLC in frame: {dlc}")
            return False
        
        # Validate data field length matches DLC
        if len(data) != dlc:
            logger.debug(f"Data length mismatch: DLC={dlc}, data_len={len(data)}")
            return False
        
        # Check for error frames
        if message.get('is_error', False):
            logger.debug(f"Error frame detected for CAN ID 0x{can_id:X}")
            return False
        
        # Check for remote frames (RTR)
        # Remote frames should have DLC but no data
        is_remote = message.get('is_remote', False)
        if is_remote and len(data) > 0:
            logger.debug(f"Remote frame with data: CAN ID 0x{can_id:X}")
            return False
        
        return True
    
    def _check_global_message_rate(self, rule: DetectionRule, timestamp: float) -> bool:
        """
        Check if global message rate exceeds threshold.
        
        Monitors total message rate across all CAN IDs to detect
        bus flooding attacks (DoS).
        
        Args:
            rule: Detection rule with global rate threshold
            timestamp: Current message timestamp
            
        Returns:
            True if rate threshold exceeded (attack detected)
            False if rate is normal
            
        Implementation: Phase 1, Parameter 3
        """
        if not rule.global_message_rate or not rule.time_window:
            return False
        
        # Add current timestamp to global tracking
        self._global_message_times.append(timestamp)
        
        # Count messages within the time window
        cutoff_time = timestamp - rule.time_window
        message_count = sum(1 for t in self._global_message_times if t >= cutoff_time)
        
        # Check if rate exceeded
        if message_count > rule.global_message_rate:
            logger.debug(f"Global message rate exceeded: {message_count} > {rule.global_message_rate} in {rule.time_window}s")
            return True
        
        return False
    
    # ========================================================================
    # END PHASE 1 METHODS
    # ========================================================================
    
    # ========================================================================
    # PHASE 2 IMPORTANT VALIDATION METHODS (Dec 2, 2025)
    # Implementing 3 important rule parameters for production deployment
    # ========================================================================
    
    def _validate_source_enhanced(self, rule: DetectionRule, message: Dict[str, Any]) -> bool:
        """
        Enhanced source validation for diagnostic requests.
        
        Validates that diagnostic messages (OBD-II, UDS) come from
        authorized sources only. Prevents unauthorized diagnostic access.
        
        Common diagnostic CAN IDs:
        - 0x7DF: Broadcast diagnostic request
        - 0x7E0-0x7E7: Physical diagnostic requests
        - 0x7E8-0x7EF: Diagnostic responses
        
        Args:
            rule: Detection rule with allowed_sources list
            message: CAN message to validate
            
        Returns:
            True if source is valid, False if unauthorized source detected
            
        Implementation: Phase 2, Parameter 1
        """
        if not rule.check_source:
            return True
            
        can_id = message['can_id']
        
        # Check if this is a diagnostic message
        is_diagnostic = (
            can_id == 0x7DF or  # Broadcast diagnostic
            (0x7E0 <= can_id <= 0x7E7) or  # Physical diagnostic request
            (0x7E8 <= can_id <= 0x7EF)     # Diagnostic response
        )
        
        if not is_diagnostic:
            return True  # Not a diagnostic message, source check not applicable
        
        # Get source identifier from message (if available)
        # In CAN, source can be derived from arbitration ID or data field
        source_id = message.get('source_id', None)
        
        # If no source_id in message, try to extract from CAN ID
        # For diagnostic messages, source is typically in the lower nibble
        if source_id is None:
            if 0x7E0 <= can_id <= 0x7E7:
                source_id = can_id - 0x7E0  # Extract ECU number
            elif 0x7E8 <= can_id <= 0x7EF:
                source_id = can_id - 0x7E8  # Extract ECU number
        
        # Track sources seen for this CAN ID
        self._source_tracking[can_id].add(source_id)
        
        # Check against allowed sources if specified
        if rule.allowed_sources:
            if source_id not in rule.allowed_sources:
                logger.debug(f"Unauthorized diagnostic source: ID={source_id}, CAN=0x{can_id:X}, allowed={rule.allowed_sources}")
                return False
        
        # Additional check: too many different sources for one CAN ID is suspicious
        if len(self._source_tracking[can_id]) > 3:
            logger.debug(f"Too many sources for CAN ID 0x{can_id:X}: {len(self._source_tracking[can_id])} sources")
            return False
        
        return True
    
    def _check_replay_attack(self, rule: DetectionRule, message: Dict[str, Any]) -> bool:
        """
        Detect replay attacks by identifying identical message patterns.
        
        Replay attacks involve capturing and retransmitting legitimate
        messages. Detection looks for:
        - Exact data field matches
        - Messages repeated within a suspicious time window
        - Unusual repetition patterns
        
        Args:
            rule: Detection rule with replay detection enabled
            message: CAN message to check
            
        Returns:
            True if replay attack detected
            False if message appears legitimate
            
        Implementation: Phase 2, Parameter 2
        """
        if not rule.check_replay:
            return False
        
        can_id = message['can_id']
        data = tuple(message['data'])  # Convert to tuple for hashing
        timestamp = message['timestamp']
        
        # Create message signature (CAN ID + data)
        signature = (can_id, data)
        
        # Get previous occurrence of this signature
        sig_info = self._message_signatures[signature]
        
        if sig_info['data'] is not None:
            # We've seen this exact message before
            time_since_last = timestamp - sig_info['timestamp']
            sig_info['count'] += 1
            
            # Check if replayed within suspicious time window
            replay_threshold = rule.replay_time_threshold if rule.replay_time_threshold else 1.0
            
            if time_since_last < replay_threshold:
                # Same message within suspicious time window
                logger.debug(f"Potential replay: CAN 0x{can_id:X}, data={data}, time_delta={time_since_last:.3f}s, count={sig_info['count']}")
                
                # Multiple rapid replays are highly suspicious
                if sig_info['count'] >= 3:
                    logger.debug(f"Replay attack confirmed: {sig_info['count']} identical messages in {replay_threshold}s")
                    return True
                
                # Single replay within very short time is suspicious for critical messages
                if time_since_last < 0.1:  # 100ms threshold for exact replays
                    logger.debug(f"Rapid replay detected: {time_since_last*1000:.1f}ms")
                    return True
        
        # Update signature tracking
        self._message_signatures[signature] = {
            'data': data,
            'timestamp': timestamp,
            'count': sig_info['count']
        }
        
        return False
    
    def _validate_data_bytes(self, rule: DetectionRule, message: Dict[str, Any]) -> bool:
        """
        Validate specific data byte values.
        
        Checks if specified data bytes match expected values. Used for:
        - Emergency brake override detection (specific byte patterns)
        - Command validation (expected byte values)
        - Safety-critical message validation
        
        Args:
            rule: Detection rule with data_byte_X specifications
            message: CAN message to validate
            
        Returns:
            True if all specified bytes match expected values
            False if any byte mismatch detected (violation)
            
        Implementation: Phase 2, Parameter 3
        """
        data = message['data']
        
        # Check each data byte that's specified in the rule
        byte_checks = [
            (0, rule.data_byte_0),
            (1, rule.data_byte_1),
            (2, rule.data_byte_2),
            (3, rule.data_byte_3),
            (4, rule.data_byte_4),
            (5, rule.data_byte_5),
            (6, rule.data_byte_6),
            (7, rule.data_byte_7),
        ]
        
        for byte_index, expected_value in byte_checks:
            if expected_value is not None:
                # Check if data has enough bytes
                if byte_index >= len(data):
                    logger.debug(f"Data byte {byte_index} check failed: insufficient data length {len(data)}")
                    return False
                
                # Check if byte matches expected value
                actual_value = data[byte_index]
                if actual_value != expected_value:
                    logger.debug(f"Data byte {byte_index} mismatch: expected=0x{expected_value:02X}, actual=0x{actual_value:02X}")
                    return False
        
        return True
    
    # ========================================================================
    # END PHASE 2 METHODS
    # ========================================================================
    
    # ========================================================================
    # PHASE 3 SPECIALIZED VALIDATION METHODS (Dec 2, 2025)
    # Implementing 4 specialized rule parameters for vehicle-specific protection
    # ========================================================================
    
    def _check_data_integrity(self, rule: DetectionRule, message: Dict[str, Any]) -> bool:
        """
        Validate data integrity for critical safety systems.
        
        Checks data integrity using checksums, CRCs, or parity bits
        for safety-critical messages (brake, steering, airbag, etc).
        
        Common integrity checks:
        - XOR checksum
        - CRC-8/CRC-16
        - Simple parity
        - Rolling counter validation
        
        Args:
            rule: Detection rule with integrity checking enabled
            message: CAN message to validate
            
        Returns:
            True if data integrity is valid
            False if integrity violation detected
            
        Implementation: Phase 3, Parameter 1
        """
        if not rule.check_data_integrity:
            return True
        
        data = message['data']
        
        # Check if data has sufficient length
        if len(data) < 2:
            logger.debug(f"Insufficient data for integrity check: {len(data)} bytes")
            return False
        
        # Determine checksum offset (default: last byte)
        checksum_offset = rule.integrity_checksum_offset if rule.integrity_checksum_offset is not None else len(data) - 1
        
        if checksum_offset >= len(data):
            logger.debug(f"Invalid checksum offset: {checksum_offset} >= {len(data)}")
            return False
        
        # Extract checksum and data to validate
        expected_checksum = data[checksum_offset]
        data_to_check = data[:checksum_offset] + data[checksum_offset+1:]
        
        # Calculate XOR checksum (simple but effective)
        calculated_checksum = 0
        for byte in data_to_check:
            calculated_checksum ^= byte
        
        # Check if checksums match
        if calculated_checksum != expected_checksum:
            logger.debug(f"Data integrity failure: expected=0x{expected_checksum:02X}, calculated=0x{calculated_checksum:02X}")
            return False
        
        return True
    
    def _check_steering_range(self, rule: DetectionRule, message: Dict[str, Any]) -> bool:
        """
        Validate steering angle is within safe range.
        
        Checks that steering angle values are within physically
        possible and safe limits for the vehicle. Detects:
        - Impossible steering angles (beyond vehicle capability)
        - Dangerous steering commands
        - Steering sensor manipulation
        
        Typical steering ranges:
        - Standard vehicles: ±540° (1.5 turns)
        - Sports vehicles: ±720° (2 turns)
        - Trucks: ±360° (1 turn)
        
        Args:
            rule: Detection rule with steering range limits
            message: CAN message containing steering data
            
        Returns:
            True if steering angle is within valid range
            False if steering angle is out of range (violation)
            
        Implementation: Phase 3, Parameter 2
        """
        if not rule.check_steering_range:
            return True
        
        if rule.steering_min is None or rule.steering_max is None:
            return True  # Range not configured
        
        data = message['data']
        
        # Steering angle typically encoded in bytes 0-1 (16-bit value)
        # Format: signed integer, little-endian, 0.1 degree resolution
        if len(data) < 2:
            logger.debug(f"Insufficient data for steering check: {len(data)} bytes")
            return False
        
        # Extract 16-bit steering angle (little-endian)
        raw_value = data[0] | (data[1] << 8)
        
        # Convert to signed integer
        if raw_value & 0x8000:
            raw_value = raw_value - 0x10000
        
        # Convert to degrees (0.1 degree resolution)
        steering_angle = raw_value * 0.1
        
        # Check if within valid range
        if steering_angle < rule.steering_min or steering_angle > rule.steering_max:
            logger.debug(f"Steering angle out of range: {steering_angle:.1f}° not in [{rule.steering_min:.1f}°, {rule.steering_max:.1f}°]")
            return False
        
        return True
    
    def _check_repetition_pattern(self, rule: DetectionRule, message: Dict[str, Any]) -> bool:
        """
        Detect repetitive data patterns indicating potential attacks.
        
        Identifies suspicious repetitive patterns such as:
        - Same message repeated excessively
        - Stuck sensor values
        - Pattern-based DoS attacks
        - Fuzzing attempts
        
        Args:
            rule: Detection rule with repetition threshold
            message: CAN message to check
            
        Returns:
            True if repetition attack detected
            False if message pattern is normal
            
        Implementation: Phase 3, Parameter 3
        """
        if not rule.check_repetition:
            return False
        
        can_id = message['can_id']
        data = tuple(message['data'])
        
        # Get repetition tracking for this CAN ID
        rep_info = self._data_repetition_counts[can_id]
        
        if rep_info['data'] == data:
            # Same data as last time - increment counter
            rep_info['count'] += 1
        else:
            # Different data - reset counter
            rep_info['data'] = data
            rep_info['count'] = 1
        
        # Check if repetition threshold exceeded
        threshold = rule.repetition_threshold if rule.repetition_threshold else 10
        
        if rep_info['count'] > threshold:
            logger.debug(f"Repetition attack detected: CAN 0x{can_id:X}, data={data}, count={rep_info['count']} > {threshold}")
            return True
        
        return False
    
    def _validate_frame_type(self, rule: DetectionRule, message: Dict[str, Any]) -> bool:
        """
        Validate CAN frame type (standard vs extended).
        
        Ensures that messages use the expected frame type.
        Some attacks involve switching between standard and
        extended frame formats to bypass filters.
        
        Frame types:
        - Standard: 11-bit CAN ID (0x000 - 0x7FF)
        - Extended: 29-bit CAN ID (0x00000000 - 0x1FFFFFFF)
        
        Args:
            rule: Detection rule with expected frame type
            message: CAN message to validate
            
        Returns:
            True if frame type matches expectation
            False if frame type violation detected
            
        Implementation: Phase 3, Parameter 4
        """
        if not rule.frame_type:
            return True
        
        is_extended = message.get('is_extended', False)
        can_id = message['can_id']
        
        # Determine expected frame type
        expected_extended = rule.frame_type.lower() == 'extended'
        
        # Check if frame type matches
        if is_extended != expected_extended:
            frame_type_str = "extended" if is_extended else "standard"
            expected_str = "extended" if expected_extended else "standard"
            logger.debug(f"Frame type mismatch: CAN 0x{can_id:X} is {frame_type_str}, expected {expected_str}")
            return False
        
        # Additional validation: ensure CAN ID is valid for frame type
        if not is_extended and can_id > 0x7FF:
            logger.debug(f"Standard frame with invalid CAN ID: 0x{can_id:X} > 0x7FF")
            return False
        
        if is_extended and can_id > 0x1FFFFFFF:
            logger.debug(f"Extended frame with invalid CAN ID: 0x{can_id:X} > 0x1FFFFFFF")
            return False
        
        return True
    
    # ========================================================================
    # END PHASE 3 METHODS
    # ========================================================================
        
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