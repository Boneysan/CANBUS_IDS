"""
Fast pre-filter for CAN messages based on academic research.

Research basis: Edge gateway pattern - fast screening before expensive analysis.
Expected improvement: Filters 80-95% of normal traffic in <0.1ms per message.

References:
- "Raspberry Pi IDS — A Fruitful Intrusion Detection System for IoT" (IEEE)
- Multi-stage detection architectures from academic literature
"""

import logging
import time
from typing import Dict, List, Tuple, Set, Any
from collections import defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PreFilterStats:
    """Statistics for pre-filter performance."""
    messages_processed: int = 0
    messages_passed: int = 0
    messages_flagged: int = 0
    total_time_ms: float = 0.0
    
    @property
    def pass_rate(self) -> float:
        """Percentage of messages that passed pre-filter."""
        if self.messages_processed == 0:
            return 0.0
        return (self.messages_passed / self.messages_processed) * 100.0
    
    @property
    def avg_time_us(self) -> float:
        """Average processing time per message in microseconds."""
        if self.messages_processed == 0:
            return 0.0
        return (self.total_time_ms * 1000) / self.messages_processed


class FastPreFilter:
    """
    Ultra-fast pre-filter for CAN messages.
    
    Implements edge gateway pattern: quickly filter benign traffic
    before expensive rule/ML analysis.
    
    Performance targets:
    - <0.1ms per message processing time
    - 80-95% of normal traffic filtered as PASS
    - 5-20% flagged for deep analysis
    """
    
    def __init__(self, known_good_ids: Set[int], 
                 timing_tolerance: float = 0.3,
                 enable_stats: bool = True):
        """
        Initialize fast pre-filter.
        
        Args:
            known_good_ids: Set of CAN IDs known to be legitimate
            timing_tolerance: Tolerance for timing deviations (0.3 = ±30%)
            enable_stats: Enable performance statistics tracking
        """
        self.known_good_ids = known_good_ids
        self.timing_tolerance = timing_tolerance
        self.enable_stats = enable_stats
        
        # Timing tracking (lightweight)
        self._timing_cache = {}  # {can_id: last_timestamp}
        self._expected_intervals = {}  # {can_id: expected_interval}
        
        # Statistics
        self.stats = PreFilterStats()
        
        logger.info(f"Fast pre-filter initialized with {len(known_good_ids)} known good IDs")
    
    def filter_batch(self, messages: List[Dict[str, Any]]) -> Tuple[List, List]:
        """
        Filter a batch of messages into PASS and SUSPICIOUS categories.
        
        Research basis: Edge gateway pattern - fast triage before deep analysis.
        
        Args:
            messages: List of CAN message dictionaries
            
        Returns:
            (pass_messages, suspicious_messages) tuple
        """
        start_time = time.time()
        
        pass_msgs = []
        suspicious_msgs = []
        
        for msg in messages:
            if self._is_likely_benign(msg):
                pass_msgs.append(msg)
                if self.enable_stats:
                    self.stats.messages_passed += 1
            else:
                suspicious_msgs.append(msg)
                if self.enable_stats:
                    self.stats.messages_flagged += 1
        
        # Update statistics
        if self.enable_stats:
            duration_ms = (time.time() - start_time) * 1000
            self.stats.messages_processed += len(messages)
            self.stats.total_time_ms += duration_ms
        
        return pass_msgs, suspicious_msgs
    
    def _is_likely_benign(self, msg: Dict[str, Any]) -> bool:
        """
        Fast benign check using simple heuristics.
        
        Checks (in order of speed):
        1. Known good CAN ID (hash lookup: O(1))
        2. Timing within expected range (simple arithmetic)
        
        Returns:
            True if likely benign, False if suspicious
        """
        can_id = msg['can_id']
        timestamp = msg['timestamp']
        
        # Check 1: Is this a known good ID?
        if can_id not in self.known_good_ids:
            return False  # Unknown ID = suspicious
        
        # Check 2: Timing check (no statistics, just range)
        if can_id in self._timing_cache:
            last_time = self._timing_cache[can_id]
            interval = timestamp - last_time
            
            # Get or learn expected interval
            if can_id in self._expected_intervals:
                expected = self._expected_intervals[can_id]
                
                # Simple range check (no std dev calculation)
                min_interval = expected * (1.0 - self.timing_tolerance)
                max_interval = expected * (1.0 + self.timing_tolerance)
                
                if min_interval <= interval <= max_interval:
                    self._timing_cache[can_id] = timestamp
                    return True  # Timing looks good
                else:
                    self._timing_cache[can_id] = timestamp
                    return False  # Timing anomaly
            else:
                # Learning mode: establish baseline
                self._expected_intervals[can_id] = interval
                self._timing_cache[can_id] = timestamp
                return True  # Learning phase - assume benign
        else:
            # First message for this ID
            self._timing_cache[can_id] = timestamp
            return True  # First message - assume benign
        
        return False  # Default: suspicious
    
    def calibrate(self, normal_messages: List[Dict[str, Any]]) -> None:
        """
        Calibrate pre-filter using normal traffic baseline.
        
        Args:
            normal_messages: List of known-normal CAN messages
        """
        logger.info(f"Calibrating pre-filter with {len(normal_messages)} normal messages")
        
        # Learn intervals for each CAN ID
        intervals_by_id = defaultdict(list)
        
        for msg in normal_messages:
            can_id = msg['can_id']
            timestamp = msg['timestamp']
            
            if can_id in self._timing_cache:
                interval = timestamp - self._timing_cache[can_id]
                intervals_by_id[can_id].append(interval)
            
            self._timing_cache[can_id] = timestamp
        
        # Calculate median interval for each ID
        for can_id, intervals in intervals_by_id.items():
            if intervals:
                intervals.sort()
                median_idx = len(intervals) // 2
                self._expected_intervals[can_id] = intervals[median_idx]
        
        logger.info(f"Calibrated {len(self._expected_intervals)} CAN IDs")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pre-filter performance statistics."""
        return {
            'messages_processed': self.stats.messages_processed,
            'messages_passed': self.stats.messages_passed,
            'messages_flagged': self.stats.messages_flagged,
            'pass_rate_percent': self.stats.pass_rate,
            'avg_time_microseconds': self.stats.avg_time_us,
            'known_good_ids': len(self.known_good_ids),
            'learned_intervals': len(self._expected_intervals)
        }
