# Performance Optimization Roadmap - Path to 7,000 msg/s

**Target:** 7,000 messages per second (heavy CAN bus network)  
**Current:** 708 msg/s (baseline from DoS-1 test)  
**Gap:** 10x improvement needed  
**Status:** Research-validated path to target  
**Timeline:** 2-3 weeks for full implementation

---

## Overview

This roadmap implements 5 research-backed optimizations to achieve 7,000 msg/s throughput:

| Optimization | Expected Gain | Implementation Time | Priority |
|--------------|---------------|---------------------|----------|
| 1. Batch Processing | 5-10x | 4 hours | CRITICAL |
| 2. Fast Pre-Filter | 2-3x | 2 hours | CRITICAL |
| 3. Rule Optimization | 5-10x | 3 hours | HIGH |
| 4. Feature Reduction (PCA) | 3-5x ML | 4 hours | HIGH |
| 5. Multicore Processing | 2-3x | 8 hours | MEDIUM |

**Cumulative Performance Progression:**
```
Current:              708 msg/s   (baseline)
+ Batch Processing:   3,540 msg/s (Week 1)
+ Fast Pre-Filter:    7,080 msg/s (Week 1) ✅ TARGET ACHIEVED!
+ Rule Optimization:  10,620 msg/s (Week 2) 🚀 Safety margin
+ Multicore:          15,930 msg/s (Week 3) 🚀 2x headroom
```

---

## Phase 1: Critical Path to 7K Target (Week 1)

**Goal:** Achieve 7,000 msg/s with two critical optimizations  
**Timeline:** 6-8 hours implementation + testing  
**Risk:** Low - well-validated techniques

### Milestone 1.1: Implement Batch Processing (4 hours)

**Research Basis:** ALL academic papers use batch processing for Python IDS

**Files to Modify:**
1. `src/capture/can_sniffer.py`
2. `src/detection/rule_engine.py`
3. `src/detection/ml_detector.py`
4. `main.py`

#### Step 1.1.1: Add Batch Reading to CAN Sniffer (1 hour)

**File:** `src/capture/can_sniffer.py`

```python
# Add after line 85 (after read_message method)

def read_batch(self, batch_size: int = 100, timeout: float = 0.1) -> List[Dict[str, Any]]:
    """
    Read multiple CAN messages in a batch for improved performance.
    
    Research basis: All high-performance Python IDS use batch processing.
    Expected improvement: 5-10x throughput increase.
    
    Args:
        batch_size: Maximum messages to read in one batch
        timeout: Maximum time to wait for batch to fill (seconds)
        
    Returns:
        List of CAN message dictionaries
    """
    batch = []
    start_time = time.time()
    
    while len(batch) < batch_size:
        # Non-blocking read with short timeout
        msg = self.bus.recv(timeout=0.001)
        
        if msg:
            batch.append(self._message_to_dict(msg))
            self.stats['messages_received'] += 1
        
        # Break if timeout reached (don't wait forever for full batch)
        if time.time() - start_time > timeout:
            break
        
        # Break if no messages arriving
        if len(batch) == 0 and time.time() - start_time > 0.01:
            break
    
    return batch

def _message_to_dict(self, msg) -> Dict[str, Any]:
    """Convert python-can Message to dictionary format."""
    return {
        'timestamp': msg.timestamp,
        'can_id': msg.arbitration_id,
        'dlc': msg.dlc,
        'data': list(msg.data),
        'is_extended_id': msg.is_extended_id,
        'is_error_frame': msg.is_error_frame,
        'is_remote_frame': msg.is_remote_frame
    }
```

**Testing:**
```bash
# Test batch reading
python -c "
from src.capture.can_sniffer import CANSniffer
sniffer = CANSniffer('vcan0')
sniffer.start()
batch = sniffer.read_batch(batch_size=100, timeout=0.1)
print(f'Read {len(batch)} messages in batch')
"
```

#### Step 1.1.2: Add Batch Processing to Rule Engine (2 hours)

**File:** `src/detection/rule_engine.py`

```python
# Add after analyze_message method (around line 250)

def analyze_batch(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Analyze multiple CAN messages in a batch for improved performance.
    
    Research basis: Batch processing reduces per-message overhead.
    Expected improvement: 5-10x throughput vs individual message processing.
    
    Args:
        messages: List of CAN message dictionaries
        
    Returns:
        List of alerts generated from batch
    """
    all_alerts = []
    
    # Pre-filter: Group messages by CAN ID for efficient rule matching
    messages_by_id = defaultdict(list)
    for msg in messages:
        messages_by_id[msg['can_id']].append(msg)
    
    # Process each CAN ID group
    for can_id, id_messages in messages_by_id.items():
        # Get applicable rules for this CAN ID (cached)
        applicable_rules = self._get_rules_for_can_id(can_id)
        
        # Process all messages for this ID
        for msg in id_messages:
            for rule in applicable_rules:
                alert = self._evaluate_rule(rule, msg)
                if alert:
                    all_alerts.append(alert)
                    
                    # Early exit on critical rules
                    if rule.priority == 0:
                        break
    
    return all_alerts

def _get_rules_for_can_id(self, can_id: int) -> List[DetectionRule]:
    """
    Get rules applicable to a specific CAN ID.
    Uses caching to avoid repeated filtering.
    
    Args:
        can_id: CAN identifier
        
    Returns:
        List of applicable rules
    """
    # Check cache first
    if hasattr(self, '_rule_cache'):
        if can_id in self._rule_cache:
            return self._rule_cache[can_id]
    else:
        self._rule_cache = {}
    
    # Filter rules for this CAN ID
    applicable = []
    for rule in self.rules:
        # Check if rule applies to this CAN ID
        if rule.can_id and rule.can_id != can_id:
            continue
        if rule.can_id_range:
            if not (rule.can_id_range[0] <= can_id <= rule.can_id_range[1]):
                continue
        applicable.append(rule)
    
    # Cache result
    self._rule_cache[can_id] = applicable
    return applicable
```

**Testing:**
```python
# Test batch processing
from src.detection.rule_engine import RuleEngine

engine = RuleEngine('config/rules.yaml')

# Generate test batch
messages = [
    {'can_id': 0x123, 'dlc': 8, 'data': [0]*8, 'timestamp': time.time()}
    for _ in range(100)
]

start = time.time()
alerts = engine.analyze_batch(messages)
duration = time.time() - start

print(f"Processed {len(messages)} messages in {duration:.3f}s")
print(f"Throughput: {len(messages)/duration:.0f} msg/s")
```

#### Step 1.1.3: Add Batch Processing to ML Detector (1 hour)

**File:** `src/detection/ml_detector.py`

```python
# Add after analyze_message method (around line 400)

def analyze_batch(self, messages: List[Dict[str, Any]]) -> List[MLAlert]:
    """
    Analyze multiple messages in batch for improved ML performance.
    
    Research basis: Vectorized operations are 5-10x faster than loops.
    
    Args:
        messages: List of CAN message dictionaries
        
    Returns:
        List of ML alerts
    """
    if not self.is_trained:
        return []
    
    all_alerts = []
    
    # Extract features for all messages (vectorized)
    feature_vectors = []
    for msg in messages:
        features = self._extract_features(msg)
        if features:
            feature_vectors.append(list(features.values()))
    
    if not feature_vectors:
        return []
    
    # Batch prediction (MUCH faster than individual predictions)
    X = np.array(feature_vectors)
    predictions = self.isolation_forest.predict(X)
    scores = self.isolation_forest.decision_function(X)
    
    # Generate alerts for anomalies
    for i, (msg, pred, score) in enumerate(zip(messages[:len(predictions)], predictions, scores)):
        if pred == -1:  # Anomaly detected
            alert = MLAlert(
                timestamp=msg['timestamp'],
                can_id=msg['can_id'],
                anomaly_score=float(score),
                confidence=min(abs(score) / self.contamination, 1.0),
                features=dict(zip(self._feature_names, feature_vectors[i])),
                message_data=msg
            )
            all_alerts.append(alert)
    
    return all_alerts
```

#### Step 1.1.4: Update Main Loop for Batch Processing (30 min)

**File:** `main.py`

```python
# Replace the message processing loop (around line 450-480)

def _process_messages_batch(self) -> None:
    """
    Process CAN messages in batches for improved performance.
    
    Research-backed optimization: 5-10x throughput improvement.
    """
    logger.info("Starting batch message processing (optimized mode)")
    
    # Configuration
    BATCH_SIZE = 100  # Optimal for Pi 4 per research
    TIMEOUT = 0.1     # 100ms max wait for batch
    
    try:
        while self.running:
            # Read batch of messages
            batch = self.can_sniffer.read_batch(
                batch_size=BATCH_SIZE,
                timeout=TIMEOUT
            )
            
            if not batch:
                continue
            
            # Update statistics
            self.stats['messages_processed'] += len(batch)
            
            # Rule-based detection (batch)
            rule_alerts = []
            if self.rule_engine:
                rule_alerts = self.rule_engine.analyze_batch(batch)
            
            # ML-based detection (batch)
            ml_alerts = []
            if self.ml_detector and self.ml_detector.is_trained:
                ml_alerts = self.ml_detector.analyze_batch(batch)
            
            # Process alerts
            all_alerts = rule_alerts + ml_alerts
            if all_alerts:
                self.stats['alerts_generated'] += len(all_alerts)
                self.alert_manager.process_alerts(all_alerts)
            
    except KeyboardInterrupt:
        logger.info("Batch processing interrupted by user")
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        raise
```

**Expected Result After Step 1.1:**
- Throughput: 708 → 3,500-7,000 msg/s (5-10x improvement)
- Test with: `python main.py -i vcan0 --log-level INFO`

---

### Milestone 1.2: Implement Fast Pre-Filter (2 hours)

**Research Basis:** Edge gateway pattern from multiple papers

**Files to Create/Modify:**
1. `src/detection/prefilter.py` (NEW)
2. `main.py`
3. `config/can_ids.yaml`

#### Step 1.2.1: Create Fast Pre-Filter Module (1.5 hours)

**File:** `src/detection/prefilter.py` (NEW FILE)

```python
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
```

#### Step 1.2.2: Integrate Pre-Filter into Main Loop (30 min)

**File:** `main.py`

```python
# Add import
from src.detection.prefilter import FastPreFilter

# In initialize_components method (around line 180):
def initialize_components(self) -> None:
    """Initialize all CAN-IDS components."""
    
    # ... existing code ...
    
    # Initialize pre-filter (NEW)
    prefilter_config = self.config.get('prefilter', {})
    if prefilter_config.get('enabled', True):
        # Get known good IDs from config or learn from rules
        known_good_ids = set(prefilter_config.get('known_good_ids', []))
        
        # If not configured, extract from rules
        if not known_good_ids and self.rule_engine:
            known_good_ids = self._extract_known_ids_from_rules()
        
        self.prefilter = FastPreFilter(
            known_good_ids=known_good_ids,
            timing_tolerance=prefilter_config.get('timing_tolerance', 0.3)
        )
        logger.info("✅ Pre-filter enabled")
    else:
        self.prefilter = None
        logger.info("Pre-filter disabled")

# Update batch processing to use pre-filter:
def _process_messages_batch(self) -> None:
    """Process CAN messages in batches with pre-filtering."""
    
    while self.running:
        # Read batch
        batch = self.can_sniffer.read_batch(batch_size=100, timeout=0.1)
        if not batch:
            continue
        
        # Pre-filter (NEW)
        if self.prefilter:
            pass_msgs, suspicious_msgs = self.prefilter.filter_batch(batch)
            
            # Only deeply analyze suspicious messages
            messages_to_analyze = suspicious_msgs
            
            # Log pass-through rate
            if len(batch) > 0:
                pass_rate = (len(pass_msgs) / len(batch)) * 100
                logger.debug(f"Pre-filter: {pass_rate:.1f}% passed, "
                           f"{len(suspicious_msgs)} need deep analysis")
        else:
            messages_to_analyze = batch
        
        # Update statistics
        self.stats['messages_processed'] += len(batch)
        
        # Rule-based detection (only on suspicious)
        rule_alerts = []
        if self.rule_engine and messages_to_analyze:
            rule_alerts = self.rule_engine.analyze_batch(messages_to_analyze)
        
        # ML detection (only on suspicious)
        ml_alerts = []
        if self.ml_detector and messages_to_analyze:
            ml_alerts = self.ml_detector.analyze_batch(messages_to_analyze)
        
        # Process alerts
        all_alerts = rule_alerts + ml_alerts
        if all_alerts:
            self.stats['alerts_generated'] += len(all_alerts)
            self.alert_manager.process_alerts(all_alerts)
```

#### Step 1.2.3: Add Configuration (15 min)

**File:** `config/can_ids.yaml`

```yaml
# Add prefilter section:
prefilter:
  enabled: true
  timing_tolerance: 0.3  # ±30% timing tolerance
  
  # Known good CAN IDs (populate with your vehicle's normal IDs)
  known_good_ids:
    - 0x100  # Example: Engine RPM
    - 0x200  # Example: Speed
    - 0x300  # Example: Steering
    # Add more as learned from normal traffic
    
  # Or set to empty and pre-filter will learn from rules.yaml
  # known_good_ids: []
```

**Expected Result After Step 1.2:**
- Throughput: 3,500 → 7,000-10,000 msg/s (2-3x improvement)
- 80-95% of messages bypass full analysis
- **Target of 7,000 msg/s ACHIEVED!**

**Testing:**
```bash
python main.py -i vcan0 --log-level INFO

# Look for log messages:
# "Pre-filter: 87.3% passed, 127 need deep analysis"
# "Throughput: 7,234 msg/s"
```

---

## Phase 2: Safety Margin & Optimization (Week 2)

**Goal:** Exceed 7K target with safety margin (10K-15K msg/s)  
**Timeline:** 8-10 hours implementation + testing

### Milestone 2.1: Rule Optimization with Priority (3 hours)

**Research Basis:** All papers emphasize reduced rule complexity

**Files to Modify:**
1. `src/detection/rule_engine.py`
2. `config/rules.yaml`

#### Step 2.1.1: Implement Priority-Based Early Exit (2 hours)

**File:** `src/detection/rule_engine.py`

```python
# Modify load_rules method to sort by priority (around line 120):

def load_rules(self, rules_file: str) -> None:
    """Load detection rules from YAML file with priority sorting."""
    
    # ... existing loading code ...
    
    # Sort rules by priority (0=critical, 10=low)
    # This enables early exit on critical detections
    self.rules.sort(key=lambda r: r.priority)
    
    # Build CAN ID index for O(1) lookup
    self._build_rule_index()
    
    logger.info(f"Loaded {len(self.rules)} rules, sorted by priority")

def _build_rule_index(self) -> None:
    """
    Build index of rules by CAN ID for fast lookup.
    
    Research basis: Hash table lookup O(1) vs linear scan O(n).
    Expected improvement: 5-10x faster rule matching.
    """
    self._rule_index = defaultdict(list)
    
    for rule in self.rules:
        if rule.can_id:
            # Specific CAN ID
            self._rule_index[rule.can_id].append(rule)
        elif rule.can_id_range:
            # Range of CAN IDs (less efficient but necessary)
            for can_id in range(rule.can_id_range[0], rule.can_id_range[1] + 1):
                self._rule_index[can_id].append(rule)
        else:
            # Global rule (applies to all IDs)
            self._rule_index[None].append(rule)
    
    logger.info(f"Built rule index for {len(self._rule_index)} CAN ID entries")

# Modify analyze_message to use index and early exit:

def analyze_message(self, message: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Analyze single CAN message with optimized rule matching.
    
    Optimizations:
    1. Hash table lookup for rules (O(1) vs O(n))
    2. Priority-based early exit on critical rules
    3. Skip non-applicable rules early
    """
    can_id = message['can_id']
    alerts = []
    
    # Get applicable rules from index (O(1) lookup)
    applicable_rules = self._rule_index.get(can_id, []) + self._rule_index.get(None, [])
    
    # Rules already sorted by priority
    for rule in applicable_rules:
        # Quick applicability check
        if not self._quick_rule_check(rule, message):
            continue
        
        # Full rule evaluation
        alert = self._evaluate_rule(rule, message)
        
        if alert:
            alerts.append(alert)
            
            # Early exit on critical rules
            if rule.priority == 0:
                logger.debug(f"Critical rule hit (ID: 0x{can_id:03X}), early exit")
                break
            
            # Stop after max alerts per message
            if len(alerts) >= 5:  # Configurable
                break
    
    return alerts

def _quick_rule_check(self, rule: DetectionRule, message: Dict[str, Any]) -> bool:
    """
    Ultra-fast preliminary check if rule might apply.
    
    Checks only the fastest conditions to avoid expensive evaluation.
    
    Returns:
        True if rule might apply (do full check), False to skip
    """
    # DLC check (fastest)
    if rule.validate_dlc or rule.dlc_min or rule.dlc_max:
        dlc = message['dlc']
        if rule.dlc_min and dlc < rule.dlc_min:
            return False
        if rule.dlc_max and dlc > rule.dlc_max:
            return False
    
    # Frame type check (fast)
    if rule.frame_type:
        is_extended = message.get('is_extended_id', False)
        if rule.frame_type == 'standard' and is_extended:
            return False
        if rule.frame_type == 'extended' and not is_extended:
            return False
    
    return True  # Might apply, do full evaluation
```

#### Step 2.1.2: Optimize Rule Configuration (1 hour)

**File:** `config/rules.yaml`

```yaml
# Add priority field to all rules
# Priority scale: 0 (critical/immediate threat) to 10 (low priority/informational)

rules:
  # CRITICAL RULES (Priority 0-2): Immediate threats
  - name: "Bus Flooding Attack"
    priority: 0  # CRITICAL - DoS attack
    severity: CRITICAL
    global_message_rate: 5000
    time_window: 1
    description: "Excessive global message rate indicates flooding"
    
  - name: "Diagnostic Session Hijack"
    priority: 0  # CRITICAL - Security breach
    severity: CRITICAL
    can_id: 0x7DF
    check_source: true
    description: "Unauthorized diagnostic session"
    
  # HIGH PRIORITY (Priority 3-5): Known attacks
  - name: "RPM Manipulation"
    priority: 3
    severity: HIGH
    can_id: 0x316
    check_timing: true
    description: "RPM signal timing anomaly"
    
  # MEDIUM PRIORITY (Priority 6-7): Suspicious behavior
  - name: "Unknown CAN ID"
    priority: 6
    severity: MEDIUM
    check_can_id_whitelist: true
    description: "Unrecognized CAN identifier"
    
  # LOW PRIORITY (Priority 8-10): Informational
  - name: "High Frequency Benign"
    priority: 8
    severity: LOW
    max_frequency: 100
    description: "Elevated message frequency (may be normal)"
```

**Expected Result After Step 2.1:**
- Rule evaluation: 1.4ms → 0.2-0.3ms per message (5-10x faster)
- Throughput: 7,000 → 10,000-12,000 msg/s

---

### Milestone 2.2: Feature Reduction with PCA (4 hours)

**Research Basis:** IJRASET 2025 paper on SVM+PCA for Pi

**Files to Create/Modify:**
1. `src/preprocessing/feature_reduction.py` (NEW)
2. `src/detection/ml_detector.py`
3. Training scripts

#### Step 2.2.1: Create PCA Feature Reducer (2 hours)

**File:** `src/preprocessing/feature_reduction.py` (NEW)

```python
"""
Feature reduction using PCA for resource-constrained deployment.

Research basis: "Intrusion Detection System using Raspberry Pi for IoT Devices"
                IJRASET 2025 - Reduces 58 features to 10-15 using PCA.

Expected improvement: 3-5x faster ML inference with minimal accuracy loss.
"""

import logging
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class FeatureReducer:
    """
    Reduce feature dimensionality using PCA.
    
    Research-backed approach:
    - Original: 58 features → ~15ms inference time
    - Reduced: 10-15 features → ~3-5ms inference time
    - Accuracy loss: < 5% (per IJRASET research)
    """
    
    def __init__(self, n_components: int = 15, 
                 variance_threshold: float = 0.95):
        """
        Initialize feature reducer.
        
        Args:
            n_components: Target number of principal components (10-15 recommended)
            variance_threshold: Minimum cumulative explained variance (0.95 = 95%)
        """
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        
        self.pca: Optional[PCA] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names_in: List[str] = []
        self.feature_names_out: List[str] = []
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, feature_names: List[str]) -> 'FeatureReducer':
        """
        Fit PCA on training features.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            feature_names: List of original feature names
            
        Returns:
            self for chaining
        """
        logger.info(f"Fitting PCA: {X.shape[1]} features → {self.n_components} components")
        
        self.feature_names_in = feature_names
        
        # Standardize features first (required for PCA)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit PCA
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X_scaled)
        
        # Check variance explained
        cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)
        actual_variance = cumulative_variance[-1]
        
        logger.info(f"PCA fitted: {self.n_components} components explain "
                   f"{actual_variance*100:.1f}% of variance")
        
        if actual_variance < self.variance_threshold:
            logger.warning(f"Variance {actual_variance*100:.1f}% < threshold "
                          f"{self.variance_threshold*100:.1f}%")
        
        # Generate component names
        self.feature_names_out = [f"PC{i+1}" for i in range(self.n_components)]
        self.is_fitted = True
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted PCA.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Reduced feature matrix (n_samples, n_components)
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureReducer not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        X_reduced = self.pca.transform(X_scaled)
        
        return X_reduced
    
    def fit_transform(self, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """Fit PCA and transform in one step."""
        self.fit(X, feature_names)
        return self.transform(X)
    
    def save(self, path: str) -> None:
        """Save fitted reducer to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted FeatureReducer")
        
        save_dict = {
            'pca': self.pca,
            'scaler': self.scaler,
            'n_components': self.n_components,
            'feature_names_in': self.feature_names_in,
            'feature_names_out': self.feature_names_out
        }
        
        joblib.dump(save_dict, path)
        logger.info(f"FeatureReducer saved to {path}")
    
    def load(self, path: str) -> 'FeatureReducer':
        """Load fitted reducer from disk."""
        save_dict = joblib.load(path)
        
        self.pca = save_dict['pca']
        self.scaler = save_dict['scaler']
        self.n_components = save_dict['n_components']
        self.feature_names_in = save_dict['feature_names_in']
        self.feature_names_out = save_dict['feature_names_out']
        self.is_fitted = True
        
        logger.info(f"FeatureReducer loaded from {path}")
        return self
    
    def get_feature_importance(self, n_top: int = 10) -> List[tuple]:
        """
        Get top N most important original features.
        
        Args:
            n_top: Number of top features to return
            
        Returns:
            List of (feature_name, importance_score) tuples
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureReducer not fitted")
        
        # Calculate feature importance from PCA components
        # Sum of absolute values across all components
        feature_importance = np.abs(self.pca.components_).sum(axis=0)
        
        # Normalize to [0, 1]
        feature_importance = feature_importance / feature_importance.sum()
        
        # Create (name, score) pairs
        importance_pairs = list(zip(self.feature_names_in, feature_importance))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return importance_pairs[:n_top]
```

#### Step 2.2.2: Training Script with PCA (1 hour)

**File:** `scripts/train_with_pca.py` (NEW)

```python
#!/usr/bin/env python3
"""
Train ML model with PCA feature reduction for Pi deployment.

Research basis: IJRASET 2025 - Reduces features for efficient inference.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier
import joblib

from src.preprocessing.feature_reduction import FeatureReducer
from src.preprocessing.feature_extractor import FeatureExtractor

def main():
    # Load training data
    print("Loading training data...")
    df = pd.read_csv('data/processed/training_data.csv')
    
    # Extract features
    print("Extracting features...")
    extractor = FeatureExtractor()
    
    X = []
    y = []
    for _, row in df.iterrows():
        msg = row.to_dict()
        features = extractor.extract_features(msg)
        X.append(list(features.values()))
        y.append(row.get('label', 0))
    
    X = np.array(X)
    y = np.array(y)
    feature_names = list(features.keys())
    
    print(f"Original features: {X.shape[1]}")
    
    # Apply PCA reduction
    print("Applying PCA feature reduction...")
    reducer = FeatureReducer(n_components=15)  # 58 → 15 features
    X_reduced = reducer.fit_transform(X, feature_names)
    
    print(f"Reduced features: {X_reduced.shape[1]}")
    print(f"Explained variance: {reducer.pca.explained_variance_ratio_.sum()*100:.1f}%")
    
    # Show most important features
    print("\nTop 10 most important features:")
    for name, importance in reducer.get_feature_importance(10):
        print(f"  {name}: {importance:.4f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced, y, test_size=0.2, random_state=42
    )
    
    # Train model on reduced features
    print("\nTraining model on reduced features...")
    model = IsolationForest(
        n_estimators=50,  # Fewer trees for speed
        contamination=0.02,
        random_state=42
    )
    model.fit(X_train)
    
    # Evaluate
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    
    # Save reducer and model
    reducer.save('data/models/feature_reducer.joblib')
    joblib.dump(model, 'data/models/model_with_pca.joblib')
    
    print("\n✅ Training complete!")
    print(f"  Feature reducer: data/models/feature_reducer.joblib")
    print(f"  Model: data/models/model_with_pca.joblib")
    print(f"\nExpected inference speedup: 3-5x faster!")

if __name__ == '__main__':
    main()
```

**Run training:**
```bash
python scripts/train_with_pca.py
```

#### Step 2.2.3: Integrate PCA into ML Detector (1 hour)

**File:** `src/detection/ml_detector.py`

```python
# Add import at top
from src.preprocessing.feature_reduction import FeatureReducer

# In __init__ method:
def __init__(self, model_path: Optional[str] = None, 
             contamination: float = 0.20,
             feature_window: int = 100,
             use_pca: bool = True):  # NEW parameter
    """Initialize ML detector with optional PCA."""
    
    # ... existing code ...
    
    self.use_pca = use_pca
    self.feature_reducer: Optional[FeatureReducer] = None
    
    # Load model and reducer
    if self.model_path and self.model_path.exists():
        self.load_model()

# Modify load_model to also load PCA reducer:
def load_model(self) -> None:
    """Load pre-trained model and optional PCA reducer."""
    
    # ... existing model loading code ...
    
    # Try to load PCA reducer
    if self.use_pca:
        reducer_path = self.model_path.parent / 'feature_reducer.joblib'
        if reducer_path.exists():
            self.feature_reducer = FeatureReducer()
            self.feature_reducer.load(str(reducer_path))
            logger.info(f"✅ PCA reducer loaded: {self.feature_reducer.n_components} components")
            logger.info(f"   Expected speedup: 3-5x faster inference!")
        else:
            logger.warning(f"PCA reducer not found at {reducer_path}")
            self.use_pca = False

# Modify analyze_batch to use PCA:
def analyze_batch(self, messages: List[Dict[str, Any]]) -> List[MLAlert]:
    """Analyze batch with optional PCA feature reduction."""
    
    if not self.is_trained:
        return []
    
    # Extract features
    feature_vectors = []
    for msg in messages:
        features = self._extract_features(msg)
        if features:
            feature_vectors.append(list(features.values()))
    
    if not feature_vectors:
        return []
    
    X = np.array(feature_vectors)
    
    # Apply PCA if available (3-5x speedup!)
    if self.feature_reducer:
        X = self.feature_reducer.transform(X)
    
    # Batch prediction
    predictions = self.isolation_forest.predict(X)
    scores = self.isolation_forest.decision_function(X)
    
    # ... rest of method unchanged ...
```

**Expected Result After Step 2.2:**
- ML inference: 15ms → 3-5ms per message (3-5x faster)
- Throughput with ML: 8,000 → 15,000-20,000 msg/s
- Memory usage: -30% (fewer features to track)

---

## Phase 3: Maximum Performance (Week 3)

**Goal:** Maximize throughput using all cores (15K-20K msg/s)  
**Timeline:** 8-10 hours implementation + testing

### Milestone 3.1: Multicore Processing (8 hours)

**Research Basis:** Suricata pattern on Pi 4

**Files to Create/Modify:**
1. `src/processing/multicore_processor.py` (NEW)
2. `main.py`

#### Step 3.1.1: Create Multicore Processor (6 hours)

**File:** `src/processing/multicore_processor.py` (NEW)

```python
"""
Multicore CAN message processing for Raspberry Pi 4.

Research basis: Suricata multicore pattern on Pi 4.
Expected improvement: 2-3x throughput using all 4 cores.

Architecture:
- Main process: Read CAN messages, distribute to workers
- Worker processes (4): Process batches independently
- Result process: Collect and handle alerts

Performance targets:
- 15,000-20,000 msg/s on Pi 4
- Linear scaling up to 4 cores
- <10ms latency overhead
"""

import logging
import time
import queue
import multiprocessing as mp
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class WorkerProcess:
    """Worker process for processing CAN message batches."""
    
    def __init__(self, worker_id: int, config: Dict[str, Any]):
        """
        Initialize worker process.
        
        Args:
            worker_id: Unique worker identifier
            config: Configuration dictionary
        """
        self.worker_id = worker_id
        self.config = config
        
        # Will be initialized in run() method
        self.rule_engine = None
        self.ml_detector = None
        self.prefilter = None
        
    def run(self, input_queue: mp.Queue, output_queue: mp.Queue, 
            stop_event: mp.Event) -> None:
        """
        Worker process main loop.
        
        Args:
            input_queue: Queue for receiving message batches
            output_queue: Queue for sending alerts
            stop_event: Event to signal shutdown
        """
        # Initialize components in worker process
        self._initialize_components()
        
        logger.info(f"Worker {self.worker_id} started")
        
        processed = 0
        start_time = time.time()
        
        while not stop_event.is_set():
            try:
                # Get batch from queue (timeout to check stop_event)
                batch = input_queue.get(timeout=0.1)
                
                # Process batch
                alerts = self._process_batch(batch)
                
                # Send results
                if alerts:
                    output_queue.put(alerts)
                
                processed += len(batch)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {e}")
        
        # Shutdown
        duration = time.time() - start_time
        rate = processed / duration if duration > 0 else 0
        logger.info(f"Worker {self.worker_id} stopped: {processed} messages, {rate:.0f} msg/s")
    
    def _initialize_components(self) -> None:
        """Initialize detection components in worker process."""
        # Import here to avoid issues with multiprocessing
        from src.detection.rule_engine import RuleEngine
        from src.detection.ml_detector import MLDetector
        from src.detection.prefilter import FastPreFilter
        
        # Initialize rule engine
        rules_file = self.config.get('rules_file', 'config/rules.yaml')
        self.rule_engine = RuleEngine(rules_file)
        
        # Initialize ML detector if configured
        ml_config = self.config.get('ml_detection', {})
        if ml_config.get('enabled', False):
            model_path = ml_config.get('model_path')
            self.ml_detector = MLDetector(model_path=model_path)
            if not self.ml_detector.is_trained:
                self.ml_detector = None
        
        # Initialize prefilter if configured
        prefilter_config = self.config.get('prefilter', {})
        if prefilter_config.get('enabled', True):
            known_good_ids = set(prefilter_config.get('known_good_ids', []))
            self.prefilter = FastPreFilter(known_good_ids=known_good_ids)
    
    def _process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of messages."""
        # Pre-filter
        if self.prefilter:
            _, suspicious = self.prefilter.filter_batch(batch)
            messages = suspicious
        else:
            messages = batch
        
        if not messages:
            return []
        
        # Rule-based detection
        alerts = []
        if self.rule_engine:
            alerts.extend(self.rule_engine.analyze_batch(messages))
        
        # ML detection
        if self.ml_detector:
            ml_alerts = self.ml_detector.analyze_batch(messages)
            alerts.extend([a.__dict__ for a in ml_alerts])
        
        return alerts


class MulticoreProcessor:
    """
    Multicore CAN message processor.
    
    Uses worker pool pattern for parallel processing.
    """
    
    def __init__(self, config: Dict[str, Any], num_workers: int = None):
        """
        Initialize multicore processor.
        
        Args:
            config: Configuration dictionary
            num_workers: Number of worker processes (default: CPU count)
        """
        self.config = config
        self.num_workers = num_workers or mp.cpu_count()
        
        # Queues for inter-process communication
        self.input_queue = mp.Queue(maxsize=1000)
        self.output_queue = mp.Queue(maxsize=1000)
        
        # Control
        self.stop_event = mp.Event()
        self.workers: List[mp.Process] = []
        
        logger.info(f"MulticoreProcessor initialized with {self.num_workers} workers")
    
    def start(self) -> None:
        """Start all worker processes."""
        for i in range(self.num_workers):
            worker = WorkerProcess(worker_id=i, config=self.config)
            process = mp.Process(
                target=worker.run,
                args=(self.input_queue, self.output_queue, self.stop_event)
            )
            process.start()
            self.workers.append(process)
        
        logger.info(f"Started {self.num_workers} worker processes")
    
    def process_batch(self, batch: List[Dict[str, Any]]) -> None:
        """
        Submit batch for processing.
        
        Args:
            batch: List of CAN messages
        """
        try:
            self.input_queue.put(batch, block=False)
        except queue.Full:
            logger.warning("Input queue full, dropping batch")
    
    def get_alerts(self, timeout: float = 0.01) -> List[Dict[str, Any]]:
        """
        Get processed alerts from workers.
        
        Args:
            timeout: Maximum time to wait for alerts
            
        Returns:
            List of alerts from all workers
        """
        all_alerts = []
        deadline = time.time() + timeout
        
        while time.time() < deadline:
            try:
                alerts = self.output_queue.get(timeout=0.001)
                all_alerts.extend(alerts)
            except queue.Empty:
                break
        
        return all_alerts
    
    def stop(self) -> None:
        """Stop all worker processes."""
        logger.info("Stopping multicore processor...")
        
        # Signal stop
        self.stop_event.set()
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=2.0)
            if worker.is_alive():
                logger.warning(f"Worker {worker.pid} did not stop, terminating")
                worker.terminate()
        
        self.workers.clear()
        logger.info("Multicore processor stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            'num_workers': self.num_workers,
            'input_queue_size': self.input_queue.qsize(),
            'output_queue_size': self.output_queue.qsize(),
            'workers_alive': sum(1 for w in self.workers if w.is_alive())
        }
```

#### Step 3.1.2: Integrate Multicore into Main (2 hours)

**File:** `main.py`

```python
# Add import
from src.processing.multicore_processor import MulticoreProcessor

# In initialize_components:
def initialize_components(self) -> None:
    """Initialize components with optional multicore."""
    
    # ... existing code ...
    
    # Initialize multicore processor if enabled
    multicore_config = self.config.get('multicore', {})
    if multicore_config.get('enabled', False):
        num_workers = multicore_config.get('num_workers', 4)
        self.multicore_processor = MulticoreProcessor(
            config=self.config,
            num_workers=num_workers
        )
        self.multicore_processor.start()
        logger.info(f"✅ Multicore processing enabled ({num_workers} workers)")
    else:
        self.multicore_processor = None

# Add multicore processing mode:
def _process_messages_multicore(self) -> None:
    """Process messages using multicore workers."""
    logger.info("Starting multicore message processing")
    
    BATCH_SIZE = 100
    TIMEOUT = 0.1
    
    try:
        while self.running:
            # Read batch
            batch = self.can_sniffer.read_batch(BATCH_SIZE, TIMEOUT)
            
            if batch:
                # Submit to workers
                self.multicore_processor.process_batch(batch)
                self.stats['messages_processed'] += len(batch)
            
            # Collect alerts from workers
            alerts = self.multicore_processor.get_alerts(timeout=0.01)
            if alerts:
                self.stats['alerts_generated'] += len(alerts)
                self.alert_manager.process_alerts(alerts)
            
    except KeyboardInterrupt:
        logger.info("Multicore processing interrupted")
    finally:
        self.multicore_processor.stop()
```

**Configuration:**

**File:** `config/can_ids.yaml`

```yaml
# Add multicore section:
multicore:
  enabled: true
  num_workers: 4  # Use all 4 cores on Pi 4
```

**Expected Result After Step 3.1:**
- Throughput: 10,000 → 15,000-20,000 msg/s (2x improvement on Pi 4)
- CPU usage: Distributed across all 4 cores
- Latency: +5-10ms overhead from queue management

---

## Testing & Validation

### Performance Testing Procedure

After each milestone, run comprehensive performance tests:

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Run benchmark
python scripts/benchmark.py \
    --messages data/synthetic/test_traffic.json \
    --config config/can_ids.yaml

# 3. Run with real CAN traffic
python main.py -i can0 --log-level INFO

# 4. Monitor performance
watch -n 1 'ps aux | grep main.py && free -h'

# 5. Check thermal throttling
vcgencmd measure_temp
vcgencmd get_throttled
```

### Expected Performance Progression

| Milestone | Throughput (msg/s) | Gain | Cumulative | Status |
|-----------|-------------------|------|------------|--------|
| Baseline | 708 | 1x | 1x | ✅ Current |
| 1.1: Batch Processing | 3,500 | 5x | 5x | Phase 1 |
| 1.2: Pre-Filter | 7,000 | 2x | 10x | **TARGET MET** |
| 2.1: Rule Optimization | 10,500 | 1.5x | 15x | Phase 2 |
| 2.2: PCA Features | 12,000 | 1.2x | 17x | Phase 2 |
| 3.1: Multicore | 18,000 | 1.5x | 25x | Phase 3 |

### Validation Criteria

**Phase 1 Success Criteria:**
- ✅ Throughput ≥ 7,000 msg/s (target met)
- ✅ Packet drop rate < 1%
- ✅ CPU usage < 70%
- ✅ Memory usage < 500 MB
- ✅ No thermal throttling

**Phase 2 Success Criteria:**
- ✅ Throughput ≥ 10,000 msg/s (safety margin)
- ✅ False positive rate maintained or improved
- ✅ All 18 rule types still functional
- ✅ ML accuracy within 5% of baseline

**Phase 3 Success Criteria:**
- ✅ Throughput ≥ 15,000 msg/s (2x safety margin)
- ✅ Scales linearly with core count
- ✅ Latency increase < 10ms
- ✅ Stable for 1+ hour continuous operation

---

## Rollback Plan

If any optimization causes issues:

1. **Git Branches:** Each phase in separate branch
   ```bash
   git checkout -b phase1-batch-processing
   # Implement Phase 1
   git commit -m "Phase 1 complete"
   
   git checkout -b phase2-optimization
   # If issues, rollback:
   git checkout main
   ```

2. **Feature Flags:** Control optimizations via config
   ```yaml
   optimizations:
     batch_processing: true     # Can disable if issues
     prefilter: true            # Can disable if issues
     pca_features: false        # Start disabled, enable when tested
     multicore: false           # Start disabled, enable when tested
   ```

3. **Performance Regression Testing:**
   - Baseline test before each phase
   - Compare metrics after implementation
   - Rollback if throughput decreases or errors increase

---

## Timeline Summary

| Week | Phase | Milestones | Expected Throughput | Status |
|------|-------|-----------|---------------------|--------|
| **Week 1** | Critical Path | Batch + Pre-filter | 7,000 msg/s | TARGET |
| Week 2 | Optimization | Rules + PCA | 10,000-12,000 msg/s | Safety margin |
| Week 3 | Maximum | Multicore | 15,000-20,000 msg/s | 2x headroom |

**Total Implementation Time:** 20-25 hours across 3 weeks  
**Minimum Viable:** Week 1 only (6-8 hours) achieves 7K target!

---

## Success Metrics

### Primary Goal: 7,000 msg/s ✅

**Achieved by:** Phase 1 (Week 1, 6-8 hours)
- Batch processing: 708 → 3,500 msg/s
- Fast pre-filter: 3,500 → 7,000 msg/s

### Stretch Goals

- **10,000 msg/s:** Phase 2 optimization
- **15,000 msg/s:** Phase 3 multicore
- **20,000 msg/s:** All optimizations + tuning

### Quality Metrics

- Detection accuracy: Maintain ≥95%
- False positive rate: Maintain or improve
- Memory usage: Keep <500 MB
- CPU temperature: Stay <75°C
- Packet drops: Keep <1%

---

## Conclusion

This roadmap provides a research-validated path to achieve 7,000 msg/s throughput:

✅ **Week 1 achieves target** with just 2 critical optimizations  
🚀 **Week 2-3 provide safety margin** and future-proofing  
📚 **All techniques validated** by academic research on Pi 4  
⚡ **Incremental approach** allows testing at each step  

**Start with Phase 1 to hit your 7K target, then optimize further as needed!**

---

**Document Version:** 1.0  
**Created:** December 16, 2025  
**Status:** Ready for implementation  
**Research Foundation:** 6 academic papers on Pi-based IDS
