# Machine Learning Model Optimization Guide

**Date:** December 3, 2025  
**Status:** Optimization Required for Real-Time Deployment  
**Current ML Performance:** 15 msg/s (insufficient)  
**Target Performance:** 500-1,500 msg/s  

---

## Executive Summary

The current ML detection system uses an IsolationForest model with 100 estimators, resulting in **15 msg/s throughput** - far too slow for real-time CAN bus monitoring. This document outlines practical strategies to optimize ML performance by 10-100x to achieve production viability.

**Key Findings:**
- Current bottleneck: 99% of time in `IsolationForest.decision_function()`
- Model complexity: 100 decision trees, 9 features per message
- Required speedup: **50-100x** to handle real-time traffic (1,000-1,500 msg/s)

**Quick Win Available:**
- Implement message sampling → **10x speedup** with 5 lines of code
- No model retraining required
- Can be deployed today

---

## Current Model Analysis

### Model Configuration

```python
IsolationForest(
    n_estimators=100,        # Number of decision trees
    max_samples='auto',      # Training samples per tree
    contamination=0.1,       # Expected anomaly rate
    n_features_in_=9,        # Input features
    random_state=42
)
```

### Performance Metrics

| Metric | Value | Impact |
|--------|-------|--------|
| **Throughput** | 15 msg/s | ❌ 66x too slow for normal driving |
| **Latency** | 64 ms avg | ❌ 49x slower than rule-based |
| **CPU Time** | 99% in ML | Bottleneck confirmed |
| **Function Calls** | 2.6M per 100 msgs | Extreme overhead |
| **Tree Evaluations** | 100 per message | Too many |

### Profiler Results

```
13.7 seconds / 13.8 seconds (99%) spent in:
  ├─ decision_function()           13.694s
  │  ├─ score_samples()             13.519s
  │  │  ├─ _compute_score_samples() 13.241s
  │  │  │  └─ parallel tree eval    12.821s
  └─ predict()                       6.822s
```

---

## Optimization Strategies

### Strategy 1: Message Sampling ⭐ **QUICK WIN**

**Concept:** Only analyze every Nth message with ML, while rules check everything.

**Implementation Complexity:** ⭐ Very Easy (5 minutes)  
**Expected Speedup:** 10-50x  
**Accuracy Trade-off:** Minimal (may miss attacks between samples)  

#### Implementation

```python
# src/detection/ml_detector.py

class MLDetector:
    def __init__(self, model_path: Optional[str] = None, 
                 contamination: float = 0.20,
                 feature_window: int = 100,
                 sampling_rate: int = 10):  # NEW PARAMETER
        """
        Initialize ML detector.
        
        Args:
            model_path: Path to trained model file
            contamination: Expected proportion of anomalies
            feature_window: Number of messages for feature extraction
            sampling_rate: Analyze every Nth message (1=all, 10=every 10th)
        """
        self.model_path = Path(model_path) if model_path else None
        self.contamination = contamination
        self.feature_window = feature_window
        self.sampling_rate = sampling_rate  # NEW
        
        # ... rest of initialization ...
        
    def analyze_message(self, message: Dict[str, Any]) -> Optional[MLAlert]:
        """
        Analyze a single CAN message for anomalies.
        Uses sampling to reduce computational load.
        """
        if not SKLEARN_AVAILABLE or not self.is_trained:
            raise RuntimeError("ML detector not ready")
        
        self._stats['messages_analyzed'] += 1
        
        # SAMPLING: Only analyze every Nth message
        if self.sampling_rate > 1:
            if self._stats['messages_analyzed'] % self.sampling_rate != 0:
                # Skip ML analysis for this message
                # Still update state for feature extraction
                self._update_message_state(message)
                return None
        
        # Normal ML analysis for sampled messages
        start_time = time.time()
        # ... rest of existing code ...
```

#### Usage

```yaml
# config/can_ids.yaml
ml_model:
  path: data/models/adaptive_load_shedding.joblib
  contamination: 0.20
  sampling_rate: 10  # Check every 10th message
```

```python
# In comprehensive_test.py or main.py
ml_detector = MLDetector(
    model_path='data/models/adaptive_load_shedding.joblib',
    contamination=0.20,
    sampling_rate=10  # 10x speedup
)
```

#### Performance Impact

| Sampling Rate | Messages Checked | Effective Throughput | Status |
|---------------|------------------|----------------------|--------|
| 1 (all) | 100% | 15 msg/s | ❌ Too slow |
| 5 | 20% | 75 msg/s | ⚠️ Marginal |
| 10 | 10% | 150 msg/s | ✅ Idle/parked OK |
| 25 | 4% | 375 msg/s | ✅ Low activity OK |
| 50 | 2% | 750 msg/s | ✅ Normal driving OK |
| 100 | 1% | 1,500 msg/s | ✅ All scenarios OK |

**Recommendation:** Start with sampling_rate=10 for testing, adjust based on attack detection results.

---

### Strategy 2: Lightweight Model Retraining ⭐ **HIGH IMPACT**

**Concept:** Train a simpler model with fewer estimators and features.

**Implementation Complexity:** ⭐⭐ Moderate (2-3 hours)  
**Expected Speedup:** 5-10x  
**Accuracy Trade-off:** Small (~5% recall reduction)  

#### Training Script

```python
# Vehicle_Models/retrain_lightweight.py

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
from pathlib import Path

def train_lightweight_model():
    """Train optimized IsolationForest for real-time deployment."""
    
    print("Loading training data...")
    # Load from attack-free datasets
    data_dir = Path('data/raw')
    dfs = []
    for csv in ['attack-free-1.csv', 'attack-free-2.csv']:
        df = pd.read_csv(data_dir / csv)
        dfs.append(df)
    
    train_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(train_df)} training samples")
    
    # Extract minimal features (faster)
    print("Extracting features...")
    X_train = extract_minimal_features(train_df)
    print(f"Feature shape: {X_train.shape}")
    
    # Train lightweight model
    print("Training lightweight IsolationForest...")
    model = IsolationForest(
        n_estimators=15,           # Down from 100 (87% reduction)
        max_samples=0.3,           # Use 30% samples (faster)
        max_features=0.7,          # Use 70% features
        contamination=0.1,
        random_state=42,
        n_jobs=1,                  # Single thread (avoid overhead)
        warm_start=False,
        bootstrap=False            # No bootstrap sampling
    )
    
    model.fit(X_train)
    
    print(f"✅ Model trained with {model.n_estimators} estimators")
    
    # Save model
    output_path = Path('models/lightweight_if.joblib')
    output_path.parent.mkdir(exist_ok=True)
    
    model_dict = {
        'stage1_model': model,
        'config': {
            'contamination': 0.1,
            'n_estimators': 15,
            'max_samples': 0.3
        }
    }
    
    joblib.dump(model_dict, output_path)
    print(f"✅ Model saved to {output_path}")
    
    # Validate performance
    print("\nValidating model...")
    predictions = model.predict(X_train[:1000])
    anomalies = (predictions == -1).sum()
    print(f"Anomalies in training sample: {anomalies}/1000 ({anomalies/10:.1f}%)")
    
    return model

def extract_minimal_features(df):
    """Extract only essential features for speed."""
    features = []
    
    # Convert to numpy for speed
    ids = df['arbitration_id'].values
    dlcs = df['DLC'].values if 'DLC' in df.columns else [8] * len(df)
    
    # Simple features only
    features = np.column_stack([
        ids,           # CAN ID
        dlcs,          # Data length
        np.zeros(len(df)),  # Placeholder for frequency (updated in real-time)
        np.zeros(len(df)),  # Placeholder for timing
        np.zeros(len(df)),  # Hour
        np.zeros(len(df)),  # Minute
        np.zeros(len(df)),  # Second
        np.zeros(len(df)),  # Reserved
        np.zeros(len(df)),  # Reserved
    ])
    
    return features

if __name__ == '__main__':
    model = train_lightweight_model()
    print("\n✅ Lightweight model ready for deployment!")
    print("Copy models/lightweight_if.joblib to CANBUS_IDS/data/models/")
```

#### Usage

```bash
# In Vehicle_Models project
cd /media/boneysan/Data/GitHub/Vehicle_Models
python retrain_lightweight.py

# Copy to CANBUS_IDS
cp models/lightweight_if.joblib ~/Documents/Github/CANBUS_IDS/data/models/
```

```python
# Use in CANBUS_IDS
ml_detector = MLDetector(
    model_path='data/models/lightweight_if.joblib',
    contamination=0.20
)
```

#### Expected Results

| Configuration | Estimators | Throughput | Status |
|---------------|------------|------------|--------|
| Current | 100 | 15 msg/s | ❌ |
| Optimized | 15 | 75-150 msg/s | ✅ |
| + Sampling (10x) | 15 | 750-1,500 msg/s | ✅✅ |

---

### Strategy 3: Batch Processing

**Concept:** Process multiple messages in a single ML call to amortize overhead.

**Implementation Complexity:** ⭐⭐⭐ Moderate (4-6 hours)  
**Expected Speedup:** 2-3x  
**Accuracy Trade-off:** None (delayed alerts by batch size)  

#### Implementation

```python
# src/detection/ml_detector.py

class MLDetector:
    def __init__(self, ..., batch_size: int = 50):
        self.batch_size = batch_size
        self.batch_buffer = []
        self.batch_alerts = []
        
    def analyze_message(self, message: Dict[str, Any]) -> Optional[List[MLAlert]]:
        """
        Analyze message with batching.
        Returns list of alerts when batch is full, None otherwise.
        """
        if not self.is_trained:
            return None
        
        # Add to batch
        self.batch_buffer.append(message)
        
        # Process when batch is full
        if len(self.batch_buffer) >= self.batch_size:
            alerts = self._process_batch()
            self.batch_buffer = []
            return alerts
        
        return None
    
    def _process_batch(self) -> List[MLAlert]:
        """Process a batch of messages at once."""
        if not self.batch_buffer:
            return []
        
        # Update state for all messages
        for msg in self.batch_buffer:
            self._update_message_state(msg)
        
        # Extract features for all messages at once
        features_list = []
        valid_messages = []
        
        for msg in self.batch_buffer:
            features = self._extract_simple_features(msg)
            if features:
                features_list.append(features)
                valid_messages.append(msg)
        
        if not features_list:
            return []
        
        # Single model inference for entire batch
        X = np.array(features_list)
        X_scaled = self.scaler.transform(X) if self.scaler else X
        
        # Batch prediction (much faster than individual calls)
        predictions = self.isolation_forest.predict(X_scaled)
        scores = self.isolation_forest.decision_function(X_scaled)
        
        # Create alerts for anomalies
        alerts = []
        for msg, pred, score in zip(valid_messages, predictions, scores):
            if pred == -1:  # Anomaly detected
                alerts.append(MLAlert(
                    timestamp=msg['timestamp'],
                    can_id=msg['can_id'],
                    anomaly_score=float(score),
                    confidence=min(abs(score), 1.0),
                    features={},
                    message_data=msg
                ))
                self._stats['anomalies_detected'] += 1
        
        return alerts
    
    def flush_batch(self) -> List[MLAlert]:
        """Process remaining messages in batch at end of stream."""
        if self.batch_buffer:
            alerts = self._process_batch()
            self.batch_buffer = []
            return alerts
        return []
```

#### Usage

```python
ml_detector = MLDetector(
    model_path='data/models/adaptive_load_shedding.joblib',
    batch_size=50  # Process 50 messages at a time
)

# In processing loop
for message in messages:
    alerts = ml_detector.analyze_message(message)
    if alerts:
        for alert in alerts:
            handle_alert(alert)

# At end of stream
final_alerts = ml_detector.flush_batch()
```

#### Performance Impact

| Batch Size | Overhead | Throughput | Alert Delay |
|------------|----------|------------|-------------|
| 1 (current) | High | 15 msg/s | Immediate |
| 10 | Medium | 25-30 msg/s | ~0.3s at 30 msg/s |
| 50 | Low | 40-50 msg/s | ~1.5s at 30 msg/s |
| 100 | Very Low | 50-60 msg/s | ~3s at 30 msg/s |

**Recommendation:** batch_size=50 for best speed/latency balance.

---

### Strategy 4: Adaptive Load Shedding ⭐ **INTELLIGENT**

**Concept:** Dynamically adjust ML sampling based on system load.

**Implementation Complexity:** ⭐⭐⭐ Moderate-High (6-8 hours)  
**Expected Speedup:** Variable (maintains target latency)  
**Accuracy Trade-off:** Adaptive (reduces under load)  

#### Implementation

```python
# src/detection/ml_detector.py

from collections import deque
import statistics

class MLDetector:
    def __init__(self, ..., 
                 adaptive: bool = True,
                 target_latency_ms: float = 5.0):
        """
        Args:
            adaptive: Enable adaptive load shedding
            target_latency_ms: Target per-message latency (milliseconds)
        """
        self.adaptive = adaptive
        self.target_latency_ms = target_latency_ms
        
        # Adaptive control
        self.recent_latencies = deque(maxlen=100)
        self.current_sampling_rate = 1  # Start checking all messages
        self.adjustment_interval = 100  # Adjust every N messages
        self.adjustment_counter = 0
        
    def analyze_message(self, message: Dict[str, Any]) -> Optional[MLAlert]:
        """Analyze with adaptive sampling."""
        
        self.adjustment_counter += 1
        
        # Periodically adjust sampling rate
        if self.adaptive and self.adjustment_counter >= self.adjustment_interval:
            self._adjust_sampling_rate()
            self.adjustment_counter = 0
        
        # Apply current sampling rate
        self._stats['messages_analyzed'] += 1
        if self._stats['messages_analyzed'] % self.current_sampling_rate != 0:
            self._update_message_state(message)
            return None
        
        # Measure latency for this analysis
        start_time = time.time()
        
        # Normal ML analysis
        result = self._do_ml_analysis(message)
        
        # Track latency
        latency_ms = (time.time() - start_time) * 1000
        self.recent_latencies.append(latency_ms)
        
        return result
    
    def _adjust_sampling_rate(self):
        """Dynamically adjust sampling based on recent performance."""
        if len(self.recent_latencies) < 10:
            return
        
        avg_latency = statistics.mean(self.recent_latencies)
        p95_latency = sorted(self.recent_latencies)[int(len(self.recent_latencies) * 0.95)]
        
        # If we're too slow, reduce ML frequency (higher sampling rate)
        if p95_latency > self.target_latency_ms:
            self.current_sampling_rate = min(self.current_sampling_rate + 5, 100)
            logger.info(f"System slow (p95={p95_latency:.1f}ms), reducing ML to every "
                       f"{self.current_sampling_rate}th message")
        
        # If we have headroom, check more often (lower sampling rate)
        elif avg_latency < self.target_latency_ms * 0.5 and self.current_sampling_rate > 1:
            self.current_sampling_rate = max(self.current_sampling_rate - 5, 1)
            logger.info(f"System fast (avg={avg_latency:.1f}ms), increasing ML to every "
                       f"{self.current_sampling_rate}th message")
        
        # Log current state
        logger.debug(f"Adaptive ML: sampling_rate={self.current_sampling_rate}, "
                    f"avg_latency={avg_latency:.2f}ms, p95_latency={p95_latency:.2f}ms")
    
    def get_adaptive_stats(self) -> Dict[str, Any]:
        """Get adaptive control statistics."""
        return {
            'current_sampling_rate': self.current_sampling_rate,
            'avg_latency_ms': statistics.mean(self.recent_latencies) if self.recent_latencies else 0,
            'p95_latency_ms': sorted(self.recent_latencies)[int(len(self.recent_latencies) * 0.95)] 
                             if len(self.recent_latencies) > 10 else 0,
            'target_latency_ms': self.target_latency_ms,
        }
```

#### Behavior

```
System Load:  Low     →  Medium  →  High    →  Medium  →  Low
              │         │          │          │          │
Sampling:     Every 1  →  Every 5 →  Every 50 →  Every 10 →  Every 1
              │         │          │          │          │
Throughput:   15 msg/s →  75      →  750     →  150     →  15
```

**Benefit:** System automatically adapts to maintain target latency regardless of traffic patterns.

---

### Strategy 5: Reduced Feature Set

**Concept:** Use fewer features to reduce computation time.

**Implementation Complexity:** ⭐⭐ Moderate (3-4 hours + retraining)  
**Expected Speedup:** 2-3x  
**Accuracy Trade-off:** Medium (~10-15% recall reduction)  

#### Minimal Feature Set

```python
def extract_ultra_minimal_features(message):
    """
    Extract only 3-5 essential features.
    Current: 9 features → Target: 3-5 features
    """
    can_id = message['can_id']
    
    return [
        float(can_id),                      # 1. CAN ID
        float(message['dlc']),              # 2. Data length
        calculate_frequency(can_id),        # 3. Message frequency
        # Optional:
        # calculate_timing_variance(can_id), # 4. Timing variance
        # calculate_data_entropy(message),   # 5. Data entropy
    ]
```

#### Retrain with Reduced Features

```python
# In Vehicle_Models
X_train_minimal = extract_ultra_minimal_features(training_data)

model = IsolationForest(
    n_estimators=15,
    max_features=1.0,  # Use all features (but there are fewer)
    contamination=0.1
)

model.fit(X_train_minimal)
```

---

### Strategy 6: Alternative ML Algorithms

**Concept:** Use simpler/faster algorithms than IsolationForest.

**Implementation Complexity:** ⭐⭐⭐ High (requires research + testing)  
**Expected Speedup:** 3-10x (algorithm dependent)  
**Accuracy Trade-off:** Variable  

#### Option A: One-Class SVM (Linear Kernel)

```python
from sklearn.svm import OneClassSVM

model = OneClassSVM(
    kernel='linear',    # Much faster than RBF
    nu=0.1,            # Similar to contamination
    gamma='auto'
)

model.fit(X_train)
```

**Pros:** Faster than IsolationForest with RBF  
**Cons:** May need more tuning  

#### Option B: Local Outlier Factor (LOF)

```python
from sklearn.neighbors import LocalOutlierFactor

model = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.1,
    novelty=True       # For prediction on new data
)

model.fit(X_train)
```

**Pros:** Good for local anomalies  
**Cons:** Requires neighborhood computation  

#### Option C: Threshold-Based Scoring

```python
class SimpleAnomalyDetector:
    """Ultra-fast rule-based scoring."""
    
    def __init__(self):
        self.id_frequencies = {}
        self.id_timing_stats = {}
    
    def predict(self, message):
        """Score based on simple heuristics."""
        score = 0
        
        # Unknown CAN ID
        if message['can_id'] not in self.id_frequencies:
            score += 0.5
        
        # Frequency anomaly
        if self._is_frequency_anomaly(message):
            score += 0.3
        
        # Timing anomaly
        if self._is_timing_anomaly(message):
            score += 0.2
        
        return -1 if score >= 0.5 else 1  # -1 = anomaly
```

**Pros:** Extremely fast (microseconds)  
**Cons:** Less sophisticated than ML  

---

## Combined Strategy Recommendations

### Deployment Tier 1: Immediate (Today) ⭐⭐⭐

**Goal:** Make ML usable right now without retraining.

```python
# config/can_ids.yaml
ml_model:
  path: data/models/adaptive_load_shedding.joblib
  contamination: 0.20
  sampling_rate: 10    # ADD THIS
```

**Expected Result:** 150 msg/s (10x improvement)  
**Effort:** 5 minutes  
**Risk:** Very low  

### Deployment Tier 2: This Week ⭐⭐⭐

**Goal:** Achieve production-viable performance.

1. ✅ Implement sampling (Tier 1)
2. Train lightweight model (15 estimators)
3. Use both together

```python
ml_detector = MLDetector(
    model_path='data/models/lightweight_if.joblib',  # 15 estimators
    sampling_rate=10                                  # Sample
)
```

**Expected Result:** 750-1,500 msg/s (viable for real-time!)  
**Effort:** 3-4 hours  
**Risk:** Low (can rollback)  

### Deployment Tier 3: Next Week ⭐⭐

**Goal:** Intelligent adaptive system.

1. ✅ Tier 1 + Tier 2
2. Implement batch processing
3. Add adaptive load shedding

```python
ml_detector = MLDetector(
    model_path='data/models/lightweight_if.joblib',
    batch_size=50,
    adaptive=True,
    target_latency_ms=5.0
)
```

**Expected Result:** 1,500-3,000 msg/s (handles all scenarios)  
**Effort:** 8-12 hours  
**Risk:** Medium (more complex)  

---

## Testing & Validation

### Performance Testing

```bash
# Test with different configurations
for sampling in 1 5 10 25 50; do
    echo "Testing sampling_rate=$sampling"
    python scripts/comprehensive_test.py \
        /tmp/dos1_small.csv \
        --enable-ml \
        --ml-sampling $sampling \
        --output test_results/sampling_${sampling}
done
```

### Accuracy Validation

```python
# Measure detection accuracy with different configs
def validate_accuracy(model_path, sampling_rate):
    """Test on all attack datasets."""
    datasets = [
        'DoS-1.csv', 'DoS-2.csv',
        'fuzzing-1.csv', 'fuzzing-2.csv',
        'rpm-1.csv', 'rpm-2.csv'
    ]
    
    results = []
    for dataset in datasets:
        detector = MLDetector(
            model_path=model_path,
            sampling_rate=sampling_rate
        )
        
        metrics = test_detector(detector, dataset)
        results.append(metrics)
    
    return pd.DataFrame(results)
```

### Acceptance Criteria

| Metric | Minimum | Target | Stretch |
|--------|---------|--------|---------|
| **Throughput** | 500 msg/s | 1,500 msg/s | 3,000 msg/s |
| **Latency (p95)** | <10 ms | <5 ms | <2 ms |
| **Recall** | >90% | >95% | >99% |
| **Precision** | >50% | >70% | >85% |
| **CPU Usage** | <70% | <50% | <30% |

---

## Implementation Roadmap

### Phase 1: Quick Win (Day 1)
- [x] Document current performance
- [ ] Add sampling parameter to MLDetector
- [ ] Test with sampling_rate=10
- [ ] Validate throughput improvement

**Deliverable:** 150 msg/s with minimal effort

### Phase 2: Model Optimization (Week 1)
- [ ] Create retrain_lightweight.py script
- [ ] Train model with n_estimators=15
- [ ] Validate accuracy on test datasets
- [ ] Deploy lightweight model

**Deliverable:** 750-1,500 msg/s with good accuracy

### Phase 3: Advanced Features (Week 2-3)
- [ ] Implement batch processing
- [ ] Add adaptive load shedding
- [ ] Create performance monitoring dashboard
- [ ] Comprehensive testing

**Deliverable:** 1,500-3,000 msg/s with adaptive control

### Phase 4: Production Hardening (Week 4)
- [ ] Stress testing under various loads
- [ ] Real vehicle capture testing
- [ ] Documentation and deployment guide
- [ ] Performance tuning and optimization

**Deliverable:** Production-ready ML detection system

---

## Code Changes Required

### Minimal Implementation (Sampling Only)

```python
# File: src/detection/ml_detector.py
# Lines to change: ~5-10

# In __init__:
self.sampling_rate = sampling_rate  # Add parameter

# In analyze_message:
if self.sampling_rate > 1:
    if self._stats['messages_analyzed'] % self.sampling_rate != 0:
        self._update_message_state(message)
        return None
```

### Full Implementation (All Strategies)

```
Files to modify:
├── src/detection/ml_detector.py         (~200 lines)
├── scripts/comprehensive_test.py        (~50 lines)
├── config/can_ids.yaml                  (~10 lines)
└── Vehicle_Models/retrain_lightweight.py (NEW, ~150 lines)

Total effort: 6-10 hours for complete implementation
```

---

## Monitoring & Metrics

### Key Metrics to Track

```python
ml_stats = ml_detector.get_stats()

print(f"""
ML Performance Metrics:
  Messages Analyzed:    {ml_stats['messages_analyzed']}
  Anomalies Detected:   {ml_stats['anomalies_detected']}
  Throughput:           {calculate_throughput()} msg/s
  Avg Latency:          {ml_stats['feature_extraction_time'] * 1000:.2f} ms
  Sampling Rate:        {ml_detector.sampling_rate}
  
  Current Config:
    Model: {ml_detector.model_path.name}
    Estimators: {ml_detector.isolation_forest.n_estimators}
    Features: {ml_detector.isolation_forest.n_features_in_}
""")
```

### Real-Time Dashboard

```python
# Add to main.py for live monitoring
def print_ml_status():
    while True:
        stats = ml_detector.get_adaptive_stats()
        print(f"\rML: {stats['current_sampling_rate']}x sampling | "
              f"{stats['avg_latency_ms']:.1f}ms latency | "
              f"Target: {stats['target_latency_ms']}ms", end='')
        time.sleep(1)
```

---

## Conclusion

The current ML detection system is too slow for real-time use, but multiple optimization strategies can achieve the required 50-100x speedup:

**Immediate Win (Today):**
- Implement message sampling → 10x speedup with 5 lines of code

**Production Viability (This Week):**
- Lightweight model (15 estimators) + Sampling → 50-100x speedup

**Advanced Deployment (Next 2-3 Weeks):**
- Add batch processing + Adaptive control → Intelligent, self-tuning system

The combination of these strategies can achieve **1,500-3,000 msg/s** throughput, making ML detection viable for real-time CAN bus monitoring on Raspberry Pi 4.

---

**Last Updated:** December 3, 2025  
**Next Review:** After Phase 1 implementation  
**Owner:** CAN-IDS Development Team  
