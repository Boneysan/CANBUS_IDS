# Solution 1: Retrain Multi-Stage Models with Proper Module Structure

**Goal**: Fix pickle/joblib compatibility by moving classes from `__main__` to proper modules  
**Time Required**: 1-2 hours  
**Performance**: Full multi-stage performance (90%+ recall, 74-100% precision)

---

## Step-by-Step Implementation

### Phase 1: Restructure Vehicle_Models Classes (30 minutes)

#### 1.1: Create Proper Module for SimpleRuleDetector

```bash
cd /mnt/d/GitHub/Vehicle_Models
```

Create `src/detectors.py`:

```python
"""
Detection classes for multi-stage pipeline.

These classes are in a proper module (not __main__) so pickle
can save/load them with proper module paths.
"""

import numpy as np


class SimpleRuleDetector:
    """
    Simple rule-based detector for Stage 2 of multi-stage pipeline.
    
    Applies fast heuristic rules to filter obvious anomalies:
    - Abnormal time deltas (< 1ms or > 1s)
    - High message frequency (> 100 Hz)
    - Invalid DLC values (not 0-8)
    """
    
    def __init__(self, rules=None):
        """
        Initialize rule detector.
        
        Args:
            rules: Optional dictionary of rule thresholds
        """
        self.rules = rules or {
            'min_time_delta': 0.001,  # 1ms
            'max_time_delta': 1.0,     # 1s
            'max_frequency': 100,       # 100 Hz
            'min_dlc': 0,
            'max_dlc': 8
        }
        
    def predict(self, X):
        """
        Predict anomalies based on simple rules.
        
        Args:
            X: Feature array (n_samples, n_features)
                Expected features at indices:
                - 0: arb_id_numeric
                - 1: data_length (DLC)
                - 2: id_frequency
                - 3: time_delta
            
        Returns:
            Predictions: 1 for anomaly, 0 for normal
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        predictions = np.zeros(len(X), dtype=int)
        
        # Rule 1: Abnormal time delta
        if X.shape[1] >= 4:
            time_delta = X[:, 3]
            anomaly_mask = (
                (time_delta < self.rules['min_time_delta']) | 
                (time_delta > self.rules['max_time_delta'])
            )
            predictions[anomaly_mask] = 1
        
        # Rule 2: High frequency
        if X.shape[1] >= 3:
            frequency = X[:, 2]
            anomaly_mask = frequency > self.rules['max_frequency']
            predictions[anomaly_mask] = 1
        
        # Rule 3: Invalid DLC
        if X.shape[1] >= 2:
            dlc = X[:, 1]
            anomaly_mask = (
                (dlc < self.rules['min_dlc']) | 
                (dlc > self.rules['max_dlc'])
            )
            predictions[anomaly_mask] = 1
            
        return predictions
    
    def fit(self, X, y=None):
        """Compatibility method for sklearn pipeline."""
        return self
    
    def score(self, X, y):
        """Calculate accuracy for compatibility."""
        predictions = self.predict(X)
        return np.mean(predictions == y)


# Export for easy import
__all__ = ['SimpleRuleDetector']
```

#### 1.2: Update MultiStageDetector Imports

Edit `src/multistage_detector.py` to import from proper module:

```python
# At the top of the file, add:
from .detectors import SimpleRuleDetector

# Remove any class SimpleRuleDetector definition in this file
```

---

### Phase 2: Update Training Script (15 minutes)

#### 2.1: Modify test_multistage_pipeline.py

Find and replace the inline `SimpleRuleDetector` class definition:

```python
# OLD (lines 31-42):
class SimpleRuleDetector:
    """Simple rule-based detector for testing."""
    def predict(self, X):
        # ... implementation ...
        
# NEW - Replace with import:
from src.detectors import SimpleRuleDetector
```

#### 2.2: Verify All Imports

Make sure the script imports from proper modules:

```python
import sys
sys.path.append('src')

from multistage_detector import MultiStageDetector, create_default_multistage_detector
from detectors import SimpleRuleDetector  # Now from module, not __main__!
```

---

### Phase 3: Retrain Models (45 minutes)

#### 3.1: Prepare Data

```bash
cd /mnt/d/GitHub/Vehicle_Models

# Verify preprocessed data exists
ls -lh data/processed/
# Should see: train_normal_comprehensive.csv, test_attacks_comprehensive.csv
```

If data doesn't exist, regenerate it:
```bash
python prepare_comprehensive_dataset.py
```

#### 3.2: Run Training

```bash
# Run the updated training script
python test_multistage_pipeline.py
```

Expected output:
```
🔧 Training simple models on basic features...
   Training Isolation Forest...
   ✅ IF trained on X normal samples
   ✅ Simple rule detector created
   Training OneClassSVM...
   ✅ SVM trained on X normal samples

💾 Saving multi-stage models...
   ✅ Saved: models/multistage/retrained_aggressive_load_shedding.joblib
   ✅ Saved: models/multistage/retrained_adaptive_load_shedding.joblib
   ...
```

#### 3.3: Test Model Loading

Verify the new models can be loaded anywhere:

```bash
python << 'EOF'
import joblib
import sys

# Test loading WITHOUT the classes being in __main__
model = joblib.load('models/multistage/retrained_aggressive_load_shedding.joblib')
print(f"✅ Model loaded: {type(model).__name__}")
print(f"✅ Stage 1: {type(model.stage1_model).__name__}")

if hasattr(model, 'stage2_detector'):
    print(f"✅ Stage 2: {type(model.stage2_detector).__name__}")
    print(f"   Module: {model.stage2_detector.__class__.__module__}")
    # Should show: "src.detectors" not "__main__"!
    
print("\n✅ SUCCESS - Model is properly portable!")
EOF
```

---

### Phase 4: Export to CANBUS_IDS (15 minutes)

#### 4.1: Copy Models

```bash
# Copy retrained models to CANBUS_IDS
cp models/multistage/retrained_*.joblib \
   /mnt/d/GitHub/CANBUS_IDS/data/models/
```

#### 4.2: Copy Detector Module

```bash
# Copy the detectors module
cp src/detectors.py \
   /mnt/d/GitHub/CANBUS_IDS/src/detection/detectors.py
```

#### 4.3: Ensure MultiStageDetector is Available

Check if already copied (it should be):
```bash
ls /mnt/d/GitHub/CANBUS_IDS/src/detection/multistage_detector.py
```

If not there:
```bash
cp src/multistage_detector.py \
   /mnt/d/GitHub/CANBUS_IDS/src/detection/
```

#### 4.4: Update CANBUS_IDS Imports

Edit `/mnt/d/GitHub/CANBUS_IDS/src/detection/ml_detector.py`:

```python
# Update the compatibility imports section (around line 18):
try:
    from .detectors import SimpleRuleDetector  # NEW - from proper module
    from .multistage_detector import MultiStageDetector
    VEHICLE_MODELS_COMPAT = True
    
    # Register globally for pickle compatibility
    import __main__
    __main__.SimpleRuleDetector = SimpleRuleDetector
    __main__.MultiStageDetector = MultiStageDetector
    
except ImportError:
    logger.debug("Vehicle_Models compatibility classes not available")
    VEHICLE_MODELS_COMPAT = False
```

---

### Phase 5: Test in CANBUS_IDS (15 minutes)

#### 5.1: Update Configuration

Enable ML detection:

```bash
cd /mnt/d/GitHub/CANBUS_IDS
```

Edit `config/can_ids_rpi4.yaml`:
```yaml
detection_modes:
  - rule_based
  - ml_based  # Re-enable!

ml_model:
  path: data/models/retrained_aggressive_load_shedding.joblib
  contamination: 0.20
```

#### 5.2: Test Model Loading

```bash
source venv/bin/activate
python scripts/test_ml_initialization.py
```

Expected output:
```
✅ ML DETECTION ENABLED
   Model: retrained_aggressive_load_shedding.joblib
   Size: 1.3 MB
   Contamination: 0.20
   Trained: True
```

#### 5.3: Test Full System

```bash
# Test with monitoring mode
python main.py --test-interface can0

# Or run live (if you have CAN traffic)
python main.py -i can0
```

Look for:
```
============================================================
INITIALIZING ML DETECTION
============================================================
Loading ML model: data/models/retrained_aggressive_load_shedding.joblib
✅ ML DETECTION ENABLED
   Model: retrained_aggressive_load_shedding.joblib
   Trained: True
============================================================
```

---

## Verification Checklist

After completing all phases:

- [ ] `SimpleRuleDetector` is in `Vehicle_Models/src/detectors.py` (not `__main__`)
- [ ] Training script imports from `src.detectors`, not defining classes inline
- [ ] Models retrained and saved to `models/multistage/retrained_*.joblib`
- [ ] Test load shows module path is `src.detectors`, not `__main__`
- [ ] Models copied to CANBUS_IDS `data/models/`
- [ ] `detectors.py` module copied to CANBUS_IDS `src/detection/`
- [ ] CANBUS_IDS config updated to enable `ml_based` detection
- [ ] Test script passes with ✅ ML DETECTION ENABLED
- [ ] Full system runs with ML alerts being generated

---

## Troubleshooting

### Issue: "Can't get attribute 'SimpleRuleDetector'"
**Cause**: Model still has `__main__` reference  
**Fix**: Make sure you actually retrained the model after moving classes to modules

### Issue: "Module 'src.detectors' not found"
**Cause**: Import path issue  
**Fix**: Ensure `src/` is in Python path and `__init__.py` files exist

### Issue: Model loads but doesn't detect anything
**Cause**: Model wasn't trained properly  
**Fix**: Check training output, ensure sufficient training data (>10K samples)

### Issue: Performance is worse than expected
**Cause**: Training data quality or quantity  
**Fix**: Retrain with more attack-free data, verify data preprocessing

---

## Expected Results

After successful implementation:

| Metric | Before (Incompatible) | After (Solution 1) |
|--------|----------------------|-------------------|
| **Model Loading** | ❌ AttributeError | ✅ Loads successfully |
| **Recall** | N/A (not running) | 90-96% |
| **Precision** | N/A (not running) | 74-100% |
| **Throughput** | Rules only | 50K+ msg/s |
| **False Positives** | N/A | 0-26% |

---

## Alternative: Quick Test First

Before full retraining, test if basic approach works:

```bash
cd /mnt/d/GitHub/Vehicle_Models

# Just create the detectors.py module
# Copy inline SimpleRuleDetector to src/detectors.py

# Try retraining just ONE model quickly
python << 'EOF'
from sklearn.ensemble import IsolationForest
from src.detectors import SimpleRuleDetector
import joblib

# Train quick test model
if_model = IsolationForest(n_estimators=10)
if_model.fit([[1, 2, 3, 4]] * 100)  # Dummy data

rule_detector = SimpleRuleDetector()

# Save both
joblib.dump(if_model, 'test_model_if.joblib')
joblib.dump(rule_detector, 'test_model_rules.joblib')

# Try loading
loaded_if = joblib.load('test_model_if.joblib')
loaded_rules = joblib.load('test_model_rules.joblib')

print(f"IF module: {loaded_if.__class__.__module__}")
print(f"Rules module: {loaded_rules.__class__.__module__}")
# Should show "sklearn.ensemble" and "src.detectors" - NOT "__main__"!

EOF
```

If this works, proceed with full retraining. If not, troubleshoot module structure first.

---

**Time Investment**: ~2 hours total  
**Benefit**: Full multi-stage ML with 90%+ detection accuracy  
**One-time effort**: Once fixed, models are portable forever
