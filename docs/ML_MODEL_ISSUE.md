# ML Model Compatibility Issue - Pickle/Joblib Class References

**Date**: December 1, 2025  
**Issue**: Pre-trained models from Vehicle_Models cannot load in CANBUS_IDS  
**Status**: Architecture fixed, models need to be retrained or simplified

---

## Problem Summary

The multi-stage ML models exported from Vehicle_Models (e.g., `aggressive_load_shedding.joblib`) cannot be loaded in CANBUS_IDS due to Python pickle's class reference mechanism.

### Error Message
```
AttributeError: Can't get attribute 'SimpleRuleDetector' on <module '__main__' 
from '/mnt/d/GitHub/CANBUS_IDS/main.py'>
```

---

## Root Cause: How Pickle Works

### What Pickle Does
When you save a Python object with `pickle.dump()` or `joblib.dump()`:

1. **Saves data**: Model weights, parameters, arrays
2. **Saves class references**: The full module path where classes were defined

Example:
```python
# In training script (test_multistage_pipeline.py)
class SimpleRuleDetector:
    pass

model = SimpleRuleDetector()
joblib.dump(model, 'model.joblib')  
# Saves: "Object is instance of __main__.SimpleRuleDetector"
```

### What Pickle Expects When Loading
```python
# In different script (main.py)
model = joblib.load('model.joblib')
# Looks for: __main__.SimpleRuleDetector
# But __main__ is NOW main.py, not test_multistage_pipeline.py!
# ❌ Error: Can't find the class!
```

### The Specific Problem

The Vehicle_Models multi-stage models were trained and saved in context where:
- `__main__` = `test_multistage_pipeline.py` (or similar training script)
- Custom classes defined: `SimpleRuleDetector`, `MultiStageDetector`
- Pickle saved references: `__main__.SimpleRuleDetector`

When loading in CANBUS_IDS:
- `__main__` = `main.py` (different script!)
- Classes don't exist in this new `__main__` context
- Pickle can't find them → AttributeError

---

## What Was Attempted

### ❌ Attempt 1: Copy Classes to CANBUS_IDS
Created `vehicle_models_compat.py` with `SimpleRuleDetector` class.
- **Problem**: Pickle still looks for `__main__.SimpleRuleDetector`, not our module

### ❌ Attempt 2: Inject Classes into __main__
```python
__main__.SimpleRuleDetector = SimpleRuleDetector
```
- **Problem**: Injects into *current* `__main__`, but pickle wants *original* `__main__`

### ❌ Attempt 3: Custom Unpickler
```python
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'SimpleRuleDetector':
            return SimpleRuleDetector
        return super().find_class(module, name)
```
- **Problem**: Joblib wraps pickle in complex ways, hard to intercept

### ❌ Attempt 4: Re-export from Vehicle_Models
Tried to load in Vehicle_Models and extract just IsolationForest.
- **Problem**: Even Vehicle_Models can't load them now! Original training script context is gone.

---

## Why This Happened

The models were likely saved during an interactive session or training script where:
1. Classes were defined in the script's global scope (`__main__`)
2. Model was trained and pickled with those class references
3. Original training script wasn't preserved or classes weren't in proper modules

**Best Practice Violated**: Classes should be defined in importable modules (not `__main__`), so pickle saves proper module paths like `src.detectors.SimpleRuleDetector` instead of `__main__.SimpleRuleDetector`.

---

## Solutions (In Order of Preference)

### ✅ Solution 1: Retrain Multi-Stage Models Properly (RECOMMENDED)

Retrain the models in Vehicle_Models with classes in proper modules instead of `__main__`.

**Steps:**
1. Move `SimpleRuleDetector` from training scripts to `src/detectors.py`
2. Import it properly: `from src.detectors import SimpleRuleDetector`
3. Retrain all models with this proper module structure
4. Models will save as `src.detectors.SimpleRuleDetector` (portable!)
5. Copy both models AND `src/detectors.py` to CANBUS_IDS

**Pros**: 
- ✅ Full performance (90%+ recall, 74-100% precision)
- ✅ All multi-stage features
- ✅ Properly portable models
- ✅ Best practice architecture

**Cons**:
- ⏱️ Requires retraining (~1-2 hours)
- 📝 Need training data access

**Implementation:**
```python
# In Vehicle_Models/src/detectors.py
class SimpleRuleDetector:
    """Properly defined in a module, not __main__"""
    pass

# In training script
from src.detectors import SimpleRuleDetector  # Not defined here!
# Now pickle saves proper path
```

---

### ✅ Solution 2: Train Basic IsolationForest in CANBUS_IDS (QUICK FIX)

Train a simple sklearn IsolationForest directly in CANBUS_IDS.

**Steps:**
1. Collect 10-15 minutes of normal CAN traffic from your Pi
2. Use `scripts/train_basic_model.py` (to be created)
3. Train simple IsolationForest (no custom classes)
4. Save to `data/models/basic_isolation_forest.joblib`

**Pros**: 
- ✅ Quick (30 minutes to implement)
- ✅ No dependencies or pickle issues
- ✅ Works immediately
- ✅ Portable across systems

**Cons**:
- ⚠️ Lower performance than multi-stage (60-70% recall vs 90%+)
- ❌ No vehicle calibration
- ❌ No advanced features

**Performance Expectation:**
- Recall: 60-75% (vs 90-96% for multi-stage)
- Precision: 40-60% (vs 74-100% for multi-stage)
- Throughput: 10K msg/s (vs 50K+ for multi-stage)

---

### ⚙️ Solution 3: Use Dill Instead of Pickle (ALTERNATIVE)

Use `dill` library which handles `__main__` classes better.

**Steps:**
1. Install dill: `pip install dill`
2. In Vehicle_Models, save with dill: `dill.dump(model, file)`
3. In CANBUS_IDS, load with dill: `dill.load(file)`

**Pros**: 
- ✅ Can serialize `__main__` classes
- ✅ Might work with existing models (try it first!)

**Cons**:
- ⚠️ Still fragile - depends on context
- ⚠️ Not standard in sklearn/joblib
- ❌ May still fail across environments

---

### 🔧 Solution 4: ONNX Export (FUTURE - BEST PRACTICE)

Export models to ONNX (Open Neural Network Exchange) format - completely portable.

**Pros**: 
- ✅ Language/platform independent
- ✅ No pickle issues ever
- ✅ Optimized for inference

**Cons**:
- ❌ sklearn support for ONNX is limited
- ❌ Complex for multi-stage pipelines
- ❌ Requires significant refactoring

---

## Current Status

### Configuration Updated (December 1, 2025)
ML detection temporarily disabled in config files:

```yaml
# config/can_ids.yaml
# config/can_ids_rpi4.yaml
detection_modes:
  - rule_based
  # - ml_based  # Disabled until models retrained
```

### Architecture Status
✅ **CANBUS_IDS architecture is FIXED and ready**:
- Proper fail-fast initialization with clear ✅/❌ messages
- ML detector will engage when compatible model available
- Proper error handling and logging
- No silent failures

### What's Missing
❌ **Compatible ML model file**:
- Need model trained with proper module structure, OR
- Need basic IsolationForest trained in CANBUS_IDS context

---

## Recommended Action Plan

### Immediate (Tonight)
1. ✅ System runs with rule-based detection (working now)
2. ✅ Architecture improvements deployed (completed)
3. ✅ Issue documented (this document)

### Short-term (This Week)
Choose one:
- **Option A**: Train basic IsolationForest in CANBUS_IDS (quick, lower performance)
- **Option B**: Retrain multi-stage in Vehicle_Models with proper modules (better performance)

### Long-term (Next Month)
1. Implement proper module structure in Vehicle_Models
2. Retrain all models with portable class references
3. Set up automated model versioning and export pipeline
4. Consider ONNX for ultimate portability

---

## Testing Verification

When ML is re-enabled, verify with:

```bash
# Test initialization
python scripts/test_ml_initialization.py

# Expected output:
# ✅ ML DETECTION ENABLED
#    Model: <model_name>.joblib
#    Trained: True
```

---

## References

- **Architecture improvements**: `docs/ARCHITECTURE_IMPROVEMENT_PLAN.md`
- **Python pickle docs**: https://docs.python.org/3/library/pickle.html
- **Joblib documentation**: https://joblib.readthedocs.io/
- **Best practices for ML serialization**: https://scikit-learn.org/stable/model_persistence.html

---

## Key Takeaway

> **The CANBUS_IDS architecture is now properly fixed and ready for ML.** The only issue is that the existing model files use pickle class references from a training context that no longer exists. Once we retrain with proper module structure or train a basic model, everything will work perfectly.

**Next step**: Choose Solution 1 (retrain properly) or Solution 2 (train basic) based on your time constraints and performance needs.
