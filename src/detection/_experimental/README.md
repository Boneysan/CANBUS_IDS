# Detection Engine — Experimental / Research Prototypes

This folder contains alternative detector implementations developed during research
and experimentation phases. These files are **not part of the active detection
pipeline** and are not imported by `main.py` or the production codebase.

They are preserved here for reference, reproducibility, and potential future use.

## Contents

| File | Purpose | Status |
|------|---------|--------|
| `advanced_detectors.py` | Time-based and cumulative timing detectors; Isolation Forest / OneClassSVM variants | Research prototype |
| `can_feature_engineering.py` | Sophisticated CAN payload feature engineering (entropy, n-grams, sequences) | Research prototype |
| `enhanced_features.py` | Shannon entropy, byte-frequency, sequence analysis as standalone extractor | Research prototype |
| `enhanced_ml_detector.py` | Wrapper combining `MLDetector` with multistage pipeline; vehicle-specific calibration | Experimental |
| `ensemble_crosscheck_detector.py` | Multi-filter ensemble voting with cross-validation | Research prototype |
| `improved_detectors.py` | Research-paper-inspired detectors (CUSUM, OCSVM, histogram-based) | Research prototype |
| `vehicle_calibration.py` | Per-vehicle adaptive calibration manager | Experimental |
| `weighted_ensemble_detector.py` | Performance-weighted voting ensemble | Research prototype |
| `utils.py` | Matplotlib/seaborn visualisation helpers for detector analysis | Analysis tooling |

## Active Detection Pipeline

The production pipeline lives in `src/detection/`:

```
FastPreFilter → RuleEngine → DecisionTreeDetector → MLDetector (multistage)
```

## Usage

Files in this folder can be run or imported independently for research purposes.
Note that internal cross-imports use relative imports (`from .module import ...`).
