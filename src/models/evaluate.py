"""
Model evaluation and validation utilities.

Provides comprehensive evaluation metrics and analysis
for CAN-IDS machine learning models.
"""

import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report, roc_curve, auc
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not available. Model evaluation will be limited.")
    SKLEARN_AVAILABLE = False


class ModelEvaluator:
    """
    Evaluate and analyze ML model performance.
    
    Provides comprehensive metrics, visualization support,
    and performance analysis for CAN-IDS models.
    """
    
    def __init__(self):
        """Initialize model evaluator."""
        self.evaluation_results = {}
        self.confusion_matrices = {}
        
    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray,
                           y_scores: Optional[np.ndarray] = None,
                           threshold: float = 0.5) -> Dict[str, Any]:
        """
        Comprehensive evaluation of model predictions.
        
        Args:
            y_true: True labels (1=anomaly, 0=normal)
            y_pred: Predicted labels (1=anomaly, 0=normal)
            y_scores: Optional anomaly scores for ROC analysis
            threshold: Decision threshold for binary classification
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn required for evaluation")
            return {}
            
        logger.info(f"Evaluating predictions on {len(y_true)} samples")
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # False positive rate and false negative rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        # Specificity and NPV
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'specificity': float(specificity),
            'false_positive_rate': float(fpr),
            'false_negative_rate': float(fnr),
            'negative_predictive_value': float(npv),
            'confusion_matrix': {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            },
            'sample_counts': {
                'total': int(len(y_true)),
                'positive': int(np.sum(y_true == 1)),
                'negative': int(np.sum(y_true == 0))
            }
        }
        
        # ROC analysis if scores provided
        if y_scores is not None:
            try:
                fpr_curve, tpr_curve, thresholds = roc_curve(y_true, y_scores)
                roc_auc = auc(fpr_curve, tpr_curve)
                
                metrics['roc_auc'] = float(roc_auc)
                metrics['roc_curve'] = {
                    'fpr': fpr_curve.tolist(),
                    'tpr': tpr_curve.tolist(),
                    'thresholds': thresholds.tolist()
                }
            except Exception as e:
                logger.warning(f"Error computing ROC curve: {e}")
                
        logger.info(f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, "
                   f"Recall: {recall:.3f}, F1: {f1:.3f}")
        
        self.evaluation_results = metrics
        return metrics
        
    def evaluate_by_attack_type(self, y_true: np.ndarray, y_pred: np.ndarray,
                                attack_types: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model performance per attack type.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            attack_types: List of attack type labels for each sample
            
        Returns:
            Dictionary of attack types and their metrics
        """
        if not SKLEARN_AVAILABLE:
            return {}
            
        results = {}
        unique_types = set(attack_types)
        
        for attack_type in unique_types:
            # Get indices for this attack type
            indices = [i for i, t in enumerate(attack_types) if t == attack_type]
            
            if not indices:
                continue
                
            y_true_type = y_true[indices]
            y_pred_type = y_pred[indices]
            
            # Calculate metrics for this type
            precision = precision_score(y_true_type, y_pred_type, zero_division=0)
            recall = recall_score(y_true_type, y_pred_type, zero_division=0)
            f1 = f1_score(y_true_type, y_pred_type, zero_division=0)
            
            results[attack_type] = {
                'samples': len(indices),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            }
            
        return results
        
    def analyze_detection_latency(self, timestamps: List[float], 
                                 detections: List[bool]) -> Dict[str, float]:
        """
        Analyze detection latency for real-time performance.
        
        Args:
            timestamps: List of message timestamps
            detections: List of detection flags (True=detected)
            
        Returns:
            Dictionary containing latency statistics
        """
        if not detections or not timestamps:
            return {}
            
        # Calculate time differences between detections
        detection_times = [t for t, d in zip(timestamps, detections) if d]
        
        if len(detection_times) < 2:
            return {'mean_latency': 0.0, 'median_latency': 0.0}
            
        latencies = np.diff(detection_times)
        
        return {
            'mean_latency': float(np.mean(latencies)),
            'median_latency': float(np.median(latencies)),
            'min_latency': float(np.min(latencies)),
            'max_latency': float(np.max(latencies)),
            'std_latency': float(np.std(latencies)),
            'total_detections': len(detection_times)
        }
        
    def calculate_detection_rate_over_time(self, timestamps: List[float],
                                          detections: List[bool],
                                          window_size: float = 60.0) -> List[Dict[str, Any]]:
        """
        Calculate detection rate in time windows.
        
        Args:
            timestamps: List of message timestamps
            detections: List of detection flags
            window_size: Time window size in seconds
            
        Returns:
            List of time windows with detection rates
        """
        if not timestamps:
            return []
            
        start_time = min(timestamps)
        end_time = max(timestamps)
        
        windows = []
        current_time = start_time
        
        while current_time < end_time:
            window_end = current_time + window_size
            
            # Get detections in this window
            window_detections = [
                d for t, d in zip(timestamps, detections)
                if current_time <= t < window_end
            ]
            
            if window_detections:
                detection_rate = sum(window_detections) / len(window_detections)
                
                windows.append({
                    'start_time': current_time,
                    'end_time': window_end,
                    'total_messages': len(window_detections),
                    'detections': sum(window_detections),
                    'detection_rate': detection_rate
                })
                
            current_time = window_end
            
        return windows
        
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            output_path: Optional path to save JSON report
            
        Returns:
            JSON string of evaluation report
        """
        if not self.evaluation_results:
            logger.warning("No evaluation results available")
            return "{}"
            
        report = {
            'timestamp': time.time(),
            'metrics': self.evaluation_results,
            'summary': self._generate_summary()
        }
        
        report_json = json.dumps(report, indent=2)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write(report_json)
                
            logger.info(f"Evaluation report saved to {output_path}")
            
        return report_json
        
    def _generate_summary(self) -> Dict[str, str]:
        """Generate human-readable summary of results."""
        if not self.evaluation_results:
            return {}
            
        metrics = self.evaluation_results
        
        # Performance assessment
        f1 = metrics.get('f1_score', 0.0)
        if f1 >= 0.9:
            performance = "Excellent"
        elif f1 >= 0.8:
            performance = "Good"
        elif f1 >= 0.7:
            performance = "Fair"
        else:
            performance = "Poor"
            
        # False positive assessment
        fpr = metrics.get('false_positive_rate', 0.0)
        if fpr <= 0.01:
            fp_assessment = "Very Low"
        elif fpr <= 0.05:
            fp_assessment = "Low"
        elif fpr <= 0.10:
            fp_assessment = "Moderate"
        else:
            fp_assessment = "High"
            
        return {
            'overall_performance': performance,
            'false_positive_rate_assessment': fp_assessment,
            'recommendation': self._get_recommendation(f1, fpr)
        }
        
    def _get_recommendation(self, f1: float, fpr: float) -> str:
        """Generate recommendation based on metrics."""
        if f1 >= 0.9 and fpr <= 0.05:
            return "Model performs well. Ready for production deployment."
        elif f1 >= 0.8:
            return "Model shows good performance. Consider fine-tuning for better results."
        elif fpr > 0.10:
            return "High false positive rate. Adjust detection threshold or retrain model."
        else:
            return "Model needs improvement. Consider collecting more training data or adjusting features."
            
    def compare_models(self, results: List[Dict[str, Any]], 
                      model_names: List[str]) -> Dict[str, Any]:
        """
        Compare multiple model evaluation results.
        
        Args:
            results: List of evaluation result dictionaries
            model_names: List of model names
            
        Returns:
            Comparison summary
        """
        if len(results) != len(model_names):
            raise ValueError("Number of results must match number of model names")
            
        comparison = {
            'models': model_names,
            'metrics': defaultdict(list)
        }
        
        # Collect metrics from each model
        metric_keys = ['accuracy', 'precision', 'recall', 'f1_score', 
                      'false_positive_rate', 'false_negative_rate']
        
        for result in results:
            for key in metric_keys:
                comparison['metrics'][key].append(result.get(key, 0.0))
                
        # Find best model for each metric
        best_models = {}
        for key in metric_keys:
            values = comparison['metrics'][key]
            # For error rates, lower is better
            if 'false' in key:
                best_idx = values.index(min(values))
            else:
                best_idx = values.index(max(values))
            best_models[key] = model_names[best_idx]
            
        comparison['best_models'] = best_models
        
        return dict(comparison)
