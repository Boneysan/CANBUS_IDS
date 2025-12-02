"""
Per-Vehicle Calibration Manager
Strategy from IMPROVEMENT_RECOMMENDATIONS.md

This module manages separate calibration and models for different vehicle types,
solving the cross-vehicle degradation problem where set_01 (Impala) calibration
doesn't work well for set_02 (Traverse) attacks.

Expected improvement: +10-15% recall on cross-vehicle attacks
"""

import numpy as np
import pandas as pd
import joblib
from typing import Dict, Optional, List
from pathlib import Path
import json


class VehicleCalibrationManager:
    """
    Manages per-vehicle model calibration and selection.
    
    This manager:
    1. Stores separate calibrations for each vehicle model
    2. Automatically selects the correct calibration based on vehicle metadata
    3. Provides fallback to default calibration if vehicle unknown
    
    Parameters
    ----------
    base_model : sklearn model
        Base ML model (e.g., OneClassSVM)
    default_vehicle : str, default='impala'
        Default vehicle to use if vehicle type cannot be determined
    """
    
    def __init__(self, base_model, default_vehicle='impala'):
        self.base_model = base_model
        self.default_vehicle = default_vehicle
        
        # Storage for per-vehicle calibrations
        self.vehicle_calibrations = {}
        self.vehicle_metadata = {}
        
        # Statistics
        self.stats = {
            'vehicles_calibrated': [],
            'calibration_counts': {},
            'detection_counts': {}
        }
    
    def calibrate_vehicle(self, vehicle_name: str, df_normal: pd.DataFrame,
                         dos_params: Optional[Dict] = None,
                         spoofing_params: Optional[Dict] = None,
                         fuzzing_params: Optional[Dict] = None):
        """
        Calibrate filters for a specific vehicle model.
        
        Parameters
        ----------
        vehicle_name : str
            Name of vehicle model (e.g., 'impala', 'traverse', 'malibu')
        df_normal : DataFrame
            Normal traffic data from this vehicle (minimum 40K samples recommended)
        dos_params : dict, optional
            Parameters for DoS filter calibration
        spoofing_params : dict, optional
            Parameters for Spoofing filter calibration
        fuzzing_params : dict, optional
            Parameters for Fuzzing filter calibration
        """
        from ensemble_crosscheck_detector import MultiAttackCrossCheckEnsemble
        
        print(f"\n{'='*70}")
        print(f"🚗 CALIBRATING FOR VEHICLE: {vehicle_name.upper()}")
        print(f"{'='*70}")
        
        # Default parameters if not provided
        if dos_params is None:
            dos_params = {'id_threshold': 0x100, 'reset_period': 10.0}
        if spoofing_params is None:
            spoofing_params = {'entropy_threshold_factor': 2.5}
        if fuzzing_params is None:
            fuzzing_params = {'randomness_threshold': 2.0}
        
        # Create ensemble for this vehicle
        ensemble = MultiAttackCrossCheckEnsemble(
            ml_model=self.base_model,
            mode='auto',
            dos_params=dos_params,
            spoofing_params=spoofing_params,
            fuzzing_params=fuzzing_params
        )
        
        # Calibrate on vehicle-specific normal data
        print(f"\nCalibrating on {len(df_normal):,} normal samples from {vehicle_name}...")
        ensemble.calibrate(df_normal)
        
        # Store calibration
        self.vehicle_calibrations[vehicle_name] = ensemble
        
        # Store metadata
        self.vehicle_metadata[vehicle_name] = {
            'calibration_samples': len(df_normal),
            'unique_ids': int(df_normal['arb_id_numeric'].nunique()),
            'dos_params': dos_params,
            'spoofing_params': spoofing_params,
            'fuzzing_params': fuzzing_params
        }
        
        # Update statistics
        self.stats['vehicles_calibrated'].append(vehicle_name)
        self.stats['calibration_counts'][vehicle_name] = len(df_normal)
        
        print(f"\n✅ Calibration complete for {vehicle_name}")
        print(f"   Unique CAN IDs: {self.vehicle_metadata[vehicle_name]['unique_ids']}")
        print(f"{'='*70}\n")
    
    def get_ensemble_for_vehicle(self, vehicle_name: Optional[str] = None):
        """
        Get the appropriate ensemble for a vehicle.
        
        Parameters
        ----------
        vehicle_name : str, optional
            Vehicle model name. If None or unknown, uses default vehicle.
            
        Returns
        -------
        ensemble : MultiAttackCrossCheckEnsemble
            Calibrated ensemble for the vehicle
        """
        # Normalize vehicle name
        if vehicle_name:
            vehicle_name = vehicle_name.lower().strip()
        
        # Check if we have calibration for this vehicle
        if vehicle_name and vehicle_name in self.vehicle_calibrations:
            print(f"🚗 Using {vehicle_name} calibration")
            return self.vehicle_calibrations[vehicle_name]
        
        # Fallback to default
        if self.default_vehicle in self.vehicle_calibrations:
            print(f"⚠️  Vehicle '{vehicle_name}' not found, using {self.default_vehicle} calibration")
            return self.vehicle_calibrations[self.default_vehicle]
        
        # No calibration available
        raise ValueError(f"No calibration found for vehicle '{vehicle_name}' or default '{self.default_vehicle}'")
    
    def predict(self, df: pd.DataFrame, vehicle_name: Optional[str] = None) -> np.ndarray:
        """
        Predict using vehicle-specific ensemble.
        
        Parameters
        ----------
        df : DataFrame
            Traffic data to analyze
        vehicle_name : str, optional
            Vehicle model name. If None, tries to infer from data or uses default.
            
        Returns
        -------
        predictions : ndarray
            Binary predictions (1=attack, 0=normal)
        """
        # Try to infer vehicle from data if not provided
        if vehicle_name is None:
            vehicle_name = self._infer_vehicle_from_data(df)
        
        # Get appropriate ensemble
        ensemble = self.get_ensemble_for_vehicle(vehicle_name)
        
        # Make predictions
        predictions = ensemble.predict(df)
        
        # Update detection statistics
        if vehicle_name not in self.stats['detection_counts']:
            self.stats['detection_counts'][vehicle_name] = 0
        self.stats['detection_counts'][vehicle_name] += len(df)
        
        return predictions
    
    def _infer_vehicle_from_data(self, df: pd.DataFrame) -> Optional[str]:
        """
        Try to infer vehicle type from data characteristics.
        
        Uses CAN ID patterns and other features to guess vehicle model.
        """
        # Check for source_file or vehicle column
        if 'vehicle' in df.columns:
            return df['vehicle'].iloc[0].lower()
        
        if 'source_file' in df.columns:
            source = df['source_file'].iloc[0].lower()
            # Try to extract vehicle name from source file
            for vehicle in self.vehicle_calibrations.keys():
                if vehicle in source:
                    return vehicle
        
        # Try to infer from CAN ID patterns
        unique_ids = set(df['arb_id_numeric'].unique())
        
        # Compare with known vehicle ID sets
        best_match = None
        best_similarity = 0
        
        for vehicle_name, ensemble in self.vehicle_calibrations.items():
            # Get IDs from this vehicle's calibration
            vehicle_ids = set(ensemble.dos_filter.all_ids if hasattr(ensemble.dos_filter, 'all_ids') else [])
            
            if len(vehicle_ids) > 0:
                # Calculate Jaccard similarity
                intersection = len(unique_ids & vehicle_ids)
                union = len(unique_ids | vehicle_ids)
                similarity = intersection / union if union > 0 else 0
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = vehicle_name
        
        # Use best match if similarity is high enough (>50%)
        if best_similarity > 0.5:
            print(f"🔍 Inferred vehicle: {best_match} (similarity: {best_similarity*100:.1f}%)")
            return best_match
        
        print(f"⚠️  Could not infer vehicle, using default: {self.default_vehicle}")
        return self.default_vehicle
    
    def save(self, directory: str):
        """
        Save all vehicle calibrations to directory.
        
        Parameters
        ----------
        directory : str
            Directory to save calibrations (will be created if doesn't exist)
        """
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Save each vehicle's ensemble
        for vehicle_name, ensemble in self.vehicle_calibrations.items():
            filepath = dir_path / f"ensemble_{vehicle_name}.joblib"
            joblib.dump(ensemble, filepath)
            print(f"✅ Saved {vehicle_name} calibration to {filepath}")
        
        # Save metadata and stats
        metadata_path = dir_path / "vehicle_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                'vehicle_metadata': self.vehicle_metadata,
                'stats': self.stats,
                'default_vehicle': self.default_vehicle
            }, f, indent=2)
        print(f"✅ Saved metadata to {metadata_path}")
        
        print(f"\n✅ All calibrations saved to {directory}")
    
    def load(self, directory: str):
        """
        Load all vehicle calibrations from directory.
        
        Parameters
        ----------
        directory : str
            Directory containing saved calibrations
        """
        dir_path = Path(directory)
        
        if not dir_path.exists():
            raise ValueError(f"Directory not found: {directory}")
        
        # Load metadata
        metadata_path = dir_path / "vehicle_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                data = json.load(f)
                self.vehicle_metadata = data.get('vehicle_metadata', {})
                self.stats = data.get('stats', self.stats)
                self.default_vehicle = data.get('default_vehicle', self.default_vehicle)
        
        # Load ensemble files
        for ensemble_file in dir_path.glob("ensemble_*.joblib"):
            vehicle_name = ensemble_file.stem.replace('ensemble_', '')
            ensemble = joblib.load(ensemble_file)
            self.vehicle_calibrations[vehicle_name] = ensemble
            print(f"✅ Loaded {vehicle_name} calibration from {ensemble_file}")
        
        print(f"\n✅ Loaded {len(self.vehicle_calibrations)} vehicle calibrations from {directory}")
    
    def print_summary(self):
        """Print summary of all vehicle calibrations."""
        print("\n" + "="*70)
        print("VEHICLE CALIBRATION MANAGER SUMMARY")
        print("="*70)
        
        print(f"\nDefault vehicle: {self.default_vehicle}")
        print(f"Total vehicles calibrated: {len(self.vehicle_calibrations)}")
        
        print("\nVehicle Calibrations:")
        for vehicle_name in sorted(self.vehicle_calibrations.keys()):
            metadata = self.vehicle_metadata.get(vehicle_name, {})
            print(f"\n  {vehicle_name.upper()}:")
            print(f"    Calibration samples: {metadata.get('calibration_samples', 0):,}")
            print(f"    Unique CAN IDs: {metadata.get('unique_ids', 0)}")
            
            if vehicle_name in self.stats['detection_counts']:
                print(f"    Detections run: {self.stats['detection_counts'][vehicle_name]:,}")
        
        print("\n" + "="*70 + "\n")
    
    def compare_vehicles(self, df_test: pd.DataFrame, y_true: np.ndarray,
                        vehicles_to_test: Optional[List[str]] = None):
        """
        Compare detection performance across different vehicle calibrations.
        
        Useful for understanding cross-vehicle performance and finding
        the best calibration for unknown vehicles.
        
        Parameters
        ----------
        df_test : DataFrame
            Test data
        y_true : array
            Ground truth labels
        vehicles_to_test : list, optional
            List of vehicle names to test. If None, tests all calibrated vehicles.
            
        Returns
        -------
        results : dict
            Performance metrics for each vehicle calibration
        """
        from sklearn.metrics import recall_score, precision_score, f1_score
        
        if vehicles_to_test is None:
            vehicles_to_test = list(self.vehicle_calibrations.keys())
        
        print("\n" + "="*70)
        print("CROSS-VEHICLE PERFORMANCE COMPARISON")
        print("="*70)
        
        results = {}
        
        for vehicle_name in vehicles_to_test:
            if vehicle_name not in self.vehicle_calibrations:
                print(f"⚠️  Skipping {vehicle_name} (not calibrated)")
                continue
            
            print(f"\nTesting with {vehicle_name} calibration...")
            
            ensemble = self.vehicle_calibrations[vehicle_name]
            predictions = ensemble.predict(df_test)
            
            # Convert predictions if needed
            if np.any(predictions == -1):
                predictions = np.where(predictions == -1, 1, 0)
            
            y_true_arr = np.asarray(y_true)
            pred_arr = np.asarray(predictions)
            
            recall = float(recall_score(y_true_arr, pred_arr, zero_division=0))
            precision = float(precision_score(y_true_arr, pred_arr, zero_division=0))
            f1 = float(f1_score(y_true_arr, pred_arr, zero_division=0))
            
            results[vehicle_name] = {
                'recall': recall,
                'precision': precision,
                'f1': f1
            }
            
            print(f"  Recall: {recall*100:6.2f}%")
            print(f"  Precision: {precision*100:6.2f}%")
            print(f"  F1-score: {f1:.4f}")
        
        # Print comparison table
        print("\n" + "="*70)
        print("COMPARISON TABLE")
        print("="*70)
        print(f"\n{'Vehicle':<15} {'Recall':>10} {'Precision':>12} {'F1':>10}")
        print("-"*70)
        
        for vehicle_name, metrics in sorted(results.items()):
            print(f"{vehicle_name:<15} {metrics['recall']*100:9.2f}% {metrics['precision']*100:11.2f}% {metrics['f1']:9.4f}")
        
        print("="*70 + "\n")
        
        return results


def create_vehicle_calibration_manager(ml_model, vehicles_data: Dict[str, pd.DataFrame],
                                      default_vehicle: str = 'impala') -> VehicleCalibrationManager:
    """
    Convenience function to create and calibrate manager for multiple vehicles.
    
    Parameters
    ----------
    ml_model : sklearn model
        Base ML model
    vehicles_data : dict
        Dictionary mapping vehicle names to their normal traffic DataFrames
        Example: {'impala': df_impala_normal, 'traverse': df_traverse_normal}
    default_vehicle : str
        Default vehicle to use
        
    Returns
    -------
    manager : VehicleCalibrationManager
        Calibrated manager ready to use
    """
    manager = VehicleCalibrationManager(ml_model, default_vehicle=default_vehicle)
    
    for vehicle_name, df_normal in vehicles_data.items():
        manager.calibrate_vehicle(vehicle_name, df_normal)
    
    return manager
