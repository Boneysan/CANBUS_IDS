#!/usr/bin/env python3
"""
Import real CAN bus dataset from can-train-and-test repository.

Converts the CSV format real vehicle data to CAN-IDS compatible JSON format
for training and testing the intrusion detection system.
"""

import argparse
import pandas as pd
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class RealCANDatasetImporter:
    """Import and convert real CAN bus dataset to CAN-IDS format."""
    
    def __init__(self, dataset_path: str):
        """
        Initialize dataset importer.
        
        Args:
            dataset_path: Path to can-train-and-test dataset
        """
        self.dataset_path = Path(dataset_path)
        self.vehicle_mapping = {
            'set_01': {
                'known': 'Chevrolet_Impala',
                'unknown': 'Chevrolet_Silverado'
            },
            'set_02': {
                'known': 'Chevrolet_Traverse', 
                'unknown': 'Subaru_Forester'
            },
            'set_03': {
                'known': 'Chevrolet_Silverado',
                'unknown': 'Subaru_Forester'
            },
            'set_04': {
                'known': 'Subaru_Forester',
                'unknown': 'Chevrolet_Traverse'
            }
        }
        
    def convert_csv_to_canids_format(self, csv_file: Path) -> List[Dict[str, Any]]:
        """
        Convert CSV file to CAN-IDS message format.
        
        Args:
            csv_file: Path to CSV file
            
        Returns:
            List of CAN-IDS compatible messages
        """
        print(f"Converting {csv_file.name}...")
        
        try:
            # Read CSV file
            df = pd.read_csv(csv_file)
            
            messages = []
            for _, row in df.iterrows():
                # Parse arbitration ID (hex string to int)
                try:
                    can_id = int(row['arbitration_id'], 16)
                except ValueError:
                    continue  # Skip malformed IDs
                
                # Parse data field (hex string to bytes)
                data_hex = row['data_field']
                data_bytes = []
                
                # Convert hex string to byte array
                for i in range(0, len(data_hex), 2):
                    byte_str = data_hex[i:i+2]
                    if len(byte_str) == 2:
                        try:
                            data_bytes.append(int(byte_str, 16))
                        except ValueError:
                            pass
                
                # Create CAN-IDS compatible message
                message = {
                    'timestamp': float(row['timestamp']),
                    'can_id': can_id,
                    'dlc': len(data_bytes),
                    'data': data_bytes,
                    'data_hex': ' '.join(f'{b:02X}' for b in data_bytes),
                    'is_extended': can_id > 0x7FF,
                    'is_remote': False,
                    'is_error': False,
                    'is_attack': bool(row['attack']),
                    'attack_type': self._determine_attack_type(csv_file.name),
                    'vehicle_type': self._determine_vehicle_type(csv_file),
                    'source_file': csv_file.name
                }
                
                messages.append(message)
                
            print(f"  Converted {len(messages)} messages from {len(df)} rows")
            return messages
            
        except Exception as e:
            print(f"Error converting {csv_file}: {e}")
            return []
    
    def _determine_attack_type(self, filename: str) -> str:
        """Determine attack type from filename."""
        if 'attack-free' in filename or 'standstill' in filename:
            return 'normal'
        elif 'DoS' in filename:
            return 'dos'
        elif 'accessory' in filename:
            return 'accessory_manipulation'
        elif 'force-neutral' in filename:
            return 'gear_manipulation'
        elif 'rpm' in filename:
            return 'rpm_manipulation'
        else:
            return 'unknown'
    
    def _determine_vehicle_type(self, csv_file: Path) -> str:
        """Determine vehicle type from file path."""
        # Extract set number from path
        for part in csv_file.parts:
            if part.startswith('set_'):
                set_name = part
                # Determine if this is from known or unknown vehicle based on test folder
                if 'unknown_vehicle' in str(csv_file):
                    return self.vehicle_mapping[set_name]['unknown']
                else:
                    return self.vehicle_mapping[set_name]['known']
        return 'unknown_vehicle'
    
    def import_training_set(self, set_name: str, output_dir: str = 'data/real_dataset') -> None:
        """
        Import a complete training set.
        
        Args:
            set_name: Set name (e.g., 'set_01')
            output_dir: Output directory for converted data
        """
        set_path = self.dataset_path / set_name
        if not set_path.exists():
            print(f"Error: Set {set_name} not found at {set_path}")
            return
        
        output_path = Path(output_dir) / set_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nImporting {set_name}...")
        print(f"Vehicle mapping: {self.vehicle_mapping[set_name]}")
        
        # Import training data
        train_path = set_path / 'train_01'
        if train_path.exists():
            self._import_folder(train_path, output_path / 'train_01')
        
        # Import test data
        for test_folder in ['test_01_known_vehicle_known_attack',
                           'test_02_unknown_vehicle_known_attack', 
                           'test_03_known_vehicle_unknown_attack',
                           'test_04_unknown_vehicle_unknown_attack']:
            test_path = set_path / test_folder
            if test_path.exists():
                self._import_folder(test_path, output_path / test_folder)
    
    def _import_folder(self, input_folder: Path, output_folder: Path) -> None:
        """Import all CSV files from a folder."""
        output_folder.mkdir(parents=True, exist_ok=True)
        
        csv_files = list(input_folder.glob('*.csv'))
        if not csv_files:
            print(f"  No CSV files found in {input_folder}")
            return
        
        print(f"\n  Processing {input_folder.name} ({len(csv_files)} files):")
        
        all_messages = []
        attack_stats = {}
        
        for csv_file in csv_files:
            messages = self.convert_csv_to_canids_format(csv_file)
            all_messages.extend(messages)
            
            # Track attack statistics
            attack_type = self._determine_attack_type(csv_file.name)
            attack_count = sum(1 for msg in messages if msg['is_attack'])
            normal_count = len(messages) - attack_count
            
            attack_stats[attack_type] = attack_stats.get(attack_type, 0) + attack_count
            attack_stats['normal'] = attack_stats.get('normal', 0) + normal_count
            
            # Save individual file
            output_file = output_folder / f"{csv_file.stem}.json"
            with open(output_file, 'w') as f:
                json.dump(messages, f, indent=2)
        
        # Save combined file
        combined_file = output_folder / 'combined_messages.json'
        with open(combined_file, 'w') as f:
            json.dump(all_messages, f, indent=2)
        
        # Save statistics
        stats = {
            'total_messages': len(all_messages),
            'attack_distribution': attack_stats,
            'unique_can_ids': len(set(msg['can_id'] for msg in all_messages)),
            'time_span': all_messages[-1]['timestamp'] - all_messages[0]['timestamp'] if all_messages else 0,
            'vehicle_type': self._determine_vehicle_type(input_folder),
            'folder_name': input_folder.name
        }
        
        stats_file = output_folder / 'statistics.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"    Total messages: {len(all_messages)}")
        print(f"    Attack distribution: {attack_stats}")
        print(f"    Unique CAN IDs: {stats['unique_can_ids']}")
        print(f"    Time span: {stats['time_span']:.2f} seconds")
    
    def create_training_datasets(self, output_dir: str = 'data/real_dataset') -> None:
        """Create ML training datasets from imported data."""
        print("\nCreating ML training datasets...")
        
        output_path = Path(output_dir)
        
        # Create combined training set for ML
        all_training_messages = []
        all_test_messages = []
        
        for set_name in ['set_01', 'set_02', 'set_03', 'set_04']:
            set_path = output_path / set_name
            
            # Collect training data
            train_file = set_path / 'train_01' / 'combined_messages.json'
            if train_file.exists():
                with open(train_file, 'r') as f:
                    messages = json.load(f)
                    all_training_messages.extend(messages)
            
            # Collect test data (from known vehicle, known attack for baseline)
            test_file = set_path / 'test_01_known_vehicle_known_attack' / 'combined_messages.json'
            if test_file.exists():
                with open(test_file, 'r') as f:
                    messages = json.load(f)
                    all_test_messages.extend(messages)
        
        # Save combined datasets
        ml_output = output_path / 'ml_datasets'
        ml_output.mkdir(exist_ok=True)
        
        with open(ml_output / 'training_data.json', 'w') as f:
            json.dump(all_training_messages, f, indent=2)
            
        with open(ml_output / 'test_data.json', 'w') as f:
            json.dump(all_test_messages, f, indent=2)
        
        # Create feature extraction ready datasets
        self._create_feature_datasets(all_training_messages, all_test_messages, ml_output)
        
        print(f"  Training messages: {len(all_training_messages)}")
        print(f"  Test messages: {len(all_test_messages)}")
        print(f"  Saved to: {ml_output}")
    
    def _create_feature_datasets(self, train_messages: List[Dict], 
                                test_messages: List[Dict], output_dir: Path) -> None:
        """Create feature-extracted datasets for ML training."""
        from src.preprocessing.feature_extractor import FeatureExtractor
        
        feature_extractor = FeatureExtractor()
        
        # Extract features from training data
        print("  Extracting features from training data...")
        train_features = []
        train_labels = []
        
        for msg in train_messages:
            features = feature_extractor.extract_features(msg)
            if features:
                train_features.append(features)
                train_labels.append(1 if msg['is_attack'] else 0)
        
        # Extract features from test data
        print("  Extracting features from test data...")
        test_features = []
        test_labels = []
        
        for msg in test_messages:
            features = feature_extractor.extract_features(msg)
            if features:
                test_features.append(features)
                test_labels.append(1 if msg['is_attack'] else 0)
        
        # Save feature datasets
        np.save(output_dir / 'train_features.npy', np.array(train_features))
        np.save(output_dir / 'train_labels.npy', np.array(train_labels))
        np.save(output_dir / 'test_features.npy', np.array(test_features))
        np.save(output_dir / 'test_labels.npy', np.array(test_labels))
        
        # Save feature metadata
        feature_info = {
            'feature_count': len(train_features[0]) if train_features else 0,
            'training_samples': len(train_features),
            'test_samples': len(test_features),
            'attack_ratio_train': sum(train_labels) / len(train_labels) if train_labels else 0,
            'attack_ratio_test': sum(test_labels) / len(test_labels) if test_labels else 0
        }
        
        with open(output_dir / 'feature_info.json', 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        print(f"    Training features: {feature_info['training_samples']} x {feature_info['feature_count']}")
        print(f"    Test features: {feature_info['test_samples']} x {feature_info['feature_count']}")
        print(f"    Attack ratio (train): {feature_info['attack_ratio_train']:.3f}")
        print(f"    Attack ratio (test): {feature_info['attack_ratio_test']:.3f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Import real CAN bus dataset for CAN-IDS training'
    )
    
    parser.add_argument('dataset_path',
                       help='Path to can-train-and-test dataset directory')
    parser.add_argument('--set', 
                       choices=['set_01', 'set_02', 'set_03', 'set_04', 'all'],
                       default='all',
                       help='Which dataset to import (default: all)')
    parser.add_argument('--output', default='data/real_dataset',
                       help='Output directory (default: data/real_dataset)')
    parser.add_argument('--create-ml-datasets', action='store_true',
                       help='Create ML training datasets from imported data')
    
    args = parser.parse_args()
    
    print("Real CAN Dataset Importer for CAN-IDS")
    print("=" * 60)
    
    # Initialize importer
    importer = RealCANDatasetImporter(args.dataset_path)
    
    # Import datasets
    if args.set == 'all':
        for set_name in ['set_01', 'set_02', 'set_03', 'set_04']:
            importer.import_training_set(set_name, args.output)
    else:
        importer.import_training_set(args.set, args.output)
    
    # Create ML datasets if requested
    if args.create_ml_datasets:
        importer.create_training_datasets(args.output)
    
    print("\n" + "=" * 60)
    print("Import complete!")
    print(f"Data saved to: {args.output}")
    print("\nNext steps:")
    print("1. Train ML model: python src/models/train_model.py")
    print("2. Test detection: python main.py --mode replay --file data/real_dataset/ml_datasets/test_data.json")


if __name__ == '__main__':
    main()