#!/usr/bin/env python3
"""
Test CAN-IDS rule engine with real CAN bus data from Vehicle_Models.

This script loads real CAN bus datasets and tests the rule engine's
detection capabilities across different attack types.
"""

import sys
import pandas as pd
import time
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.rule_engine import RuleEngine


class RealDataTester:
    """Test rule engine with real CAN bus data."""
    
    def __init__(self, rules_file: str, data_dir: str):
        """
        Initialize tester.
        
        Args:
            rules_file: Path to rules configuration
            data_dir: Path to Vehicle_Models data directory
        """
        self.rules_file = Path(rules_file)
        self.data_dir = Path(data_dir)
        self.engine = RuleEngine(str(self.rules_file))
        
        # Statistics
        self.stats = {
            'messages_processed': 0,
            'alerts_generated': 0,
            'by_attack_type': defaultdict(lambda: {'processed': 0, 'alerts': 0}),
            'by_rule': defaultdict(int),
            'processing_time': 0
        }
    
    def load_csv_file(self, csv_file: Path, max_messages: int = None) -> List[Dict[str, Any]]:
        """
        Load CAN messages from CSV file.
        
        Args:
            csv_file: Path to CSV file
            max_messages: Maximum messages to load (None = all)
            
        Returns:
            List of CAN messages in rule engine format
        """
        print(f"\nLoading {csv_file.name}...")
        
        try:
            df = pd.read_csv(csv_file)
            
            # Limit messages if specified
            if max_messages:
                df = df.head(max_messages)
            
            print(f"  Loaded {len(df)} messages")
            
            messages = []
            for _, row in df.iterrows():
                # Convert CSV row to CAN message format
                message = {
                    'timestamp': float(row.get('Timestamp', time.time())),
                    'can_id': int(row.get('ID', row.get('CAN_ID', 0)), 16) if isinstance(row.get('ID', row.get('CAN_ID', 0)), str) else int(row.get('ID', row.get('CAN_ID', 0))),
                    'dlc': int(row.get('DLC', 8)),
                    'data': self._parse_data_field(row),
                    'is_extended': False,  # Most automotive CAN uses standard frames
                    'is_error_frame': False,
                    'is_remote_frame': False
                }
                
                # Add attack label if present
                if 'Flag' in row:
                    message['is_attack'] = row['Flag'] == 'T'  # T = attack, R = normal
                elif 'Label' in row:
                    message['is_attack'] = row['Label'] != 0
                
                messages.append(message)
            
            return messages
            
        except Exception as e:
            print(f"  Error loading {csv_file.name}: {e}")
            return []
    
    def _parse_data_field(self, row: pd.Series) -> List[int]:
        """Parse data field from CSV row."""
        # Try different column names for data
        data_str = None
        if 'Data' in row:
            data_str = row['Data']
        elif 'DATA' in row:
            data_str = row['DATA']
        else:
            # Try individual byte columns (DATA[0], DATA[1], etc.)
            data_bytes = []
            for i in range(8):
                col_name = f'DATA[{i}]'
                if col_name in row:
                    data_bytes.append(int(row[col_name]))
            if data_bytes:
                return data_bytes
        
        # Parse hex string (e.g., "00 01 02 03 04 05 06 07")
        if data_str and isinstance(data_str, str):
            try:
                return [int(b, 16) for b in data_str.split()]
            except:
                pass
        
        # Default to zeros
        return [0] * 8
    
    def test_file(self, csv_file: Path, max_messages: int = None) -> Dict[str, Any]:
        """
        Test rule engine on a single CSV file.
        
        Args:
            csv_file: Path to CSV file
            max_messages: Maximum messages to process
            
        Returns:
            Test results dictionary
        """
        # Determine attack type from filename
        attack_type = self._get_attack_type(csv_file.name)
        
        # Load messages
        messages = self.load_csv_file(csv_file, max_messages)
        if not messages:
            return {'error': 'Failed to load messages'}
        
        # Process messages
        print(f"  Testing {attack_type} detection...")
        start_time = time.time()
        
        alerts = []
        attack_count = 0
        
        for msg in messages:
            # Track ground truth
            if msg.get('is_attack', False):
                attack_count += 1
            
            # Analyze with rule engine
            msg_alerts = self.engine.analyze_message(msg)
            alerts.extend(msg_alerts)
            
            # Update statistics
            self.stats['by_attack_type'][attack_type]['processed'] += 1
            if msg_alerts:
                self.stats['by_attack_type'][attack_type]['alerts'] += len(msg_alerts)
                for alert in msg_alerts:
                    self.stats['by_rule'][alert.rule_name] += 1
        
        processing_time = time.time() - start_time
        
        # Calculate results
        results = {
            'file': csv_file.name,
            'attack_type': attack_type,
            'messages_processed': len(messages),
            'attack_messages': attack_count,
            'normal_messages': len(messages) - attack_count,
            'alerts_generated': len(alerts),
            'processing_time': processing_time,
            'messages_per_second': len(messages) / processing_time if processing_time > 0 else 0,
            'detection_rate': (len(alerts) / attack_count * 100) if attack_count > 0 else 0,
            'rules_triggered': len(set(a.rule_name for a in alerts))
        }
        
        # Update global stats
        self.stats['messages_processed'] += len(messages)
        self.stats['alerts_generated'] += len(alerts)
        self.stats['processing_time'] += processing_time
        
        return results
    
    def _get_attack_type(self, filename: str) -> str:
        """Determine attack type from filename."""
        filename_lower = filename.lower()
        
        if 'dos' in filename_lower:
            return 'DoS'
        elif 'fuzzing' in filename_lower or 'fuzz' in filename_lower:
            return 'Fuzzing'
        elif 'rpm' in filename_lower:
            return 'RPM Spoofing'
        elif 'gear' in filename_lower or 'neutral' in filename_lower:
            return 'Gear Spoofing'
        elif 'interval' in filename_lower:
            return 'Interval Attack'
        elif 'accessory' in filename_lower:
            return 'Accessory Attack'
        elif 'standstill' in filename_lower:
            return 'Standstill'
        elif 'attack-free' in filename_lower or 'normal' in filename_lower:
            return 'Normal Traffic'
        else:
            return 'Unknown'
    
    def test_all_files(self, max_messages_per_file: int = 10000):
        """
        Test all CSV files in data directory.
        
        Args:
            max_messages_per_file: Limit messages per file for faster testing
        """
        print("="*70)
        print("CAN-IDS Rule Engine Test with Real Data")
        print("="*70)
        print(f"\nData Directory: {self.data_dir}")
        print(f"Rules File: {self.rules_file}")
        print(f"Max Messages Per File: {max_messages_per_file}")
        
        # Find all CSV files
        csv_files = sorted(self.data_dir.glob("*.csv"))
        
        if not csv_files:
            print("\nNo CSV files found!")
            return
        
        print(f"\nFound {len(csv_files)} CSV files")
        
        # Test each file
        results = []
        for csv_file in csv_files:
            result = self.test_file(csv_file, max_messages_per_file)
            if 'error' not in result:
                results.append(result)
                self._print_result(result)
        
        # Print summary
        self._print_summary(results)
    
    def _print_result(self, result: Dict[str, Any]):
        """Print individual file results."""
        print(f"\n  Results:")
        print(f"    Messages: {result['messages_processed']:,} ({result['attack_messages']:,} attacks, {result['normal_messages']:,} normal)")
        print(f"    Alerts: {result['alerts_generated']:,}")
        print(f"    Detection Rate: {result['detection_rate']:.1f}%")
        print(f"    Rules Triggered: {result['rules_triggered']}")
        print(f"    Speed: {result['messages_per_second']:,.0f} msg/s")
    
    def _print_summary(self, results: List[Dict[str, Any]]):
        """Print overall summary."""
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        # Overall statistics
        print(f"\nTotal Messages Processed: {self.stats['messages_processed']:,}")
        print(f"Total Alerts Generated: {self.stats['alerts_generated']:,}")
        print(f"Total Processing Time: {self.stats['processing_time']:.2f}s")
        print(f"Average Speed: {self.stats['messages_processed']/self.stats['processing_time']:,.0f} msg/s")
        
        # By attack type
        print("\nDetection by Attack Type:")
        print("-" * 70)
        print(f"{'Attack Type':<25} {'Messages':<12} {'Alerts':<12} {'Rate':<10}")
        print("-" * 70)
        
        for attack_type, stats in sorted(self.stats['by_attack_type'].items()):
            rate = (stats['alerts'] / stats['processed'] * 100) if stats['processed'] > 0 else 0
            print(f"{attack_type:<25} {stats['processed']:<12,} {stats['alerts']:<12,} {rate:>9.1f}%")
        
        # Top triggered rules
        print("\nTop Triggered Rules:")
        print("-" * 70)
        for rule_name, count in sorted(self.stats['by_rule'].items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {rule_name:<50} {count:>8,} alerts")
        
        print("\n" + "="*70)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test CAN-IDS with real data')
    parser.add_argument('--data-dir', 
                       default='/home/mike/Documents/GitHub/Vehicle_Models/data/raw',
                       help='Path to Vehicle_Models data directory')
    parser.add_argument('--rules',
                       default='config/rules.yaml',
                       help='Path to rules file')
    parser.add_argument('--max-messages',
                       type=int,
                       default=10000,
                       help='Maximum messages per file (for faster testing)')
    parser.add_argument('--file',
                       help='Test specific file only')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = RealDataTester(args.rules, args.data_dir)
    
    # Test
    if args.file:
        # Test single file
        file_path = Path(args.data_dir) / args.file
        if file_path.exists():
            result = tester.test_file(file_path, args.max_messages)
            tester._print_result(result)
            tester._print_summary([result])
        else:
            print(f"File not found: {file_path}")
    else:
        # Test all files
        tester.test_all_files(args.max_messages)


if __name__ == '__main__':
    main()
