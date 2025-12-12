#!/usr/bin/env python3
"""
Generate Vehicle-Specific CAN-IDS Rules from Attack-Free Baseline Data

This script analyzes attack-free (normal) CAN traffic to extract timing and frequency
statistics per CAN ID, then generates optimized detection rules with vehicle-specific
thresholds instead of generic hardcoded values.

Usage:
    # Use default Vehicle_Models data
    python scripts/generate_rules_from_baseline.py
    
    # Use custom baseline data
    python scripts/generate_rules_from_baseline.py --input /path/to/normal_traffic.csv
    
    # Adjust confidence level
    python scripts/generate_rules_from_baseline.py --confidence 0.997  # 3-sigma
    python scripts/generate_rules_from_baseline.py --confidence 0.95   # 2-sigma (more sensitive)
    
    # Output to custom location
    python scripts/generate_rules_from_baseline.py --output config/rules_tuned.yaml

Author: CAN-IDS Project
Date: December 8, 2025
"""

import argparse
import pandas as pd
import numpy as np
import yaml
import statistics
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.feature_extractor import FeatureExtractor


class BaselineAnalyzer:
    """Analyze attack-free CAN traffic to extract baseline statistics."""
    
    def __init__(self, confidence_level: float = 0.997):
        """
        Initialize analyzer.
        
        Args:
            confidence_level: Statistical confidence level (default 0.997 = 3-sigma)
                             0.997 = ±3σ (99.7% coverage)
                             0.954 = ±2σ (95.4% coverage) 
                             0.683 = ±1σ (68.3% coverage)
        """
        self.confidence_level = confidence_level
        self.sigma_multiplier = self._calculate_sigma_multiplier(confidence_level)
        self.extractor = FeatureExtractor(history_size=100)
        
    def _calculate_sigma_multiplier(self, confidence: float) -> float:
        """Convert confidence level to sigma multiplier."""
        if confidence >= 0.997:
            return 3.0
        elif confidence >= 0.954:
            return 2.0
        elif confidence >= 0.683:
            return 1.0
        else:
            return 1.5  # Default fallback
    
    def load_data(self, input_paths: List[str]) -> pd.DataFrame:
        """
        Load CAN data from CSV file(s).
        
        Args:
            input_paths: List of CSV file paths
            
        Returns:
            DataFrame with columns: timestamp, arbitration_id, data_field, attack
        """
        print(f"Loading data from {len(input_paths)} file(s)...")
        
        dfs = []
        for path in input_paths:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Data file not found: {path}")
            
            df = pd.read_csv(path)
            print(f"  Loaded {len(df):,} messages from {path.name}")
            
            # Filter to attack-free only if attack column exists
            if 'attack' in df.columns:
                original_len = len(df)
                df = df[df['attack'] == 0].copy()
                if len(df) < original_len:
                    print(f"    Filtered to {len(df):,} attack-free messages")
            
            dfs.append(df)
        
        combined = pd.concat(dfs, ignore_index=True)
        print(f"\nTotal attack-free messages: {len(combined):,}")
        
        return combined
    
    def analyze_baseline(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Extract timing and frequency statistics per CAN ID.
        
        Args:
            df: DataFrame with CAN messages
            
        Returns:
            Dictionary mapping CAN ID to statistics:
            {
                'interval_mean': float,
                'interval_std': float,
                'interval_min': float,
                'interval_max': float,
                'frequency_mean': float,  # messages per second
                'frequency_std': float,
                'message_count': int,
                'unique_data_patterns': int
            }
        """
        print("\nAnalyzing baseline statistics per CAN ID...")
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        can_id_stats = defaultdict(lambda: {
            'intervals': [],
            'frequencies': [],
            'data_patterns': set(),
            'message_count': 0
        })
        
        # Track last timestamp per CAN ID
        last_timestamps = {}
        
        # Track messages in 1-second windows for frequency calculation
        frequency_windows = defaultdict(lambda: deque(maxlen=1000))
        
        for idx, row in df.iterrows():
            # Convert arbitration_id to int if it's a hex string
            can_id_raw = row['arbitration_id']
            if isinstance(can_id_raw, str):
                can_id = int(can_id_raw, 16)
            else:
                can_id = int(can_id_raw)
            
            timestamp = row['timestamp']
            data = row['data_field']
            
            stats = can_id_stats[can_id]
            stats['message_count'] += 1
            stats['data_patterns'].add(data)
            
            # Calculate interval
            if can_id in last_timestamps:
                interval = timestamp - last_timestamps[can_id]
                if interval > 0:  # Skip duplicate timestamps
                    stats['intervals'].append(interval)
            
            last_timestamps[can_id] = timestamp
            
            # Track for frequency calculation
            frequency_windows[can_id].append(timestamp)
            
            if (idx + 1) % 100000 == 0:
                print(f"  Processed {idx + 1:,} messages...")
        
        print(f"  Completed analysis of {len(df):,} messages")
        
        # Calculate statistics
        results = {}
        
        for can_id, stats in can_id_stats.items():
            intervals = stats['intervals']
            
            if len(intervals) < 10:
                print(f"  Warning: CAN ID 0x{can_id:03X} has only {len(intervals)} intervals, skipping")
                continue
            
            # Calculate frequency (messages per second)
            timestamps = list(frequency_windows[can_id])
            if len(timestamps) >= 2:
                time_span = timestamps[-1] - timestamps[0]
                if time_span > 0:
                    frequency = len(timestamps) / time_span
                else:
                    frequency = 0
            else:
                frequency = 0
            
            # Calculate interval statistics (convert to milliseconds)
            interval_ms = [i * 1000 for i in intervals]
            
            results[can_id] = {
                'interval_mean': statistics.mean(interval_ms),
                'interval_std': statistics.stdev(interval_ms) if len(interval_ms) > 1 else 0,
                'interval_min': min(interval_ms),
                'interval_max': max(interval_ms),
                'frequency_mean': frequency,
                'frequency_std': statistics.stdev([frequency] * len(timestamps)) if len(timestamps) > 1 else 0,
                'message_count': stats['message_count'],
                'unique_data_patterns': len(stats['data_patterns'])
            }
        
        print(f"\nGenerated statistics for {len(results)} CAN IDs")
        
        return results
    
    def generate_rules(self, baseline_stats: Dict[str, Dict], 
                      existing_rules_path: Optional[str] = None) -> Dict:
        """
        Generate YAML rules from baseline statistics.
        
        Args:
            baseline_stats: Statistics per CAN ID from analyze_baseline()
            existing_rules_path: Optional path to existing rules.yaml to preserve non-timing rules
            
        Returns:
            Dictionary suitable for YAML export
        """
        print("\nGenerating detection rules...")
        
        rules = []
        
        # Load existing rules if provided
        existing_rules = []
        timing_rule_ids = set()
        
        if existing_rules_path:
            path = Path(existing_rules_path)
            if path.exists():
                with open(path, 'r') as f:
                    existing = yaml.safe_load(f)
                    if existing and 'rules' in existing:
                        existing_rules = existing['rules']
                        print(f"  Loaded {len(existing_rules)} existing rules")
                        
                        # Identify timing-based rules to replace
                        for rule in existing_rules:
                            if any(k in rule for k in ['expected_interval', 'interval_variance', 
                                                       'max_frequency', 'check_timing']):
                                if 'can_id' in rule:
                                    timing_rule_ids.add(rule['can_id'])
        
        # Preserve non-timing rules
        for rule in existing_rules:
            is_timing_rule = any(k in rule for k in ['expected_interval', 'interval_variance', 
                                                      'max_frequency', 'check_timing'])
            
            # Keep rule if it's not a timing rule, or if we don't have baseline data for it
            if not is_timing_rule or (rule.get('can_id') not in baseline_stats):
                # Update whitelist rule with discovered CAN IDs
                if rule.get('whitelist_mode') and 'allowed_can_ids' in rule:
                    rule['allowed_can_ids'] = sorted(baseline_stats.keys())
                    print(f"  Updated whitelist rule with {len(baseline_stats)} CAN IDs")
                
                rules.append(rule)
                print(f"  Preserved rule: {rule['name']}")
        
        # Generate timing rules from baseline
        for can_id, stats in sorted(baseline_stats.items()):
            mean_interval = stats['interval_mean']
            std_interval = stats['interval_std']
            frequency = stats['frequency_mean']
            message_count = stats['message_count']
            
            # Calculate coefficient of variation (normalized jitter)
            cv = std_interval / mean_interval if mean_interval > 0 else 0
            
            # Skip if variance is too high (unstable timing)
            if cv > 2.0:  # More than 200% variance
                print(f"  Skipping 0x{can_id:03X}: unstable timing (CV={cv:.1f})")
                continue
            
            # ============================================================
            # ADAPTIVE THRESHOLD CALCULATION (Per-CAN-ID)
            # ============================================================
            # Determine traffic category and adjust thresholds based on:
            # 1. Message rate (more samples = tighter thresholds)
            # 2. Jitter level (high CV = looser thresholds)
            # ============================================================
            
            if frequency > 50:
                # High-traffic CAN IDs (>50 msg/s)
                # Tier 1: Loose for obvious attacks (DoS), Tier 2: Tight for subtle attacks
                traffic_category = "high-traffic"
                sigma_extreme_base = 2.5  # Tier 1: Very loose for DoS/flood detection
                sigma_moderate_base = 1.3  # Tier 2: Tight for interval manipulation
                consecutive_base = 3
            elif frequency > 10:
                # Medium-traffic CAN IDs (10-50 msg/s)
                # Balance detection and FPR for medium traffic
                traffic_category = "medium-traffic"
                sigma_extreme_base = 2.8  # Tier 1: Loose for extreme violations
                sigma_moderate_base = 1.5  # Tier 2: Moderate for sustained attacks
                consecutive_base = 4
            else:
                # Low-traffic CAN IDs (<10 msg/s)
                # Higher tolerance due to sparse sampling
                traffic_category = "low-traffic"
                sigma_extreme_base = 3.0  # Tier 1: Very loose
                sigma_moderate_base = 1.7  # Tier 2: Moderate tolerance
                consecutive_base = 3
            
            # Adjust for high natural jitter (coefficient of variation)
            if cv > 0.5:
                # >50% jitter - add tolerance to both tiers and require one more consecutive
                sigma_extreme = sigma_extreme_base + 0.3
                sigma_moderate = sigma_moderate_base + 0.1  # Minimal adjustment for Tier 2
                consecutive_required = consecutive_base + 1
                jitter_note = "high-jitter"
            else:
                sigma_extreme = sigma_extreme_base
                sigma_moderate = sigma_moderate_base
                consecutive_required = consecutive_base
                jitter_note = "low-jitter"
            
            # Calculate thresholds using adaptive sigma
            # Note: interval_variance stores 1-sigma, detection engine multiplies by sigma_extreme
            interval_variance_1sigma = std_interval
            interval_tolerance = self.sigma_multiplier * std_interval  # For display only
            frequency_threshold = frequency + (self.sigma_multiplier * frequency)
            
            # Generate timing rule with adaptive parameters
            rule = {
                'name': f"Timing Anomaly - CAN ID 0x{can_id:03X} ({traffic_category}, {jitter_note})",
                'can_id': can_id,
                'severity': 'MEDIUM',
                'description': (
                    f"Message timing deviates from baseline "
                    f"(learned from {message_count:,} messages, "
                    f"rate={frequency:.1f} msg/s, jitter={cv*100:.1f}%)"
                ),
                'action': 'alert',
                'check_timing': True,
                'expected_interval': round(mean_interval, 2),  # milliseconds
                'interval_variance': round(max(interval_variance_1sigma, 2.0), 2),  # 1-sigma, minimum 2ms
                'sigma_extreme': round(sigma_extreme, 1),  # Tier 1: Multiplier for extreme threshold
                'sigma_moderate': round(sigma_moderate, 1),  # Tier 2: Multiplier for moderate threshold
                'consecutive_required': consecutive_required,  # N for consecutive violations
            }
            
            rules.append(rule)
            print(f"  Generated timing rule for 0x{can_id:03X} ({traffic_category}, {jitter_note}): "
                  f"interval={mean_interval:.1f}±{interval_variance_1sigma:.1f}ms (1σ), "
                  f"extreme={sigma_extreme}σ, consecutive={consecutive_required}")
            
            # Generate frequency rule if traffic rate is significant
            if frequency >= 10:  # At least 10 msg/s average
                freq_rule = {
                    'name': f"High Frequency - CAN ID 0x{can_id:03X}",
                    'can_id': can_id,
                    'severity': 'HIGH',
                    'description': f"Message rate exceeds baseline (normal: {frequency:.1f} msg/s)",
                    'action': 'alert',
                    'max_frequency': round(frequency_threshold, 1),
                    'time_window': 1
                }
                
                rules.append(freq_rule)
                print(f"  Generated frequency rule for 0x{can_id:03X}: "
                      f"max={frequency_threshold:.1f} msg/s")
        
        print(f"\nGenerated {len(rules)} total rules ({len(rules) - len(existing_rules)} new)")
        
        return {
            'rules': rules,
            '_metadata': {
                'generated_by': 'generate_rules_from_baseline.py',
                'confidence_level': self.confidence_level,
                'sigma_multiplier': self.sigma_multiplier,
                'baseline_message_count': sum(s['message_count'] for s in baseline_stats.values()),
                'can_ids_analyzed': len(baseline_stats)
            }
        }
    
    def save_rules(self, rules: Dict, output_path: str):
        """Save rules to YAML file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(rules, f, default_flow_style=False, sort_keys=False)
        
        print(f"\n✅ Rules saved to: {output_path}")
        print(f"   Total rules: {len(rules['rules'])}")
        print(f"   Baseline messages: {rules['_metadata']['baseline_message_count']:,}")
        print(f"   Confidence level: {rules['_metadata']['confidence_level']} ({rules['_metadata']['sigma_multiplier']}σ)")


def main():
    parser = argparse.ArgumentParser(
        description='Generate vehicle-specific CAN-IDS rules from attack-free baseline data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default Vehicle_Models data (3.2M messages)
  python scripts/generate_rules_from_baseline.py
  
  # Use custom baseline data
  python scripts/generate_rules_from_baseline.py --input /path/to/normal_traffic.csv
  
  # Multiple input files
  python scripts/generate_rules_from_baseline.py --input file1.csv file2.csv file3.csv
  
  # Adjust sensitivity (lower = more sensitive, more alerts)
  python scripts/generate_rules_from_baseline.py --confidence 0.95  # 2-sigma
  
  # Output to custom location
  python scripts/generate_rules_from_baseline.py --output config/rules_vehicle_specific.yaml
  
  # Preserve existing non-timing rules
  python scripts/generate_rules_from_baseline.py --existing-rules config/rules.yaml
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        nargs='+',
        help='Input CSV file(s) with attack-free CAN data. If not specified, uses Vehicle_Models data.'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='config/rules_generated.yaml',
        help='Output YAML file for generated rules (default: config/rules_generated.yaml)'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.997,
        choices=[0.683, 0.954, 0.997],
        help='Statistical confidence level: 0.683 (1σ), 0.954 (2σ), 0.997 (3σ) [default: 0.997]'
    )
    
    parser.add_argument(
        '--existing-rules',
        help='Path to existing rules.yaml to preserve non-timing rules'
    )
    
    parser.add_argument(
        '--vehicle-models-path',
        default='../Vehicle_Models/data/raw',
        help='Path to Vehicle_Models data directory (default: ../Vehicle_Models/data/raw)'
    )
    
    args = parser.parse_args()
    
    # Determine input files
    if args.input:
        input_files = args.input
    else:
        # Use default Vehicle_Models attack-free data
        vm_path = Path(__file__).parent.parent / args.vehicle_models_path
        input_files = [
            str(vm_path / 'attack-free-1.csv'),
            str(vm_path / 'attack-free-2.csv')
        ]
        print(f"No input specified, using Vehicle_Models data:")
        for f in input_files:
            print(f"  {f}")
    
    # Initialize analyzer
    analyzer = BaselineAnalyzer(confidence_level=args.confidence)
    
    try:
        # Load data
        df = analyzer.load_data(input_files)
        
        # Analyze baseline
        baseline_stats = analyzer.analyze_baseline(df)
        
        if not baseline_stats:
            print("\n❌ Error: No baseline statistics generated. Check your input data.")
            return 1
        
        # Generate rules
        rules = analyzer.generate_rules(baseline_stats, args.existing_rules)
        
        # Save rules
        analyzer.save_rules(rules, args.output)
        
        print("\n✅ Rule generation complete!")
        print(f"\nNext steps:")
        print(f"  1. Review generated rules: cat {args.output}")
        print(f"  2. Test with: python main.py --config config/can_ids.yaml --rules {args.output} --pcap data.pcap")
        print(f"  3. Evaluate false positive rate on attack-free test data")
        print(f"  4. If FP rate is still high, try lower confidence: --confidence 0.95")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
