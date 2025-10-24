#!/usr/bin/env python3
"""
Generate synthetic CAN bus traffic dataset for testing and training.

Creates realistic CAN traffic patterns including normal operation
and various attack scenarios.
"""

import argparse
import random
import time
import json
from pathlib import Path
from typing import List, Dict, Any
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class CANDatasetGenerator:
    """Generate synthetic CAN bus traffic for testing."""
    
    def __init__(self, seed: int = 42):
        """
        Initialize dataset generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        
        # Common automotive CAN IDs with realistic patterns
        self.normal_ids = {
            0x100: {'freq': 100, 'dlc': 8, 'name': 'Engine_RPM'},
            0x110: {'freq': 50, 'dlc': 8, 'name': 'Vehicle_Speed'},
            0x120: {'freq': 100, 'dlc': 8, 'name': 'Brake_Status'},
            0x130: {'freq': 20, 'dlc': 8, 'name': 'Steering_Angle'},
            0x200: {'freq': 10, 'dlc': 8, 'name': 'Airbag_Status'},
            0x210: {'freq': 5, 'dlc': 8, 'name': 'Door_Status'},
            0x220: {'freq': 100, 'dlc': 8, 'name': 'ABS_Status'},
            0x300: {'freq': 50, 'dlc': 8, 'name': 'Transmission'},
            0x400: {'freq': 1, 'dlc': 8, 'name': 'Climate_Control'},
            0x500: {'freq': 10, 'dlc': 8, 'name': 'Battery_Status'},
        }
        
    def generate_normal_traffic(self, duration: float = 60.0, 
                               output_file: str = 'normal_traffic.json') -> None:
        """
        Generate normal CAN traffic.
        
        Args:
            duration: Duration in seconds
            output_file: Output file path
        """
        print(f"Generating {duration}s of normal traffic...")
        
        messages = []
        start_time = time.time()
        current_time = 0.0
        
        while current_time < duration:
            for can_id, config in self.normal_ids.items():
                # Calculate message interval based on frequency
                interval = 1.0 / config['freq'] if config['freq'] > 0 else 1.0
                
                # Generate message if it's time
                if current_time % interval < 0.01:  # Small tolerance
                    message = self._generate_normal_message(can_id, config, start_time + current_time)
                    messages.append(message)
            
            current_time += 0.01  # 10ms resolution
        
        # Save to file
        self._save_messages(messages, output_file)
        print(f"Generated {len(messages)} messages -> {output_file}")
        
    def generate_dos_attack(self, duration: float = 10.0, 
                           target_id: int = 0x100,
                           output_file: str = 'dos_attack.json') -> None:
        """
        Generate DoS attack traffic (high frequency flooding).
        
        Args:
            duration: Attack duration in seconds
            target_id: Target CAN ID to flood
            output_file: Output file path
        """
        print(f"Generating {duration}s DoS attack on ID 0x{target_id:03X}...")
        
        messages = []
        start_time = time.time()
        current_time = 0.0
        
        # Mix normal traffic with attack
        while current_time < duration:
            # Generate attack messages (high frequency)
            if random.random() < 0.8:  # 80% attack traffic
                message = {
                    'timestamp': start_time + current_time,
                    'can_id': target_id,
                    'dlc': 8,
                    'data': [random.randint(0, 255) for _ in range(8)],
                    'is_extended': False,
                    'is_remote': False,
                    'is_error': False
                }
                messages.append(message)
            
            # Some normal traffic
            else:
                can_id = random.choice(list(self.normal_ids.keys()))
                config = self.normal_ids[can_id]
                message = self._generate_normal_message(can_id, config, start_time + current_time)
                messages.append(message)
            
            current_time += 0.001  # Very short interval for DoS
        
        self._save_messages(messages, output_file)
        print(f"Generated {len(messages)} messages -> {output_file}")
        
    def generate_fuzzing_attack(self, duration: float = 10.0,
                               output_file: str = 'fuzzing_attack.json') -> None:
        """
        Generate fuzzing attack (random CAN IDs and data).
        
        Args:
            duration: Attack duration in seconds
            output_file: Output file path
        """
        print(f"Generating {duration}s fuzzing attack...")
        
        messages = []
        start_time = time.time()
        current_time = 0.0
        
        while current_time < duration:
            # Mix fuzzing with normal traffic
            if random.random() < 0.3:  # 30% fuzzing
                message = {
                    'timestamp': start_time + current_time,
                    'can_id': random.randint(0, 0x7FF),  # Random CAN ID
                    'dlc': random.randint(0, 8),  # Random DLC
                    'data': [random.randint(0, 255) for _ in range(random.randint(0, 8))],
                    'is_extended': random.choice([True, False]),
                    'is_remote': False,
                    'is_error': False
                }
                messages.append(message)
            else:
                # Normal traffic
                can_id = random.choice(list(self.normal_ids.keys()))
                config = self.normal_ids[can_id]
                message = self._generate_normal_message(can_id, config, start_time + current_time)
                messages.append(message)
            
            current_time += 0.01
        
        self._save_messages(messages, output_file)
        print(f"Generated {len(messages)} messages -> {output_file}")
        
    def generate_replay_attack(self, duration: float = 10.0,
                              output_file: str = 'replay_attack.json') -> None:
        """
        Generate replay attack (repeated message patterns).
        
        Args:
            duration: Attack duration in seconds
            output_file: Output file path
        """
        print(f"Generating {duration}s replay attack...")
        
        messages = []
        start_time = time.time()
        
        # Capture a "legitimate" message to replay
        replay_message = {
            'can_id': 0x100,
            'dlc': 8,
            'data': [0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0],
            'is_extended': False,
            'is_remote': False,
            'is_error': False
        }
        
        current_time = 0.0
        while current_time < duration:
            # Mix replay with normal traffic
            if random.random() < 0.4:  # 40% replay
                message = replay_message.copy()
                message['timestamp'] = start_time + current_time
                messages.append(message)
            else:
                # Normal traffic
                can_id = random.choice(list(self.normal_ids.keys()))
                config = self.normal_ids[can_id]
                message = self._generate_normal_message(can_id, config, start_time + current_time)
                messages.append(message)
            
            current_time += 0.01
        
        self._save_messages(messages, output_file)
        print(f"Generated {len(messages)} messages -> {output_file}")
        
    def generate_diagnostic_attack(self, duration: float = 10.0,
                                   output_file: str = 'diagnostic_attack.json') -> None:
        """
        Generate diagnostic/UDS attack traffic.
        
        Args:
            duration: Attack duration in seconds
            output_file: Output file path
        """
        print(f"Generating {duration}s diagnostic attack...")
        
        messages = []
        start_time = time.time()
        current_time = 0.0
        
        # UDS diagnostic IDs
        diagnostic_ids = [0x7DF, 0x7E0, 0x7E1, 0x7E2, 0x7E3]
        
        # Common UDS service IDs
        uds_services = [
            [0x10, 0x01],  # Diagnostic Session Control
            [0x27, 0x01],  # Security Access - Request Seed
            [0x27, 0x02],  # Security Access - Send Key
            [0x22, 0xF1, 0x90],  # Read Data By ID
            [0x2E, 0xF1, 0x90],  # Write Data By ID
            [0x31, 0x01, 0xFF, 0x00],  # Routine Control
        ]
        
        while current_time < duration:
            if random.random() < 0.2:  # 20% diagnostic messages
                message = {
                    'timestamp': start_time + current_time,
                    'can_id': random.choice(diagnostic_ids),
                    'dlc': 8,
                    'data': random.choice(uds_services) + [0x00] * (8 - len(random.choice(uds_services))),
                    'is_extended': False,
                    'is_remote': False,
                    'is_error': False
                }
                messages.append(message)
            else:
                # Normal traffic
                can_id = random.choice(list(self.normal_ids.keys()))
                config = self.normal_ids[can_id]
                message = self._generate_normal_message(can_id, config, start_time + current_time)
                messages.append(message)
            
            current_time += 0.01
        
        self._save_messages(messages, output_file)
        print(f"Generated {len(messages)} messages -> {output_file}")
        
    def _generate_normal_message(self, can_id: int, config: Dict[str, Any], 
                                 timestamp: float) -> Dict[str, Any]:
        """Generate a normal CAN message."""
        # Generate realistic data based on message type
        data = []
        for i in range(config['dlc']):
            # Add some variation to simulate real sensor data
            if 'Engine' in config['name']:
                data.append(random.randint(80, 120))  # RPM variation
            elif 'Speed' in config['name']:
                data.append(random.randint(0, 100))
            else:
                data.append(random.randint(0, 255))
        
        return {
            'timestamp': timestamp,
            'can_id': can_id,
            'dlc': config['dlc'],
            'data': data,
            'is_extended': False,
            'is_remote': False,
            'is_error': False
        }
        
    def _save_messages(self, messages: List[Dict[str, Any]], output_file: str) -> None:
        """Save messages to JSON file."""
        output_path = Path('data/synthetic') / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(messages, f, indent=2)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate synthetic CAN bus traffic datasets'
    )
    
    parser.add_argument('--type', 
                       choices=['normal', 'dos', 'fuzzing', 'replay', 'diagnostic', 'all'],
                       default='all',
                       help='Type of dataset to generate')
    parser.add_argument('--duration', type=float, default=60.0,
                       help='Duration in seconds (default: 60)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--output-dir', default='data/synthetic',
                       help='Output directory (default: data/synthetic)')
    
    args = parser.parse_args()
    
    # Create generator
    generator = CANDatasetGenerator(seed=args.seed)
    
    print(f"CAN Dataset Generator")
    print(f"{'=' * 50}")
    
    # Generate requested datasets
    if args.type == 'all' or args.type == 'normal':
        generator.generate_normal_traffic(duration=args.duration, 
                                         output_file='normal_traffic.json')
    
    if args.type == 'all' or args.type == 'dos':
        generator.generate_dos_attack(duration=args.duration / 6,
                                     output_file='dos_attack.json')
    
    if args.type == 'all' or args.type == 'fuzzing':
        generator.generate_fuzzing_attack(duration=args.duration / 6,
                                         output_file='fuzzing_attack.json')
    
    if args.type == 'all' or args.type == 'replay':
        generator.generate_replay_attack(duration=args.duration / 6,
                                        output_file='replay_attack.json')
    
    if args.type == 'all' or args.type == 'diagnostic':
        generator.generate_diagnostic_attack(duration=args.duration / 6,
                                            output_file='diagnostic_attack.json')
    
    print(f"\n{'=' * 50}")
    print("Dataset generation complete!")
    print(f"Files saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
