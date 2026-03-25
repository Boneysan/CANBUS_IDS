#!/usr/bin/env python3
"""
Convert candump log files to JSON format for analysis.

Supports various candump log formats and outputs structured JSON
compatible with CAN-IDS.
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any
import sys


class CANDumpConverter:
    """Convert candump logs to JSON format."""
    
    def __init__(self):
        """Initialize converter."""
        # Regex patterns for different candump formats
        self.patterns = {
            # (timestamp) interface can_id#data
            'standard': re.compile(r'\((\d+\.\d+)\)\s+(\w+)\s+([0-9A-Fa-f]+)#([0-9A-Fa-f]*)'),
            # timestamp interface can_id [dlc] data
            'detailed': re.compile(r'(\d+\.\d+)\s+(\w+)\s+([0-9A-Fa-f]+)\s+\[(\d+)\]\s+([0-9A-Fa-f\s]*)'),
            # Simple: can_id#data
            'simple': re.compile(r'([0-9A-Fa-f]+)#([0-9A-Fa-f]*)'),
        }
        
    def parse_line(self, line: str, base_timestamp: float = 0.0) -> Dict[str, Any]:
        """
        Parse a single candump log line.
        
        Args:
            line: Log line to parse
            base_timestamp: Base timestamp for simple format
            
        Returns:
            Parsed message dictionary or None if parse failed
        """
        line = line.strip()
        
        if not line or line.startswith('#'):
            return None
        
        # Try standard format first
        match = self.patterns['standard'].search(line)
        if match:
            timestamp, interface, can_id, data = match.groups()
            return self._create_message(
                float(timestamp),
                int(can_id, 16),
                data
            )
        
        # Try detailed format
        match = self.patterns['detailed'].search(line)
        if match:
            timestamp, interface, can_id, dlc, data = match.groups()
            return self._create_message(
                float(timestamp),
                int(can_id, 16),
                data.replace(' ', '')
            )
        
        # Try simple format
        match = self.patterns['simple'].search(line)
        if match:
            can_id, data = match.groups()
            return self._create_message(
                base_timestamp,
                int(can_id, 16),
                data
            )
        
        return None
        
    def _create_message(self, timestamp: float, can_id: int, 
                       data_hex: str) -> Dict[str, Any]:
        """
        Create message dictionary from parsed components.
        
        Args:
            timestamp: Message timestamp
            can_id: CAN identifier
            data_hex: Hex string of data bytes
            
        Returns:
            Message dictionary
        """
        # Parse data bytes
        data_bytes = []
        if data_hex:
            for i in range(0, len(data_hex), 2):
                byte_str = data_hex[i:i+2]
                if byte_str:
                    data_bytes.append(int(byte_str, 16))
        
        return {
            'timestamp': timestamp,
            'can_id': can_id,
            'dlc': len(data_bytes),
            'data': data_bytes,
            'data_hex': ' '.join(f'{b:02X}' for b in data_bytes),
            'is_extended': can_id > 0x7FF,
            'is_remote': False,
            'is_error': False
        }
        
    def convert_file(self, input_file: str, output_file: str = None) -> List[Dict[str, Any]]:
        """
        Convert candump log file to JSON.
        
        Args:
            input_file: Input candump log file
            output_file: Output JSON file (optional)
            
        Returns:
            List of parsed messages
        """
        input_path = Path(input_file)
        
        if not input_path.exists():
            print(f"Error: Input file not found: {input_file}")
            sys.exit(1)
        
        print(f"Converting: {input_file}")
        
        messages = []
        base_timestamp = 0.0
        line_count = 0
        error_count = 0
        
        with open(input_path, 'r') as f:
            for line in f:
                line_count += 1
                message = self.parse_line(line, base_timestamp)
                
                if message:
                    messages.append(message)
                    base_timestamp = message['timestamp'] + 0.001  # Increment for simple format
                else:
                    if line.strip() and not line.strip().startswith('#'):
                        error_count += 1
        
        print(f"  Total lines: {line_count}")
        print(f"  Parsed messages: {len(messages)}")
        print(f"  Parse errors: {error_count}")
        
        # Save to output file if specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(messages, f, indent=2)
            
            print(f"  Saved to: {output_file}")
        
        return messages
        
    def convert_directory(self, input_dir: str, output_dir: str, 
                         pattern: str = '*.log') -> None:
        """
        Convert all candump logs in a directory.
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            pattern: File pattern to match (default: *.log)
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            print(f"Error: Input directory not found: {input_dir}")
            sys.exit(1)
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all matching files
        files = list(input_path.glob(pattern))
        
        if not files:
            print(f"No files matching pattern '{pattern}' found in {input_dir}")
            return
        
        print(f"Found {len(files)} files to convert\n")
        
        for file in files:
            output_file = output_path / f"{file.stem}.json"
            self.convert_file(str(file), str(output_file))
            print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Convert candump log files to JSON format'
    )
    
    parser.add_argument('input',
                       help='Input candump log file or directory')
    parser.add_argument('-o', '--output',
                       help='Output JSON file or directory')
    parser.add_argument('-d', '--directory', action='store_true',
                       help='Process directory of files')
    parser.add_argument('-p', '--pattern', default='*.log',
                       help='File pattern for directory mode (default: *.log)')
    parser.add_argument('--stats', action='store_true',
                       help='Show statistics about converted data')
    
    args = parser.parse_args()
    
    print("CANDump to JSON Converter")
    print("=" * 50)
    print()
    
    converter = CANDumpConverter()
    
    if args.directory:
        # Convert directory
        output_dir = args.output or 'data/converted'
        converter.convert_directory(args.input, output_dir, args.pattern)
    else:
        # Convert single file
        output_file = args.output
        if not output_file:
            input_path = Path(args.input)
            output_file = f"data/converted/{input_path.stem}.json"
        
        messages = converter.convert_file(args.input, output_file)
        
        # Show statistics if requested
        if args.stats and messages:
            print("\nStatistics:")
            print(f"  Total messages: {len(messages)}")
            print(f"  Duration: {messages[-1]['timestamp'] - messages[0]['timestamp']:.2f}s")
            
            # Count unique CAN IDs
            unique_ids = set(msg['can_id'] for msg in messages)
            print(f"  Unique CAN IDs: {len(unique_ids)}")
            
            # Message rate
            duration = messages[-1]['timestamp'] - messages[0]['timestamp']
            if duration > 0:
                rate = len(messages) / duration
                print(f"  Average rate: {rate:.1f} msg/s")
            
            # Show top 5 most common IDs
            from collections import Counter
            id_counts = Counter(msg['can_id'] for msg in messages)
            print("\n  Top 5 CAN IDs:")
            for can_id, count in id_counts.most_common(5):
                percentage = (count / len(messages)) * 100
                print(f"    0x{can_id:03X}: {count} messages ({percentage:.1f}%)")
    
    print("\n" + "=" * 50)
    print("Conversion complete!")


if __name__ == '__main__':
    main()
