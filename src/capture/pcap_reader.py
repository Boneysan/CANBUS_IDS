"""
PCAP file reader for offline CAN traffic analysis.

Supports reading CAN messages from PCAP files for forensic
analysis and detection rule testing.
"""

import logging
import struct
from typing import Generator, Optional, Dict, Any, List
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class PCAPReader:
    """
    PCAP file reader for CAN bus traffic analysis.
    
    Supports socketcan PCAP format commonly used by
    candump and Wireshark.
    """
    
    def __init__(self, filepath: str):
        """
        Initialize PCAP reader.
        
        Args:
            filepath: Path to PCAP file
        """
        self.filepath = Path(filepath)
        self._validate_file()
        
        self._stats = {
            'total_packets': 0,
            'can_packets': 0,
            'errors': 0,
            'file_size': self.filepath.stat().st_size
        }
        
    def _validate_file(self) -> None:
        """Validate PCAP file exists and is readable."""
        if not self.filepath.exists():
            raise FileNotFoundError(f"PCAP file not found: {self.filepath}")
            
        if not self.filepath.is_file():
            raise ValueError(f"Path is not a file: {self.filepath}")
            
        if self.filepath.stat().st_size == 0:
            raise ValueError(f"PCAP file is empty: {self.filepath}")
            
    def read_messages(self) -> Generator[Dict[str, Any], None, None]:
        """
        Read CAN messages from PCAP file.
        
        Yields:
            Dictionary containing CAN message data
        """
        logger.info(f"Reading PCAP file: {self.filepath}")
        
        try:
            with open(self.filepath, 'rb') as f:
                # Read PCAP header
                header = f.read(24)
                if len(header) < 24:
                    raise ValueError("Invalid PCAP file: header too short")
                    
                magic = struct.unpack('I', header[:4])[0]
                if magic not in [0xa1b2c3d4, 0xd4c3b2a1, 0xa1b23c4d, 0x4d3cb2a1]:
                    raise ValueError("Invalid PCAP file: bad magic number")
                    
                # Determine endianness and nanosecond precision
                if magic in [0xa1b2c3d4, 0xa1b23c4d]:
                    endian = '<'  # Little endian
                else:
                    endian = '>'  # Big endian
                    
                nsec_precision = magic in [0xa1b23c4d, 0x4d3cb2a1]
                
                # Read packets
                while True:
                    packet_header = f.read(16)
                    if len(packet_header) < 16:
                        break  # End of file
                        
                    self._stats['total_packets'] += 1
                    
                    # Parse packet header
                    if endian == '<':
                        ts_sec, ts_usec, caplen, origlen = struct.unpack('<IIII', packet_header)
                    else:
                        ts_sec, ts_usec, caplen, origlen = struct.unpack('>IIII', packet_header)
                        
                    # Calculate timestamp
                    if nsec_precision:
                        timestamp = ts_sec + (ts_usec / 1_000_000_000)
                    else:
                        timestamp = ts_sec + (ts_usec / 1_000_000)
                        
                    # Read packet data
                    packet_data = f.read(caplen)
                    if len(packet_data) < caplen:
                        break  # Truncated file
                        
                    # Try to parse as CAN frame
                    try:
                        can_message = self._parse_socketcan_frame(packet_data, timestamp)
                        if can_message:
                            self._stats['can_packets'] += 1
                            yield can_message
                    except Exception as e:
                        self._stats['errors'] += 1
                        logger.debug(f"Error parsing packet: {e}")
                        
        except Exception as e:
            logger.error(f"Error reading PCAP file: {e}")
            raise
            
    def _parse_socketcan_frame(self, data: bytes, timestamp: float) -> Optional[Dict[str, Any]]:
        """
        Parse SocketCAN frame from packet data.
        
        Args:
            data: Raw packet bytes
            timestamp: Packet timestamp
            
        Returns:
            Parsed CAN message or None if not a CAN frame
        """
        # Skip Ethernet and other headers - look for CAN data
        # This is a simplified parser - real implementation would need
        # to handle various link layer types properly
        
        # For SocketCAN PCAP, look for CAN frame structure
        if len(data) < 16:
            return None
            
        # Try different offsets to find CAN data
        for offset in [0, 14, 16, 18]:  # Common header sizes
            if offset + 16 > len(data):
                continue
                
            try:
                frame_data = data[offset:offset + 16]
                
                # Parse CAN frame header (simplified)
                can_id = struct.unpack('>I', frame_data[:4])[0]
                dlc = frame_data[4] & 0x0F
                
                # Extract data bytes
                payload = frame_data[8:8 + min(dlc, 8)]
                
                # Validate CAN ID and DLC
                if dlc > 8:
                    continue
                    
                # Check if this looks like a valid CAN frame
                if can_id > 0x7FF and not (can_id & 0x80000000):  # Standard ID check
                    continue
                    
                return {
                    'timestamp': timestamp,
                    'can_id': can_id & 0x7FF if not (can_id & 0x80000000) else can_id & 0x1FFFFFFF,
                    'dlc': dlc,
                    'data': list(payload),
                    'data_hex': ' '.join(f"{b:02X}" for b in payload),
                    'is_extended': bool(can_id & 0x80000000),
                    'is_remote': bool(can_id & 0x40000000),
                    'is_error': False
                }
                
            except (struct.error, IndexError):
                continue
                
        return None
        
    def get_message_count(self) -> int:
        """
        Get total number of CAN messages in file (requires full read).
        
        Returns:
            Number of CAN messages
        """
        count = 0
        for _ in self.read_messages():
            count += 1
        return count
        
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get PCAP file statistics.
        
        Returns:
            Dictionary containing file statistics
        """
        return self._stats.copy()
        
    def get_file_info(self) -> Dict[str, Any]:
        """
        Get PCAP file information.
        
        Returns:
            Dictionary with file details
        """
        stat = self.filepath.stat()
        
        return {
            'filepath': str(self.filepath),
            'filename': self.filepath.name,
            'size_bytes': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'modified_time': stat.st_mtime,
            'created_time': stat.st_ctime,
            'exists': self.filepath.exists(),
            'readable': self.filepath.is_file()
        }
        
    def extract_time_range(self, start_time: float, end_time: float) -> List[Dict[str, Any]]:
        """
        Extract messages within a specific time range.
        
        Args:
            start_time: Start timestamp (Unix time)
            end_time: End timestamp (Unix time)
            
        Returns:
            List of CAN messages in the time range
        """
        messages = []
        
        for message in self.read_messages():
            if start_time <= message['timestamp'] <= end_time:
                messages.append(message)
            elif message['timestamp'] > end_time:
                break  # Assuming chronological order
                
        return messages
        
    def filter_by_can_id(self, can_id: int) -> Generator[Dict[str, Any], None, None]:
        """
        Filter messages by specific CAN ID.
        
        Args:
            can_id: CAN ID to filter for
            
        Yields:
            CAN messages with matching ID
        """
        for message in self.read_messages():
            if message['can_id'] == can_id:
                yield message


class CANDumpReader:
    """
    Reader for candump log files (text format).
    
    Handles the text format produced by the candump utility:
    (timestamp) interface can_id#data
    """
    
    def __init__(self, filepath: str):
        """
        Initialize candump reader.
        
        Args:
            filepath: Path to candump log file
        """
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Candump file not found: {self.filepath}")
            
    def read_messages(self) -> Generator[Dict[str, Any], None, None]:
        """
        Read CAN messages from candump log file.
        
        Yields:
            Dictionary containing CAN message data
        """
        logger.info(f"Reading candump file: {self.filepath}")
        
        try:
            with open(self.filepath, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                        
                    try:
                        message = self._parse_candump_line(line)
                        if message:
                            yield message
                    except Exception as e:
                        logger.warning(f"Error parsing line {line_num}: {e}")
                        
        except Exception as e:
            logger.error(f"Error reading candump file: {e}")
            raise
            
    def _parse_candump_line(self, line: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single candump log line.
        
        Format: (timestamp) interface can_id#data
        Example: (1609459200.123456) can0 123#DEADBEEF
        
        Args:
            line: Candump log line
            
        Returns:
            Parsed CAN message or None if invalid
        """
        # Parse timestamp
        if not line.startswith('('):
            return None
            
        timestamp_end = line.find(')')
        if timestamp_end == -1:
            return None
            
        try:
            timestamp_str = line[1:timestamp_end]
            timestamp = float(timestamp_str)
        except ValueError:
            return None
            
        # Parse interface and message
        rest = line[timestamp_end + 1:].strip()
        parts = rest.split(' ', 2)
        
        if len(parts) < 2:
            return None
            
        interface = parts[0]
        message_part = parts[1]
        
        # Parse CAN ID and data
        if '#' not in message_part:
            return None
            
        can_id_str, data_str = message_part.split('#', 1)
        
        try:
            # Handle extended IDs (might have flags)
            can_id = int(can_id_str, 16)
            is_extended = can_id > 0x7FF
            
            # Parse data bytes
            data_bytes = []
            if data_str and data_str != 'R':  # 'R' indicates remote frame
                for i in range(0, len(data_str), 2):
                    if i + 1 < len(data_str):
                        byte_str = data_str[i:i+2]
                        data_bytes.append(int(byte_str, 16))
                        
            return {
                'timestamp': timestamp,
                'interface': interface,
                'can_id': can_id,
                'dlc': len(data_bytes),
                'data': data_bytes,
                'data_hex': ' '.join(f"{b:02X}" for b in data_bytes),
                'is_extended': is_extended,
                'is_remote': data_str == 'R',
                'is_error': False
            }
            
        except ValueError:
            return None