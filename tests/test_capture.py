"""
Test suite for CAN message capture modules.

Tests for CANSniffer and PCAP readers.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from collections import deque

# Import modules to test
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.capture.can_sniffer import CANSniffer
from src.capture.pcap_reader import PCAPReader, CANDumpReader


class TestCANSniffer:
    """Test cases for CANSniffer class."""
    
    def test_initialization(self):
        """Test CANSniffer initialization."""
        sniffer = CANSniffer(interface='can0', bitrate=500000, buffer_size=100)
        
        assert sniffer.interface == 'can0'
        assert sniffer.bitrate == 500000
        assert sniffer.buffer_size == 100
        assert sniffer._bus is None
        
    def test_statistics_initialization(self):
        """Test that statistics are properly initialized."""
        sniffer = CANSniffer()
        stats = sniffer.get_statistics()
        
        assert stats['messages_received'] == 0
        assert stats['messages_dropped'] == 0
        assert stats['errors'] == 0
        assert stats['start_time'] is None
        
    @patch('src.capture.can_sniffer.can.Bus')
    def test_start(self, mock_bus):
        """Test starting CAN sniffer."""
        sniffer = CANSniffer(interface='can0')
        sniffer.start()
        
        # Verify bus was created
        mock_bus.assert_called_once()
        assert sniffer._bus is not None
        
        # Verify statistics were updated
        stats = sniffer.get_statistics()
        assert stats['start_time'] is not None
        
    @patch('src.capture.can_sniffer.can.Bus')
    def test_stop(self, mock_bus):
        """Test stopping CAN sniffer."""
        mock_bus_instance = MagicMock()
        mock_bus.return_value = mock_bus_instance
        
        sniffer = CANSniffer()
        sniffer.start()
        sniffer.stop()
        
        # Verify shutdown was called
        mock_bus_instance.shutdown.assert_called_once()
        
    @patch('src.capture.can_sniffer.can.Bus')
    def test_statistics_calculation(self, mock_bus):
        """Test statistics calculation."""
        sniffer = CANSniffer()
        sniffer.start()
        
        # Simulate some time passing
        time.sleep(0.1)
        
        stats = sniffer.get_statistics()
        
        assert 'runtime_seconds' in stats
        assert 'messages_per_second' in stats
        assert stats['runtime_seconds'] >= 0


class TestPCAPReader:
    """Test cases for PCAPReader class."""
    
    def test_initialization(self):
        """Test PCAPReader initialization."""
        reader = PCAPReader('test.pcap')
        
        assert reader.pcap_file == Path('test.pcap')
        assert reader._message_count == 0
        
    def test_invalid_file(self):
        """Test handling of non-existent file."""
        reader = PCAPReader('nonexistent.pcap')
        
        messages = list(reader.read_messages())
        assert len(messages) == 0


class TestCANDumpReader:
    """Test cases for CANDumpReader class."""
    
    def test_initialization(self):
        """Test CANDumpReader initialization."""
        reader = CANDumpReader('test.log')
        
        assert reader.log_file == Path('test.log')
        assert reader._message_count == 0
        
    def test_parse_candump_line(self):
        """Test parsing candump log line."""
        reader = CANDumpReader('test.log')
        
        # Test standard format: (timestamp) interface can_id#data
        line = "(1234.567890) can0 123#DEADBEEF"
        message = reader._parse_candump_line(line)
        
        assert message is not None
        assert message['can_id'] == 0x123
        assert message['data'] == [0xDE, 0xAD, 0xBE, 0xEF]
        assert message['dlc'] == 4
        assert message['timestamp'] == 1234.567890
        
    def test_parse_invalid_line(self):
        """Test parsing invalid log line."""
        reader = CANDumpReader('test.log')
        
        line = "invalid line format"
        message = reader._parse_candump_line(line)
        
        assert message is None


class TestMessageFormats:
    """Test different CAN message format handling."""
    
    def test_extended_id_flag(self):
        """Test extended ID flag handling."""
        reader = CANDumpReader('test.log')
        
        # Standard ID (11-bit)
        line = "(0.0) can0 123#00"
        message = reader._parse_candump_line(line)
        assert message['is_extended'] == False
        
        # Extended ID (29-bit) - IDs > 0x7FF
        line = "(0.0) can0 12345678#00"
        message = reader._parse_candump_line(line)
        assert message['is_extended'] == True
        
    def test_empty_data(self):
        """Test handling of messages with no data."""
        reader = CANDumpReader('test.log')
        
        line = "(0.0) can0 100#"
        message = reader._parse_candump_line(line)
        
        assert message is not None
        assert message['dlc'] == 0
        assert len(message['data']) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
