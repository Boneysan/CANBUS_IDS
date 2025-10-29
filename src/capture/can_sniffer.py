"""
CAN traffic capture from live SocketCAN interfaces.

This module provides real-time CAN bus monitoring capabilities
using the python-can library with SocketCAN backend.
"""

import logging
import time
from typing import Generator, Optional, Dict, Any
import can
from can.message import Message
from threading import Event, Lock
import queue

logger = logging.getLogger(__name__)


class CANSniffer:
    """
    Real-time CAN bus traffic capture using SocketCAN.
    
    Optimized for Raspberry Pi 4 deployment with configurable
    buffer sizes and performance settings.
    """
    
    def __init__(self, interface: str = 'can0', bitrate: int = 500000,
                 buffer_size: int = 1000, bustype: str = 'socketcan'):
        """
        Initialize CAN sniffer.
        
        Args:
            interface: CAN interface name (e.g., 'can0')
            bitrate: CAN bus bitrate in bps
            buffer_size: Internal message buffer size
            bustype: CAN bus type ('socketcan', 'pcan', etc.)
        """
        self.interface = interface
        self.bitrate = bitrate
        self.buffer_size = buffer_size
        self.bustype = bustype
        
        self._bus: Optional[can.Bus] = None
        self._stop_event = Event()
        self._message_buffer = queue.Queue(maxsize=buffer_size)
        self._stats_lock = Lock()
        self._stats = {
            'messages_received': 0,
            'messages_dropped': 0,
            'errors': 0,
            'start_time': None,
            'last_message_time': None
        }
        
    def start(self) -> None:
        """Initialize and start CAN bus connection."""
        try:
            logger.info(f"Starting CAN sniffer on {self.interface}")
            
            self._bus = can.Bus(
                channel=self.interface,
                bustype=self.bustype
            )
            
            with self._stats_lock:
                self._stats['start_time'] = time.time()
                
            logger.info(f"CAN sniffer started on {self.interface}")
            
        except Exception as e:
            logger.error(f"Failed to start CAN sniffer: {e}")
            raise
    
    def stop(self) -> None:
        """Stop CAN bus monitoring and close connection."""
        logger.info("Stopping CAN sniffer")
        self._stop_event.set()
        
        if self._bus:
            self._bus.shutdown()
            self._bus = None
            
    def capture_messages(self) -> Generator[Message, None, None]:
        """
        Generator that yields CAN messages as they arrive.
        
        Yields:
            can.Message: Individual CAN messages
        """
        if not self._bus:
            raise RuntimeError("CAN sniffer not started. Call start() first.")
            
        logger.info("Starting message capture")
        
        try:
            while not self._stop_event.is_set():
                message = self._bus.recv(timeout=1.0)
                
                if message is not None:
                    with self._stats_lock:
                        self._stats['messages_received'] += 1
                        self._stats['last_message_time'] = time.time()
                    
                    # Add to buffer for other consumers
                    try:
                        self._message_buffer.put_nowait(message)
                    except queue.Full:
                        with self._stats_lock:
                            self._stats['messages_dropped'] += 1
                        logger.warning("Message buffer full, dropping message")
                    
                    yield message
                    
        except Exception as e:
            with self._stats_lock:
                self._stats['errors'] += 1
            logger.error(f"Error during message capture: {e}")
            raise
            
    def get_buffered_messages(self) -> Generator[Message, None, None]:
        """
        Get messages from internal buffer.
        
        Yields:
            can.Message: Buffered CAN messages
        """
        while not self._message_buffer.empty():
            try:
                message = self._message_buffer.get_nowait()
                yield message
            except queue.Empty:
                break
                
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get capture statistics.
        
        Returns:
            Dictionary containing capture statistics
        """
        with self._stats_lock:
            stats = self._stats.copy()
            
        # Calculate additional metrics
        if stats['start_time']:
            runtime = time.time() - stats['start_time']
            stats['runtime_seconds'] = runtime
            
            if runtime > 0:
                stats['messages_per_second'] = stats['messages_received'] / runtime
            else:
                stats['messages_per_second'] = 0.0
                
        return stats
        
    def is_connected(self) -> bool:
        """Check if CAN interface is connected and operational."""
        return self._bus is not None and self._bus.state == can.BusState.ACTIVE
        
    def get_can_info(self) -> Dict[str, Any]:
        """
        Get CAN interface information.
        
        Returns:
            Dictionary with interface details
        """
        info = {
            'interface': self.interface,
            'bitrate': self.bitrate,
            'bustype': self.bustype,
            'connected': self.is_connected(),
            'buffer_size': self.buffer_size,
            'buffer_usage': self._message_buffer.qsize()
        }
        
        if self._bus:
            info['state'] = str(self._bus.state)
            
        return info
        
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


class CANMessage:
    """
    Enhanced CAN message wrapper with additional metadata.
    """
    
    def __init__(self, message: Message, timestamp: Optional[float] = None):
        """
        Initialize enhanced CAN message.
        
        Args:
            message: Original can.Message
            timestamp: Custom timestamp (uses current time if None)
        """
        self.message = message
        self.timestamp = timestamp or time.time()
        
        # Extract common fields for easy access
        self.can_id = message.arbitration_id
        self.dlc = message.dlc
        self.data = message.data
        self.is_extended = message.is_extended_id
        self.is_remote = message.is_remote_frame
        self.is_error = message.is_error_frame
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format."""
        return {
            'timestamp': self.timestamp,
            'can_id': f"0x{self.can_id:03X}",
            'dlc': self.dlc,
            'data': [f"0x{b:02X}" for b in self.data],
            'data_hex': ' '.join(f"{b:02X}" for b in self.data),
            'is_extended': self.is_extended,
            'is_remote': self.is_remote,
            'is_error': self.is_error
        }
        
    def __str__(self) -> str:
        """String representation of the message."""
        data_str = ' '.join(f"{b:02X}" for b in self.data)
        return f"CAN ID: 0x{self.can_id:03X}, DLC: {self.dlc}, Data: {data_str}"