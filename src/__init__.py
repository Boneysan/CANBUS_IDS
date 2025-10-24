"""
CAN-IDS: Controller Area Network Intrusion Detection System

A real-time intrusion detection system for CAN bus networks,
optimized for Raspberry Pi 4 deployment.
"""

__version__ = "1.0.0"
__author__ = "CAN-IDS Development Team"
__license__ = "MIT"

from .capture import CANSniffer, PCAPReader
from .detection import RuleEngine, MLDetector
from .preprocessing import FeatureExtractor, Normalizer
from .alerts import AlertManager

__all__ = [
    'CANSniffer',
    'PCAPReader', 
    'RuleEngine',
    'MLDetector',
    'FeatureExtractor',
    'Normalizer',
    'AlertManager'
]