"""
CAN-IDS: Controller Area Network Intrusion Detection System

A real-time intrusion detection system for CAN bus networks,
optimized for Raspberry Pi 4 deployment.
"""

__version__ = "1.0.0"
__author__ = "CAN-IDS Development Team"
__license__ = "MIT"

from .capture import CANSniffer, PCAPReader, CANDumpReader
from .detection import RuleEngine, MLDetector, DecisionTreeDetector, FastPreFilter
from .preprocessing import FeatureExtractor, Normalizer
from .alerts import AlertManager

__all__ = [
    'CANSniffer',
    'PCAPReader',
    'CANDumpReader',
    'RuleEngine',
    'MLDetector',
    'DecisionTreeDetector',
    'FastPreFilter',
    'FeatureExtractor',
    'Normalizer',
    'AlertManager',
]