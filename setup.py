#!/usr/bin/env python3
"""
Setup script for CAN-IDS: Controller Area Network Intrusion Detection System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "CAN-IDS: Controller Area Network Intrusion Detection System"

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, "r") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = []

setup(
    name="can-ids",
    version="1.0.0",
    author="CAN-IDS Development Team",
    author_email="canids@example.com",
    description="Real-time intrusion detection system for CAN bus networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/can-ids",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/can-ids/issues",
        "Source": "https://github.com/yourusername/can-ids",
        "Documentation": "https://can-ids.readthedocs.io",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Security",
        "Topic :: System :: Networking :: Monitoring",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Environment :: Console",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.910",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
        "raspberry-pi": [
            "RPi.GPIO>=0.7.0",
            "gpiozero>=1.6.0",
        ],
        "email": [
            "smtplib-ssl>=1.0",
        ],
        "webhook": [
            "requests>=2.25.0",
        ],
        "all": [
            "RPi.GPIO>=0.7.0",
            "gpiozero>=1.6.0",
            "smtplib-ssl>=1.0", 
            "requests>=2.25.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "can-ids=main:main",
            "canids=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml", "config/*.conf", "data/samples/*.pcap"],
    },
    data_files=[
        ("share/can-ids/config", ["config/can_ids.conf", "config/can_ids_rpi4.conf"]),
        ("share/can-ids/config", ["config/rules.yaml", "config/example_rules.yaml"]),
        ("share/can-ids/systemd", ["raspberry-pi/systemd/can-ids.service"]),
    ],
    zip_safe=False,
    keywords=[
        "can-bus",
        "intrusion-detection",
        "automotive-security", 
        "industrial-security",
        "anomaly-detection",
        "machine-learning",
        "raspberry-pi",
        "socketcan",
        "cybersecurity",
        "network-security",
    ],
    platforms=["Linux"],
)