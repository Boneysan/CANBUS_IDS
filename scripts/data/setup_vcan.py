#!/usr/bin/env python3
"""
Setup virtual CAN (vcan) interface for testing on Linux.

Creates a virtual CAN interface for development and testing
without requiring physical CAN hardware.
"""

import subprocess
import sys
import argparse


def run_command(command: str, check: bool = True) -> tuple:
    """
    Run a shell command.
    
    Args:
        command: Command to run
        check: Whether to check return code
        
    Returns:
        Tuple of (returncode, stdout, stderr)
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=check
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return e.returncode, e.stdout, e.stderr


def check_root():
    """Check if running as root."""
    if subprocess.run(['id', '-u'], capture_output=True).stdout.decode().strip() != '0':
        print("Error: This script must be run as root")
        print("Usage: sudo python scripts/setup_vcan.py")
        sys.exit(1)


def load_vcan_module():
    """Load vcan kernel module."""
    print("Loading vcan kernel module...")
    returncode, stdout, stderr = run_command('modprobe vcan', check=False)
    
    if returncode != 0:
        print(f"Warning: Could not load vcan module: {stderr}")
        print("vcan module may already be loaded or not available")
    else:
        print("✓ vcan module loaded")


def create_vcan_interface(interface: str = 'vcan0'):
    """
    Create virtual CAN interface.
    
    Args:
        interface: Interface name (default: vcan0)
    """
    print(f"\nCreating virtual CAN interface: {interface}")
    
    # Check if interface already exists
    returncode, stdout, stderr = run_command(f'ip link show {interface}', check=False)
    
    if returncode == 0:
        print(f"Interface {interface} already exists, removing it first...")
        run_command(f'ip link delete {interface}', check=False)
    
    # Create interface
    returncode, stdout, stderr = run_command(f'ip link add dev {interface} type vcan')
    if returncode != 0:
        print(f"Error creating interface: {stderr}")
        sys.exit(1)
    
    print(f"✓ Interface {interface} created")
    
    # Bring interface up
    returncode, stdout, stderr = run_command(f'ip link set up {interface}')
    if returncode != 0:
        print(f"Error bringing up interface: {stderr}")
        sys.exit(1)
    
    print(f"✓ Interface {interface} is up")


def verify_interface(interface: str = 'vcan0'):
    """
    Verify interface is working.
    
    Args:
        interface: Interface name to verify
    """
    print(f"\nVerifying {interface}...")
    
    # Check interface status
    returncode, stdout, stderr = run_command(f'ip link show {interface}')
    
    if returncode == 0:
        print(f"✓ Interface {interface} is active")
        print("\nInterface details:")
        print(stdout)
        return True
    else:
        print(f"✗ Interface {interface} verification failed")
        return False


def setup_persistent_vcan(interface: str = 'vcan0'):
    """
    Setup vcan to load automatically on boot (systemd).
    
    Args:
        interface: Interface name
    """
    print(f"\nSetting up persistent vcan interface...")
    
    service_content = f"""[Unit]
Description=Virtual CAN interface {interface}
After=network.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/sbin/modprobe vcan
ExecStart=/sbin/ip link add dev {interface} type vcan
ExecStart=/sbin/ip link set up {interface}
ExecStop=/sbin/ip link delete {interface}

[Install]
WantedBy=multi-user.target
"""
    
    service_file = f'/etc/systemd/system/vcan-{interface}.service'
    
    try:
        with open(service_file, 'w') as f:
            f.write(service_content)
        
        print(f"✓ Created systemd service: {service_file}")
        
        # Reload systemd and enable service
        run_command('systemctl daemon-reload')
        run_command(f'systemctl enable vcan-{interface}.service', check=False)
        
        print(f"✓ Service enabled (will start on boot)")
        
    except PermissionError:
        print("Error: Permission denied writing systemd service file")
        print("Make sure you're running as root")
        sys.exit(1)


def remove_vcan_interface(interface: str = 'vcan0'):
    """
    Remove virtual CAN interface.
    
    Args:
        interface: Interface name to remove
    """
    print(f"\nRemoving interface {interface}...")
    returncode, stdout, stderr = run_command(f'ip link delete {interface}', check=False)
    
    if returncode == 0:
        print(f"✓ Interface {interface} removed")
    else:
        print(f"Interface {interface} not found or already removed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Setup virtual CAN interface for testing'
    )
    
    parser.add_argument('--interface', default='vcan0',
                       help='Interface name (default: vcan0)')
    parser.add_argument('--remove', action='store_true',
                       help='Remove interface instead of creating')
    parser.add_argument('--persistent', action='store_true',
                       help='Setup interface to persist across reboots')
    parser.add_argument('--no-root-check', action='store_true',
                       help='Skip root user check (for testing)')
    
    args = parser.parse_args()
    
    print("Virtual CAN Interface Setup")
    print("=" * 50)
    
    # Check if running as root (skip on Windows or if flag set)
    if sys.platform != 'win32' and not args.no_root_check:
        check_root()
    
    if args.remove:
        # Remove interface
        remove_vcan_interface(args.interface)
    else:
        # Create interface
        load_vcan_module()
        create_vcan_interface(args.interface)
        verify_interface(args.interface)
        
        if args.persistent:
            setup_persistent_vcan(args.interface)
        
        print("\n" + "=" * 50)
        print(f"✓ Virtual CAN interface {args.interface} is ready!")
        print(f"\nYou can now use it with CAN-IDS:")
        print(f"  python main.py -i {args.interface}")
        print(f"\nTo test the interface:")
        print(f"  cansend {args.interface} 123#DEADBEEF")
        print(f"  candump {args.interface}")


if __name__ == '__main__':
    main()
