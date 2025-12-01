"""
Notification handlers for different alert delivery methods.

Provides various ways to deliver alerts including console output,
JSON logging, email, and SIEM integration.
"""

import logging
import json
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseNotifier(ABC):
    """Base class for all notifiers."""
    
    @abstractmethod
    def notify(self, alert) -> None:
        """
        Send notification for an alert.
        
        Args:
            alert: Alert object to send
        """
        pass
        
    def shutdown(self) -> None:
        """Cleanup resources when shutting down."""
        pass


class ConsoleNotifier(BaseNotifier):
    """
    Console output notifier.
    
    Prints alerts to stdout with color coding based on severity.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize console notifier.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.use_colors = self.config.get('use_colors', True)
        
        # ANSI color codes
        self.colors = {
            'CRITICAL': '\033[91m',  # Red
            'HIGH': '\033[93m',      # Yellow
            'MEDIUM': '\033[94m',    # Blue
            'LOW': '\033[92m',       # Green
            'INFO': '\033[96m',      # Cyan
            'RESET': '\033[0m'       # Reset
        }
        
    def notify(self, alert) -> None:
        """Print alert to console."""
        severity = alert.severity.value
        
        # Format timestamp
        timestamp_str = time.strftime('%Y-%m-%d %H:%M:%S', 
                                     time.localtime(alert.timestamp))
        
        # Build alert message
        message_parts = [
            f"[{timestamp_str}]",
            f"[{severity}]",
            f"[{alert.source.upper()}]",
            f"CAN ID: 0x{alert.can_id:03X}",
            f"Rule: {alert.rule_name}",
            f"Description: {alert.description}"
        ]
        
        if alert.confidence < 1.0:
            message_parts.append(f"Confidence: {alert.confidence:.2f}")
            
        message = " | ".join(message_parts)
        
        # Apply color coding
        if self.use_colors and severity in self.colors:
            color_code = self.colors[severity]
            reset_code = self.colors['RESET']
            message = f"{color_code}{message}{reset_code}"
            
        print(message)
        
        # Print additional details for high severity alerts
        if severity in ['CRITICAL', 'HIGH']:
            self._print_alert_details(alert)
            
    def _print_alert_details(self, alert) -> None:
        """Print detailed information for critical alerts."""
        print("  Alert Details:")
        print(f"    Alert ID: {alert.id}")
        print(f"    CAN Message Data: {alert.message_data.get('data_hex', 'N/A')}")
        print(f"    DLC: {alert.message_data.get('dlc', 'N/A')}")
        
        if alert.additional_info:
            print("    Additional Info:")
            for key, value in alert.additional_info.items():
                print(f"      {key}: {value}")


class JSONNotifier(BaseNotifier):
    """
    JSON file notifier.
    
    Logs alerts to structured JSON files for analysis and archival.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize JSON notifier.
        
        Args:
            config: Configuration with 'log_file' path
        """
        self.config = config
        self.log_file = Path(config['log_file'])
        
        # Create directory if it doesn't exist
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Rotation settings
        self.max_size_mb = config.get('max_size_mb', 100)
        self.max_files = config.get('max_files', 10)
        
        logger.info(f"JSON notifier initialized: {self.log_file}")
        
    def notify(self, alert) -> None:
        """Write alert to JSON log file."""
        try:
            # Check file size and rotate if needed
            self._check_rotation()
            
            # Convert alert to JSON
            alert_data = alert.to_dict()
            
            # Write to file
            with open(self.log_file, 'a') as f:
                json.dump(alert_data, f, default=str)
                f.write('\n')
                
        except Exception as e:
            logger.error(f"Error writing JSON alert: {e}")
            
    def _check_rotation(self) -> None:
        """Check if log file needs rotation."""
        if not self.log_file.exists():
            return
            
        # Check file size
        size_mb = self.log_file.stat().st_size / (1024 * 1024)
        
        if size_mb >= self.max_size_mb:
            self._rotate_logs()
            
    def _rotate_logs(self) -> None:
        """Rotate log files."""
        try:
            # Move existing files
            for i in range(self.max_files - 1, 0, -1):
                old_file = self.log_file.with_suffix(f'.{i}.json')
                new_file = self.log_file.with_suffix(f'.{i+1}.json')
                
                if old_file.exists():
                    if new_file.exists():
                        new_file.unlink()
                    old_file.rename(new_file)
                    
            # Move current file to .1
            if self.log_file.exists():
                backup_file = self.log_file.with_suffix('.1.json')
                self.log_file.rename(backup_file)
                
            logger.info(f"Rotated log files: {self.log_file}")
            
        except Exception as e:
            logger.error(f"Error rotating logs: {e}")


class EmailNotifier(BaseNotifier):
    """
    Email notifier for critical alerts.
    
    Sends email notifications for high-priority alerts.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize email notifier.
        
        Args:
            config: Email configuration
        """
        self.config = config
        self.enabled = config.get('enabled', False)
        
        if not self.enabled:
            logger.info("Email notifier disabled")
            return
            
        self.smtp_server = config.get('smtp_server', 'localhost')
        self.smtp_port = config.get('smtp_port', 587)
        self.username = config.get('username')
        self.password = config.get('password')
        self.from_email = config.get('from_email', 'canids@localhost')
        self.recipients = config.get('recipients', [])
        self.use_tls = config.get('use_tls', True)
        
        # Rate limiting for emails
        self.min_interval = config.get('min_interval_minutes', 15) * 60
        self.last_sent = {}
        
        logger.info(f"Email notifier initialized: {len(self.recipients)} recipients")
        
    def notify(self, alert) -> None:
        """Send email for critical alerts."""
        if not self.enabled:
            return
            
        # Only send emails for HIGH and CRITICAL alerts
        if alert.severity.value not in ['HIGH', 'CRITICAL']:
            return
            
        # Check rate limiting
        if not self._should_send_email(alert):
            return
            
        try:
            self._send_email(alert)
            self.last_sent[alert.rule_name] = time.time()
            
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
            
    def _should_send_email(self, alert) -> bool:
        """Check if email should be sent based on rate limiting."""
        rule_name = alert.rule_name
        
        if rule_name in self.last_sent:
            time_since_last = time.time() - self.last_sent[rule_name]
            return time_since_last >= self.min_interval
            
        return True
        
    def _send_email(self, alert) -> None:
        """Send email notification."""
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
        except ImportError:
            logger.error("Email functionality requires smtplib - not available")
            return
            
        # Create message
        msg = MIMEMultipart()
        msg['From'] = self.from_email
        msg['To'] = ', '.join(self.recipients)
        msg['Subject'] = f"CAN-IDS Alert: {alert.severity.value} - {alert.rule_name}"
        
        # Email body
        body = f"""
CAN-IDS Security Alert

Severity: {alert.severity.value}
Rule: {alert.rule_name}
Description: {alert.description}
CAN ID: 0x{alert.can_id:03X}
Confidence: {alert.confidence:.2f}
Timestamp: {time.ctime(alert.timestamp)}
Source: {alert.source}

Message Data:
  DLC: {alert.message_data.get('dlc', 'N/A')}
  Data: {alert.message_data.get('data_hex', 'N/A')}
  Extended: {alert.message_data.get('is_extended', False)}
  Remote: {alert.message_data.get('is_remote', False)}

Alert ID: {alert.id}

This alert was generated by CAN-IDS monitoring system.
Please investigate immediately if this is a CRITICAL alert.
"""
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            if self.use_tls:
                server.starttls()
                
            if self.username and self.password:
                server.login(self.username, self.password)
                
            server.send_message(msg)
            
        logger.info(f"Email alert sent: {alert.severity.value} - {alert.rule_name}")


class SyslogNotifier(BaseNotifier):
    """
    Syslog notifier for integration with system logs.
    
    Sends alerts to system syslog facility.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize syslog notifier.
        
        Args:
            config: Syslog configuration
        """
        self.config = config
        self.enabled = config.get('enabled', False)
        
        if not self.enabled:
            return
            
        try:
            import syslog
            self.syslog = syslog
            
            # Priority mapping
            self.priority_map = {
                'CRITICAL': syslog.LOG_CRIT,
                'HIGH': syslog.LOG_WARNING,
                'MEDIUM': syslog.LOG_NOTICE,
                'LOW': syslog.LOG_INFO,
                'INFO': syslog.LOG_INFO
            }
            
            # Open syslog
            facility = config.get('facility', 'LOG_USER')
            facility_value = getattr(syslog, facility, syslog.LOG_USER)
            syslog.openlog("canids", syslog.LOG_PID, facility_value)
            
            logger.info("Syslog notifier initialized")
            
        except ImportError:
            logger.warning("Syslog not available on this platform")
            self.enabled = False
            
    def notify(self, alert) -> None:
        """Send alert to syslog."""
        if not self.enabled:
            return
            
        try:
            priority = self.priority_map.get(alert.severity.value, 
                                           self.syslog.LOG_INFO)
            
            message = (
                f"CAN-IDS Alert: {alert.severity.value} | "
                f"Rule: {alert.rule_name} | "
                f"CAN ID: 0x{alert.can_id:03X} | "
                f"Description: {alert.description}"
            )
            
            self.syslog.syslog(priority, message)
            
        except Exception as e:
            logger.error(f"Error sending syslog alert: {e}")
            
    def shutdown(self) -> None:
        """Close syslog connection."""
        if self.enabled and hasattr(self, 'syslog'):
            self.syslog.closelog()


class WebhookNotifier(BaseNotifier):
    """
    Webhook notifier for integration with external systems.
    
    Sends alerts via HTTP POST to configured webhook URLs.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize webhook notifier.
        
        Args:
            config: Webhook configuration
        """
        self.config = config
        self.enabled = config.get('enabled', False)
        self.urls = config.get('urls', [])
        self.timeout = config.get('timeout', 30)
        self.headers = config.get('headers', {'Content-Type': 'application/json'})
        
        if self.enabled and not self.urls:
            logger.warning("Webhook notifier enabled but no URLs configured")
            self.enabled = False
            
        logger.info(f"Webhook notifier initialized: {len(self.urls)} URLs")
        
    def notify(self, alert) -> None:
        """Send alert via webhook."""
        if not self.enabled:
            return
            
        try:
            import requests
        except ImportError:
            logger.error("Webhook functionality requires requests library")
            return
            
        alert_data = alert.to_dict()
        
        for url in self.urls:
            try:
                response = requests.post(
                    url,
                    json=alert_data,
                    headers=self.headers,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    logger.debug(f"Webhook sent successfully: {url}")
                else:
                    logger.warning(f"Webhook failed: {url} - {response.status_code}")
                    
            except Exception as e:
                logger.error(f"Error sending webhook to {url}: {e}")


def create_notifiers(config: Dict[str, Any]) -> List[BaseNotifier]:
    """
    Create notifiers based on configuration.
    
    Args:
        config: Alert configuration dictionary
        
    Returns:
        List of configured notifiers
    """
    notifiers = []
    
    # Console notifier
    if config.get('console_output', True):
        console_config = config.get('console', {})
        notifiers.append(ConsoleNotifier(console_config))
        
    # JSON file notifier
    if config.get('log_file'):
        json_config = {
            'log_file': config['log_file'],
            'max_size_mb': config.get('max_log_size_mb', 100),
            'max_files': config.get('max_log_files', 10)
        }
        notifiers.append(JSONNotifier(json_config))
        
    # Email notifier
    if config.get('email_alerts', False):
        email_config = config.get('email', {})
        email_config['enabled'] = True
        notifiers.append(EmailNotifier(email_config))
        
    # Syslog notifier
    if config.get('syslog_enabled', False):
        syslog_config = config.get('syslog', {})
        syslog_config['enabled'] = True
        notifiers.append(SyslogNotifier(syslog_config))
        
    # Webhook notifier
    if config.get('webhook_enabled', False):
        webhook_config = config.get('webhook', {})
        webhook_config['enabled'] = True
        notifiers.append(WebhookNotifier(webhook_config))
        
    logger.info(f"Created {len(notifiers)} notifiers")
    return notifiers