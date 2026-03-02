# Logs Directory

CAN-IDS log files are stored here.

## Log Files

- `can_ids.log` - Main application log
- `alerts.json` - Alert log in structured JSON format
- `*.log.*` - Rotated log files

## Configuration

Log settings are configured in `config/can_ids.yaml`:

```yaml
alerts:
  log_file: logs/alerts.json
  max_size_mb: 100
  max_files: 10
```

## Log Rotation

Logs are automatically rotated when they reach the configured size limit.
On Raspberry Pi, consider using tmpfs to reduce SD card wear:

```bash
sudo mount -t tmpfs -o size=100M tmpfs /path/to/can-ids/logs
```

## Viewing Logs

```bash
# Tail alert log
tail -f logs/alerts.json | jq .

# View systemd logs (if running as service)
sudo journalctl -u can-ids.service -f

# Search for critical alerts
grep "CRITICAL" logs/alerts.json
```

Note: Log files are excluded from git tracking.