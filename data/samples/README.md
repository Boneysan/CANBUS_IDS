# Sample Data Directory

This directory contains example CAN traffic datasets for testing and demonstration purposes.

## Sample Files (to be added)

### Normal Traffic
- `normal_traffic.pcap` - Baseline normal CAN traffic (24 hours)
- `vehicle_startup.pcap` - Vehicle startup sequence
- `highway_driving.pcap` - Highway driving scenario

### Attack Scenarios
- `dos_attack.pcap` - Denial of Service attack
- `replay_attack.pcap` - Message replay attack
- `fuzzing_attack.pcap` - CAN fuzzing attack
- `diagnostic_injection.pcap` - Unauthorized diagnostic messages

### Test Cases
- `test_frequency_violation.pcap` - High frequency messages
- `test_timing_anomaly.pcap` - Irregular timing patterns
- `test_unknown_ids.pcap` - Unauthorized CAN IDs

## Generating Sample Data

Use the provided scripts to generate synthetic CAN traffic:

```bash
# Generate normal traffic
python scripts/generate_dataset.py --type normal --output data/samples/normal.pcap --duration 60

# Generate attack scenarios
python scripts/generate_attack_data.py --attack dos --output data/samples/dos_attack.pcap
python scripts/generate_attack_data.py --attack replay --output data/samples/replay_attack.pcap
```

## Using Sample Data

Test CAN-IDS with sample PCAP files:

```bash
# Analyze normal traffic
python main.py --mode replay --file data/samples/normal_traffic.pcap

# Analyze attack traffic
python main.py --mode replay --file data/samples/dos_attack.pcap
```

## Data Format

All PCAP files should contain SocketCAN format captures. You can create them using:

```bash
# Capture from live interface
candump -l can0

# Or with tcpdump
tcpdump -i can0 -w capture.pcap
```

## Notes

- PCAP files are not tracked in git (see .gitignore)
- Large datasets should be stored externally
- Always sanitize real vehicle data before sharing
- Follow responsible disclosure for security vulnerabilities