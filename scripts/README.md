# Utility Scripts

Helper scripts for CAN-IDS development, testing, and data generation.

## Available Scripts

### Data Generation
- `generate_dataset.py` - Generate synthetic CAN traffic
- `generate_attack_data.py` - Generate attack scenarios
- `convert_candump.py` - Convert candump logs to CSV

### Testing & Benchmarking
- `benchmark.py` - Performance benchmarking
- `visualize_traffic.py` - Visualize CAN traffic patterns

### Setup & Configuration
- `setup_can_interface.sh` - Configure CAN interfaces (in raspberry-pi/scripts/)
- See raspberry-pi/scripts/ for Pi-specific utilities

## TODO: Create Scripts

Scripts to be implemented:
- [ ] generate_dataset.py
- [ ] generate_attack_data.py
- [ ] benchmark.py
- [ ] convert_candump.py
- [ ] visualize_traffic.py
- [ ] train_model_cli.py

## Usage Examples

```bash
# Generate normal CAN traffic
python scripts/generate_dataset.py --type normal --output data/raw/normal.csv

# Generate DoS attack
python scripts/generate_attack_data.py --attack dos --output data/samples/dos.pcap

# Benchmark system
python scripts/benchmark.py --interface vcan0 --duration 60

# Convert candump to CSV
python scripts/convert_candump.py input.log output.csv
```