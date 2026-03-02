# CAN Packet Format

## Visual Representation

### Standard CAN 2.0A Frame (11-bit Identifier)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                             CAN 2.0A STANDARD FRAME                             │
└─────────────────────────────────────────────────────────────────────────────────┘

 SOF  │←────── Arbitration Field ──────→│ Control │←─────── Data Field ─────→│CRC│A│EOF│IFS│
  ↓   │                                  │         │                          │   │ │   │   │
┌───┬─────────────────┬───┬───┬───┬─────┬─────────┬────────────────────────┬─────┬─┬───┬───┐
│ 0 │  Identifier     │RTR│IDE│r0 │ DLC │ Data    │                        │ CRC │A│111│111│
│   │   (11 bits)     │   │   │   │(4b) │ Bytes   │      (0-8 bytes)       │(15b)│ │   │   │
└───┴─────────────────┴───┴───┴───┴─────┴─────────┴────────────────────────┴─────┴─┴───┴───┘
  1        0x000-0x7FF   1   0   0   0-8    D0 D1 D2 D3 D4 D5 D6 D7          15   1   7   3
 bit       (2048 IDs)   bit bit bit bits   (up to 64 bits total)           bits bit bits bits

  │                      │   │   │    │                                      │    │   │    │
  │                      │   │   │    └─ Data Length Code (0-8)             │    │   │    │
  │                      │   │   └────── Reserved bit (must be 0)           │    │   │    │
  │                      │   └────────── Identifier Extension (0=std)       │    │   │    │
  │                      └────────────── Remote Transmission Request        │    │   │    │
  │                                      (0=data, 1=remote)                 │    │   │    │
  └──────────────────────────────────────────────────────────────────────────┴────┴───┴────┘
   Start Of Frame         Message Priority (Lower ID = Higher Priority)      CRC ACK EOF IFS
```

### Extended CAN 2.0B Frame (29-bit Identifier)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                             CAN 2.0B EXTENDED FRAME                             │
└─────────────────────────────────────────────────────────────────────────────────┘

 SOF  │←────────────── Arbitration Field ──────────────→│ Control │←── Data ──→│CRC│A│EOF│IFS│
  ↓   │                                                  │         │            │   │ │   │   │
┌───┬─────────┬───┬───┬─────────────────────┬───┬───────┬─────────┬────────────┬───┬─┬───┬───┐
│ 0 │Base ID  │SRR│IDE│Extended Identifier  │RTR│  r1r0 │   DLC   │    Data    │CRC│A│111│111│
│   │(11 bits)│   │   │    (18 bits)        │   │ (2b)  │  (4b)   │  (0-8 B)   │   │ │   │   │
└───┴─────────┴───┴───┴─────────────────────┴───┴───────┴─────────┴────────────┴───┴─┴───┴───┘
  1   11 bits   1   1       18 bits          1    2 bits  4 bits    0-64 bits  15b 1  7b  3b

  │      │       │   │           │            │      │       │          │        │   │   │   │
  │      │       │   │           │            │      │       │          │        │   │   │   │
  │      │       │   │           │            │      │       └─ DLC     │        │   │   │   │
  │      │       │   │           │            │      └───────── Reserved│        │   │   │   │
  │      │       │   │           │            └──────────────── RTR     │        │   │   │   │
  │      │       │   │           └───────────────────────────── Ext ID  │        │   │   │   │
  │      │       │   └───────────────────────────────────────── IDE=1   │        │   │   │   │
  │      │       └───────────────────────────────────────────── SRR     │        │   │   │   │
  │      └───────────────────────────────────────────────────── Base ID │        │   │   │   │
  └────────────────────────────────────────────────────────────────────┴────────┴───┴───┴───┘
                     Total: 29-bit Identifier (0x00000000 - 0x1FFFFFFF)
                           536,870,912 possible IDs
```

## Field Breakdown

### Frame Components

| Field | Bits | Description |
|-------|------|-------------|
| **SOF** | 1 | Start Of Frame - Dominant bit (0) marks beginning |
| **Identifier** | 11 or 29 | Message priority and addressing (lower = higher priority) |
| **RTR** | 1 | Remote Transmission Request (0=data frame, 1=remote frame) |
| **IDE** | 1 | Identifier Extension (0=standard 11-bit, 1=extended 29-bit) |
| **r0/r1** | 1-2 | Reserved bits (must be dominant/0) |
| **DLC** | 4 | Data Length Code (0-8 bytes) |
| **Data** | 0-64 | Payload data (0-8 bytes) |
| **CRC** | 15 | Cyclic Redundancy Check for error detection |
| **ACK** | 1 | Acknowledge bit (receivers set to dominant if CRC valid) |
| **EOF** | 7 | End Of Frame - 7 recessive bits (1) |
| **IFS** | 3 | Inter-Frame Space - 3 recessive bits (1) |

### Bit Representation

- **Dominant bit:** Logic 0 (overwrites recessive on bus)
- **Recessive bit:** Logic 1 (can be overwritten by dominant)

## Example CAN Messages

### Example 1: Engine RPM (Standard ID)

```
Message: Engine RPM = 3000 (0x0BB8)
CAN ID: 0x0C0 (192 decimal)
DLC: 2 bytes

┌─────────────────────────────────────────────────────────┐
│  ID: 0x0C0   │ RTR │ DLC │      Data Bytes              │
├──────────────┼─────┼─────┼──────────────────────────────┤
│ 00011000000  │  0  │  2  │  0x0B    0xB8                │
│  (11 bits)   │ (1) │ (4) │  (byte 0) (byte 1)           │
└──────────────┴─────┴─────┴──────────────────────────────┘

Hex Representation: 0C0#0BB8
Binary Data: 00001011 10111000
           =    11       184  (3000 in big-endian)
```

### Example 2: Diagnostic Request (Extended ID)

```
Message: OBD-II Diagnostic Request
CAN ID: 0x18DA10F1 (Extended)
DLC: 8 bytes (full frame)

┌────────────────────────────────────────────────────────────────────┐
│    Extended ID: 0x18DA10F1        │RTR│ DLC │      Data Bytes      │
├───────────────────────────────────┼───┼─────┼──────────────────────┤
│ Base: 0x31B  │  Extended: 0x2843D │ 0 │  8  │ 02 01 0C 55 55 55 55 │
│  (11 bits)   │    (18 bits)       │   │     │ 55                   │
└──────────────┴────────────────────┴───┴─────┴──────────────────────┘

Hex Representation: 18DA10F1#02010C5555555555
Service: $01 (Show current data)
PID: $0C (Engine RPM)
Padding: 0x55 (unused bytes)
```

### Example 3: Steering Angle (Real Vehicle Data)

```
Message: Steering Wheel Angle
CAN ID: 0x025
DLC: 8 bytes

┌─────────────────────────────────────────────────────────────────────┐
│  ID: 0x025  │ RTR │ DLC │           Data Bytes                      │
├─────────────┼─────┼─────┼───────────────────────────────────────────┤
│ 00000100101 │  0  │  8  │ 0x3F 0xF8 0x00 0x10 0xFF 0xE7 0x00 0x00  │
└─────────────┴─────┴─────┴───────────────────────────────────────────┘

Parsing:
Byte 0-1: Angle value (0x3FF8 = 16376)
          Angle = (16376 - 32768) * 0.1 = -1639.2°
Byte 2-3: Angle rate (0x0010 = 16)
          Rate = 16 * 0.5 = 8°/s
Byte 4-7: Status/checksum fields
```

## CAN Bus Arbitration Example

When multiple nodes transmit simultaneously, arbitration determines priority:

```
Time ──────────────────────────────────────────────────────────►

Node A (ID: 0x100 = 0b00100000000):  0  0  1  0  0  [WINS] ──► continues
                                     │  │  │
Node B (ID: 0x120 = 0b00100100000):  0  0  1  0  1  [LOSES] ──┐ stops
                                     │  │  │  │  ↑              │
Node C (ID: 0x200 = 0b01000000000):  0  1  [LOSES] ──────────┐ │ stops
                                     │  ↑                      │ │
                                     │  └─ recessive bit      │ │
                                     │     (loses to          │ │
                                     │      dominant 0)       │ │
                                     │                        │ │
                                     └─ all transmit same     │ │
                                        dominant bit          │ │
                                                               │ │
Lower ID = Higher Priority                Node C backs off ──┘ │
Node A wins arbitration                   Node B backs off ────┘

Result: Node A (0x100) transmits first
        Node B (0x120) transmits second
        Node C (0x200) transmits last
```

## Bit Timing Diagram

```
Nominal Bit Time (typically 1 μs at 1 Mbps)
┌────────────────────────────────────────────────────────┐
│                   One Bit Period                       │
└────────────────────────────────────────────────────────┘

├─────┼────────────────┼────┼────────┤
 Sync    Prop + Phase1  Phase2  SJW
 Seg         Seg          Seg

 │       Sample Point ────────┘
 │            ↑
 │         (typically at 75-87.5% of bit time)
 │
 └─ Synchronization edge

Typical 1 Mbps Configuration:
- Bit Rate: 1,000,000 bits/sec
- Bit Time: 1 μs (1000 ns)
- Sample Point: 87.5% (875 ns)
- Sync Seg: 1 TQ (125 ns)
- Prop Seg: 2 TQ (250 ns)
- Phase1 Seg: 4 TQ (500 ns)
- Phase2 Seg: 1 TQ (125 ns)
- Total: 8 TQ = 1 bit time
```

## CAN Bus States

```
┌──────────────────────────────────────────────────────────────┐
│                    CAN Bus Voltage Levels                    │
└──────────────────────────────────────────────────────────────┘

CAN_H  ────┐     ┌─────     ──────┐     ┌─────
           │     │                │     │
   3.5V ───┤     ├─── Dominant    │     │
           │     │                │     │
   2.5V ───┼─────┼────────────────┼─────┼──────  Differential
           │     │                │     │
   1.5V ───┤     ├─── Dominant    │     │
           │     │                │     │
CAN_L  ────┘     └─────     ──────┘     └─────

       Recessive  Dominant   Recessive  Dominant
           1         0           1         0

Differential Voltage:
- Recessive: ~0V (CAN_H ≈ CAN_L ≈ 2.5V)
- Dominant: ~2V (CAN_H ≈ 3.5V, CAN_L ≈ 1.5V)
```

## Error Frame Visualization

```
Normal Data Frame:
┌───┬─────┬───┬─────┬──────────┬─────┬───┬───┐
│SOF│ ID  │CTL│ DLC │   Data   │ CRC │ACK│EOF│
└───┴─────┴───┴─────┴──────────┴─────┴───┴───┘

Error Detected → Error Frame:
┌───┬─────┬───┬─────┬──────────┬─────┬
│SOF│ ID  │CTL│ DLC │   Data   │ CRC │
└───┴─────┴───┴─────┴──────────┴─────┴
                            ↓
                      ┌────────────────────┬──────────┐
                      │ Error Flag (6 bits)│ EOF (8b) │
                      │   (all dominant)   │          │
                      └────────────────────┴──────────┘
                            ↓
                      Frame retransmitted automatically

Error Counters:
- Transmit Error Counter (TEC): 0-255
- Receive Error Counter (REC): 0-255

Bus States:
- Error Active: TEC < 128, REC < 128
- Error Passive: TEC ≥ 128 or REC ≥ 128
- Bus Off: TEC > 255
```

## Data Byte Order Examples

### Big-Endian (Motorola) - Most Common in Automotive

```
16-bit value: 0x1234 (4660 decimal)

CAN Frame: ID#12 34
           ↑  ↑  ↑
           │  │  └─ Low byte (0x34 = 52)
           │  └──── High byte (0x12 = 18)
           └─────── Message ID

Memory Layout: [0x12] [0x34]
                 MSB    LSB

Value = (0x12 << 8) | 0x34 = 4660
```

### Little-Endian (Intel) - Less Common

```
16-bit value: 0x1234 (4660 decimal)

CAN Frame: ID#34 12
           ↑  ↑  ↑
           │  │  └─ High byte (0x12 = 18)
           │  └──── Low byte (0x34 = 52)
           └─────── Message ID

Memory Layout: [0x34] [0x12]
                 LSB    MSB

Value = 0x34 | (0x12 << 8) = 4660
```

## Signal Encoding Example

### Temperature Sensor Signal

```
Raw CAN Message:
ID: 0x3D1
Data: [0x00, 0xFA, 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC]

Signal: Engine_Temp
- Start Bit: 8
- Length: 16 bits
- Byte Order: Big-Endian
- Factor: 0.1
- Offset: -40
- Unit: °C

Extraction:
┌────────────────────────────────────────────────────┐
│ Byte 0 │ Byte 1 │ Byte 2 │ Byte 3 │ Byte 4 │ ...  │
├────────┼────────┼────────┼────────┼────────┼──────┤
│  0x00  │  0xFA  │  0x12  │  0x34  │  0x56  │ ...  │
└────────┴────────┴────────┴────────┴────────┴──────┘
            ↑        ↑
            └────────┘
          Start bit 8, 16 bits

Raw Value: 0xFA12 = 64018
Physical Value: (64018 * 0.1) - 40 = 6401.8 - 40 = 6361.8°C

Note: This example shows the encoding process. 
      Real temperature would have different raw value!
```

## CAN FD (Flexible Data-Rate) Comparison

```
Classical CAN:
┌───┬────┬───┬───┬──────────────┬─────┬───┬───┐
│SOF│ ID │CTL│DLC│ Data (0-8 B) │ CRC │ACK│EOF│
└───┴────┴───┴───┴──────────────┴─────┴───┴───┘
Max: 8 bytes @ 1 Mbps

CAN FD:
┌───┬────┬───┬───┬────────────────────────┬─────┬───┬───┐
│SOF│ ID │CTL│DLC│ Data (0-64 bytes)      │ CRC │ACK│EOF│
└───┴────┴───┴───┴────────────────────────┴─────┴───┴───┘
Max: 64 bytes @ 8 Mbps (data phase)

Key Differences:
- Payload: 8 bytes → 64 bytes
- Bit Rate: Fixed → Dual (arbitration + data phase)
- CRC: 15 bits → 17/21 bits (depends on payload)
```

## Common CAN IDs in Vehicles

```
┌──────────────────────────────────────────────────────┐
│              Standard Automotive CAN IDs              │
├───────────┬──────────────────────────────────────────┤
│   Range   │              Description                 │
├───────────┼──────────────────────────────────────────┤
│ 0x000-0x07F │ High Priority (safety, powertrain)    │
│ 0x080-0x0FF │ Engine & transmission                 │
│ 0x100-0x1FF │ Chassis (ABS, ESP, steering)          │
│ 0x200-0x2FF │ Body (lights, doors, HVAC)            │
│ 0x300-0x3FF │ Infotainment & comfort                │
│ 0x400-0x5FF │ Network management                    │
│ 0x600-0x6FF │ Reserved                              │
│ 0x700-0x7FF │ Diagnostic messages (OBD-II)          │
└───────────┴──────────────────────────────────────────┘

Example Messages:
0x0C0: Engine RPM
0x0C1: Engine Temperature
0x0C2: Throttle Position
0x0D0: Vehicle Speed
0x025: Steering Angle
0x240: Door Status
0x7DF: OBD-II Functional Request (broadcast)
0x7E0-0x7E7: OBD-II Physical Request
0x7E8-0x7EF: OBD-II Response
```

## Tools for Visualization

This visualization can be used with:
- **candump**: `candump can0` - Display CAN messages in real-time
- **cansniffer**: `cansniffer can0` - Highlight changing bytes
- **can-utils**: Suite of CAN tools for Linux
- **Wireshark**: Packet capture with CAN protocol dissector
- **CANalyzer/CANoe**: Professional Vector tools
- **python-can**: Python library (used by this project)

## References

- **ISO 11898-1**: CAN specification for data link layer
- **ISO 11898-2**: CAN physical layer specification  
- **SAE J1939**: Heavy-duty vehicle CAN protocol
- **ISO 15765-2**: Diagnostic communication (ISO-TP)
- **python-can docs**: https://python-can.readthedocs.io/

---

**Document Version:** 1.0  
**Last Updated:** November 11, 2025  
**Related:** See [traffic_monitoring_guide.md](traffic_monitoring_guide.md) for analysis techniques
