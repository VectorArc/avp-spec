# AVP Binary Format Specification

## Message Structure

Every AVP message consists of:
```
+------------------+
|   Header (64b)   |
+------------------+
|   Metadata       |
+------------------+
|   Payload        |
+------------------+
```

### Header (8 bytes)
```
Byte 0-1: Magic number (0x4156 = "AV" in ASCII)
Byte 2:   Protocol version (0x01 for v1)
Byte 3:   Flags
          Bit 0: Compressed (0=no, 1=yes)
          Bit 1-7: Reserved
Byte 4-7: Payload length (uint32, little-endian)
```

### Metadata (Variable length)

Protocol Buffer encoded metadata:
```protobuf
message Metadata {
  string model_id = 1;        // e.g., "claude-sonnet-4"
  uint32 embedding_dim = 2;   // e.g., 4096
  string data_type = 3;       // e.g., "float32"
  optional string compression = 4;  // e.g., "zstd"
}
```

### Payload

Raw embedding bytes:
- For float32: 4 bytes per dimension
- For float16: 2 bytes per dimension
- Little-endian byte order

## Example

4096-dimensional float32 embedding:
- Header: 8 bytes
- Metadata: ~50 bytes
- Payload: 16,384 bytes (4096 × 4)
- **Total: ~16,442 bytes**

Compare to JSON: ~200,000+ bytes

**Savings: 92% smaller**
