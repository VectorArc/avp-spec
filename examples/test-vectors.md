# AVP Test Vectors

These test vectors can be used to verify AVP encoder/decoder implementations.

## Vector 1: 4-dimensional float32 hidden state (uncompressed)

**Input:**
- Payload: `[0.1, 0.2, 0.3, 0.4]` (float32)
- model_id: `"test-model"`
- source_agent_id: `"agent-a"`
- hidden_dim: 4
- payload_type: HIDDEN_STATE (0)
- dtype: FLOAT32 (0)
- tensor_shape: [4]
- Compression: none

**Expected output (hex):**
```
415601002a0000001a00000012076167656e742d61220a746573742d6d6f64656c28044a0104cdcccc3dcdcc4c3e9a99993ecdcccc3e
```

**Breakdown:**

| Field            | Bytes  | Hex                | Value                       |
|------------------|--------|--------------------|-----------------------------|
| Magic            | 0-1    | `4156`             | "AV"                        |
| Version          | 2      | `01`               | 1                           |
| Flags            | 3      | `00`               | No flags set                |
| Payload length   | 4-7    | `2a000000`         | 42 (LE)                     |
| Metadata length  | 8-11   | `1a000000`         | 26 (LE)                     |
| Metadata         | 12-37  | (protobuf)         | agent-a, test-model, dim=4  |
| Tensor data      | 38-53  | `cdcccc3d...cc3e`  | [0.1, 0.2, 0.3, 0.4]       |

**Total size:** 54 bytes

## Vector 2: Minimal (no model_id, no agent_id)

**Input:**
- Payload: `[1.0, 2.0]` (float32)
- hidden_dim: 2
- payload_type: HIDDEN_STATE (0)
- dtype: FLOAT32 (0)
- tensor_shape: [2]
- Compression: none

**Expected output (hex):**
```
415601000d0000000500000028024a01020000803f00000040
```

**Header bytes:** `4156 01 00` (Magic "AV", version 1, no flags)

**Total size:** 25 bytes

## Validation Rules

1. First 2 bytes MUST be `0x4156`
2. Version byte MUST be `0x01`
3. Payload length MUST equal `metadata_length + tensor_data_length`
4. If flags bit 0 is set, tensor data is zstd-compressed
5. If flags bit 3 is set, payload_type MUST be KV_CACHE (1)
6. Metadata MUST be valid protobuf conforming to `schemas/avp.proto`
