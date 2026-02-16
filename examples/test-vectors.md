# AVP Test Vectors

These test vectors can be used to verify AVP encoder/decoder implementations across languages.

## Vector 1: Simple 4-dimensional float32 (uncompressed)

**Input:**
- Embedding: `[0.1, 0.2, 0.3, 0.4]` (float32)
- model_id: `"test-model"`
- agent_id: `"agent-a"`
- Compression: none

**Expected output (hex):**
```
4156010030000000200000000a0a746573742d6d6f64656c10041a07666c6f617433322a076167656e742d61cdcccc3dcdcc4c3e9a99993ecdcccc3e
```

**Breakdown:**

| Field            | Bytes  | Hex                | Value                    |
|------------------|--------|--------------------|--------------------------|
| Magic            | 0-1    | `4156`             | "AV"                     |
| Version          | 2      | `01`               | 1                        |
| Flags            | 3      | `00`               | No compression           |
| Payload length   | 4-7    | `30000000`         | 48 (LE)                  |
| Metadata length  | 8-11   | `20000000`         | 32 (LE)                  |
| Metadata         | 12-43  | (protobuf)         | model_id, dim=4, float32 |
| Embedding        | 44-59  | `cdcccc3d...cc3e`  | [0.1, 0.2, 0.3, 0.4]    |

**Total size:** 60 bytes

## Vector 2: Minimal (empty model_id)

**Input:**
- Embedding: `[1.0, 2.0]` (float32)
- model_id: `""`
- Compression: none

**Expected header bytes:**
```
4156 01 00
```
(Magic "AV", version 1, no flags)

## Validation Rules

1. First 2 bytes MUST be `0x4156`
2. Version byte MUST be `0x01`
3. Payload length MUST equal `metadata_length + embedding_bytes_length`
4. Embedding bytes length MUST equal `embedding_dim * sizeof(data_type)`
5. If flags bit 0 is set, embedding bytes are zstd-compressed before length calculation
