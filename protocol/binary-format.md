# AVP Binary Format Specification

## Message Structure

Every AVP message consists of three parts:
```
+---------------------+
|   Header (12 bytes) |
+---------------------+
|   Metadata          |
+---------------------+
|   Payload           |
+---------------------+
```

### Header (12 bytes)
```
Byte 0-1:  Magic number (0x4156 = "AV" in ASCII)
Byte 2:    Protocol version (0x01)
Byte 3:    Flags
           Bit 0: Compressed (0=no, 1=yes, zstd)
           Bit 1: Has AVP map (0=no, 1=yes, cross-model projection)
           Bit 2: KV-cache payload (0=no, 1=yes)
           Bit 3-7: Reserved
Byte 4-7:  Payload length (uint32, little-endian)
           Total bytes after header (metadata + tensor data)
Byte 8-11: Metadata length (uint32, little-endian)
           Length of the Protocol Buffer metadata section
```

**Note on payload length:** The `payload_length` field (bytes 4-7) encodes the total number of bytes following the header, i.e. `len(metadata) + len(tensor_bytes)`. The `metadata_length` field (bytes 8-11) allows the decoder to locate the boundary between metadata and tensor data without parsing the protobuf first.

### Metadata (Variable length)

Protocol Buffer encoded metadata. See `schemas/avp.proto` for the canonical schema.

Fields:

| Field | Number | Type | Description |
|-------|--------|------|-------------|
| session_id | 1 | string | Session identifier from handshake |
| source_agent_id | 2 | string | Sender agent identifier |
| target_agent_id | 3 | string | Recipient agent identifier |
| model_id | 4 | string | Model that produced the payload, e.g. "meta-llama/Llama-2-7b" |
| hidden_dim | 5 | uint32 | Hidden state dimensionality, e.g. 4096 |
| num_layers | 6 | uint32 | Number of transformer layers |
| payload_type | 7 | PayloadType | HIDDEN_STATE (0), KV_CACHE (1), or EMBEDDING (2) |
| dtype | 8 | DataType | FLOAT32 (0), FLOAT16 (1), BFLOAT16 (2), INT8 (3) |
| tensor_shape | 9 | repeated uint32 | Shape of the tensor payload |
| mode | 10 | CommunicationMode | LATENT (0) or JSON_MODE (1) |
| compression | 11 | string | Compression algorithm if compressed, e.g. "zstd" |
| avp_map_id | 13 | string | Cross-model projection map identifier. Format: `"vocab:{tokenizer_hash[:16]}"` for vocabulary-mediated, `"vocab_overlap:{overlap_count}"` for vocabulary-overlap, `"{src_hash[:16]}_{tgt_hash[:16]}"` for pre-calibrated maps. Empty for same-model communication. |
| extra | 14 | map<string,string> | Extensible key-value pairs |

### Payload Types

**HIDDEN_STATE (0):** Raw hidden state tensor bytes from a transformer layer. Little-endian, dtype specified in metadata. Used for same-model latent communication where agents share intermediate representations.

**KV_CACHE (1):** Serialized key-value cache from transformer attention layers. Layout: `[K_l0][V_l0][K_l1][V_l1]...` where each tensor is contiguous little-endian. Preceded by a 17-byte KV-cache header (num_layers, num_kv_heads, head_dim, seq_len, dtype).

**EMBEDDING (2):** Raw embedding vector bytes (e.g. from sentence-transformers). 4 bytes per dimension for float32, 2 bytes per dimension for float16. Little-endian byte order.

If the compressed flag is set, the payload is zstd-compressed.

### Decoding Algorithm

```
1. Read 12 bytes -> header
2. Verify magic == 0x4156
3. Verify version == 0x01
4. Read payload_length bytes after header
5. First metadata_length bytes -> parse as Protobuf Metadata
6. Remaining bytes -> raw tensor payload (decompress if flag bit 0 set)
7. Interpret payload using payload_type, dtype, and tensor_shape from metadata
```

## Example

4096-dimensional float32 hidden state:
- Header: 12 bytes
- Metadata: ~50 bytes
- Payload: 16,384 bytes (4096 x 4)
- **Total: ~16,446 bytes**

## Size Comparison (measured, embedding payloads)

| Dimensions | dtype   | AVP (bytes) | AVP+zstd | JSON (bytes) | Ratio |
|------------|---------|-------------|----------|--------------|-------|
| 384        | float32 | 1,567       | 1,515    | 7,963        | 5.1x  |
| 768        | float32 | 3,103       | 2,931    | 15,930       | 5.1x  |
| 1,024      | float32 | 4,127       | 3,855    | 21,190       | 5.1x  |
| 4,096      | float32 | 16,415      | 15,212   | 84,654       | 5.2x  |
| 384        | float16 | 799         | 815      | 5,802        | 7.3x  |
| 4,096      | float16 | 8,223       | 7,638    | 61,213       | 7.4x  |

Note: zstd compression provides significant savings for embeddings but is less effective for hidden states and KV-cache data (typically 1-7% reduction). The primary value of latent communication is skipping autoregressive generation, not bandwidth savings.
