# AVP Transport Layer: HTTP/2 Reference Binding

## Overview

AVP is transport-agnostic. This document specifies the **reference HTTP/2 binding** -- how AVP messages are carried over HTTP/2. Other transports (gRPC, WebSockets, A2A DataParts, shared memory) can carry the same AVP binary messages using their own framing.

## Endpoints

### POST /avp/v2/handshake

Model identity exchange. Both agents send their `HelloMessage` (JSON) to negotiate the communication mode.

```http
POST /avp/v2/handshake
Host: agent-receiver.example.com
Content-Type: application/json
AVP-Version: 0.4
AVP-Agent-ID: agent-a-123
```

**Request body (JSON):**
```json
{
  "agent_id": "agent-a-123",
  "avp_version": "0.4.2",
  "identity": {
    "model_family": "llama",
    "model_id": "meta-llama/Llama-2-7b",
    "model_hash": "sha256:abc123...",
    "hidden_dim": 4096,
    "num_layers": 32,
    "num_kv_heads": 32,
    "head_dim": 128,
    "tokenizer_hash": "sha256:def456..."
  }
}
```

**Response:**
```json
{
  "success": true,
  "agent_id": "agent-b-456",
  "session_id": "sess-...",
  "identity": {
    "model_family": "llama",
    "model_id": "meta-llama/Llama-2-7b",
    "model_hash": "sha256:abc123...",
    "hidden_dim": 4096,
    "num_layers": 32,
    "num_kv_heads": 32,
    "head_dim": 128,
    "tokenizer_hash": "sha256:def456..."
  }
}
```

The client uses the response identity to resolve the communication mode (see Specification section 3). The `tokenizer_hash` field is optional but required for cross-model vocabulary-mediated projection (Rosetta Stone v2).

### POST /avp/v2/transmit

Binary payload transfer (hidden states, KV-cache, or embeddings).

```http
POST /avp/v2/transmit
Host: agent-receiver.example.com
Content-Type: application/avp+binary
AVP-Version: 0.4
AVP-Agent-ID: agent-a-123
```

Request body is a raw AVP binary message (header + protobuf metadata + payload).

### POST /avp/v2/text

JSON fallback for agents that cannot communicate via latent mode.

```http
POST /avp/v2/text
Host: agent-receiver.example.com
Content-Type: application/json
AVP-Version: 0.4
AVP-Agent-ID: agent-a-123
```

**Request body:**
```json
{
  "avp_version": "0.4.2",
  "session_id": "sess-...",
  "source_agent_id": "agent-a-123",
  "target_agent_id": "agent-b-456",
  "content": "The analysis results indicate...",
  "extra": {}
}
```

### GET /health

Health check endpoint. Returns `{"status": "ok"}`.

## Required Headers

| Header | Description |
|--------|-------------|
| `Content-Type` | `application/avp+binary` for transmit, `application/json` for handshake/text |
| `AVP-Version` | Protocol version, currently `"0.4"` |
| `AVP-Agent-ID` | Sender agent identifier |

## Security

- **TLS 1.2+** required (HTTPS only)
- Recommended: Bearer token authentication
```http
Authorization: Bearer <token>
```

## Example

```python
from avp import AVPClient, ModelIdentity

client = AVPClient(
    "https://receiver.example.com",
    agent_id="agent-a-123",
    model_identity=my_identity,
    token="bearer-token",
)

# Negotiate communication mode
session = client.handshake()

# Send binary payload (latent mode)
resp = client.transmit(payload, metadata)

# Or send text (JSON fallback)
resp = client.send_text(json_message)
```
