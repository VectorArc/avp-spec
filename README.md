# Agent Vector Protocol (AVP)

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-draft-yellow.svg)]()

## Overview

Agent Vector Protocol (AVP) is a binary protocol for same-model LLM agent communication. When two agents run the same model, AVP lets them exchange hidden states and KV-cache directly, skipping autoregressive text generation entirely. When models are incompatible, agents fall back to JSON.

AVP complements existing agent protocols like [A2A](https://github.com/google/A2A) -- it handles the latent communication layer, not discovery or orchestration.

### How It Works

1. **Handshake** -- Agents exchange model identity (architecture, dimensions, weight hash)
2. **Resolve** -- If models match, use latent mode. If not, fall back to JSON.
3. **Communicate** -- Latent mode: binary tensor payloads. JSON mode: text messages.

### What Latent Mode Skips

In a standard agent-to-agent exchange, each message requires full autoregressive generation (token-by-token decoding). For same-model agents, this is redundant -- the receiving agent already operates in the same representation space. AVP eliminates this step by transmitting intermediate hidden states and KV-cache directly.

### Binary Format

AVP uses a compact 12-byte header followed by protobuf metadata and raw tensor bytes:

```
Bytes 0-1:   Magic (0x4156 = "AV")
Byte 2:      Version (0x01)
Byte 3:      Flags (compressed, hybrid, has_map, kv_cache)
Bytes 4-7:   Payload length (uint32 LE)
Bytes 8-11:  Metadata length (uint32 LE)
Bytes 12..N: Protobuf metadata
Bytes N..:   Raw tensor bytes
```

## Documentation

- [Full Specification](SPECIFICATION.md)
- [Binary Format](protocol/binary-format.md)
- [Compression](protocol/compression.md)
- [Transport](protocol/transport.md)
- [Security](protocol/security.md)
- [Test Vectors](examples/test-vectors.md)

## Status

**Version**: 0.1.0-draft

Current scope: same-model latent communication with HuggingFace Transformers. Cross-model communication via learned projection maps is a planned extension.

## Implementation

- [Python SDK](https://github.com/vectorarc/avp-python) -- codec, handshake, session management, realignment, KV-cache serialization, HuggingFace connector, HTTP/2 transport

## Research Foundation

Based on [LatentMAS: Latent Collaboration in Multi-Agent Systems](https://arxiv.org/abs/2511.20639) -- same-model latent communication via hidden state transfer and KV-cache sharing, with realignment for untied-weight models.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)

## License

Apache 2.0
