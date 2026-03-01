# Agent Vector Protocol (AVP)

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/spec-v0.2.2-yellow.svg)]()
[![SDK Tests](https://img.shields.io/badge/SDK%20tests-377%20passing-brightgreen)](https://github.com/VectorArc/avp-python)

A binary protocol for transferring KV-cache and hidden states between LLM agents, eliminating redundant text re-processing in multi-agent systems. **51-78% token savings, 1.5-5x faster** across 4 benchmarks and 5 models. Cross-model projection with zero training.

## Overview

Agent Vector Protocol (AVP) is a binary protocol for LLM agent communication via latent representations. When two agents run the same model, AVP lets them exchange hidden states and KV-cache directly, skipping autoregressive text generation entirely. When agents run different models -- same family or different families -- AVP uses vocabulary-mediated projection to bridge between their latent spaces with zero training. When no compatible projection path exists, agents fall back to JSON.

AVP is transport-agnostic -- it defines the binary format, handshake, and codec, not the transport. The reference implementation uses HTTP/2, but AVP messages can be carried over A2A, MCP, gRPC, WebSockets, or any channel that supports binary payloads. AVP handles the latent communication layer, not discovery or orchestration.

### How It Works

1. **Handshake** -- Agents exchange model identity (architecture, dimensions, weight hash, tokenizer hash)
2. **Resolve** -- Same model: latent mode. Same family: cross-model projection. Otherwise: JSON fallback.
3. **Communicate** -- Latent mode: binary tensor payloads. Cross-model: projected hidden states. JSON mode: text messages.

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

**Version**: 0.2.2

Current scope: same-model latent communication and cross-model communication via vocabulary-mediated projection (Rosetta Stone v2). Same-family models project through shared vocabulary; cross-family models project through overlapping BPE tokens (~85% overlap for Qwen/Llama). Both require zero training.

## Implementation

- **[Python SDK](https://github.com/vectorarc/avp-python)** -- `pip install avp` (v0.2.2). Easy API (`pack()`/`unpack()`/`generate()`), connector API (`think()`/`generate()`/`AVPContext`), `ContextStore`, observability metrics, codec, handshake, session management, realignment, KV-cache serialization, Rosetta Stone cross-model projection, HuggingFace + vLLM connectors, HTTP/2 transport, 7 benchmark suites (377 tests)

## Ecosystem

AVP is complementary to existing agent protocols:

- **[A2A](https://github.com/google/A2A)** -- AVP provides a transport binding for A2A via `multipart/related` with binary payloads
- **[MCP](https://github.com/modelcontextprotocol)** -- MCP handles tools and context; AVP handles tensor transfer between agents
- **[vLLM](https://github.com/vllm-project/vllm)** -- AVP integrates via KVConnectorBase_V1 plugin for production serving
- **[HuggingFace Transformers](https://github.com/huggingface/transformers)** -- Full hidden state and KV-cache access for development and benchmarking

## Research Foundation

Based on [LatentMAS: Latent Collaboration in Multi-Agent Systems](https://arxiv.org/abs/2511.20639) -- same-model latent communication via hidden state transfer and KV-cache sharing, with realignment for untied-weight models.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)

## License

Apache 2.0
