# Agent Vector Protocol (AVP) Specification

**Version:** 0.1.0-draft
**Status:** Draft
**Last Updated:** February 2026

## Abstract

Agent Vector Protocol (AVP) is a binary protocol that enables same-model LLM agents to communicate via latent representations (hidden states and KV-cache) instead of text. When agents run the same model, they skip autoregressive generation entirely and exchange intermediate tensors directly. When models are incompatible, agents fall back to JSON text.

## 1. Introduction

### 1.1 Motivation

Current agent-to-agent communication requires each agent to:
1. Generate a full text response (autoregressive token-by-token)
2. Serialize the text (JSON, MessagePack, etc.)
3. Transmit over the network
4. Parse and re-encode text into the receiving model's embedding space

For same-model agents, this is wasteful -- the receiving agent already shares the same representation space. AVP lets these agents skip steps 1 and 4 by transmitting hidden states and KV-cache directly.

### 1.2 Design Goals

- **Skip generation**: Same-model agents bypass autoregressive decoding
- **Graceful fallback**: Incompatible models automatically fall back to JSON
- **Complementary to A2A**: AVP extends existing agent protocols, not replaces them
- **Engine-agnostic**: Works with HuggingFace Transformers, vLLM, and other inference engines
- **Extensible**: Handshake carries enough structural info for future cross-model communication

### 1.3 Scope

This specification covers same-model latent communication. Cross-model communication via learned projection maps (Rosetta Stone) is a planned extension and is not part of this version.

## 2. Protocol Overview

### 2.1 Communication Modes

AVP supports three communication modes, negotiated during handshake:

| Mode | When | What's transmitted |
|------|------|-------------------|
| **Latent** | Same model (hash or structure match) | Hidden states, KV-cache |
| **Hybrid** | Partial compatibility (future) | Mix of latent chunks and text |
| **JSON** | Incompatible models | Plain text messages |

### 2.2 Message Flow

```
Agent A                              Agent B
   |                                    |
   |--[1] POST /avp/v2/handshake ----->|  (exchange model identity)
   |<-[2] Identity response -----------|
   |                                    |
   |   [3] Resolve compatibility        |  (same hash -> latent, else -> json)
   |                                    |
   |--[4] POST /avp/v2/transmit ------>|  (binary: hidden state or KV-cache)
   |<-[5] Response --------------------|
   |                                    |
   |   ... or if JSON mode ...          |
   |                                    |
   |--[4] POST /avp/v2/text ---------->|  (JSON fallback message)
   |<-[5] Response --------------------|
```

### 2.3 Core Components

1. **Handshake**: Model identity exchange and compatibility resolution
2. **Binary codec**: Hidden states, KV-cache, and embeddings serialized with protobuf metadata
3. **Compression**: Optional zstd (mainly useful for embeddings)
4. **Session management**: Track active agent pairs with TTL
5. **Realignment**: Project hidden states from output to input embedding space
6. **JSON fallback**: Text communication for incompatible models
7. **Transport**: HTTP/2 with binary payloads

## 3. Handshake Protocol

### 3.1 Model Identity

Each agent advertises its model identity during handshake:

| Field | Type | Description |
|-------|------|-------------|
| model_family | string | Architecture family (e.g. "llama", "qwen", "mistral") |
| model_id | string | Full model identifier (e.g. "meta-llama/Llama-2-7b") |
| model_hash | string | SHA-256 of sorted model config |
| hidden_dim | uint32 | Hidden state dimensionality |
| num_layers | uint32 | Number of transformer layers |
| num_kv_heads | uint32 | Number of key-value attention heads |
| head_dim | uint32 | Dimension per attention head |

### 3.2 Compatibility Resolution

The resolver determines the communication mode:

1. **Model hash matches** -> Latent mode (identical models)
2. **Same family + matching hidden_dim + num_layers** -> Latent mode (structurally identical)
3. **No match** -> JSON fallback

### 3.3 Session

A successful handshake creates a session with:
- Unique session_id
- Negotiated communication mode
- Both agent identities
- TTL (default 1 hour)

Sessions expire automatically. The session manager handles cleanup.

## 4. Latent Communication

### 4.1 Hidden States

Agents extract hidden states from intermediate transformer layers and transmit them as raw tensor bytes. The receiving agent injects these via `inputs_embeds` to continue generation without re-encoding from text.

Hidden states require **realignment** -- projection from the model's output space back to the input embedding space. This is computed from the model's embedding and language model head weights:

```
W_realign = (E_out^T E_out + lambda * I)^{-1} E_out^T E_in
```

Models with tied weights (`tie_word_embeddings=True`) do not need realignment because their input and output embedding spaces are already identical.

Realignment matrices are cached to disk (`~/.avp/realign/{model_hash}.pt`) since they only depend on the model weights.

### 4.2 KV-Cache Transfer

Agents can transfer attention key-value caches to share context without re-processing input tokens. The KV-cache is serialized as contiguous little-endian tensor bytes:

```
[K_layer0][V_layer0][K_layer1][V_layer1]...
```

A 17-byte header precedes the tensor data:
- num_layers (uint32)
- num_kv_heads (uint32)
- head_dim (uint32)
- seq_len (uint32)
- dtype (uint8)

### 4.3 Payload Sizes

Representative KV-cache sizes per token (fp16):

| Model size | Per-token size |
|-----------|---------------|
| 7B | ~256 KB |
| 14B | ~320 KB |
| 70B | ~640 KB |

Latent communication is designed for datacenter environments with high-bandwidth interconnects (>1 Gbps).

## 5. Binary Format

See [protocol/binary-format.md](protocol/binary-format.md)

## 6. Compression

See [protocol/compression.md](protocol/compression.md)

## 7. Transport Layer

See [protocol/transport.md](protocol/transport.md)

## 8. Security Considerations

See [protocol/security.md](protocol/security.md)

## 9. A2A Compatibility

AVP is designed to work alongside the [Agent-to-Agent (A2A) protocol](https://github.com/google/A2A). Integration approach:

- AVP capabilities advertised via URI-namespaced A2A extensions
- Binary payloads transmitted as `multipart/related` HTTP parts with `cid:` URIs
- Handshake data carried in A2A DataParts

This allows agents to use A2A for discovery and orchestration while using AVP for latent communication when both agents support it.

## 10. Versioning

AVP follows semantic versioning (MAJOR.MINOR.PATCH).

Current version: **0.1.0**

## 11. References

- [LatentMAS: Latent Collaboration in Multi-Agent Systems](https://arxiv.org/abs/2511.20639) -- research foundation for same-model latent communication and realignment

## 12. Authors

VectorArc Team

## 13. License

Apache 2.0 -- See LICENSE file
