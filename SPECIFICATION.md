# Agent Vector Protocol (AVP) Specification

**Version:** 0.2.0-draft
**Status:** Draft
**Last Updated:** February 2026

## Abstract

Agent Vector Protocol (AVP) is a binary protocol that enables LLM agents to communicate via latent representations (hidden states and KV-cache) instead of text. Same-model agents skip autoregressive generation entirely and exchange intermediate tensors directly. Same-family agents with different sizes (e.g. Qwen2.5-1.5B and Qwen2.5-0.5B) communicate via vocabulary-mediated cross-model projection. Incompatible models fall back to JSON text.

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
- **Transport-agnostic**: AVP defines the binary format, handshake, and codec -- not the transport. The reference implementation uses HTTP/2, but AVP messages can be carried over any transport (A2A DataParts, gRPC, WebSockets, shared memory, etc.)
- **Complementary**: AVP is a latent communication layer, not an orchestration protocol. It works alongside A2A, MCP, or any agent framework.
- **Engine-agnostic**: Works with HuggingFace Transformers, vLLM, and other inference engines
- **Extensible**: Handshake carries enough structural info for cross-model communication

### 1.3 Scope

This specification covers same-model latent communication and same-family cross-model communication via vocabulary-mediated projection (Rosetta Stone v2). Cross-family communication via learned projection maps is a planned extension.

## 2. Protocol Overview

### 2.1 Communication Modes

AVP supports three communication modes, negotiated during handshake:

| Mode | When | What's transmitted |
|------|------|-------------------|
| **Latent** | Same model (hash or structure match) | Hidden states, KV-cache |
| **Latent (cross-model)** | Same family, shared tokenizer | Vocabulary-mediated projected hidden states |
| **Hybrid** | Partial compatibility (future) | Mix of latent chunks and text |
| **JSON** | Incompatible models | Plain text messages |

### 2.2 Message Flow

```
Agent A                              Agent B
   |                                    |
   |--[1] Handshake ------------------>|  (exchange model identity)
   |<-[2] Identity response -----------|
   |                                    |
   |   [3] Resolve compatibility        |  (same hash -> latent, else -> json)
   |                                    |
   |--[4] AVP binary message --------->|  (hidden state or KV-cache)
   |<-[5] Response --------------------|
   |                                    |
   |   ... or if JSON mode ...          |
   |                                    |
   |--[4] JSON text message ---------->|  (fallback)
   |<-[5] Response --------------------|
```

The diagram above is transport-independent. The reference HTTP/2 binding maps these to `POST /avp/v2/handshake`, `/avp/v2/transmit`, and `/avp/v2/text`. Other transports (gRPC, A2A DataParts, shared memory) can carry the same messages.

### 2.3 Core Components

1. **Handshake**: Model identity exchange and compatibility resolution
2. **Binary codec**: Hidden states, KV-cache, and embeddings serialized with protobuf metadata
3. **Compression**: Optional zstd (mainly useful for embeddings)
4. **Session management**: Track active agent pairs with TTL
5. **Realignment**: Project hidden states from output to input embedding space
6. **JSON fallback**: Text communication for incompatible models
7. **Transport**: Transport-agnostic; reference binding is HTTP/2

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
| tokenizer_hash | string | SHA-256 of sorted tokenizer vocabulary (optional, enables cross-model projection) |

### 3.2 Compatibility Resolution

The resolver determines the communication mode by evaluating rules in priority order. The first matching rule wins:

1. **Model hash matches** -> Latent mode (identical models)
2. **Same family + matching hidden_dim + num_layers** -> Latent mode (structurally identical)
3. **Shared tokenizer_hash** -> Latent mode with `avp_map_id="vocab:{hash[:16]}"` (vocabulary-mediated cross-model projection)
4. **Pre-calibrated .avp-map file exists** -> Latent mode with `avp_map_id="{src_hash[:16]}_{tgt_hash[:16]}"` (learned cross-model projection)
5. **No match** -> JSON fallback

Rules 1-2 resolve same-model communication (no projection needed). Rule 3 enables zero-parameter cross-model communication between same-family models that share a tokenizer (e.g. Qwen2.5-1.5B and Qwen2.5-0.5B). Rule 4 enables pre-calibrated cross-model communication via learned linear maps. When `avp_map_id` is non-empty, the session requires a Rosetta Stone projection map (see Section 4.3).

### 3.3 Session

A successful handshake creates a session with:
- Unique session_id
- Negotiated communication mode
- Both agent identities
- `avp_map_id` (non-empty if cross-model projection is required)
- TTL (default 1 hour)

Sessions expire automatically. The session manager handles cleanup.

## 4. Latent Communication

### 4.1 Hidden States

Agents extract hidden states from intermediate transformer layers and transmit them as raw tensor bytes. The receiving agent injects these via `inputs_embeds` to continue generation without re-encoding from text.

Hidden states require **realignment** -- projection from the model's output space back to the input embedding space. This is computed from the model's embedding and language model head weights:

```
W_realign = (E_out^T E_out + lambda * I)^{-1} E_out^T E_in
```

Models with tied weights (`tie_word_embeddings=True`) do not need the W_realign projection. However, hidden states from the last transformer layer still have different directional structure than input embeddings (cosine similarity ~0.24). For these models, hidden states are projected through the vocabulary via softmax soft embedding:

```
logits = hidden @ W_embed^T         (project to vocabulary logits)
probs  = softmax(logits)            (probability distribution over tokens)
embed  = probs @ W_embed            (weighted average of embeddings)
```

This produces vectors with cosine similarity ~1.0 to the nearest input embedding.

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

### 4.3 Transfer Modes

KV-cache payloads are large. AVP defines multiple transfer modes so users can choose the right bandwidth/compute tradeoff for their environment. The transfer mode is selected by the sender and indicated in the message metadata.

#### Mode 1: Full KV-cache (default)

Transmit the complete KV-cache as contiguous fp16 tensor bytes. Lossless. No additional receiver compute. Best for same-host or high-bandwidth datacenter (>1 Gbps).

#### Mode 2: Quantized KV-cache

Transmit KV-cache in int8 or int4 representation. Reduces payload by 2-4x with negligible quality impact (int8) or small quality impact (int4). No additional receiver compute. Best for moderate bandwidth (500 Mbps - 1 Gbps).

#### Mode 3: Hidden-state transfer

Transmit hidden state vectors only, without the full KV-cache. The receiver reconstructs context by running latent steps (forward passes with the injected hidden states). Reduces payload by 16x or more. Trades bandwidth for receiver compute. Best for lower bandwidth (<500 Mbps).

This is what LatentMAS uses for Agent 4 (Judger) -- it receives hidden states via `inputs_embeds`, not KV-cache.

#### Mode 4: Delta transfer

Transmit only KV-cache entries beyond a shared prefix. When agents share a common system prompt, the KV-cache for that prefix is identical and does not need to be transferred. Lossless. Combinable with modes 1-3.

#### Payload Size Reference

Representative KV-cache sizes per token (fp16):

| Model size | Per token | 200 tokens (full) | 200 tokens (int8) | 200 hidden states |
|-----------|----------|-------------------|-------------------|------------------|
| 7B | ~256 KB | 50 MB | 25 MB | 1.6 MB |
| 14B | ~320 KB | 64 MB | 32 MB | 2.0 MB |
| 70B | ~640 KB | 128 MB | 64 MB | 3.2 MB |

#### Choosing a Transfer Mode

The choice depends on available bandwidth and acceptable receiver compute:

| Environment | Recommended mode | Transfer overhead | Rationale |
|------------|-----------------|-------------------|-----------|
| Same process | In-memory (tensor reference) | ~5ms | No serialization needed |
| Same machine (multi-process) | Full KV-cache via shared memory | ~15-40ms | Local memory bandwidth (~5 GB/s) |
| Datacenter (>1 Gbps) | Full KV-cache or int8 | ~50-200ms | Network bandwidth is cheap |
| Cloud / cross-region (100 Mbps - 1 Gbps) | Quantized (int8/int4) + delta | ~0.5-2s | Balance bandwidth and quality |
| Edge / limited bandwidth (<100 Mbps) | Hidden-state transfer | ~0.04s network + latent step compute | Minimize payload, trade for compute |

Latent communication works well in two deployment scenarios:
- **Local**: Agents on the same machine (same process, multi-process, or containers). Transfer overhead is under 40ms even for 70B models. This is the simplest deployment and requires no network considerations.
- **Datacenter**: Agents on different machines with high-bandwidth interconnects (>500 Mbps). Transfer modes allow tuning the bandwidth/compute tradeoff.

Below ~50 Mbps over a network, JSON text mode is likely more practical unless hidden-state transfer mode is used.

### 4.4 Cross-Model Communication (Rosetta Stone)

When agents run different models from the same family (e.g. Qwen2.5-1.5B and Qwen2.5-0.5B), they can communicate via latent projection instead of JSON fallback. The handshake sets `avp_map_id` to indicate which projection method to use.

#### Vocabulary-Mediated Projection (avp_map_id = "vocab:...")

Same-family models share the same tokenizer -- same vocabulary, same token indices. The vocabulary serves as a natural shared coordinate system with dimensionality equal to the vocabulary size (e.g. 151K for Qwen2). The projection requires zero learned parameters and no calibration:

```
Source model (D_src):  hidden @ W_src^T  -> logits [vocab_size]
                       softmax(logits)   -> token probabilities [vocab_size]
                       probs @ W_tgt     -> target embedding [D_tgt]
Target model (D_tgt):  inject via inputs_embeds
```

Where `W_src` is the source model's output head (lm_head) weights and `W_tgt` is the target model's input embedding weights. This is the cross-model generalization of the tied-weight soft embedding projection in Section 4.1.

The method is identified by `avp_map_id` starting with `"vocab:"`, followed by the first 16 characters of the shared tokenizer hash.

#### Learned Projection (avp_map_id = "{src}_{tgt}")

For model pairs that do not share a tokenizer, a learned linear map can be pre-calibrated via ridge regression or orthogonal Procrustes alignment:

```
target_hidden = source_hidden @ W_map + bias
```

Calibration runs anchor texts through both models, collects hidden states, and fits `W_map` to minimize reconstruction error. Calibrated maps are stored as `.avp-map` files in `~/.avp/maps/` and identified by `avp_map_id = "{source_hash[:16]}_{target_hash[:16]}"`.

| Method | Cross-dim? | Training | Quality | Use case |
|--------|-----------|----------|---------|----------|
| Vocabulary-mediated | Yes | None (instant) | High (cos sim ~1.0) | Same-family, shared tokenizer |
| Ridge regression | Yes | ~60s calibration | Low for large dim gaps | Cross-family (experimental) |
| Orthogonal Procrustes | Same dim only | ~1s calibration | Good | Same-dim model variants |

#### Projection Method Enum

Maps carry a `method` field indicating the projection algorithm:

| Value | Description |
|-------|-------------|
| `vocab_mediated` | Vocabulary-mediated projection (zero parameters) |
| `ridge` | Ridge regression learned map |
| `procrustes` | Orthogonal Procrustes alignment |

## 5. Binary Format

See [protocol/binary-format.md](protocol/binary-format.md)

## 6. Compression

See [protocol/compression.md](protocol/compression.md)

## 7. Transport Layer

AVP is transport-agnostic. The binary format and handshake protocol do not depend on any specific transport. Implementations can carry AVP messages over HTTP/2, gRPC, WebSockets, A2A DataParts, shared memory, or any other channel that supports binary payloads.

The reference HTTP/2 transport binding is documented in [protocol/transport.md](protocol/transport.md).

## 8. Security Considerations

See [protocol/security.md](protocol/security.md)

## 9. Integration with Agent Protocols

AVP is a latent communication layer, not an orchestration protocol. It is designed to work alongside any agent protocol that handles discovery, delegation, and task management.

### A2A

Integration with [A2A](https://github.com/google/A2A):
- AVP capabilities advertised via URI-namespaced A2A extensions
- Binary payloads transmitted as `multipart/related` HTTP parts with `cid:` URIs
- Handshake data carried in A2A DataParts

### MCP

Agents connected via [MCP](https://modelcontextprotocol.io/) (tool/resource access) can use AVP for latent communication when both agents run the same model. MCP handles tool invocation; AVP handles the tensor transfer.

### Other Protocols

Any orchestration layer that can pass binary payloads between agents can use AVP. The binary format and handshake are self-contained and do not depend on the transport or orchestration protocol.

## 10. Versioning

AVP follows semantic versioning (MAJOR.MINOR.PATCH).

Current version: **0.2.0**

## 11. References

- [LatentMAS: Latent Collaboration in Multi-Agent Systems](https://arxiv.org/abs/2511.20639) -- research foundation for same-model latent communication and realignment

## 12. Authors

VectorArc Team

## 13. License

Apache 2.0 -- See LICENSE file
