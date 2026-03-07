# Agent Vector Protocol (AVP) Specification

**Version:** 0.3
**Status:** Draft
**Last Updated:** March 2026

## Abstract

Agent Vector Protocol (AVP) is a binary protocol that enables LLM agents to communicate via latent representations (hidden states and KV-cache) instead of text. Same-model agents skip autoregressive generation entirely and exchange intermediate tensors directly. Cross-model agents -- same family or different families -- communicate via vocabulary-mediated projection with zero training. Models with no compatible projection path fall back to JSON text.

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

This specification covers same-model latent communication and cross-model communication via vocabulary-mediated projection (Rosetta Stone v2). Same-family models project through shared vocabulary; cross-family models project through overlapping BPE tokens. Both require zero training.

## 2. Protocol Overview

### 2.1 Communication Modes

AVP supports three communication modes, negotiated during handshake:

| Mode | When | What's transmitted |
|------|------|-------------------|
| **Latent** | Same model (hash or structure match) | Hidden states, KV-cache |
| **Latent (cross-model)** | Same or different family | Vocabulary-mediated projected hidden states |
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
5. **Sufficient vocabulary overlap (>= 100 tokens)** -> Latent mode with `avp_map_id="vocab_overlap:{overlap_count}"` (vocabulary-overlap cross-model projection)
6. **No match** -> JSON fallback

Rules 1-2 resolve same-model communication (no projection needed). Rule 3 enables zero-parameter cross-model communication between same-family models that share a tokenizer (e.g. Qwen2.5-1.5B and Qwen2.5-0.5B). Rule 4 enables pre-calibrated cross-model communication via learned linear maps. Rule 5 enables zero-parameter cross-family communication by projecting through the overlapping portion of two different BPE vocabularies (e.g. Qwen to Llama, ~85% token overlap). When `avp_map_id` is non-empty, the session requires a Rosetta Stone projection map (see Section 4.3).

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

#### Mode 2: Quantized KV-cache (specified, not yet implemented)

Transmit KV-cache in int8 or int4 representation. Reduces payload by 2-4x with negligible quality impact (int8) or small quality impact (int4). No additional receiver compute. Best for moderate bandwidth (500 Mbps - 1 Gbps).

#### Mode 3: Hidden-state transfer

Transmit hidden state vectors only, without the full KV-cache. The receiver reconstructs context by running latent steps (forward passes with the injected hidden states). Reduces payload by 16x or more. Trades bandwidth for receiver compute. Best for lower bandwidth (<500 Mbps).

This is what LatentMAS uses for Agent 4 (Judger) -- it receives hidden states via `inputs_embeds`, not KV-cache.

#### Mode 4: Delta transfer (specified, not yet implemented)

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

When agents run different models -- same family or different families -- they can communicate via latent projection instead of JSON fallback. The handshake sets `avp_map_id` to indicate which projection method to use.

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

#### Vocabulary-Overlap Projection (avp_map_id = "vocab_overlap:...")

Cross-family models have different tokenizers, but BPE tokenizers share many tokens (ASCII characters, common English words, punctuation). The vocabulary overlap bridge identifies shared tokens between the two vocabularies and projects through only the overlapping portion:

```
Source model (D_src):  hidden @ W_src^T           -> full_logits [vocab_size_src]
                       full_logits[src_indices]    -> shared_logits [N_shared]
                       softmax(shared_logits)      -> shared_probs [N_shared]  (renormalized)
                       shared_probs @ W_tgt_shared -> target embedding [D_tgt]
Target model (D_tgt):  inject via inputs_embeds
```

Where `src_indices` are the source token IDs for tokens that exist in both vocabularies, and `W_tgt_shared` are the corresponding target embedding rows. The softmax renormalization over shared tokens is semantically correct: "given the source model's beliefs about the next token, restricted to tokens both models understand, what's the expected target embedding?"

Typical overlap ratios: ~85% for Qwen/Llama, varying by family pair. Minimum overlap threshold: 100 shared tokens (below this, fall through to learned projection or JSON fallback).

This is a strict generalization of vocabulary-mediated projection -- at 100% vocabulary overlap (same tokenizer), it produces identical results. The method requires zero learned parameters and no calibration.

The method is identified by `avp_map_id` starting with `"vocab_overlap:"`, followed by the overlap count.

#### Learned Projection (avp_map_id = "{src}_{tgt}")

For model pairs that do not share a tokenizer, a learned linear map can be pre-calibrated via ridge regression or orthogonal Procrustes alignment:

```
target_hidden = source_hidden @ W_map + bias
```

Calibration runs anchor texts through both models, collects hidden states, and fits `W_map` to minimize reconstruction error. Calibrated maps are stored as `.avp-map` files in `~/.avp/maps/` and identified by `avp_map_id = "{source_hash[:16]}_{target_hash[:16]}"`.

| Method | Cross-dim? | Training | Quality | Use case |
|--------|-----------|----------|---------|----------|
| Vocabulary-mediated | Yes | None (instant) | High (cos sim ~1.0) | Same-family, shared tokenizer |
| Vocabulary-overlap | Yes | None (instant) | High on structured tasks | Cross-family, different tokenizers |
| Ridge regression | Yes | ~60s calibration | Low for large dim gaps | Fallback for low-overlap pairs |
| Orthogonal Procrustes | Same dim only | ~1s calibration | Good | Same-dim model variants |

#### Projection Method Enum

Maps carry a `method` field indicating the projection algorithm:

| Value | Description |
|-------|-------------|
| `vocab_mediated` | Vocabulary-mediated projection, shared tokenizer (zero parameters) |
| `vocab_overlap` | Vocabulary-overlap projection, different tokenizers (zero parameters) |
| `ridge` | Ridge regression learned map |
| `procrustes` | Orthogonal Procrustes alignment |

#### Projection Validation

Before using a cross-model projection in production, implementations SHOULD validate projection quality. AVP defines a two-tier validation gate:

**Tier 1: Cosine similarity (fast, ~1ms)**

Project source hidden states through the projection and compare to target embeddings. For vocabulary-mediated projection, `projected[i]` should predict `target_embed[token_ids[i+1]]` (next-token prediction). Reject projections with cosine similarity below 0.5 (instant JSON fallback).

**Tier 2: Pseudo-perplexity (~30ms, requires shared tokenizer)**

Inject a single projected embedding to prime the target model's context, then feed actual text tokens and measure cross-entropy. This mirrors the actual pipeline behavior.

| Perplexity | Recommendation |
|------------|---------------|
| ≤ 100 | LATENT — projection quality sufficient for direct use |
| > 100 | JSON — projection too lossy, fall back to text |

Thresholds are calibrated against real pipeline results: Qwen2.5-1.5B→0.5B achieves pseudo-perplexity of 25.8 and produces coherent output in the latent pipeline.

#### Per-Transfer Quality Gate

Cross-model projection accuracy depends on prompt length. Single-embedding rosetta works well for short structured prompts but degrades for longer prompts:

| Prompt tokens | GSM8K cross-family | HumanEval same-family |
|---------------|--------------------|-----------------------|
| < 300 | 65% | 61% |
| 300-500 | 41% | 40% |
| 500+ | — | 19% |

Implementations SHOULD provide an advisory quality gate that recommends latent vs JSON fallback based on prompt token count. The default threshold is 300 tokens. The gate is advisory — callers decide how to act on the recommendation.

```
from avp.rosetta.quality import assess_transfer

result = assess_transfer(prompt_tokens=len(input_ids[0]))
if result.recommend_latent:
    # proceed with rosetta projection
else:
    # fall back to JSON text transfer
```

### 4.5 Engine Support

AVP is engine-agnostic. The binary format, handshake, and codec do not depend on any specific inference engine. Implementations provide engine-specific connectors that handle hidden state extraction, KV-cache access, and embedding injection.

#### Connector Interface

All engine connectors implement the `EngineConnector` abstract base class, which defines both a low-level interface (hidden state extraction, embedding injection, KV-cache access) and a high-level API for common integration patterns.

##### High-Level API

The high-level API reduces AVP integration from ~50 lines of boilerplate to ~5 lines:

| Method | Description | Returns |
|--------|-------------|---------|
| `think(prompt, steps, context)` | Run latent thinking steps, accumulating a KV-cache without producing text | `AVPContext` |
| `generate(prompt, context, source, ...)` | Generate text, optionally conditioned on latent context from `think()`. `source=` enables automatic cross-model projection. | `str` |
| `can_think` | Whether this connector supports `think()` (requires hidden state access) | `bool` |

`AVPContext` is a lightweight wrapper around a KV-cache (tensor references, no copy) with metadata for compatibility checking:
- `past_key_values` -- DynamicCache or legacy tuple
- `model_hash` -- SHA-256 of source model config (checked implicitly by `think()`/`generate()`)
- `num_steps` -- accumulated latent thinking steps
- `seq_len` -- current KV-cache sequence length
- `last_hidden_state` -- final hidden state vector from the last latent step (used for cross-model projection)
- `to_bytes()` / `from_bytes()` -- serialize to/from AVP wire format for cross-process transfer

Same-process usage never requires serialization -- `AVPContext` holds tensor references directly. `to_bytes()` invokes the standard AVP codec (Section 5) and stores `model_hash`, `model_family`, and `num_steps` in the protobuf metadata `extra` fields.

##### Cross-Model via source=

When `generate()` receives a `source=` parameter pointing to a different connector, it automatically:
1. Detects model mismatch via `model_hash`
2. Calibrates or loads a projection map (memory cache -> disk cache -> calibrate)
3. Projects `context.last_hidden_state` through the Rosetta Stone projection
4. Primes the target model's KV-cache with the projected embedding
5. Generates text conditioned on the projected context

```
researcher = HuggingFaceConnector.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
solver = HuggingFaceConnector.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
context = researcher.think(prompt, steps=10)
answer = solver.generate(prompt, context=context, source=researcher)
```

##### Capability Discovery

Not all engines support all operations. Connectors advertise capabilities via properties:

| Capability | HuggingFace | vLLM SDK | Description |
|-----------|-------------|----------|-------------|
| `can_think` | Yes | No | Latent thinking requires per-step hidden state access |
| `generate()` with context | Yes | No | KV-cache injection requires direct cache access |
| `generate()` without context | Yes | Yes | Text-only generation |

Calling `think()` on a connector with `can_think=False` raises `EngineNotAvailableError` with a message guiding the user to the correct connector. Calling `generate()` with a `context` argument on a connector that doesn't support it raises the same error.

#### HuggingFace Transformers

Full hidden state and KV-cache access via the standard `model()` and `model.generate()` APIs. Supports `output_hidden_states=True` for hidden state extraction, `past_key_values` (DynamicCache or legacy tuple) for KV-cache injection, and `inputs_embeds` for embedding injection. This is the primary development and benchmarking backend.

The HuggingFace connector supports the full high-level API:

```
connector = HuggingFaceConnector.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
context = connector.think("Analyze this problem: ...", steps=20)
answer  = connector.generate("Solve it.", context=context)
```

`from_pretrained()` auto-detects device (CUDA > MPS > CPU) and dtype (bfloat16 > float16 > float32).

#### vLLM

Production serving integration via two components at different levels:

1. **SDK connector** (`VLLMConnector`): Wraps the vLLM `LLM` engine for identity extraction, tokenization, text generation, and embedding injection via the `prompt_embeds` API. Supports `generate()` for text-only generation. Does not support `think()` or context injection -- vLLM is a serving engine that manages KV-cache internally and does not expose per-step hidden states.

2. **KV connector plugin** (`AVPKVConnectorV1Dynamic`): Implements `KVConnectorBase_V1` for intercepting vLLM's attention pipeline. Uses file-based storage (`{request_id}.avp`) for KV-cache save/load between requests. Includes PagedAttention ↔ contiguous tensor conversion for bridging vLLM's paged memory layout with AVP's contiguous binary format. This is where latent transfer happens for vLLM -- transparently at the engine level, not at the SDK level.

The KV connector plugin is loaded dynamically via vLLM's `--kv-connector` configuration -- no vLLM fork required.

The two components serve different roles: the SDK connector provides handshake, identity, and text generation for application code. The KV connector plugin handles latent transfer between vLLM instances, transparent to the application.

### 4.6 Easy API

The easy API provides a zero-friction entry point for common use cases. It manages model loading, connector creation, handshake, and serialization internally.

**Primary API (v0.3.0):**

| Function | Description | Returns |
|----------|-------------|---------|
| `think(content, model, steps, ...)` | Load model, run latent thinking steps | `AVPContext` |
| `generate(content, model, steps, source_model, store, store_key, ...)` | Think + generate in one call. `source_model=` enables cross-model projection. | `str` |

**Deprecated API (v0.2.x, still exported):**

| Function | Description | Returns |
|----------|-------------|---------|
| `pack(content, model, think_steps, ...)` | Load model, run latent thinking steps, return a serializable message | `PackedMessage` |
| `unpack(data, model, ...)` | Receive a packed message, generate a text response | `str` |

```
import avp

# Same-model
answer = avp.generate("Solve: 24 * 17 + 3", model="Qwen/Qwen2.5-7B-Instruct")

# Cross-model (automatic rosetta projection)
answer = avp.generate("Solve: 24 * 17 + 3",
                       model="meta-llama/Llama-3.2-3B-Instruct",
                       source_model="Qwen/Qwen2.5-7B-Instruct")
```

#### ContextStore

`ContextStore` is a thread-safe, TTL-backed store for `PackedMessage` objects. It enables multi-turn latent conversations where each agent stores its context under a key and retrieves prior context from other agents:

```
store = ContextStore(default_ttl=300)
result = avp.generate(content, model=model, store=store, store_key="agent_a", prior_key="agent_b")
```

### 4.7 Observability

Implementations SHOULD provide timing metrics for key operations. The SDK exposes metrics via `collect_metrics=True` on `pack()`, `unpack()`, and `generate()`:

- `PackMetrics`: identity extraction time, think duration, total duration
- `UnpackMetrics`: decode duration, generate duration, total duration
- `GenerateMetrics`: pack + unpack combined metrics
- `HandshakeMetrics`: resolution time, mode selected

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

Current version: **0.3.0**

## 11. References

- [LatentMAS: Latent Collaboration in Multi-Agent Systems](https://arxiv.org/abs/2511.20639) -- research foundation for same-model latent communication and realignment
- [AVP Benchmark Results](https://github.com/VectorArc/avp-python/blob/main/docs/BENCHMARKS.md) -- 8 benchmarks, 5 models, 2 families, same-model + cross-model (46-78% token savings, 2-4x faster, +8.6pp on code generation)

## 12. Authors

VectorArc Team

## 13. License

Apache 2.0 -- See LICENSE file
