# AVP Compression Specification

## Overview

AVP supports optional compression using **zstd** (Zstandard).

## Compression Levels

| Level | Zstd Level | Use Case | Trade-off |
|-------|-----------|----------|-----------|
| **fast** | 1 | Real-time | Fastest, lowest ratio |
| **balanced** | 3 | Default | Good balance |
| **max** | 19 | Archival/batch | Slowest, best ratio |

## When to Compress

Compress:
- Embedding payloads > 1KB
- Network-constrained environments
- Batch transmission

Don't compress:
- Hidden state or KV-cache payloads (zstd achieves only 1-7% on random tensor data)
- Payloads < 1KB (overhead exceeds savings)
- Ultra-low latency requirements

## Typical Results

**Embeddings** (compressible -- repeated patterns in floating point):

4096-dim float32 embedding:
- Uncompressed: ~16KB
- Compressed (level 3): ~11KB
- Savings: ~30%

**Hidden states and KV-cache** (effectively random data):

4096-dim float32 hidden state:
- Uncompressed: ~16KB
- Compressed (level 3): ~15KB
- Savings: 1-7%

For latent communication, the primary value is skipping autoregressive generation, not bandwidth reduction. Compression is most useful for embedding payloads.
