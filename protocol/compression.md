# AVP Compression Specification

## Overview

AVP supports optional compression using **zstd** (Zstandard).

## Compression Levels

| Level | Zstd Level | Use Case | Speed | Ratio |
|-------|-----------|----------|-------|-------|
| **fast** | 1 | Real-time | ⚡⚡⚡ | 📦📦 |
| **balanced** | 3 | Default | ⚡⚡ | 📦📦📦 |
| **max** | 19 | Archival | ⚡ | 📦📦📦📦 |

## When to Compress

**DO compress:**
- ✅ Embeddings > 1KB
- ✅ Network-constrained environments
- ✅ Batch transmission

**DON'T compress:**
- ❌ Embeddings < 1KB
- ❌ Ultra-low latency requirements

## Typical Results

4096-dim float32 embedding:
- Uncompressed: 16KB
- Compressed (level 3): ~11KB
- **Savings: 30%**

Combined with binary format:
- JSON: 200KB
- AVP + compression: 11KB
- **Total savings: 94.5%**
