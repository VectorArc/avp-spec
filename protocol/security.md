# AVP Security Considerations

## Transport Security

- **Minimum**: TLS 1.2
- **Recommended**: TLS 1.3
- **Required**: HTTPS (never HTTP)

## Authentication

Recommended approaches:
1. API Keys (service-to-service)
2. OAuth 2.0 (user-delegated)
3. mTLS (high-security)

## Data Privacy

Embeddings MAY contain sensitive information.

**DO log:**
- ✅ Metadata (timestamps, agent IDs, sizes)
- ✅ Headers

**DO NOT log:**
- ❌ Raw embedding payloads
- ❌ Binary data

## DoS Protection

- Limit message size (10MB max recommended)
- Rate limiting
- Validate headers before payload
- Set decompression timeouts
