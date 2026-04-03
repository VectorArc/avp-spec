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

Hidden states and KV-cache payloads may contain sensitive information derived from model inputs.

**DO log:**
- Metadata (timestamps, agent IDs, sizes, payload types)
- Headers

**DO NOT log:**
- Raw tensor payloads (hidden states, KV-cache)
- Binary data

## DoS Protection

- Limit message size (10MB max recommended)
- Rate limiting per agent
- Validate headers before reading payload
- Set decompression timeouts
- Enforce session TTL and cleanup expired sessions
