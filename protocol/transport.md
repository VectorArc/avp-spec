# AVP Transport Layer Specification

## Overview

AVP messages transmitted over **HTTP/2** with binary payloads.

## Endpoint Format
```
POST /avp/v1/transmit
Host: agent-receiver.example.com
Content-Type: application/avp+binary
AVP-Version: 1.0
```

## Required Headers

| Header | Description |
|--------|-------------|
| `Content-Type` | Must be `application/avp+binary` |
| `AVP-Version` | Protocol version (e.g., "1.0") |
| `AVP-Agent-ID` | Sender agent identifier |

## Security

- **TLS 1.2+** required (HTTPS only)
- Recommended: Bearer token authentication
```http
Authorization: Bearer <api-key>
```

## Example Request
```python
import requests

headers = {
    'Content-Type': 'application/avp+binary',
    'AVP-Version': '1.0',
    'AVP-Agent-ID': 'agent-a-123',
}

response = requests.post(
    'https://receiver.example.com/avp/v1/transmit',
    headers=headers,
    data=binary_payload,
)
```
