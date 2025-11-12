# Service Security Guidelines

## Input Validation

All services should validate and sanitize all user inputs:

### Session IDs
- Must be alphanumeric with hyphens and underscores only
- Maximum length: 100 characters
- Cannot be empty

### JSON Input
- Must be valid JSON objects
- Maximum size: 10KB (configurable)
- Required fields must be present
- Type validation using Pydantic models

### String Input
- Remove null bytes and control characters
- Maximum length: 1000 characters (configurable)
- Sanitize to prevent injection attacks

## Rate Limiting

**Development:**
- In-memory rate limiting (100 requests per minute per IP)
- Simple implementation for development

**Production:**
- Redis-based rate limiting
- Configurable limits per endpoint
- Per-user and per-IP rate limiting
- Exponential backoff for repeated violations

## Authentication

**Development:**
- No authentication required
- Open CORS for development

**Production:**
- JWT token-based authentication
- API key authentication for service-to-service communication
- OAuth2 for user authentication
- Role-based access control (RBAC)

## Error Handling

- Never expose internal error messages to clients
- Log detailed errors server-side
- Return generic error messages to clients
- Sanitize error messages before sending

## Data Privacy

- Never log sensitive data (API keys, passwords, neural data)
- Encrypt sensitive data at rest and in transit
- Implement data retention policies
- Provide data deletion capabilities
- Anonymize data for logging and analytics

## Service Communication

**Development:**
- HTTP communication between services
- No authentication between services

**Production:**
- HTTPS/TLS for all service communication
- Mutual TLS (mTLS) for service-to-service authentication
- Service mesh for secure communication
- Network segmentation and firewalls

## Monitoring and Logging

- Log all authentication attempts
- Monitor for suspicious activity
- Track failed requests and errors
- Implement alerting for security events
- Regular security log reviews

## Dependency Security

- Regularly update dependencies
- Scan for known vulnerabilities
- Use dependency pinning in production
- Review and approve dependency updates

## Example: Adding Input Validation

```python
from fastapi import HTTPException
from .utils.security import validate_session_id, validate_json_input

@app.post("/sessions/start")
async def start_session(request: StreamRequest):
    # Validate session ID
    session_id = validate_session_id(request.session_id)
    
    # Validate request data
    request_data = validate_json_input(
        request.dict(),
        required_fields=["session_id", "signal_type"],
        max_size=10000
    )
    
    # Process request...
```

## Example: Adding Rate Limiting

```python
from fastapi import Request, HTTPException
from .utils.security import check_rate_limit, get_client_ip

@app.post("/sessions/start")
async def start_session(request: Request, stream_request: StreamRequest):
    # Check rate limit
    client_ip = get_client_ip(request)
    if not check_rate_limit(client_ip, max_requests=100, window_seconds=60):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )
    
    # Process request...
```

## Security Checklist

- [ ] Input validation on all endpoints
- [ ] Rate limiting implemented
- [ ] Authentication enabled (production)
- [ ] HTTPS/TLS enabled (production)
- [ ] Error messages sanitized
- [ ] Sensitive data not logged
- [ ] Dependencies up to date
- [ ] Security scanning in CI/CD
- [ ] Regular security audits
- [ ] Monitoring and alerting configured

