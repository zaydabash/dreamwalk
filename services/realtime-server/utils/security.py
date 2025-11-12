"""
Security utilities for input validation and rate limiting
"""
import re
from typing import Any, Dict, Optional
from fastapi import HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import time
from collections import defaultdict

# Rate limiting storage (in-memory for development, use Redis in production)
rate_limit_storage: Dict[str, list] = defaultdict(list)


def validate_session_id(session_id: str) -> str:
    """
    Validate session ID format.
    
    Args:
        session_id: Session ID to validate
        
    Returns:
        Validated session ID
        
    Raises:
        HTTPException: If session ID is invalid
    """
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID cannot be empty")
    
    # Session ID should be alphanumeric with hyphens and underscores
    if not re.match(r'^[a-zA-Z0-9_-]+$', session_id):
        raise HTTPException(
            status_code=400,
            detail="Session ID can only contain alphanumeric characters, hyphens, and underscores"
        )
    
    # Limit length
    if len(session_id) > 100:
        raise HTTPException(status_code=400, detail="Session ID is too long")
    
    return session_id


def validate_json_input(data: Dict[str, Any], required_fields: list, max_size: int = 10000) -> Dict[str, Any]:
    """
    Validate JSON input data.
    
    Args:
        data: Input data to validate
        required_fields: List of required field names
        max_size: Maximum size of JSON data in bytes
        
    Returns:
        Validated data
        
    Raises:
        HTTPException: If validation fails
    """
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Input must be a JSON object")
    
    # Check size (approximate)
    import json
    data_size = len(json.dumps(data).encode('utf-8'))
    if data_size > max_size:
        raise HTTPException(status_code=400, detail=f"Input data too large (max {max_size} bytes)")
    
    # Check required fields
    for field in required_fields:
        if field not in data:
            raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
    
    return data


def sanitize_string(value: str, max_length: int = 1000) -> str:
    """
    Sanitize string input.
    
    Args:
        value: String to sanitize
        max_length: Maximum length of string
        
    Returns:
        Sanitized string
        
    Raises:
        HTTPException: If string is invalid
    """
    if not isinstance(value, str):
        raise HTTPException(status_code=400, detail="Value must be a string")
    
    # Remove null bytes and control characters
    value = value.replace('\x00', '')
    value = ''.join(char for char in value if ord(char) >= 32 or char in '\n\r\t')
    
    # Limit length
    if len(value) > max_length:
        raise HTTPException(status_code=400, detail=f"String too long (max {max_length} characters)")
    
    return value


def check_rate_limit(identifier: str, max_requests: int = 100, window_seconds: int = 60) -> bool:
    """
    Check if request should be rate limited.
    
    Args:
        identifier: Unique identifier (IP address, session ID, etc.)
        max_requests: Maximum number of requests allowed
        window_seconds: Time window in seconds
        
    Returns:
        True if request should be allowed, False if rate limited
        
    Note:
        This is a simple in-memory implementation for development.
        Use Redis-based rate limiting in production.
    """
    current_time = time.time()
    
    # Clean old entries
    rate_limit_storage[identifier] = [
        req_time for req_time in rate_limit_storage[identifier]
        if current_time - req_time < window_seconds
    ]
    
    # Check if limit exceeded
    if len(rate_limit_storage[identifier]) >= max_requests:
        return False
    
    # Add current request
    rate_limit_storage[identifier].append(current_time)
    return True


def get_client_ip(request: Request) -> str:
    """
    Get client IP address from request.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Client IP address
    """
    # Check for forwarded IP (behind proxy)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Fall back to direct client IP
    if request.client:
        return request.client.host
    
    return "unknown"


# Security bearer token (placeholder for future authentication)
security = HTTPBearer(auto_error=False)


async def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = None) -> Optional[str]:
    """
    Verify authentication token.
    
    Args:
        credentials: HTTP bearer credentials
        
    Returns:
        User ID if token is valid, None otherwise
        
    Note:
        This is a placeholder. Implement proper token verification in production.
    """
    if credentials is None:
        return None
    
    # TODO: Implement proper token verification
    # For development, accept any token
    # In production, verify JWT token or API key
    token = credentials.credentials
    if token:
        return "user"  # Placeholder
    
    return None

