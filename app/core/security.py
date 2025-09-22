"""
Security utilities and middleware for HIPAA compliance
Enhanced security headers and utilities
"""

import secrets
import hashlib
from typing import Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class SecurityHeaders(BaseHTTPMiddleware):
    """
    Middleware to add security headers for HIPAA compliance and general security.
    """
    
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        
        # Generate nonce for CSP
        nonce = secrets.token_urlsafe(16)
        
        # Security headers for medical application
        security_headers = {
            # Prevent clickjacking
            "X-Frame-Options": "DENY",
            
            # Prevent MIME type sniffing
            "X-Content-Type-Options": "nosniff",
            
            # XSS protection
            "X-XSS-Protection": "1; mode=block",
            
            # Prevent information disclosure
            "X-Powered-By": "",  # Remove server identification
            "Server": "Medical-Server",  # Generic server name
            
            # HTTPS enforcement (when in production behind reverse proxy)
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            
            # Content Security Policy for medical app
            "Content-Security-Policy": (
                f"default-src 'self'; "
                f"script-src 'self' 'nonce-{nonce}'; "
                f"style-src 'self' 'unsafe-inline'; "
                f"img-src 'self' data: blob:; "
                f"font-src 'self'; "
                f"connect-src 'self'; "
                f"frame-ancestors 'none'; "
                f"base-uri 'self'; "
                f"form-action 'self'"
            ),
            
            # Referrer policy to prevent information leakage
            "Referrer-Policy": "strict-origin-when-cross-origin",
            
            # Permissions policy (disable unnecessary features)
            "Permissions-Policy": (
                "accelerometer=(), "
                "camera=(), "
                "geolocation=(), "
                "gyroscope=(), "
                "magnetometer=(), "
                "microphone=(), "
                "payment=(), "
                "usb=()"
            ),
            
            # Cache control for sensitive medical data
            "Cache-Control": "no-store, no-cache, must-revalidate, private",
            "Pragma": "no-cache",
            "Expires": "0",
            
            # Custom headers for audit trail
            "X-Request-ID": getattr(request.state, 'request_id', 'unknown'),
            "X-Content-Security-Policy": f"nonce-{nonce}",
        }
        
        # Add all security headers to response
        for header, value in security_headers.items():
            response.headers[header] = value
        
        return response


def generate_secure_token(length: int = 32) -> str:
    """Generate cryptographically secure random token."""
    return secrets.token_urlsafe(length)


def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    """
    Hash password using PBKDF2 with SHA-256 (HIPAA compliant).
    
    Returns:
        tuple: (hashed_password, salt)
    """
    if salt is None:
        salt = secrets.token_hex(32)
    
    # Use PBKDF2 with 100,000 iterations (OWASP recommendation)
    hashed = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000  # iterations
    )
    
    return hashed.hex(), salt


def verify_password(password: str, hashed_password: str, salt: Optional[str] = None) -> bool:
    """Verify password against hash. Supports both bcrypt and PBKDF2."""
    # Check if this is a bcrypt hash (starts with $2b$)
    if hashed_password.startswith('$2b$'):
        try:
            from passlib.context import CryptContext
            pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
            return pwd_context.verify(password, hashed_password)
        except ImportError:
            # Fallback if passlib is not available
            import bcrypt
            return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
    
    # Original PBKDF2 verification
    if salt is None:
        return False
    computed_hash, _ = hash_password(password, salt)
    return secrets.compare_digest(computed_hash, hashed_password)


def generate_request_id() -> str:
    """Generate unique request ID for audit trail."""
    return f"req_{secrets.token_hex(8)}"


class RateLimiter:
    """Simple in-memory rate limiter for medical API protection."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 3600):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}  # ip -> [(timestamp, count), ...]
    
    def is_allowed(self, client_ip: str) -> tuple[bool, int]:
        """
        Check if request is allowed and return remaining requests.
        
        Returns:
            tuple: (is_allowed, requests_remaining)
        """
        import time
        
        now = time.time()
        window_start = now - self.window_seconds
        
        # Clean old entries
        if client_ip in self.requests:
            self.requests[client_ip] = [
                (timestamp, count) for timestamp, count in self.requests[client_ip]
                if timestamp > window_start
            ]
        
        # Count requests in current window
        current_requests = sum(
            count for timestamp, count in self.requests.get(client_ip, [])
        )
        
        if current_requests >= self.max_requests:
            return False, 0
        
        # Add current request
        if client_ip not in self.requests:
            self.requests[client_ip] = []
        self.requests[client_ip].append((now, 1))
        
        remaining = self.max_requests - current_requests - 1
        return True, remaining


# Global rate limiter instance
rate_limiter = RateLimiter()


def validate_npi_number(npi: str) -> bool:
    """
    Validate National Provider Identifier (NPI) number.
    NPI is a 10-digit number used to identify healthcare providers.
    """
    if not npi or len(npi) != 10 or not npi.isdigit():
        return False
    
    # NPI uses Luhn algorithm for validation
    def luhn_checksum(card_num):
        def digits_of(number):
            return [int(d) for d in str(number)]
        
        digits = digits_of(card_num)
        odd_digits = digits[-1::-2]
        even_digits = digits[-2::-2]
        checksum = sum(odd_digits)
        for digit in even_digits:
            checksum += sum(digits_of(digit * 2))
        return checksum % 10
    
    return luhn_checksum(npi) == 0


def validate_dea_number(dea: str) -> bool:
    """
    Validate Drug Enforcement Administration (DEA) number.
    Format: 2 letters + 7 digits, with specific checksum validation.
    """
    if not dea or len(dea) != 9:
        return False
    
    letters = dea[:2].upper()
    digits = dea[2:]
    
    if not letters.isalpha() or not digits.isdigit():
        return False
    
    # DEA validation algorithm
    if letters[0] not in ['A', 'B', 'F', 'G', 'M', 'P', 'R', 'S']:
        return False
    
    # Calculate checksum
    sum1 = sum(int(digits[i]) for i in [0, 2, 4])
    sum2 = sum(int(digits[i]) for i in [1, 3, 5]) * 2
    total = sum1 + sum2
    
    return str(total)[-1] == digits[6]