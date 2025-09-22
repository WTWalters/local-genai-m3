"""
Pydantic schemas for authentication API requests and responses
HIPAA-compliant data validation and serialization
"""

from typing import Optional
from pydantic import BaseModel, EmailStr, Field, validator
from datetime import datetime


class LoginRequest(BaseModel):
    """Login request with medical professional credentials."""
    
    username: str = Field(
        ..., 
        min_length=3, 
        max_length=50,
        description="Username or email address"
    )
    password: str = Field(
        ..., 
        min_length=8, 
        description="Password (minimum 8 characters)"
    )
    mfa_code: Optional[str] = Field(
        None,
        pattern=r'^\d{6}$',
        description="6-digit MFA/TOTP code (required for PHI access)"
    )
    remember_me: bool = Field(
        default=False,
        description="Extended session duration"
    )
    
    @validator('username')
    def validate_username(cls, v):
        """Sanitize username input."""
        return v.strip().lower()
    
    class Config:
        json_schema_extra = {
            "example": {
                "username": "dr.smith",
                "password": "SecurePassword123!",
                "mfa_code": "123456",
                "remember_me": False
            }
        }


class UserInfo(BaseModel):
    """User information returned after authentication."""
    
    user_id: str
    username: str
    full_name: str
    email: EmailStr
    role: str
    can_access_phi: bool
    department: Optional[str] = None
    title: Optional[str] = None
    npi_number: Optional[str] = None
    mfa_enabled: bool
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "550e8400-e29b-41d4-a716-446655440000",
                "username": "dr.smith",
                "full_name": "Dr. John Smith",
                "email": "j.smith@orthoclinic.com",
                "role": "attending_physician",
                "can_access_phi": True,
                "department": "Orthopedics",
                "title": "Attending Physician",
                "npi_number": "1234567890",
                "mfa_enabled": True
            }
        }


class LoginResponse(BaseModel):
    """Successful login response with tokens and user information."""
    
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Access token expiration in seconds")
    user_info: UserInfo
    
    class Config:
        json_schema_extra = {
            "example": {
                "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
                "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
                "token_type": "bearer",
                "expires_in": 900,
                "user_info": {
                    "user_id": "550e8400-e29b-41d4-a716-446655440000",
                    "username": "dr.smith",
                    "full_name": "Dr. John Smith",
                    "email": "j.smith@orthoclinic.com",
                    "role": "attending_physician",
                    "can_access_phi": True,
                    "department": "Orthopedics",
                    "title": "Attending Physician",
                    "npi_number": "1234567890",
                    "mfa_enabled": True
                }
            }
        }


class LogoutRequest(BaseModel):
    """Logout request."""
    
    session_id: Optional[str] = Field(
        None,
        description="Session ID to logout (optional, uses current session if not provided)"
    )


class RefreshTokenRequest(BaseModel):
    """Token refresh request."""
    
    refresh_token: str = Field(..., description="Valid refresh token")


class RefreshTokenResponse(BaseModel):
    """Token refresh response with new access token."""
    
    access_token: str = Field(..., description="New JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Access token expiration in seconds")


class MFASetupRequest(BaseModel):
    """MFA setup request for generating QR code."""
    
    issuer_name: str = Field(
        default="Orthopedics EMR",
        description="MFA issuer name"
    )


class MFASetupResponse(BaseModel):
    """MFA setup response with secret and QR code."""
    
    secret: str = Field(..., description="Base32 encoded TOTP secret")
    qr_code_url: str = Field(..., description="QR code data URL for setup")
    backup_codes: list[str] = Field(..., description="One-time backup codes")


class MFAVerifyRequest(BaseModel):
    """MFA verification request."""
    
    code: str = Field(
        ...,
        pattern=r'^\d{6}$',
        description="6-digit TOTP code"
    )


class PasswordChangeRequest(BaseModel):
    """Password change request with medical compliance."""
    
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(
        ..., 
        min_length=12,
        description="New password (minimum 12 characters for medical compliance)"
    )
    confirm_password: str = Field(..., description="Confirm new password")
    
    @validator('new_password')
    def validate_password_strength(cls, v):
        """Validate password meets medical security standards."""
        if len(v) < 12:
            raise ValueError('Password must be at least 12 characters long')
        
        # Check for required character types
        has_upper = any(c.isupper() for c in v)
        has_lower = any(c.islower() for c in v)
        has_digit = any(c.isdigit() for c in v)
        has_special = any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in v)
        
        if not all([has_upper, has_lower, has_digit, has_special]):
            raise ValueError(
                'Password must contain uppercase, lowercase, digit, and special character'
            )
        
        return v
    
    @validator('confirm_password')
    def passwords_match(cls, v, values):
        """Ensure password confirmation matches."""
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('Passwords do not match')
        return v


class AuthError(BaseModel):
    """Authentication error response."""
    
    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[dict] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "INVALID_CREDENTIALS",
                "message": "Invalid username or password",
                "details": {"attempts_remaining": 2},
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class SessionInfo(BaseModel):
    """Current session information."""
    
    session_id: str
    user_id: str
    username: str
    role: str
    created_at: datetime
    last_activity: datetime
    expires_at: datetime
    ip_address: str
    user_agent: str
    risk_score: int
    is_privileged: bool
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "ses_abc123xyz789",
                "user_id": "550e8400-e29b-41d4-a716-446655440000",
                "username": "dr.smith",
                "role": "attending_physician",
                "created_at": "2024-01-15T09:00:00Z",
                "last_activity": "2024-01-15T10:30:00Z",
                "expires_at": "2024-01-15T17:00:00Z",
                "ip_address": "192.168.1.100",
                "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)...",
                "risk_score": 15,
                "is_privileged": False
            }
        }