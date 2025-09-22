"""
Configuration management for Orthopedics EMR RAG System
HIPAA-compliant settings with security-first defaults
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with security-focused defaults."""
    
    # Application
    APP_NAME: str = "Orthopedics EMR RAG System"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default="production", env="ENVIRONMENT")
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # Server
    HOST: str = Field(default="127.0.0.1", env="HOST")  # Localhost only for airgapped
    PORT: int = Field(default=8000, env="PORT")
    
    # Security
    SECRET_KEY: str = Field(default="dev-secret-key-change-in-production", env="SECRET_KEY")
    ALLOWED_HOSTS: List[str] = Field(default=["localhost", "127.0.0.1", "*.local"], env="ALLOWED_HOSTS")
    
    # CORS settings for JavaScript frontend
    CORS_ORIGINS: List[str] = Field(
        default=[
            "http://localhost:3000",  # React dev server
            "http://127.0.0.1:3000",
            "http://localhost:5173",  # Vite dev server  
            "http://127.0.0.1:5173",
            "https://localhost",      # Production HTTPS
        ],
        env="CORS_ORIGINS"
    )
    
    # Database (inherited from database module)
    DATABASE_URL: Optional[str] = Field(default=None, env="DATABASE_URL")
    DB_ECHO: bool = Field(default=False, env="DB_ECHO")
    
    # Authentication & Sessions
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=15, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, env="REFRESH_TOKEN_EXPIRE_DAYS") 
    SESSION_TIMEOUT_MINUTES: int = Field(default=15, env="SESSION_TIMEOUT_MINUTES")
    PASSWORD_MIN_LENGTH: int = Field(default=12, env="PASSWORD_MIN_LENGTH")
    MAX_LOGIN_ATTEMPTS: int = Field(default=5, env="MAX_LOGIN_ATTEMPTS")
    LOCKOUT_DURATION_MINUTES: int = Field(default=30, env="LOCKOUT_DURATION_MINUTES")
    
    # MFA Settings
    MFA_ISSUER_NAME: str = Field(default="Orthopedics EMR", env="MFA_ISSUER_NAME")
    MFA_REQUIRED_FOR_PHI: bool = Field(default=True, env="MFA_REQUIRED_FOR_PHI")
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = Field(default=100, env="RATE_LIMIT_REQUESTS")  # per hour
    RATE_LIMIT_WINDOW: int = Field(default=3600, env="RATE_LIMIT_WINDOW")    # seconds
    
    # File Storage (for future document uploads)
    MAX_UPLOAD_SIZE_MB: int = Field(default=50, env="MAX_UPLOAD_SIZE_MB")
    ALLOWED_FILE_TYPES: List[str] = Field(
        default=[".pdf", ".doc", ".docx", ".txt", ".png", ".jpg", ".jpeg", ".dcm"],
        env="ALLOWED_FILE_TYPES"
    )
    
    # Logging & Audit
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    AUDIT_LOG_RETENTION_DAYS: int = Field(default=2555, env="AUDIT_LOG_RETENTION_DAYS")  # 7 years HIPAA
    
    # AI/ML Settings (for future integration)
    MODEL_NAME: str = Field(default="gemma-2-9b", env="MODEL_NAME")
    MAX_QUERY_LENGTH: int = Field(default=1000, env="MAX_QUERY_LENGTH")
    MAX_RESPONSE_LENGTH: int = Field(default=4000, env="MAX_RESPONSE_LENGTH")
    
    # ChromaDB Settings
    CHROMA_HOST: str = Field(default="localhost", env="CHROMA_HOST")
    CHROMA_PORT: int = Field(default=8001, env="CHROMA_PORT")
    
    # HIPAA Compliance
    REQUIRE_REASON_FOR_ACCESS: bool = Field(default=True, env="REQUIRE_REASON_FOR_ACCESS")
    AUTO_LOGOUT_INACTIVE_MINUTES: int = Field(default=15, env="AUTO_LOGOUT_INACTIVE_MINUTES")
    FAILED_LOGIN_ALERT_THRESHOLD: int = Field(default=3, env="FAILED_LOGIN_ALERT_THRESHOLD")
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Generate secret key warning if not set
        if not self.SECRET_KEY:
            if self.ENVIRONMENT == "development":
                self.SECRET_KEY = "dev-secret-key-change-in-production"
                print("⚠️  WARNING: Using development secret key")
            else:
                raise ValueError("SECRET_KEY environment variable must be set in production")


# Global settings instance
settings = Settings()

# Environment-specific validation (moved to after settings instantiation)
def validate_production_settings():
    """Validate settings for production environment."""
    if settings.ENVIRONMENT == "production":
        assert settings.SECRET_KEY != "dev-secret-key-change-in-production", "Production SECRET_KEY required"
        assert settings.DEBUG is False, "DEBUG must be False in production"
        assert settings.HOST in ["127.0.0.1", "localhost"], "Production should only bind to localhost for airgapped deployment"

# Run validation after settings are loaded
validate_production_settings()