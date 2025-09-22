"""
SQLAlchemy models for Orthopedics EMR RAG System.
HIPAA-compliant database models with security and audit features.
"""

import uuid
from datetime import datetime, date
from typing import Optional, List
from enum import Enum as PyEnum
from sqlalchemy import (
    String, Text, Boolean, Integer, DateTime, Date, Interval,
    JSON, ARRAY, Enum, ForeignKey, Index, CheckConstraint,
    func, text
)
from sqlalchemy.dialects.postgresql import ENUM
from sqlalchemy.dialects.postgresql import UUID, INET, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .connection import Base

# Enums matching PostgreSQL types
class UserStatus(PyEnum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    LOCKED = "locked"
    SUSPENDED = "suspended"

class AuditAction(PyEnum):
    LOGIN = "LOGIN"
    LOGOUT = "LOGOUT"
    QUERY = "QUERY"
    ACCESS = "ACCESS"
    MODIFY = "MODIFY"
    DELETE = "DELETE"
    EXPORT = "EXPORT"
    ADMIN = "ADMIN"

class ResourceType(PyEnum):
    PATIENT = "PATIENT"
    DOCUMENT = "DOCUMENT"
    SYSTEM = "SYSTEM"
    USER = "USER"
    REPORT = "REPORT"
    QUERY = "QUERY"

class SecuritySeverity(PyEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class SecurityStatus(PyEnum):
    OPEN = "open"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"

# ====================================================================
# CORE MODELS
# ====================================================================

class Role(Base):
    """System roles with granular permissions."""
    __tablename__ = "roles"
    
    role_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    role_name: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    permissions: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    max_session_duration: Mapped[Optional[str]] = mapped_column(Interval, default=text("'8 hours'"))
    can_access_phi: Mapped[bool] = mapped_column(Boolean, default=False)
    can_export_data: Mapped[bool] = mapped_column(Boolean, default=False)
    query_rate_limit: Mapped[int] = mapped_column(Integer, default=100)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    created_by: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Relationships
    users: Mapped[List["User"]] = relationship("User", back_populates="role")
    
    def __repr__(self) -> str:
        return f"<Role(role_name='{self.role_name}', can_access_phi={self.can_access_phi})>"

class User(Base):
    """System users with medical professional identifiers."""
    __tablename__ = "users"
    
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    salt: Mapped[str] = mapped_column(String(255), nullable=False)
    role_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("roles.role_id", ondelete="RESTRICT"))
    
    # Personal Information
    first_name: Mapped[str] = mapped_column(String(100), nullable=False)
    last_name: Mapped[str] = mapped_column(String(100), nullable=False)
    
    # Medical Professional Identifiers
    npi_number: Mapped[Optional[str]] = mapped_column(String(20), unique=True)  # National Provider Identifier
    license_number: Mapped[Optional[str]] = mapped_column(String(50))
    dea_number: Mapped[Optional[str]] = mapped_column(String(20))  # Drug Enforcement Administration
    department: Mapped[Optional[str]] = mapped_column(String(100))
    title: Mapped[Optional[str]] = mapped_column(String(100))
    
    # Account Security (temporarily using string to test)
    status: Mapped[str] = mapped_column(String(20), default='active')
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    password_changed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())
    password_expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    failed_login_attempts: Mapped[int] = mapped_column(Integer, default=0)
    locked_until: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    lockout_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # MFA Settings
    mfa_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    mfa_secret: Mapped[Optional[str]] = mapped_column(String(255))  # TOTP secret
    backup_codes: Mapped[Optional[List[str]]] = mapped_column(ARRAY(Text))  # Encrypted backup codes
    
    # Session Management
    max_concurrent_sessions: Mapped[int] = mapped_column(Integer, default=3)
    session_timeout: Mapped[Optional[str]] = mapped_column(Interval, default=text("'15 minutes'"))
    
    # Audit Fields
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    created_by: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True))
    last_modified_by: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True))
    
    # HIPAA Compliance
    terms_accepted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    privacy_policy_accepted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    last_training_date: Mapped[Optional[date]] = mapped_column(Date)
    training_expires_date: Mapped[Optional[date]] = mapped_column(Date)
    
    # Relationships
    role: Mapped["Role"] = relationship("Role", back_populates="users")
    sessions: Mapped[List["Session"]] = relationship("Session", back_populates="user")
    audit_logs: Mapped[List["AuditLog"]] = relationship("AuditLog", back_populates="user")
    security_events: Mapped[List["SecurityEvent"]] = relationship("SecurityEvent", back_populates="user", foreign_keys="SecurityEvent.user_id")
    
    # Indexes
    __table_args__ = (
        Index('idx_users_username', 'username'),
        Index('idx_users_email', 'email'),
        Index('idx_users_role', 'role_id'),
        Index('idx_users_status', 'status'),
        Index('idx_users_last_login', 'last_login'),
        Index('idx_users_npi', 'npi_number'),
    )
    
    @property
    def full_name(self) -> str:
        """Get user's full name."""
        return f"{self.first_name} {self.last_name}"
    
    @property
    def is_active(self) -> bool:
        """Check if user account is active."""
        return self.status == UserStatus.ACTIVE
    
    @property
    def is_locked(self) -> bool:
        """Check if user account is locked."""
        if self.status == UserStatus.LOCKED:
            return True
        if self.locked_until and self.locked_until > datetime.utcnow():
            return True
        return False
    
    def __repr__(self) -> str:
        return f"<User(username='{self.username}', role='{self.role.role_name if self.role else None}')>"

class Session(Base):
    """Active user sessions with security tracking."""
    __tablename__ = "sessions"
    
    session_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"))
    
    # Session Security
    token_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    refresh_token_hash: Mapped[Optional[str]] = mapped_column(String(255))
    csrf_token: Mapped[Optional[str]] = mapped_column(String(255))
    
    # Connection Information
    ip_address: Mapped[Optional[str]] = mapped_column(INET)
    user_agent: Mapped[Optional[str]] = mapped_column(Text)
    device_fingerprint: Mapped[Optional[str]] = mapped_column(String(255))
    
    # Geographic Information
    country_code: Mapped[Optional[str]] = mapped_column(String(2))
    region: Mapped[Optional[str]] = mapped_column(String(100))
    city: Mapped[Optional[str]] = mapped_column(String(100))
    
    # Session State
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_activity: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())
    
    # Activity Tracking
    page_views: Mapped[int] = mapped_column(Integer, default=0)
    queries_executed: Mapped[int] = mapped_column(Integer, default=0)
    documents_accessed: Mapped[int] = mapped_column(Integer, default=0)
    
    # Security Flags
    is_privileged: Mapped[bool] = mapped_column(Boolean, default=False)
    requires_reauth: Mapped[bool] = mapped_column(Boolean, default=False)
    risk_score: Mapped[int] = mapped_column(Integer, default=0)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="sessions")
    audit_logs: Mapped[List["AuditLog"]] = relationship("AuditLog", back_populates="session")
    
    # Table constraints
    __table_args__ = (
        CheckConstraint('risk_score >= 0 AND risk_score <= 100', name='valid_risk_score'),
        CheckConstraint('expires_at > created_at', name='valid_session_duration'),
        Index('idx_sessions_user_id', 'user_id'),
        Index('idx_sessions_active', 'is_active'),
        Index('idx_sessions_expires', 'expires_at'),
        Index('idx_sessions_ip', 'ip_address'),
        Index('idx_sessions_last_activity', 'last_activity'),
    )
    
    @property
    def is_expired(self) -> bool:
        """Check if session is expired."""
        return datetime.utcnow() > self.expires_at
    
    def __repr__(self) -> str:
        return f"<Session(session_id='{self.session_id[:8]}...', user='{self.user.username}')>"

class AuditLog(Base):
    """Comprehensive audit trail for HIPAA compliance."""
    __tablename__ = "audit_logs"
    
    log_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # User and Session Context
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="SET NULL"))
    session_id: Mapped[Optional[str]] = mapped_column(String(255), ForeignKey("sessions.session_id", ondelete="SET NULL"))
    
    # Action Information
    action_type: Mapped[AuditAction] = mapped_column(ENUM(AuditAction, name='audit_action'), nullable=False)
    action_description: Mapped[Optional[str]] = mapped_column(Text)
    
    # Resource Information
    resource_type: Mapped[Optional[ResourceType]] = mapped_column(ENUM(ResourceType, name='resource_type'))
    resource_id: Mapped[Optional[str]] = mapped_column(String(255))
    resource_description: Mapped[Optional[str]] = mapped_column(Text)
    
    # Query Details (for RAG system)
    query_text: Mapped[Optional[str]] = mapped_column(Text)
    query_results_count: Mapped[Optional[int]] = mapped_column(Integer)
    query_execution_time_ms: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Request Details
    endpoint: Mapped[Optional[str]] = mapped_column(String(255))
    http_method: Mapped[Optional[str]] = mapped_column(String(10))
    request_size_bytes: Mapped[Optional[int]] = mapped_column(Integer)
    response_size_bytes: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Network Information
    ip_address: Mapped[Optional[str]] = mapped_column(INET)
    user_agent: Mapped[Optional[str]] = mapped_column(Text)
    referer: Mapped[Optional[str]] = mapped_column(Text)
    
    # Outcome
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    error_code: Mapped[Optional[str]] = mapped_column(String(50))
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    
    # Security Assessment
    phi_accessed: Mapped[bool] = mapped_column(Boolean, default=False)
    risk_score: Mapped[Optional[int]] = mapped_column(Integer)
    requires_review: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Additional Context
    audit_metadata: Mapped[dict] = mapped_column(JSONB, default=dict)
    tags: Mapped[Optional[List[str]]] = mapped_column(ARRAY(Text))
    
    # Compliance Fields
    retention_date: Mapped[Optional[date]] = mapped_column(Date)
    is_archived: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Timestamp (immutable)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now(), nullable=False)
    
    # Relationships
    user: Mapped[Optional["User"]] = relationship("User", back_populates="audit_logs")
    session: Mapped[Optional["Session"]] = relationship("Session", back_populates="audit_logs")
    
    # Table constraints
    __table_args__ = (
        CheckConstraint('risk_score IS NULL OR (risk_score >= 0 AND risk_score <= 100)', name='valid_audit_risk_score'),
        Index('idx_audit_logs_user_id', 'user_id'),
        Index('idx_audit_logs_session_id', 'session_id'),
        Index('idx_audit_logs_created_at', 'created_at'),
        Index('idx_audit_logs_action_type', 'action_type'),
        Index('idx_audit_logs_resource_type', 'resource_type'),
        Index('idx_audit_logs_phi_accessed', 'phi_accessed'),
        Index('idx_audit_logs_success', 'success'),
        Index('idx_audit_logs_risk_score', 'risk_score'),
        Index('idx_audit_logs_requires_review', 'requires_review'),
        Index('idx_audit_logs_audit_metadata', 'audit_metadata', postgresql_using='gin'),
        Index('idx_audit_logs_tags', 'tags', postgresql_using='gin'),
    )
    
    def __repr__(self) -> str:
        return f"<AuditLog(action_type='{self.action_type}', user='{self.user.username if self.user else None}')>"

class SystemSetting(Base):
    """System configuration and security settings."""
    __tablename__ = "system_settings"
    
    setting_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    setting_key: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    setting_value: Mapped[Optional[dict]] = mapped_column(JSONB)
    description: Mapped[Optional[str]] = mapped_column(Text)
    is_sensitive: Mapped[bool] = mapped_column(Boolean, default=False)
    requires_restart: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    updated_by: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("users.user_id"))
    
    __table_args__ = (
        Index('idx_system_settings_key', 'setting_key'),
    )
    
    def __repr__(self) -> str:
        return f"<SystemSetting(key='{self.setting_key}', sensitive={self.is_sensitive})>"

class SecurityEvent(Base):
    """Security incidents and anomalies."""
    __tablename__ = "security_events"
    
    event_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Event Classification
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    severity: Mapped[SecuritySeverity] = mapped_column(ENUM(SecuritySeverity, name='security_severity'), nullable=False)
    
    # Context
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="SET NULL"))
    session_id: Mapped[Optional[str]] = mapped_column(String(255))
    ip_address: Mapped[Optional[str]] = mapped_column(INET)
    
    # Event Details
    description: Mapped[str] = mapped_column(Text, nullable=False)
    raw_data: Mapped[Optional[dict]] = mapped_column(JSONB)
    
    # Response Status
    status: Mapped[SecurityStatus] = mapped_column(ENUM(SecurityStatus, name='security_status'), default=SecurityStatus.OPEN)
    assigned_to: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("users.user_id"))
    resolution_notes: Mapped[Optional[str]] = mapped_column(Text)
    
    # Timestamps
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())
    
    # Relationships
    user: Mapped[Optional["User"]] = relationship("User", back_populates="security_events", foreign_keys=[user_id])
    assigned_user: Mapped[Optional["User"]] = relationship("User", foreign_keys=[assigned_to])
    
    __table_args__ = (
        Index('idx_security_events_type', 'event_type'),
        Index('idx_security_events_severity', 'severity'),
        Index('idx_security_events_status', 'status'),
        Index('idx_security_events_detected', 'detected_at'),
        Index('idx_security_events_user', 'user_id'),
    )
    
    def __repr__(self) -> str:
        return f"<SecurityEvent(type='{self.event_type}', severity='{self.severity}', status='{self.status}')>"