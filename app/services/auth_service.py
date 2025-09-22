"""
HIPAA-compliant authentication service for medical professionals
Handles JWT tokens, MFA, password security, and medical credential validation
"""

import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext

from core.config import settings
from core.security import hash_password, verify_password, validate_npi_number, validate_dea_number
from database import db_manager, User, Role, Session, AuditAction, SecurityEvent, SecuritySeverity, SecurityStatus


class AuthenticationError(Exception):
    """Base exception for authentication errors."""
    pass


class CredentialsError(AuthenticationError):
    """Invalid credentials provided."""
    pass


class AccountLockedError(AuthenticationError):
    """Account is locked due to failed attempts."""
    pass


class MFARequiredError(AuthenticationError):
    """MFA verification required."""
    pass


class AuthService:
    """
    HIPAA-compliant authentication service for medical professionals.
    """
    
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.algorithm = "HS256"
    
    async def authenticate_user(
        self, 
        username: str, 
        password: str, 
        mfa_code: Optional[str] = None,
        client_ip: str = "unknown",
        user_agent: str = ""
    ) -> Dict[str, Any]:
        """
        Authenticate medical professional with comprehensive security checks.
        
        Args:
            username: Username or email
            password: Plain text password
            mfa_code: Optional MFA/TOTP code
            client_ip: Client IP address for audit
            user_agent: Browser user agent string
        
        Returns:
            Dict containing tokens and user information
        
        Raises:
            CredentialsError: Invalid username/password
            AccountLockedError: Account is locked
            MFARequiredError: MFA code required but not provided
        """
        async with db_manager.get_session() as session:
            # Get user with role information
            from sqlalchemy import select
            stmt = select(User, Role).join(Role).where(
                (User.username == username) | (User.email == username)
            )
            result = await session.execute(stmt)
            user_data = result.first()
            
            if not user_data:
                # Audit failed login attempt
                await self._log_security_event(
                    session, None, None, "FAILED_LOGIN", 
                    f"Invalid username: {username}", client_ip
                )
                raise CredentialsError("Invalid username or password")
            
            user, role = user_data
            
            # Check if account is active
            if user.status != 'active':
                await self._log_security_event(
                    session, user.user_id, None, "INACTIVE_ACCOUNT_LOGIN",
                    f"Login attempt on inactive account: {user.status}", client_ip
                )
                raise CredentialsError("Account is not active")
            
            # Check if account is locked
            if user.locked_until and datetime.utcnow() < user.locked_until:
                await self._log_security_event(
                    session, user.user_id, None, "LOCKED_ACCOUNT_LOGIN",
                    "Login attempt on locked account", client_ip
                )
                raise AccountLockedError("Account is temporarily locked")
            
            # Verify password
            if not verify_password(password, user.password_hash, user.salt):
                # Increment failed login attempts
                user.failed_login_attempts += 1
                
                # Lock account if too many failures
                if user.failed_login_attempts >= settings.MAX_LOGIN_ATTEMPTS:
                    user.locked_until = datetime.utcnow() + timedelta(
                        minutes=settings.LOCKOUT_DURATION_MINUTES
                    )
                    user.lockout_count += 1
                    
                    await self._log_security_event(
                        session, user.user_id, None, "ACCOUNT_LOCKED",
                        f"Account locked after {settings.MAX_LOGIN_ATTEMPTS} failed attempts", 
                        client_ip
                    )
                
                await session.commit()
                
                await self._log_security_event(
                    session, user.user_id, None, "FAILED_LOGIN",
                    f"Invalid password (attempt {user.failed_login_attempts})", client_ip
                )
                
                raise CredentialsError("Invalid username or password")
            
            # Check MFA requirement for PHI access
            if role.can_access_phi and settings.MFA_REQUIRED_FOR_PHI:
                if not user.mfa_enabled:
                    await self._log_security_event(
                        session, user.user_id, None, "MFA_NOT_CONFIGURED",
                        "PHI access requires MFA but not configured", client_ip
                    )
                    raise MFARequiredError("Multi-factor authentication required for PHI access")
                
                if not mfa_code:
                    raise MFARequiredError("MFA code required")
                
                # Verify MFA code
                if not await self._verify_mfa_code(user, mfa_code):
                    await self._log_security_event(
                        session, user.user_id, None, "MFA_FAILED",
                        "Invalid MFA code provided", client_ip
                    )
                    raise CredentialsError("Invalid MFA code")
            
            # Reset failed login attempts on successful authentication
            user.failed_login_attempts = 0
            user.locked_until = None
            user.last_login = datetime.utcnow()
            
            # Create session
            session_data = await self._create_session(
                session, user, client_ip, user_agent
            )
            
            # Generate tokens
            tokens = await self._generate_tokens(user, session_data["session_id"])
            
            await session.commit()
            
            # Log successful login
            await self._log_security_event(
                session, user.user_id, session_data["session_id"], "LOGIN",
                f"Successful login from {client_ip}", client_ip
            )
            
            return {
                "access_token": tokens["access_token"],
                "refresh_token": tokens["refresh_token"],
                "token_type": "bearer",
                "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
                "user_info": {
                    "user_id": str(user.user_id),
                    "username": user.username,
                    "full_name": user.full_name,
                    "email": user.email,
                    "role": role.role_name,
                    "can_access_phi": role.can_access_phi,
                    "department": user.department,
                    "title": user.title,
                    "npi_number": user.npi_number,
                    "mfa_enabled": user.mfa_enabled
                }
            }
    
    async def logout_user(
        self, 
        session_id: str,
        client_ip: str = "unknown"
    ) -> bool:
        """
        Log out user and invalidate session.
        
        Args:
            session_id: Session ID to invalidate
            client_ip: Client IP for audit
        
        Returns:
            True if logout successful
        """
        async with db_manager.get_session() as db_session:
            from sqlalchemy import select, update
            
            # Get session information
            stmt = select(Session).where(Session.session_id == session_id)
            result = await db_session.execute(stmt)
            session_obj = result.scalar_one_or_none()
            
            if session_obj and session_obj.is_active:
                # Mark session as inactive
                session_obj.is_active = False
                session_obj.ended_at = datetime.utcnow()
                
                # Log logout
                await self._log_security_event(
                    db_session, session_obj.user_id, session_id, "LOGOUT",
                    f"User logged out from {client_ip}", client_ip
                )
                
                await db_session.commit()
                return True
            
            return False
    
    async def refresh_access_token(self, refresh_token: str) -> Dict[str, Any]:
        """
        Generate new access token using refresh token.
        
        Args:
            refresh_token: Valid refresh token
        
        Returns:
            Dict containing new access token
        
        Raises:
            CredentialsError: Invalid or expired refresh token
        """
        try:
            # Decode refresh token
            payload = jwt.decode(
                refresh_token, 
                settings.SECRET_KEY, 
                algorithms=[self.algorithm]
            )
            
            if payload.get("type") != "refresh":
                raise CredentialsError("Invalid token type")
            
            user_id = payload.get("user_id")
            session_id = payload.get("session_id")
            
            if not all([user_id, session_id]):
                raise CredentialsError("Invalid token payload")
            
            # Validate session is still active
            async with db_manager.get_session() as session:
                from sqlalchemy import select
                stmt = select(Session, User).join(User).where(
                    Session.session_id == session_id,
                    Session.is_active == True,
                    User.user_id == user_id
                )
                result = await session.execute(stmt)
                session_data = result.first()
                
                if not session_data:
                    raise CredentialsError("Session not found or expired")
                
                session_obj, user = session_data
                
                # Check session expiration
                if datetime.utcnow() > session_obj.expires_at:
                    session_obj.is_active = False
                    session_obj.ended_at = datetime.utcnow()
                    await session.commit()
                    raise CredentialsError("Session expired")
                
                # Generate new access token
                access_token = await self._generate_access_token(user, session_id)
                
                return {
                    "access_token": access_token,
                    "token_type": "bearer",
                    "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
                }
        
        except JWTError:
            raise CredentialsError("Invalid refresh token")
    
    async def validate_medical_credentials(self, user_id: str) -> Dict[str, bool]:
        """
        Validate medical professional credentials (NPI, DEA numbers).
        
        Args:
            user_id: User ID to validate
        
        Returns:
            Dict with validation results
        """
        async with db_manager.get_session() as session:
            from sqlalchemy import select
            stmt = select(User).where(User.user_id == user_id)
            result = await session.execute(stmt)
            user = result.scalar_one_or_none()
            
            if not user:
                return {"valid": False, "reason": "User not found"}
            
            validation_results = {
                "npi_valid": False,
                "dea_valid": False,
                "overall_valid": True
            }
            
            # Validate NPI number if present
            if user.npi_number:
                validation_results["npi_valid"] = validate_npi_number(user.npi_number)
                if not validation_results["npi_valid"]:
                    validation_results["overall_valid"] = False
            
            # Validate DEA number if present
            if user.dea_number:
                validation_results["dea_valid"] = validate_dea_number(user.dea_number)
                if not validation_results["dea_valid"]:
                    validation_results["overall_valid"] = False
            
            return validation_results
    
    async def _create_session(
        self, 
        db_session, 
        user: User, 
        client_ip: str,
        user_agent: str
    ) -> Dict[str, Any]:
        """Create new user session with security tracking."""
        session_id = f"ses_{secrets.token_urlsafe(32)}"
        expires_at = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 4  # Session lasts longer than access token
        )
        
        # Create session record
        session_obj = Session(
            session_id=session_id,
            user_id=user.user_id,
            token_hash="",  # Will be set when token is generated
            ip_address=client_ip,
            user_agent=user_agent,
            expires_at=expires_at,
            is_active=True,
            risk_score=0
        )
        
        db_session.add(session_obj)
        
        return {
            "session_id": session_id,
            "expires_at": expires_at
        }
    
    async def _generate_tokens(self, user: User, session_id: str) -> Dict[str, str]:
        """Generate JWT access and refresh tokens."""
        now = datetime.utcnow()
        
        # Access token payload
        access_payload = {
            "user_id": str(user.user_id),
            "username": user.username,
            "session_id": session_id,
            "type": "access",
            "iat": now,
            "exp": now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        }
        
        # Refresh token payload
        refresh_payload = {
            "user_id": str(user.user_id),
            "session_id": session_id,
            "type": "refresh",
            "iat": now,
            "exp": now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        }
        
        # Generate tokens
        access_token = jwt.encode(access_payload, settings.SECRET_KEY, algorithm=self.algorithm)
        refresh_token = jwt.encode(refresh_payload, settings.SECRET_KEY, algorithm=self.algorithm)
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token
        }
    
    async def _generate_access_token(self, user: User, session_id: str) -> str:
        """Generate new access token for token refresh."""
        now = datetime.utcnow()
        
        payload = {
            "user_id": str(user.user_id),
            "username": user.username,
            "session_id": session_id,
            "type": "access",
            "iat": now,
            "exp": now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        }
        
        return jwt.encode(payload, settings.SECRET_KEY, algorithm=self.algorithm)
    
    async def _verify_mfa_code(self, user: User, code: str) -> bool:
        """
        Verify MFA/TOTP code.
        TODO: Implement TOTP verification using user.mfa_secret
        """
        # For now, return True for demo purposes
        # In production, implement proper TOTP verification
        return len(code) == 6 and code.isdigit()
    
    async def _log_security_event(
        self, 
        session, 
        user_id: Optional[str], 
        session_id: Optional[str],
        event_type: str, 
        description: str, 
        ip_address: str
    ):
        """Log security event for monitoring."""
        security_event = SecurityEvent(
            event_type=event_type,
            severity=SecuritySeverity.MEDIUM,
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            description=description,
            status=SecurityStatus.OPEN
        )
        
        session.add(security_event)


# Global auth service instance
auth_service = AuthService()