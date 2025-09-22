"""
Session management middleware for HIPAA-compliant user tracking
Handles session validation, timeout, and security monitoring
"""

import time
from datetime import datetime, timedelta, timezone
from typing import Optional
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from jose import jwt, JWTError

from core.config import settings
from database import db_manager


class SessionMiddleware(BaseHTTPMiddleware):
    """
    Middleware to handle user sessions and authentication state.
    Validates JWT tokens and manages session security.
    """
    
    # Endpoints that don't require authentication
    EXEMPT_PATHS = {
        "/", "/health", "/docs", "/redoc", "/openapi.json",
        "/api/v1/auth/login", "/api/v1/auth/register"
    }
    
    async def dispatch(self, request: Request, call_next):
        # Skip authentication for exempt paths
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)
        
        # Extract token from Authorization header
        authorization = request.headers.get("Authorization")
        token = None
        
        if authorization:
            try:
                scheme, token = authorization.split()
                if scheme.lower() != "bearer":
                    token = None
            except ValueError:
                token = None
        
        # Validate session if token provided
        user_context = None
        if token:
            user_context = await self._validate_session(request, token)
        
        # Set user context in request state
        if user_context:
            request.state.user_id = user_context["user_id"]
            request.state.session_id = user_context["session_id"]
            request.state.username = user_context["username"]
            request.state.user_role = user_context["role"]
            request.state.can_access_phi = user_context["can_access_phi"]
            request.state.is_authenticated = True
        else:
            request.state.user_id = None
            request.state.session_id = None
            request.state.username = None
            request.state.user_role = None
            request.state.can_access_phi = False
            request.state.is_authenticated = False
        
        # Process request
        response = await call_next(request)
        
        # Update session activity if user is authenticated
        if user_context:
            try:
                await self._update_session_activity(user_context["session_id"])
            except Exception as e:
                # Log but don't fail request
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Failed to update session activity: {e}")
        
        return response
    
    async def _validate_session(self, request: Request, token: str) -> Optional[dict]:
        """Validate JWT token and session."""
        try:
            # Decode JWT token
            payload = jwt.decode(
                token, 
                settings.SECRET_KEY, 
                algorithms=["HS256"]
            )
            
            session_id = payload.get("session_id")
            user_id = payload.get("user_id")
            token_type = payload.get("type")
            
            if not all([session_id, user_id, token_type]) or token_type != "access":
                return None
            
            # Validate session in database
            async with db_manager.get_session() as db_session:
                from sqlalchemy import select
                from database import Session, User, Role
                
                # Get session with user and role information
                stmt = select(Session).where(
                    Session.session_id == session_id,
                    Session.is_active == True
                ).join(User).join(Role)
                
                result = await db_session.execute(stmt)
                session_obj = result.scalar_one_or_none()
                
                if not session_obj:
                    return None
                
                # Get current UTC time with timezone info
                now_utc = datetime.now(timezone.utc)
                
                # Check session expiration
                expires_at = session_obj.expires_at
                if expires_at.tzinfo is None:
                    expires_at = expires_at.replace(tzinfo=timezone.utc)
                
                if now_utc > expires_at:
                    # Mark session as expired
                    session_obj.is_active = False
                    session_obj.ended_at = now_utc.replace(tzinfo=None)
                    await db_session.commit()
                    return None
                
                # Check for session timeout (inactivity)
                timeout_minutes = settings.SESSION_TIMEOUT_MINUTES
                if session_obj.last_activity:
                    last_activity = session_obj.last_activity
                    if last_activity.tzinfo is None:
                        last_activity = last_activity.replace(tzinfo=timezone.utc)
                    
                    timeout_threshold = last_activity + timedelta(minutes=timeout_minutes)
                    if now_utc > timeout_threshold:
                        # Mark session as expired due to inactivity
                        session_obj.is_active = False
                        session_obj.ended_at = now_utc.replace(tzinfo=None)
                        await db_session.commit()
                        return None
                
                # Get user information
                user_stmt = select(User, Role).join(Role).where(User.user_id == user_id)
                user_result = await db_session.execute(user_stmt)
                user_data = user_result.first()
                
                if not user_data:
                    return None
                
                user, role = user_data
                
                # Check if user account is active
                if user.status != 'active':
                    return None
                
                # Check account lockout
                if user.locked_until:
                    locked_until = user.locked_until
                    if locked_until.tzinfo is None:
                        locked_until = locked_until.replace(tzinfo=timezone.utc)
                    if now_utc < locked_until:
                        return None
                
                # Validate client IP and user agent for security
                client_ip = self._get_client_ip(request)
                user_agent = request.headers.get("user-agent", "")
                
                # Check for suspicious activity (IP or user agent change)
                if session_obj.ip_address and str(session_obj.ip_address) != client_ip:
                    # Log security event for IP change
                    await self._log_security_event(
                        user_id, session_id, "IP_ADDRESS_CHANGE",
                        f"IP changed from {session_obj.ip_address} to {client_ip}"
                    )
                    
                    # Increase risk score
                    session_obj.risk_score = min(session_obj.risk_score + 25, 100)
                
                if session_obj.user_agent and session_obj.user_agent != user_agent:
                    # Log security event for user agent change
                    await self._log_security_event(
                        user_id, session_id, "USER_AGENT_CHANGE",
                        "User agent string changed during session"
                    )
                    
                    # Increase risk score
                    session_obj.risk_score = min(session_obj.risk_score + 15, 100)
                
                await db_session.commit()
                
                return {
                    "user_id": user.user_id,
                    "session_id": session_obj.session_id,
                    "username": user.username,
                    "role": role.role_name,
                    "can_access_phi": role.can_access_phi,
                    "risk_score": session_obj.risk_score
                }
                
        except JWTError:
            return None
        except Exception as e:
            # Log error but don't expose details
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Session validation error: {e}")
            return None
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        if request.client:
            return request.client.host
        
        return "unknown"
    
    async def _update_session_activity(self, session_id: str):
        """Update session last activity timestamp."""
        async with db_manager.get_session() as db_session:
            from sqlalchemy import select, update
            from database import Session
            
            # Update last activity
            stmt = update(Session).where(
                Session.session_id == session_id
            ).values(
                last_activity=datetime.now(timezone.utc).replace(tzinfo=None),
                page_views=Session.page_views + 1
            )
            
            await db_session.execute(stmt)
            await db_session.commit()
    
    async def _log_security_event(self, user_id: str, session_id: str, event_type: str, description: str):
        """Log security event for monitoring."""
        try:
            async with db_manager.get_session() as db_session:
                from database import SecurityEvent, SecuritySeverity, SecurityStatus
                
                security_event = SecurityEvent(
                    event_type=event_type,
                    severity=SecuritySeverity.MEDIUM,
                    user_id=user_id,
                    session_id=session_id,
                    description=description,
                    status=SecurityStatus.OPEN
                )
                
                db_session.add(security_event)
                await db_session.commit()
                
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to log security event: {e}")