"""
Authentication endpoints for HIPAA-compliant medical system
Handles login, logout, token refresh, and MFA
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, status, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from schemas.auth import (
    LoginRequest, LoginResponse, LogoutRequest, 
    RefreshTokenRequest, RefreshTokenResponse,
    MFASetupRequest, MFASetupResponse, MFAVerifyRequest,
    PasswordChangeRequest, AuthError
)
from services.auth_service import (
    auth_service, AuthenticationError, CredentialsError,
    AccountLockedError, MFARequiredError
)

logger = logging.getLogger(__name__)
router = APIRouter()
security = HTTPBearer()

def get_client_ip(request: Request) -> str:
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

@router.post("/login", 
    response_model=LoginResponse,
    responses={
        400: {"model": AuthError, "description": "Invalid credentials"},
        423: {"model": AuthError, "description": "Account locked"},
        428: {"model": AuthError, "description": "MFA required"}
    }
)
async def login(request: LoginRequest, http_request: Request):
    """
    Authenticate medical professional with comprehensive security checks.
    
    - Validates username/password credentials
    - Enforces MFA for PHI access roles
    - Tracks failed login attempts
    - Implements account lockout protection
    - Creates secure session with JWT tokens
    """
    client_ip = get_client_ip(http_request)
    user_agent = http_request.headers.get("user-agent", "")
    
    logger.info(f"Login attempt for user: {request.username} from IP: {client_ip}")
    
    try:
        result = await auth_service.authenticate_user(
            username=request.username,
            password=request.password,
            mfa_code=request.mfa_code,
            client_ip=client_ip,
            user_agent=user_agent
        )
        
        logger.info(f"Successful login for user: {request.username}")
        return LoginResponse(**result)
        
    except MFARequiredError as e:
        logger.warning(f"MFA required for user: {request.username}")
        raise HTTPException(
            status_code=status.HTTP_428_PRECONDITION_REQUIRED,
            detail={
                "error": "MFA_REQUIRED",
                "message": str(e),
                "mfa_required": True
            }
        )
    
    except AccountLockedError as e:
        logger.warning(f"Account locked for user: {request.username}")
        raise HTTPException(
            status_code=status.HTTP_423_LOCKED,
            detail={
                "error": "ACCOUNT_LOCKED",
                "message": str(e),
                "locked": True
            }
        )
    
    except CredentialsError as e:
        logger.warning(f"Invalid credentials for user: {request.username}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "INVALID_CREDENTIALS",
                "message": str(e)
            }
        )
    
    except Exception as e:
        logger.error(f"Authentication error for user {request.username}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "AUTHENTICATION_ERROR", 
                "message": "Authentication service temporarily unavailable"
            }
        )

@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    request: LogoutRequest,
    http_request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Log out user and invalidate session.
    
    - Invalidates current session
    - Logs logout event for audit
    - Clears authentication tokens
    """
    client_ip = get_client_ip(http_request)
    
    # Get session ID from token or request
    session_id = request.session_id
    if not session_id:
        # Extract from current session if not provided
        session_id = getattr(http_request.state, 'session_id', None)
    
    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "NO_SESSION", "message": "No active session to logout"}
        )
    
    try:
        success = await auth_service.logout_user(session_id, client_ip)
        
        if success:
            logger.info(f"User logged out successfully, session: {session_id}")
            return None  # 204 No Content
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": "SESSION_NOT_FOUND", "message": "Session not found"}
            )
    
    except Exception as e:
        logger.error(f"Logout error for session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "LOGOUT_ERROR", "message": "Logout failed"}
        )

@router.post("/refresh", response_model=RefreshTokenResponse)
async def refresh_token(request: RefreshTokenRequest):
    """
    Generate new access token using refresh token.
    
    - Validates refresh token
    - Checks session status
    - Issues new access token
    """
    try:
        result = await auth_service.refresh_access_token(request.refresh_token)
        logger.info("Access token refreshed successfully")
        return RefreshTokenResponse(**result)
        
    except CredentialsError as e:
        logger.warning(f"Invalid refresh token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "INVALID_REFRESH_TOKEN",
                "message": str(e)
            }
        )
    
    except Exception as e:
        logger.error(f"Token refresh error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "REFRESH_ERROR", "message": "Token refresh failed"}
        )

@router.post("/mfa/setup", response_model=MFASetupResponse)
async def setup_mfa(
    request: MFASetupRequest,
    http_request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Set up Multi-Factor Authentication for enhanced security.
    
    - Generates TOTP secret
    - Creates QR code for authenticator app
    - Provides backup codes
    """
    # TODO: Implement MFA setup
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="MFA setup not yet implemented"
    )

@router.post("/mfa/verify")
async def verify_mfa(
    request: MFAVerifyRequest,
    http_request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Verify MFA code and enable MFA for account.
    
    - Validates TOTP code
    - Enables MFA for user account
    - Logs security event
    """
    # TODO: Implement MFA verification
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="MFA verification not yet implemented"
    )

@router.post("/change-password")
async def change_password(
    request: PasswordChangeRequest,
    http_request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Change user password with medical security compliance.
    
    - Validates current password
    - Enforces strong password policy
    - Updates password securely
    - Invalidates other sessions
    """
    # TODO: Implement password change
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Password change not yet implemented"
    )

@router.get("/me/session")
async def get_current_session(
    http_request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Get current session information for monitoring.
    
    - Returns session details
    - Shows security metrics
    - Displays last activity
    """
    # TODO: Implement session info
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Session info not yet implemented"
    )