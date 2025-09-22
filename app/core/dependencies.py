"""
FastAPI dependencies for authentication and authorization
Provides injectable user context and permission checking
"""

from typing import Optional
from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from middleware.rbac import Permission, MedicalRole

security = HTTPBearer()


class CurrentUser:
    """Current user context from authentication middleware."""
    
    def __init__(
        self,
        user_id: str,
        username: str,
        role: str,
        can_access_phi: bool,
        session_id: str,
        permissions: set
    ):
        self.user_id = user_id
        self.username = username
        self.role = role
        self.can_access_phi = can_access_phi
        self.session_id = session_id
        self.permissions = permissions
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has a specific permission."""
        return permission in self.permissions
    
    def has_any_permission(self, *permissions: str) -> bool:
        """Check if user has any of the specified permissions."""
        return any(perm in self.permissions for perm in permissions)
    
    def has_all_permissions(self, *permissions: str) -> bool:
        """Check if user has all of the specified permissions."""
        return all(perm in self.permissions for perm in permissions)
    
    def is_admin(self) -> bool:
        """Check if user is an administrator."""
        return self.role == "admin"
    
    def is_physician(self) -> bool:
        """Check if user is a physician (attending or resident)."""
        return self.role in ["attending_physician", "resident"]
    
    def can_access_patient_data(self) -> bool:
        """Check if user can access patient PHI data."""
        return self.can_access_phi and self.has_permission(Permission.PHI_READ)


def get_current_user(request: Request) -> CurrentUser:
    """
    Get current authenticated user context.
    Raises 401 if user is not authenticated.
    """
    if not getattr(request.state, 'is_authenticated', False):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "NOT_AUTHENTICATED",
                "message": "Authentication required"
            },
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    user_id = getattr(request.state, 'user_id', None)
    username = getattr(request.state, 'username', None) 
    role = getattr(request.state, 'user_role', None)
    can_access_phi = getattr(request.state, 'can_access_phi', False)
    session_id = getattr(request.state, 'session_id', None)
    
    if not all([user_id, role, session_id]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "INVALID_USER_CONTEXT",
                "message": "Invalid user authentication context"
            }
        )
    
    # Get permissions based on role
    from middleware.rbac import RBACMiddleware
    permissions = RBACMiddleware.ROLE_PERMISSIONS.get(role, set())
    
    return CurrentUser(
        user_id=user_id,
        username=username or "unknown",
        role=role,
        can_access_phi=can_access_phi,
        session_id=session_id,
        permissions=permissions
    )


def require_permission(permission: str):
    """
    Dependency to require a specific permission.
    Usage: current_user = Depends(require_permission(Permission.PHI_READ))
    """
    def check_permission(current_user: CurrentUser = Depends(get_current_user)):
        if not current_user.has_permission(permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "INSUFFICIENT_PERMISSIONS",
                    "message": f"Permission required: {permission}",
                    "required_permission": permission,
                    "user_role": current_user.role
                }
            )
        return current_user
    
    return check_permission


def require_phi_access():
    """
    Dependency to require PHI access capability.
    Usage: current_user = Depends(require_phi_access())
    """
    def check_phi_access(current_user: CurrentUser = Depends(get_current_user)):
        if not current_user.can_access_patient_data():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "PHI_ACCESS_DENIED", 
                    "message": "Access to Protected Health Information denied",
                    "phi_access_required": True,
                    "user_role": current_user.role
                }
            )
        return current_user
    
    return check_phi_access


def require_role(*allowed_roles: str):
    """
    Dependency to require specific roles.
    Usage: current_user = Depends(require_role("admin", "attending_physician"))
    """
    def check_role(current_user: CurrentUser = Depends(get_current_user)):
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "INSUFFICIENT_ROLE",
                    "message": f"Role required: {' or '.join(allowed_roles)}",
                    "required_roles": list(allowed_roles),
                    "user_role": current_user.role
                }
            )
        return current_user
    
    return check_role


def require_admin():
    """
    Dependency to require admin role.
    Usage: current_user = Depends(require_admin())
    """
    return require_role("admin")


def require_physician():
    """
    Dependency to require physician role (attending or resident).
    Usage: current_user = Depends(require_physician())
    """
    return require_role("attending_physician", "resident")


def get_optional_user(request: Request) -> Optional[CurrentUser]:
    """
    Get current user context if authenticated, None otherwise.
    Useful for endpoints that work for both authenticated and unauthenticated users.
    """
    try:
        return get_current_user(request)
    except HTTPException:
        return None