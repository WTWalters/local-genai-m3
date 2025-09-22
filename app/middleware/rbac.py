"""
Role-Based Access Control (RBAC) middleware for HIPAA-compliant medical system
Enforces medical staff permissions and PHI access controls
"""

import logging
from typing import List, Optional, Set
from fastapi import HTTPException, status, Request
from starlette.middleware.base import BaseHTTPMiddleware

from database import db_manager, AuditAction

logger = logging.getLogger(__name__)


class Permission:
    """Medical system permissions for RBAC."""
    
    # Patient Health Information (PHI) Access
    PHI_READ = "phi:read"
    PHI_WRITE = "phi:write"
    PHI_EXPORT = "phi:export"
    
    # User Management
    USER_READ = "user:read"
    USER_WRITE = "user:write"
    USER_DELETE = "user:delete"
    
    # System Administration
    ADMIN_READ = "admin:read"
    ADMIN_WRITE = "admin:write"
    ADMIN_SYSTEM = "admin:system"
    
    # Query and Search
    QUERY_BASIC = "query:basic"
    QUERY_ADVANCED = "query:advanced"
    QUERY_ANALYTICS = "query:analytics"
    
    # Document Management
    DOC_READ = "doc:read"
    DOC_WRITE = "doc:write"
    DOC_DELETE = "doc:delete"
    
    # Audit and Compliance
    AUDIT_READ = "audit:read"
    AUDIT_EXPORT = "audit:export"


class MedicalRole:
    """Predefined medical roles with appropriate permissions."""
    
    ADMIN = {
        Permission.PHI_READ,
        Permission.PHI_WRITE,
        Permission.PHI_EXPORT,
        Permission.USER_READ,
        Permission.USER_WRITE,
        Permission.USER_DELETE,
        Permission.ADMIN_READ,
        Permission.ADMIN_WRITE,
        Permission.ADMIN_SYSTEM,
        Permission.QUERY_BASIC,
        Permission.QUERY_ADVANCED,
        Permission.QUERY_ANALYTICS,
        Permission.DOC_READ,
        Permission.DOC_WRITE,
        Permission.DOC_DELETE,
        Permission.AUDIT_READ,
        Permission.AUDIT_EXPORT,
    }
    
    ATTENDING_PHYSICIAN = {
        Permission.PHI_READ,
        Permission.PHI_WRITE,
        Permission.PHI_EXPORT,
        Permission.USER_READ,  # Can view staff under supervision
        Permission.QUERY_BASIC,
        Permission.QUERY_ADVANCED,
        Permission.QUERY_ANALYTICS,
        Permission.DOC_READ,
        Permission.DOC_WRITE,
        Permission.DOC_DELETE,
        Permission.AUDIT_READ,  # Can review own audit trail
    }
    
    RESIDENT = {
        Permission.PHI_READ,
        Permission.PHI_WRITE,  # Under supervision
        Permission.QUERY_BASIC,
        Permission.QUERY_ADVANCED,
        Permission.DOC_READ,
        Permission.DOC_WRITE,
    }
    
    NURSE = {
        Permission.PHI_READ,
        Permission.PHI_WRITE,  # Limited scope
        Permission.QUERY_BASIC,
        Permission.DOC_READ,
        Permission.DOC_WRITE,
    }
    
    READ_ONLY = {
        Permission.PHI_READ,  # Limited PHI read for QA/audit
        Permission.QUERY_BASIC,
        Permission.DOC_READ,
        Permission.AUDIT_READ,
    }


class RBACMiddleware(BaseHTTPMiddleware):
    """
    Role-Based Access Control middleware for medical API endpoints.
    Enforces permissions based on user roles and endpoint requirements.
    """
    
    # Endpoint permission requirements
    ENDPOINT_PERMISSIONS = {
        # Authentication endpoints (no permissions required)
        "/api/v1/auth/login": set(),
        "/api/v1/auth/logout": set(),
        "/api/v1/auth/refresh": set(),
        
        # User management endpoints
        "/api/v1/users/me": {Permission.USER_READ},
        "/api/v1/users": {Permission.USER_READ},
        "/api/v1/users/{user_id}": {Permission.USER_READ},
        "/api/v1/users/{user_id}/update": {Permission.USER_WRITE},
        "/api/v1/users/{user_id}/delete": {Permission.USER_DELETE},
        
        # System administration
        "/api/v1/system/status": {Permission.ADMIN_READ},
        "/api/v1/system/metrics": {Permission.ADMIN_READ},
        "/api/v1/system/audit": {Permission.AUDIT_READ},
        "/api/v1/system/settings": {Permission.ADMIN_WRITE},
        
        # Future medical endpoints (examples)
        "/api/v1/patients": {Permission.PHI_READ},
        "/api/v1/patients/{patient_id}": {Permission.PHI_READ},
        "/api/v1/patients/{patient_id}/update": {Permission.PHI_WRITE},
        "/api/v1/documents": {Permission.DOC_READ},
        "/api/v1/documents/{doc_id}": {Permission.DOC_READ},
        "/api/v1/documents/{doc_id}/update": {Permission.DOC_WRITE},
        "/api/v1/query/basic": {Permission.QUERY_BASIC},
        "/api/v1/query/advanced": {Permission.QUERY_ADVANCED},
        "/api/v1/query/analytics": {Permission.QUERY_ANALYTICS},
    }
    
    # Role to permissions mapping
    ROLE_PERMISSIONS = {
        "admin": MedicalRole.ADMIN,
        "attending_physician": MedicalRole.ATTENDING_PHYSICIAN,
        "resident": MedicalRole.RESIDENT,
        "nurse": MedicalRole.NURSE,
        "read_only": MedicalRole.READ_ONLY,
    }
    
    # Endpoints that don't require authentication
    PUBLIC_ENDPOINTS = {
        "/", "/health", "/docs", "/redoc", "/openapi.json"
    }
    
    async def dispatch(self, request: Request, call_next):
        # Skip RBAC for public endpoints
        if request.url.path in self.PUBLIC_ENDPOINTS:
            return await call_next(request)
        
        # Skip RBAC for preflight OPTIONS requests
        if request.method == "OPTIONS":
            return await call_next(request)
        
        # Check if user is authenticated
        is_authenticated = getattr(request.state, 'is_authenticated', False)
        if not is_authenticated:
            # Let the endpoint handle authentication error
            return await call_next(request)
        
        # Get user context from middleware
        user_role = getattr(request.state, 'user_role', None)
        can_access_phi = getattr(request.state, 'can_access_phi', False)
        user_id = getattr(request.state, 'user_id', None)
        
        if not user_role:
            logger.error(f"User {user_id} missing role information")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "MISSING_ROLE",
                    "message": "User role information not available"
                }
            )
        
        # Get required permissions for this endpoint
        required_permissions = self._get_required_permissions(
            request.url.path, 
            request.method
        )
        
        # Skip permission check if no permissions required
        if not required_permissions:
            return await call_next(request)
        
        # Get user permissions based on role
        user_permissions = self.ROLE_PERMISSIONS.get(user_role, set())
        
        # Check PHI access permissions
        phi_permissions = {Permission.PHI_READ, Permission.PHI_WRITE, Permission.PHI_EXPORT}
        if required_permissions & phi_permissions and not can_access_phi:
            await self._log_access_denied(request, "PHI_ACCESS_DENIED", "User lacks PHI access")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "PHI_ACCESS_DENIED",
                    "message": "Access to Protected Health Information denied",
                    "required_permissions": list(required_permissions),
                    "phi_access_required": True
                }
            )
        
        # Check if user has required permissions
        missing_permissions = required_permissions - user_permissions
        if missing_permissions:
            await self._log_access_denied(
                request, 
                "INSUFFICIENT_PERMISSIONS", 
                f"Missing permissions: {missing_permissions}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "INSUFFICIENT_PERMISSIONS",
                    "message": "Insufficient permissions for this operation",
                    "required_permissions": list(required_permissions),
                    "missing_permissions": list(missing_permissions),
                    "user_role": user_role
                }
            )
        
        # Log successful access for PHI operations
        if required_permissions & phi_permissions:
            await self._log_phi_access(request, required_permissions)
        
        # Continue to endpoint
        response = await call_next(request)
        
        return response
    
    def _get_required_permissions(self, path: str, method: str) -> Set[str]:
        """
        Get required permissions for an endpoint path and HTTP method.
        Supports parameterized paths like /api/v1/users/{user_id}
        """
        # Try exact match first
        if path in self.ENDPOINT_PERMISSIONS:
            return self.ENDPOINT_PERMISSIONS[path]
        
        # Try pattern matching for parameterized paths
        for pattern, permissions in self.ENDPOINT_PERMISSIONS.items():
            if self._matches_pattern(path, pattern):
                # Adjust permissions based on HTTP method
                return self._adjust_permissions_for_method(permissions, method)
        
        # Default: require basic authentication for unknown endpoints
        return {Permission.USER_READ}
    
    def _matches_pattern(self, path: str, pattern: str) -> bool:
        """Check if path matches a parameterized pattern."""
        if "{" not in pattern:
            return path == pattern
        
        path_parts = path.split("/")
        pattern_parts = pattern.split("/")
        
        if len(path_parts) != len(pattern_parts):
            return False
        
        for path_part, pattern_part in zip(path_parts, pattern_parts):
            if pattern_part.startswith("{") and pattern_part.endswith("}"):
                continue  # Parameter match
            elif path_part != pattern_part:
                return False
        
        return True
    
    def _adjust_permissions_for_method(self, base_permissions: Set[str], method: str) -> Set[str]:
        """Adjust permissions based on HTTP method."""
        if method in ["GET", "HEAD"]:
            # Read operations
            return base_permissions
        elif method in ["POST", "PUT", "PATCH"]:
            # Write operations - upgrade read to write permissions
            adjusted = set(base_permissions)
            if Permission.PHI_READ in adjusted:
                adjusted.add(Permission.PHI_WRITE)
            if Permission.DOC_READ in adjusted:
                adjusted.add(Permission.DOC_WRITE)
            if Permission.USER_READ in adjusted and "/users/" in str(adjusted):
                adjusted.add(Permission.USER_WRITE)
            return adjusted
        elif method == "DELETE":
            # Delete operations
            adjusted = set(base_permissions)
            adjusted.add(Permission.DOC_DELETE)
            return adjusted
        
        return base_permissions
    
    async def _log_access_denied(self, request: Request, event_type: str, description: str):
        """Log access denied events for security monitoring."""
        try:
            user_id = getattr(request.state, 'user_id', None)
            session_id = getattr(request.state, 'session_id', None)
            client_ip = self._get_client_ip(request)
            
            async with db_manager.get_session() as session:
                from database import AuditLog, ResourceType
                
                audit_log = AuditLog(
                    user_id=user_id,
                    session_id=session_id,
                    action_type=AuditAction.ACCESS,
                    action_description=f"Access denied: {description}",
                    resource_type=ResourceType.SYSTEM,
                    endpoint=f"{request.method} {request.url.path}",
                    http_method=request.method,
                    ip_address=client_ip,
                    success=False,
                    error_code="403",
                    error_message=description,
                    requires_review=True,
                    audit_metadata={
                        "event_type": event_type,
                        "requested_endpoint": str(request.url.path),
                        "user_agent": request.headers.get("user-agent", ""),
                        "referer": request.headers.get("referer", "")
                    }
                )
                
                session.add(audit_log)
                await session.commit()
                
        except Exception as e:
            logger.error(f"Failed to log access denied event: {e}")
    
    async def _log_phi_access(self, request: Request, permissions: Set[str]):
        """Log PHI access for HIPAA compliance."""
        try:
            user_id = getattr(request.state, 'user_id', None)
            session_id = getattr(request.state, 'session_id', None)
            client_ip = self._get_client_ip(request)
            
            async with db_manager.get_session() as session:
                from database import AuditLog, ResourceType
                
                audit_log = AuditLog(
                    user_id=user_id,
                    session_id=session_id,
                    action_type=AuditAction.ACCESS,
                    action_description=f"PHI access granted for {request.url.path}",
                    resource_type=ResourceType.PATIENT,
                    endpoint=f"{request.method} {request.url.path}",
                    http_method=request.method,
                    ip_address=client_ip,
                    success=True,
                    phi_accessed=True,
                    requires_review=False,
                    audit_metadata={
                        "permissions": list(permissions),
                        "phi_access_type": "direct",
                        "endpoint": str(request.url.path),
                        "user_agent": request.headers.get("user-agent", "")
                    }
                )
                
                session.add(audit_log)
                await session.commit()
                
        except Exception as e:
            logger.error(f"Failed to log PHI access: {e}")
    
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


# Convenience functions for endpoint-level permission checks
def require_permissions(*permissions: str):
    """
    Decorator to require specific permissions for an endpoint.
    Usage: @require_permissions(Permission.PHI_READ, Permission.DOC_WRITE)
    """
    def decorator(func):
        func._required_permissions = set(permissions)
        return func
    return decorator


def require_phi_access(func):
    """
    Decorator to require PHI access for an endpoint.
    Usage: @require_phi_access
    """
    func._requires_phi = True
    return func


def require_role(*roles: str):
    """
    Decorator to require specific roles for an endpoint.
    Usage: @require_role("attending_physician", "admin")
    """
    def decorator(func):
        func._required_roles = set(roles)
        return func
    return decorator