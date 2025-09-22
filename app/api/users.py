"""
User management endpoints for medical staff
Handles user profiles, roles, and account management with RBAC
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
from datetime import datetime

from core.dependencies import (
    CurrentUser, get_current_user, require_admin, 
    require_permission, require_role
)
from middleware.rbac import Permission
from schemas.auth import UserInfo

router = APIRouter()

# Pydantic models
class UserProfile(BaseModel):
    """Extended user profile with medical information."""
    user_id: str
    username: str
    email: str
    first_name: str
    last_name: str
    role: str
    department: Optional[str] = None
    title: Optional[str] = None
    npi_number: Optional[str] = None
    license_number: Optional[str] = None
    dea_number: Optional[str] = None
    status: str
    last_login: Optional[datetime] = None
    can_access_phi: bool
    mfa_enabled: bool
    created_at: datetime
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "550e8400-e29b-41d4-a716-446655440000",
                "username": "dr.smith",
                "email": "j.smith@orthoclinic.com",
                "first_name": "John",
                "last_name": "Smith",
                "role": "attending_physician",
                "department": "Orthopedics",
                "title": "Attending Physician",
                "npi_number": "1234567890",
                "license_number": "MD123456",
                "dea_number": "AS1234567",
                "status": "active",
                "last_login": "2024-01-15T10:30:00Z",
                "can_access_phi": True,
                "mfa_enabled": True,
                "created_at": "2024-01-01T09:00:00Z"
            }
        }

class UserUpdate(BaseModel):
    """User profile update model."""
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    department: Optional[str] = None
    title: Optional[str] = None

class UserListResponse(BaseModel):
    """Response for user listing with metadata."""
    users: List[UserProfile]
    total: int
    page: int
    limit: int

@router.get("/me", 
    response_model=UserProfile,
    summary="Get current user profile",
    description="Get the profile of the currently authenticated user"
)
async def get_current_user_profile(current_user: CurrentUser = Depends(get_current_user)):
    """
    Get current user profile.
    
    Returns detailed profile information for the authenticated user including:
    - Personal information
    - Medical credentials (NPI, DEA, license)
    - Role and permissions
    - Account status and security settings
    """
    try:
        from database import db_manager, User, Role
        from sqlalchemy import select
        
        async with db_manager.get_session() as session:
            # Get user with role information
            stmt = select(User, Role).join(Role).where(User.user_id == current_user.user_id)
            result = await session.execute(stmt)
            user_data = result.first()
            
            if not user_data:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User profile not found"
                )
            
            user, role = user_data
            
            return UserProfile(
                user_id=str(user.user_id),
                username=user.username,
                email=user.email,
                first_name=user.first_name,
                last_name=user.last_name,
                role=role.role_name,
                department=user.department,
                title=user.title,
                npi_number=user.npi_number,
                license_number=user.license_number,
                dea_number=user.dea_number,
                status=user.status,
                last_login=user.last_login,
                can_access_phi=role.can_access_phi,
                mfa_enabled=user.mfa_enabled,
                created_at=user.created_at
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve user profile: {str(e)}"
        )

@router.put("/me", 
    response_model=UserProfile,
    summary="Update current user profile",
    description="Update the profile of the currently authenticated user"
)
async def update_current_user_profile(
    user_update: UserUpdate,
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Update current user profile.
    
    Allows users to update their own profile information including:
    - Name and contact information
    - Department and title
    
    Note: Critical fields like role, credentials, and status require admin access.
    """
    # TODO: Implement user profile update with audit logging
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="User profile update not yet implemented"
    )

@router.get("/", 
    response_model=UserListResponse,
    summary="List all users",
    description="List all system users (admin or user management permission required)"
)
async def list_users(
    page: int = 1,
    limit: int = 20,
    current_user: CurrentUser = Depends(require_permission(Permission.USER_READ))
):
    """
    List all users with pagination.
    
    Requires USER_READ permission. Returns user profiles with:
    - Basic profile information
    - Role and department
    - Account status
    - Last activity
    
    Sensitive information (credentials, passwords) is excluded.
    """
    # TODO: Implement user listing with pagination and filtering
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="User listing not yet implemented"
    )

@router.get("/{user_id}", 
    response_model=UserProfile,
    summary="Get user by ID", 
    description="Get detailed user profile by user ID (admin or supervisor access required)"
)
async def get_user_by_id(
    user_id: str,
    current_user: CurrentUser = Depends(require_permission(Permission.USER_READ))
):
    """
    Get user profile by ID.
    
    Requires USER_READ permission. Returns detailed profile for specified user.
    Additional access controls may apply based on organizational hierarchy.
    """
    # TODO: Implement user retrieval by ID with hierarchy checks
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="User retrieval by ID not yet implemented"
    )

@router.put("/{user_id}", 
    response_model=UserProfile,
    summary="Update user by ID",
    description="Update user profile by ID (admin access required)"
)
async def update_user_by_id(
    user_id: str,
    user_update: UserUpdate,
    current_user: CurrentUser = Depends(require_permission(Permission.USER_WRITE))
):
    """
    Update user profile by ID.
    
    Requires USER_WRITE permission (typically admin only).
    Allows updating all user profile fields including sensitive information.
    All changes are audited for compliance.
    """
    # TODO: Implement user update by ID with comprehensive audit logging
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="User update by ID not yet implemented"
    )

@router.delete("/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Deactivate user",
    description="Deactivate user account (admin access required)"
)
async def deactivate_user(
    user_id: str,
    current_user: CurrentUser = Depends(require_permission(Permission.USER_DELETE))
):
    """
    Deactivate user account.
    
    Requires USER_DELETE permission (admin only).
    Does not physically delete user data for audit compliance.
    Instead, marks account as inactive and invalidates all sessions.
    """
    # TODO: Implement user deactivation with session cleanup
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="User deactivation not yet implemented"
    )

@router.get("/roles/permissions",
    summary="Get role permissions",
    description="Get permission matrix for all medical roles"
)
async def get_role_permissions(
    current_user: CurrentUser = Depends(require_admin())
):
    """
    Get role permissions matrix.
    
    Admin only endpoint that returns the complete permission matrix
    for all medical roles in the system. Useful for role management
    and permission auditing.
    """
    from middleware.rbac import RBACMiddleware
    
    return {
        "roles": {
            role: list(permissions) 
            for role, permissions in RBACMiddleware.ROLE_PERMISSIONS.items()
        },
        "current_user": {
            "role": current_user.role,
            "permissions": list(current_user.permissions)
        }
    }