"""
Main API router for Orthopedics EMR RAG System
Organizes all API endpoints with proper versioning
"""

from fastapi import APIRouter

# Import route modules (will be created next)
from .auth import router as auth_router
from .users import router as users_router
from .system import router as system_router
from .test_rbac import router as test_rbac_router
from .patients import router as patients_router
from .search import router as search_router

# Create main API router
api_router = APIRouter()

# Include sub-routers with tags for organization
api_router.include_router(
    auth_router,
    prefix="/auth", 
    tags=["Authentication"],
    responses={401: {"description": "Unauthorized"}}
)

api_router.include_router(
    users_router,
    prefix="/users",
    tags=["User Management"], 
    responses={401: {"description": "Unauthorized"}}
)

api_router.include_router(
    system_router,
    prefix="/system",
    tags=["System Management"],
    responses={401: {"description": "Unauthorized"}}
)

# Test RBAC router (temporary for POC)
api_router.include_router(
    test_rbac_router,
    prefix="/test-rbac",
    tags=["RBAC Testing (POC)"]
)

# Patient management router
api_router.include_router(
    patients_router,
    prefix="/patients",
    tags=["Patient Management"],
    responses={401: {"description": "Unauthorized"}, 403: {"description": "Insufficient permissions"}}
)

# Semantic search router
api_router.include_router(
    search_router,
    prefix="/search",
    tags=["Semantic Search"],
    responses={401: {"description": "Unauthorized"}, 403: {"description": "Insufficient permissions"}}
)