"""
Orthopedics EMR RAG System - FastAPI Application
HIPAA-compliant medical AI system for orthopedic practice
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from core.config import settings
from core.security import SecurityHeaders
from core.embeddings import embedding_service
from database import db_manager
from api.router import api_router
from middleware.audit import AuditMiddleware
from middleware.session import SessionMiddleware
from middleware.rbac import RBACMiddleware

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure logging for medical compliance
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ortho_emr.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Application lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup application resources."""
    # Startup
    logger.info("🏥 Starting Orthopedics EMR RAG System...")
    
    # Initialize database
    await db_manager.initialize()
    logger.info("✅ Database connection established")
    
    # Initialize embedding service
    await embedding_service.initialize()
    logger.info("✅ Embedding service initialized")
    
    logger.info("🚀 Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down Orthopedics EMR RAG System...")
    await db_manager.close()
    logger.info("✅ Database connections closed")
    logger.info("👋 Application shutdown complete")

# Create FastAPI application
app = FastAPI(
    title="Orthopedics EMR RAG System",
    description="HIPAA-compliant AI-powered medical information system for orthopedic practice",
    version="1.0.0",
    docs_url="/docs" if settings.ENVIRONMENT == "development" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT == "development" else None,
    openapi_url="/openapi.json" if settings.ENVIRONMENT == "development" else None,
    lifespan=lifespan
)

# Security middleware - add trusted hosts for airgapped deployment
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=settings.ALLOWED_HOSTS
)

# CORS middleware for JavaScript frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"]
)

# Custom security headers middleware
app.add_middleware(SecurityHeaders)

# Session middleware for user tracking
app.add_middleware(SessionMiddleware)

# RBAC middleware for permission enforcement
app.add_middleware(RBACMiddleware)

# Audit middleware for HIPAA compliance
app.add_middleware(AuditMiddleware)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with audit logging."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error_id": "ERR_INTERNAL_500",
            "timestamp": "utc_now()"
        }
    )

# Health check endpoint (always available)
@app.get("/health", tags=["System"])
async def health_check():
    """System health check for monitoring."""
    try:
        from database import check_database_health
        db_health = await check_database_health()
        
        return {
            "status": "healthy",
            "system": "Orthopedics EMR RAG System",
            "version": "1.0.0",
            "database": db_health,
            "timestamp": "utc_now()"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": "utc_now()"
            }
        )

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include API router
app.include_router(api_router, prefix="/api/v1")

# Root endpoint - serve frontend
@app.get("/", tags=["System"])
async def root():
    """Serve the frontend application."""
    return FileResponse("static/index.html")

# API info endpoint
@app.get("/api", tags=["System"])
async def api_info():
    """API information endpoint."""
    return {
        "system": "Orthopedics EMR RAG System",
        "version": "1.0.0",
        "status": "operational",
        "description": "HIPAA-compliant AI-powered medical information system",
        "docs": "/docs" if settings.ENVIRONMENT == "development" else "disabled",
        "timestamp": "utc_now()"
    }

if __name__ == "__main__":
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Run application
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.ENVIRONMENT == "development",
        access_log=True,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
                "file": {
                    "formatter": "default",
                    "class": "logging.FileHandler",
                    "filename": "logs/access.log",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["default", "file"],
            },
        },
    )