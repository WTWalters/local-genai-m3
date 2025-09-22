"""
HIPAA-compliant audit logging middleware
Tracks all requests and responses for medical compliance
"""

import time
import json
from typing import Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from core.security import generate_request_id
from database import db_manager, AuditAction


class AuditMiddleware(BaseHTTPMiddleware):
    """
    Middleware to audit all HTTP requests for HIPAA compliance.
    Logs all access attempts, queries, and data modifications.
    """
    
    async def dispatch(self, request: Request, call_next):
        # Generate request ID for tracking
        request_id = generate_request_id()
        request.state.request_id = request_id
        
        # Start timing
        start_time = time.time()
        
        # Get client information
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "")
        referer = request.headers.get("referer", "")
        
        # Get user context (if authenticated)
        user_id = getattr(request.state, 'user_id', None)
        session_id = getattr(request.state, 'session_id', None)
        
        # Get request details
        method = request.method
        url = str(request.url)
        endpoint = f"{method} {request.url.path}"
        
        # Calculate request size
        request_size = 0
        if hasattr(request, '_body'):
            request_size = len(request._body or b'')
        
        # Determine action type based on method and endpoint
        action_type = self._determine_action_type(method, request.url.path)
        
        # Process request
        response = await call_next(request)
        
        # Calculate timing and response size
        process_time = time.time() - start_time
        response_size = 0
        if hasattr(response, 'body'):
            response_size = len(response.body or b'')
        
        # Determine if PHI was potentially accessed
        phi_accessed = self._check_phi_access(request.url.path, response.status_code)
        
        # Log audit entry
        try:
            await self._create_audit_log(
                user_id=user_id,
                session_id=session_id,
                action_type=action_type,
                endpoint=endpoint,
                http_method=method,
                ip_address=client_ip,
                user_agent=user_agent,
                referer=referer,
                request_size_bytes=request_size,
                response_size_bytes=response_size,
                query_execution_time_ms=int(process_time * 1000),
                success=(200 <= response.status_code < 400),
                error_code=str(response.status_code) if response.status_code >= 400 else None,
                phi_accessed=phi_accessed,
                audit_metadata={
                    "request_id": request_id,
                    "url": url,
                    "process_time_ms": int(process_time * 1000),
                    "status_code": response.status_code,
                }
            )
        except Exception as e:
            # Don't fail request if audit logging fails, but log the error
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Audit logging failed: {e}", exc_info=True)
        
        # Add audit headers to response
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{process_time:.3f}s"
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers (from reverse proxy)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fall back to direct client IP
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _determine_action_type(self, method: str, path: str) -> AuditAction:
        """Determine audit action type based on HTTP method and path."""
        if "login" in path.lower():
            return AuditAction.LOGIN
        elif "logout" in path.lower():
            return AuditAction.LOGOUT
        elif method == "GET":
            if any(term in path.lower() for term in ["patient", "record", "document"]):
                return AuditAction.ACCESS
            return AuditAction.QUERY
        elif method in ["POST", "PUT", "PATCH"]:
            return AuditAction.MODIFY
        elif method == "DELETE":
            return AuditAction.DELETE
        elif "export" in path.lower():
            return AuditAction.EXPORT
        elif "admin" in path.lower():
            return AuditAction.ADMIN
        else:
            return AuditAction.ACCESS
    
    def _check_phi_access(self, path: str, status_code: int) -> bool:
        """Determine if request potentially accessed PHI."""
        if status_code >= 400:
            return False  # Failed requests didn't access PHI
        
        # Define PHI-related endpoints
        phi_endpoints = [
            "patient", "record", "document", "clinical", "medical",
            "diagnosis", "treatment", "prescription", "lab", "imaging"
        ]
        
        return any(term in path.lower() for term in phi_endpoints)
    
    async def _create_audit_log(self, **kwargs):
        """Create audit log entry in database."""
        try:
            async with db_manager.get_session() as session:
                from database import AuditLog
                
                audit_log = AuditLog(**kwargs)
                session.add(audit_log)
                await session.commit()
                
        except Exception as e:
            # Re-raise to be handled by caller
            raise e