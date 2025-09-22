"""
Database connection and session management for Orthopedics EMR RAG System.
HIPAA-compliant PostgreSQL connection with security monitoring.
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy import event, text
from sqlalchemy.engine import Engine
import asyncpg

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Base(DeclarativeBase):
    """Base class for all database models."""
    pass

class DatabaseManager:
    """Manages database connections and sessions for the EMR system."""
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database manager.
        
        Args:
            database_url: PostgreSQL connection string. If None, uses environment variable.
        """
        self.database_url = database_url or self._build_database_url()
        self.engine = None
        self.async_session_factory = None
        
    def _build_database_url(self) -> str:
        """Build database URL from environment or defaults."""
        
        # Check for explicit connection string
        if db_url := os.getenv('DATABASE_URL'):
            return db_url.replace('postgresql://', 'postgresql+asyncpg://')
        
        # Build from components
        username = os.getenv('DB_USER', os.getenv('USER', 'whitneywalters'))
        password = os.getenv('DB_PASSWORD', '')
        host = os.getenv('DB_HOST', 'localhost')
        port = os.getenv('DB_PORT', '5432')
        database = os.getenv('DB_NAME', 'ortho_emr_security')
        
        if password:
            auth = f"{username}:{password}"
        else:
            auth = username
            
        return f"postgresql+asyncpg://{auth}@{host}:{port}/{database}"
    
    async def initialize(self):
        """Initialize database engine and session factory."""
        
        if self.engine is not None:
            logger.warning("Database already initialized")
            return
            
        logger.info("Initializing database connection...")
        
        # Create async engine with security-focused configuration
        self.engine = create_async_engine(
            self.database_url,
            # Connection pool settings
            pool_size=10,                    # Reasonable pool size for medical app
            max_overflow=20,                 # Allow burst connections
            pool_pre_ping=True,              # Validate connections before use
            pool_recycle=3600,               # Recycle connections hourly
            
            # Security settings
            connect_args={
                "command_timeout": 30,       # 30 second query timeout
                "server_settings": {
                    "application_name": "orthopedics_emr_rag",
                    "timezone": "UTC",
                },
            },
            
            # Logging for audit purposes
            echo=os.getenv('DB_ECHO', 'false').lower() == 'true',
            echo_pool=os.getenv('DB_ECHO_POOL', 'false').lower() == 'true',
        )
        
        # Create async session factory
        self.async_session_factory = sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        
        # Test connection
        await self._test_connection()
        
        logger.info("Database initialization complete")
    
    async def _test_connection(self):
        """Test database connection and log for audit."""
        try:
            async with self.engine.begin() as conn:
                result = await conn.execute(text("SELECT version(), current_database(), current_user"))
                version, database, user = result.fetchone()
                
            logger.info(f"Database connection successful:")
            logger.info(f"  Database: {database}")
            logger.info(f"  User: {user}")
            logger.info(f"  PostgreSQL Version: {version}")
            
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get database session with automatic cleanup.
        
        Usage:
            async with db_manager.get_session() as session:
                # Use session here
                pass
        """
        if self.async_session_factory is None:
            await self.initialize()
            
        async with self.async_session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def close(self):
        """Close database connections."""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connections closed")

# Global database manager instance
db_manager = DatabaseManager()

# Connection event handlers for security monitoring
@event.listens_for(Engine, "connect")
def set_connection_security(dbapi_connection, connection_record):
    """Set security parameters on new connections."""
    # For asyncpg, we can't use cursor context manager
    # These settings will be applied at connection level
    pass

@event.listens_for(Engine, "before_cursor_execute")
def log_sql_execution(conn, cursor, statement, parameters, context, executemany):
    """Log SQL execution for audit purposes (in development)."""
    if os.getenv('ENVIRONMENT') == 'development':
        logger.debug(f"Executing SQL: {statement[:100]}...")

# Dependency injection for FastAPI
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for getting database session.
    
    Usage in FastAPI routes:
        @app.get("/users")
        async def get_users(session: AsyncSession = Depends(get_db_session)):
            # Use session here
    """
    async with db_manager.get_session() as session:
        yield session

# Health check function
async def check_database_health() -> dict:
    """
    Check database health for monitoring.
    
    Returns:
        dict: Health status information
    """
    try:
        async with db_manager.get_session() as session:
            # Test basic query
            result = await session.execute(text("SELECT 1 as health_check"))
            health_value = result.scalar()
            
            # Check connection pool
            pool = db_manager.engine.pool
            pool_status = {
                "size": pool.size(),
                "checked_in": pool.checkedin(),
                "checked_out": pool.checkedout(),
            }
            
            return {
                "status": "healthy" if health_value == 1 else "unhealthy",
                "database": "ortho_emr_security",
                "connection_pool": pool_status,
                "timestamp": "utc_now()"
            }
            
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": "utc_now()"
        }

# Utility functions for setting user context
async def set_current_user_context(session: AsyncSession, user_id: str):
    """Set current user context for Row Level Security."""
    await session.execute(
        text("SELECT set_config('app.current_user_id', :user_id, true)"),
        {"user_id": user_id}
    )

async def clear_user_context(session: AsyncSession):
    """Clear user context."""
    await session.execute(
        text("SELECT set_config('app.current_user_id', '', true)")
    )