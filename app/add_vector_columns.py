"""
Add vector embedding columns to existing medical tables.
This script adds pgvector columns for semantic search functionality.
"""

import asyncio
import asyncpg
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def add_vector_columns():
    """Add vector embedding columns to medical tables."""
    
    # Database connection details
    username = os.getenv('DB_USER', os.getenv('USER', 'whitneywalters'))
    password = os.getenv('DB_PASSWORD', '')
    host = os.getenv('DB_HOST', 'localhost')
    port = os.getenv('DB_PORT', '5432')
    database = os.getenv('DB_NAME', 'ortho_emr_security')
    
    try:
        # Connect to database
        conn = await asyncpg.connect(
            user=username,
            password=password if password else None,
            host=host,
            port=port,
            database=database
        )
        
        logger.info("Connected to database successfully")
        
        # Check if vector extension is available
        result = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
        )
        
        if not result:
            logger.error("pgvector extension is not installed")
            return
            
        logger.info("pgvector extension is available")
        
        # Add vector columns to tables
        vector_updates = [
            {
                'table': 'patients',
                'column': 'summary_embedding',
                'description': 'Patient summary embedding for semantic search'
            },
            {
                'table': 'medical_notes',
                'column': 'content_embedding',
                'description': 'Medical note content embedding for semantic search'
            },
            {
                'table': 'procedures',
                'column': 'content_embedding',
                'description': 'Procedure content embedding for semantic search'
            },
            {
                'table': 'imaging_studies',
                'column': 'content_embedding',
                'description': 'Imaging study content embedding for semantic search'
            }
        ]
        
        for update in vector_updates:
            table = update['table']
            column = update['column']
            description = update['description']
            
            # Check if table exists
            table_exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = $1)",
                table
            )
            
            if not table_exists:
                logger.warning(f"Table {table} does not exist, skipping...")
                continue
                
            # Check if column already exists
            column_exists = await conn.fetchval(
                """
                SELECT EXISTS(
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name = $1 AND column_name = $2
                )
                """,
                table, column
            )
            
            if column_exists:
                logger.info(f"Column {table}.{column} already exists, skipping...")
                continue
                
            # Add the vector column
            try:
                await conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} vector(384)"
                )
                logger.info(f"✅ Added {column} to {table} - {description}")
                
                # Add comment to column for documentation
                await conn.execute(
                    f"COMMENT ON COLUMN {table}.{column} IS '{description}'"
                )
                
            except Exception as e:
                logger.error(f"Failed to add {column} to {table}: {e}")
        
        # Create indexes for vector similarity search
        index_updates = [
            {
                'table': 'patients',
                'column': 'summary_embedding',
                'index_name': 'idx_patients_summary_embedding_cosine'
            },
            {
                'table': 'medical_notes',
                'column': 'content_embedding',
                'index_name': 'idx_medical_notes_content_embedding_cosine'
            },
            {
                'table': 'procedures',
                'column': 'content_embedding',
                'index_name': 'idx_procedures_content_embedding_cosine'
            },
            {
                'table': 'imaging_studies',
                'column': 'content_embedding',
                'index_name': 'idx_imaging_studies_content_embedding_cosine'
            }
        ]
        
        for index_update in index_updates:
            table = index_update['table']
            column = index_update['column']
            index_name = index_update['index_name']
            
            # Check if table and column exist
            column_exists = await conn.fetchval(
                """
                SELECT EXISTS(
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name = $1 AND column_name = $2
                )
                """,
                table, column
            )
            
            if not column_exists:
                logger.warning(f"Column {table}.{column} does not exist, skipping index...")
                continue
            
            # Check if index already exists
            index_exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM pg_indexes WHERE indexname = $1)",
                index_name
            )
            
            if index_exists:
                logger.info(f"Index {index_name} already exists, skipping...")
                continue
                
            try:
                # Create vector index using cosine distance
                await conn.execute(
                    f"CREATE INDEX {index_name} ON {table} USING ivfflat ({column} vector_cosine_ops) WITH (lists = 100)"
                )
                logger.info(f"✅ Created vector index {index_name} on {table}.{column}")
                
            except Exception as e:
                logger.error(f"Failed to create index {index_name}: {e}")
        
        logger.info("✅ Vector column migration completed successfully")
        
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        
    finally:
        if 'conn' in locals():
            await conn.close()
            logger.info("Database connection closed")

if __name__ == "__main__":
    asyncio.run(add_vector_columns())