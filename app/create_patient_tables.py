#!/usr/bin/env python3
"""
Create patient tables in the database
"""
import asyncio
from database import db_manager, Base
from database.patient_models import *

async def create_patient_tables():
    """Create all patient-related tables."""
    print("Creating patient tables...")
    
    await db_manager.initialize()
    
    # Create all tables
    async with db_manager.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    print("✅ Patient tables created successfully!")
    await db_manager.close()

if __name__ == "__main__":
    asyncio.run(create_patient_tables())