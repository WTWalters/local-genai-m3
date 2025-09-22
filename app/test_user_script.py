#!/usr/bin/env python3
"""
Temporary script to create test user with proper password hashing for RBAC testing
"""
import asyncio
from passlib.context import CryptContext
from database import db_manager
from sqlalchemy import select, text
import uuid

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

async def create_test_user():
    """Create a test user for RBAC authentication testing."""
    await db_manager.initialize()
    
    # User details
    user_id = str(uuid.uuid4())
    username = "dr_smith"  # Use existing user
    password = "TestPass123"
    password_hash = pwd_context.hash(password)
    
    print(f"Creating test user: {username}")
    print(f"Password: {password}")
    print(f"Password hash: {password_hash[:30]}...")
    
    async with db_manager.get_session() as session:
        # First check if user exists
        check_sql = text("SELECT username FROM users WHERE username = :username")
        result = await session.execute(check_sql, {"username": username})
        if result.fetchone():
            print(f"User {username} already exists, updating password...")
            update_sql = text("UPDATE users SET password_hash = :password_hash WHERE username = :username")
            await session.execute(update_sql, {"username": username, "password_hash": password_hash})
        else:
            # Create new user with attending physician role
            insert_sql = text("""
                INSERT INTO users (
                    user_id, username, email, first_name, last_name, 
                    password_hash, status, role_id, department, 
                    mfa_enabled, created_at
                ) 
                SELECT 
                    :user_id, :username, :email, :first_name, :last_name,
                    :password_hash, 'active', r.role_id, 'Orthopedics',
                    false, NOW()
                FROM roles r 
                WHERE r.role_name = 'attending_physician'
            """)
            
            await session.execute(insert_sql, {
                "user_id": user_id,
                "username": username,
                "email": f"{username}@orthotest.com",
                "first_name": "Test",
                "last_name": "Doctor",
                "password_hash": password_hash
            })
            
        await session.commit()
        print(f"✅ Test user created successfully!")
    
    await db_manager.close()

if __name__ == "__main__":
    asyncio.run(create_test_user())