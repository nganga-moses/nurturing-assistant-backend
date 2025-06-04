import os
import sys
import hashlib
from sqlalchemy.orm import Session
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.models.user import User, Role
from database.session import get_db
from database.base import Base
from database.engine import engine

load_dotenv()

def hash_password(password: str) -> str:
    """Hash a password using SHA-256."""
    salt = os.urandom(32)
    key = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt,
        100000
    )
    return salt.hex() + key.hex()

def init_db():
    """Initialize the database by creating all tables."""
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")

def seed_roles_and_admin():
    """Seed default roles and admin user."""
    session = next(get_db())
    
    try:
        # Create default roles
        roles = [
            Role(name="super_admin", description="System Super administrator with full access"),
            Role(name="admin", description="System administrator with full access"),
            Role(name="admissions_assistant", description="Can view and manage student profiles"),
            Role(name="manager", description="Can view analytics and manage recruiters"),
            Role(name="readonly", description="Can only view data, no modifications")
        ]
        
        for role in roles:
            existing = session.query(Role).filter_by(name=role.name).first()
            if not existing:
                session.add(role)
        
        session.commit()
        print("Roles seeded successfully!")
        
        # Create admin user if it doesn't exist

            
    except Exception as e:
        session.rollback()
        print(f"Error seeding database: {str(e)}")
        raise
    finally:
        session.close()

if __name__ == "__main__":
    print("Initializing database...")
    init_db()
    print("\nSeeding roles and admin user...")
    seed_roles_and_admin() 