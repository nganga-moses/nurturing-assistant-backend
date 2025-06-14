import os
import sys
import hashlib
from sqlalchemy.orm import Session
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.models.user import User, Role
from database.session import get_db

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

def seed_roles_and_admin():
    """Seed default roles and admin user."""
    session = next(get_db())
    
    # Create default roles
    roles = [
        Role(name="super_admin", description="System Super administrator with full access"),
        Role(name="admin", description="System administrator with full access"),
        Role(name="admissions assistant", description="Can view and manage student profiles"),
        Role(name="manager", description="Can view analytics and manage recruiters"),
        Role(name="vp", description="Has Strategic oversight"),
        Role(name="readonly", description="Can only view data, no modifications")
    ]
    
    for role in roles:
        existing = session.query(Role).filter_by(name=role.name).first()
        if not existing:
            session.add(role)
    
    session.commit()
    

    
    print("Roles and admin user seeded successfully!")

if __name__ == "__main__":
    seed_roles_and_admin() 