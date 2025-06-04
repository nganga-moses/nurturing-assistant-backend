from supabase import create_client, Client
import os
from typing import Optional
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from database.session import get_db
from data.models.user import User
import jwt

from dotenv import load_dotenv

load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security),
    db: Session = Depends(get_db)
) -> User:
    """Get current user from Supabase JWT token."""
    try:
        # Verify the JWT token with Supabase
        user_response = supabase.auth.get_user(credentials.credentials)
        user = user_response.user if hasattr(user_response, 'user') else user_response
        print("Supabase user.id:", user.id)
        db_user = db.query(User).filter(User.supabase_id == user.id).first()
        print("db_user:", db_user)
        if not db_user:
            print("User not found in DB")
            raise HTTPException(status_code=401, detail="User not found")
        if not db_user.is_active:
            print("User is inactive")
            raise HTTPException(status_code=401, detail="Inactive user")
        print("Returning db_user:", db_user)
        return db_user
    except Exception as e:
        print("Exception in get_current_user:", e)
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

def check_admin(user: User = Depends(get_current_user)):
    """Check if user has admin role."""
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return user

def check_manager(user: User = Depends(get_current_user)):
    """Check if user has manager or admin role."""
    if user.role not in ["admin", "manager"]:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return user

def check_vp(user: User = Depends(get_current_user)):
    """Check if user has VP or admin role."""
    if user.role not in ["admin", "vp"]:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return user

def check_admin_or_vp(user: User = Depends(get_current_user)):
    """Check if user has admin or VP role."""
    if user.role not in ["admin", "vp"]:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return user

def check_recruiter_or_admin(user: User = Depends(get_current_user)):
    """Check if user has recruiter, VP, or admin role."""
    if user.role not in ["admin", "vp", "recruiter"]:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return user

async def get_supabase_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Validate Supabase JWT and return user info from the token (for registration)."""
    try:
        user = supabase.auth.get_user(credentials.credentials)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        # Return a dict with the info needed for registration
        user_info = user.user if hasattr(user, 'user') else user
        return {
            'supabase_id': user_info.id,
            'email': user_info.email,
            'username': user_info.user_metadata.get('username') if user_info.user_metadata else None,
            'first_name': user_info.user_metadata.get('first_name') if user_info.user_metadata else None,
            'last_name': user_info.user_metadata.get('last_name') if user_info.user_metadata else None,
        }
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

async def get_current_user_id(user: User = Depends(get_current_user)) -> str:
    """Extract the user ID from the current user."""
    return user.id 