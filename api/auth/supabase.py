from supabase import create_client, Client
import os
from typing import Optional
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from database.session import get_db
from data.models.user import User
import jwt
from jwt.exceptions import InvalidTokenError
import time

from dotenv import load_dotenv

load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")

# Global variable for lazy initialization
_supabase_client: Optional[Client] = None

def get_supabase_client() -> Client:
    """Get or create Supabase client with lazy initialization."""
    global _supabase_client
    if _supabase_client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            print("Warning: SUPABASE_URL and SUPABASE_KEY not set - running without auth")
            return None
        try:
            _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        except Exception as e:
            print(f"Warning: Could not initialize Supabase client: {e}")
            print("Running in development mode without authentication")
            # For development, we'll return None and handle it gracefully
            return None
    return _supabase_client


security = HTTPBearer()

def get_user_from_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """
    Extracts the user ID from the Supabase JWT token.
    """
    token = credentials.credentials
    try:
        # Decode without verification since we trust Supabase's signing
        decoded_token = jwt.decode(token, options={"verify_signature": False})
        
        # Check if token is expired
        if decoded_token.get('exp', 0) < time.time():
            raise HTTPException(status_code=401, detail="Token has expired")
            
        user_id = decoded_token.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token: missing user ID")
        return user_id
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Token validation failed: {str(e)}")

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security),
    db: Session = Depends(get_db)
) -> User:
    """Get current user from JWT token."""
    try:
        # Get user ID from token
        user_id = get_user_from_token(credentials)
        
        # Get user from database
        db_user = db.query(User).filter(User.supabase_id == user_id).first()
        if not db_user:
            raise HTTPException(status_code=401, detail="User not found")
        if not db_user.is_active:
            raise HTTPException(status_code=401, detail="Inactive user")
            
        return db_user
    except HTTPException:
        raise
    except Exception as e:
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
    """Get user info from the token (for registration)."""
    try:
        token = credentials.credentials
        decoded_token = jwt.decode(token, options={"verify_signature": False})
        
        # Check if token is expired
        if decoded_token.get('exp', 0) < time.time():
            raise HTTPException(status_code=401, detail="Token has expired")
            
        return {
            'supabase_id': decoded_token.get('sub'),
            'email': decoded_token.get('email'),
            'username': decoded_token.get('user_metadata', {}).get('username'),
            'first_name': decoded_token.get('user_metadata', {}).get('first_name'),
            'last_name': decoded_token.get('user_metadata', {}).get('last_name'),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

async def get_current_user_id(user: User = Depends(get_current_user)) -> str:
    """Extract the user ID from the current user."""
    return user.id 