from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from database.session import get_db
from ..auth.supabase import supabase
from data.models.user import User

router = APIRouter(prefix="/auth", tags=["authentication"])

@router.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Login with Supabase."""
    try:
        # Authenticate with Supabase
        auth_response = supabase.auth.sign_in_with_password({
            "email": form_data.username,
            "password": form_data.password
        })
        
        if not auth_response.user:
            raise HTTPException(status_code=401, detail="Invalid credentials")
            
        # Get the user from our database
        user = db.query(User).filter(User.id == auth_response.user.id).first()
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        if not user.is_active:
            raise HTTPException(status_code=401, detail="Inactive user")
            
        return {
            "access_token": auth_response.session.access_token,
            "refresh_token": auth_response.session.refresh_token,
            "token_type": "bearer",
            "user": user.to_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid credentials")

@router.post("/refresh")
async def refresh_token(refresh_token: str, db: Session = Depends(get_db)):
    """Refresh access token using Supabase."""
    try:
        # Refresh the session with Supabase
        auth_response = supabase.auth.refresh_session(refresh_token)
        
        if not auth_response.user:
            raise HTTPException(status_code=401, detail="Invalid refresh token")
            
        return {
            "access_token": auth_response.session.access_token,
            "refresh_token": auth_response.session.refresh_token,
            "token_type": "bearer"
        }
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

@router.post("/logout")
async def logout():
    """Logout from Supabase."""
    try:
        supabase.auth.sign_out()
        return {"message": "Successfully logged out"}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error during logout")

@router.post("/signup")
async def signup(email: str, password: str, db: Session = Depends(get_db)):
    """Sign up a new user with Supabase."""
    try:
        # Create user in Supabase
        auth_response = supabase.auth.sign_up({
            "email": email,
            "password": password
        })
        
        if not auth_response.user:
            raise HTTPException(status_code=400, detail="Error creating user")
            
        # Create user in our database
        user = User(
            id=auth_response.user.id,
            email=email,
            role="recruiter",  # Default role
            is_active=True
        )
        db.add(user)
        db.commit()
        
        return {
            "message": "User created successfully",
            "user": user.to_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error creating user")

@router.post("/password-reset-request")
async def password_reset_request(email: str):
    """Request a password reset email via Supabase."""
    try:
        supabase.auth.reset_password_email(email)
        return {"message": "Password reset email sent"}
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error sending password reset email")

@router.post("/password-reset-confirm")
async def password_reset_confirm(access_token: str, new_password: str):
    """Confirm password reset with token and new password via Supabase."""
    try:
        supabase.auth.update_user(access_token, {"password": new_password})
        return {"message": "Password has been reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error resetting password")

@router.post("/verify-email-request")
async def verify_email_request(email: str):
    """Trigger email verification via Supabase."""
    try:
        # Supabase automatically sends verification on signup, but you can resend
        supabase.auth.resend(email)
        return {"message": "Verification email sent"}
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error sending verification email")

@router.post("/verify-email-confirm")
async def verify_email_confirm(access_token: str):
    """Confirm email verification via Supabase."""
    try:
        # Supabase verifies email via magic link, but you can check status
        user = supabase.auth.get_user(access_token)
        if user and user.email_confirmed_at:
            return {"message": "Email verified successfully"}
        else:
            raise HTTPException(status_code=400, detail="Email not verified yet")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error verifying email") 