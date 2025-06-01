from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional, List
from ..dependencies import get_db
from ..auth.jwt import get_current_user, check_admin, check_manager
from data.models.user import User, Role

router = APIRouter(prefix="/roles", tags=["roles"])

@router.get("/")
def list_roles(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all roles."""
    roles = db.query(Role).all()
    return {"roles": [r.to_dict() for r in roles]}

@router.get("/users/{role_name}")
def get_users_by_role(
    role_name: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(check_manager)
):
    """Get all users with a specific role."""
    users = db.query(User).filter(User.role == role_name).all()
    return {"users": [u.to_dict() for u in users]}

@router.post("/assign")
def assign_role(
    user_id: int = Query(...),
    role_name: str = Query(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(check_admin)
):
    """Assign a role to a user."""
    user = db.query(User).filter(User.id == user_id).first()
    role = db.query(Role).filter(Role.name == role_name).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if not role:
        raise HTTPException(status_code=404, detail="Role not found")
    user.role = role_name
    db.commit()
    return {"success": True, "user": user.to_dict()} 