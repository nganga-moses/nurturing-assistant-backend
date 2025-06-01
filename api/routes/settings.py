from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional, List
from ..dependencies import get_db
from ..auth.jwt import get_current_user, check_admin, check_manager
from data.models.settings import Settings
from data.models.user import User

router = APIRouter(prefix="/settings", tags=["settings"])

@router.get("/")
def get_all_settings(
    db: Session = Depends(get_db),
    current_user: User = Depends(check_manager)
):
    """Get all global settings."""
    settings = db.query(Settings).filter(Settings.user_id == None).all()
    return {"settings": [s.to_dict() for s in settings]}

@router.get("/user/{user_id}")
def get_user_settings(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get settings for a specific user."""
    # Users can only view their own settings unless they're admin/manager
    if current_user.id != user_id and current_user.role not in ["admin", "manager"]:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    settings = db.query(Settings).filter(Settings.user_id == user_id).all()
    return {"settings": [s.to_dict() for s in settings]}

@router.post("/user/{user_id}")
def set_user_setting(
    user_id: int,
    key: str = Query(...),
    value: str = Query(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Set or update a user setting."""
    # Users can only modify their own settings unless they're admin
    if current_user.id != user_id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not enough permissions")
    setting = db.query(Settings).filter(Settings.user_id == user_id, Settings.key == key).first()
    if setting:
        setting.value = value
    else:
        setting = Settings(user_id=user_id, key=key, value=value)
        db.add(setting)
    db.commit()
    return {"success": True, "setting": setting.to_dict()}

@router.post("/global")
def set_global_setting(
    key: str = Query(...),
    value: str = Query(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(check_admin)
):
    """Set or update a global (system-wide) setting."""
    setting = db.query(Settings).filter(Settings.user_id == None, Settings.key == key).first()
    if setting:
        setting.value = value
    else:
        setting = Settings(user_id=None, key=key, value=value)
        db.add(setting)
    db.commit()
    return {"success": True, "setting": setting.to_dict()} 