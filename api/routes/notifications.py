from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Optional, List

from ..dependencies import get_db
from data.models import ErrorLog

router = APIRouter(prefix="/notifications", tags=["notifications"])

@router.get("/")
def list_notifications(
    resolved: Optional[bool] = Query(None),
    error_type: Optional[str] = Query(None),
    days: int = Query(7, description="Number of days to look back"),
    db: Session = Depends(get_db)
):
    """List recent notifications (import failures, integration issues, etc.)."""
    since = datetime.now() - timedelta(days=days)
    query = db.query(ErrorLog).filter(ErrorLog.created_at >= since)
    if resolved is not None:
        query = query.filter(ErrorLog.resolved == resolved)
    if error_type:
        query = query.filter(ErrorLog.error_type == error_type)
    logs = query.order_by(ErrorLog.created_at.desc()).all()
    return {"notifications": [log.to_dict() for log in logs]}

@router.post("/mark-read/{log_id}")
def mark_notification_read(
    log_id: int,
    db: Session = Depends(get_db)
):
    """Mark a notification/error log as read/resolved."""
    log = db.query(ErrorLog).filter(ErrorLog.id == log_id).first()
    if not log:
        raise HTTPException(status_code=404, detail="Error log not found")
    log.resolved = True
    log.resolved_at = datetime.now()
    db.commit()
    return {"success": True, "log": log.to_dict()}

@router.get("/error-logs")
def list_error_logs(
    error_type: Optional[str] = Query(None),
    file_name: Optional[str] = Query(None),
    days: int = Query(30, description="Number of days to look back"),
    db: Session = Depends(get_db)
):
    """List error logs for retry/correction workflow."""
    since = datetime.now() - timedelta(days=days)
    query = db.query(ErrorLog).filter(ErrorLog.created_at >= since)
    if error_type:
        query = query.filter(ErrorLog.error_type == error_type)
    if file_name:
        query = query.filter(ErrorLog.file_name == file_name)
    logs = query.order_by(ErrorLog.created_at.desc()).all()
    return {"error_logs": [log.to_dict() for log in logs]}

@router.get("/error-logs/{log_id}")
def get_error_log_details(
    log_id: int,
    db: Session = Depends(get_db)
):
    """Get details for a specific error log."""
    log = db.query(ErrorLog).filter(ErrorLog.id == log_id).first()
    if not log:
        raise HTTPException(status_code=404, detail="Error log not found")
    return log.to_dict() 