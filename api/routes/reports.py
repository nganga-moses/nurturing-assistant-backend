from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Optional, List

from ..dependencies import get_db
from data.models import (
    EngagementHistory, NudgeAction, NudgeFeedbackMetrics, StudentProfile, RecommendationSettings, StatusChange
)

router = APIRouter(prefix="/reports", tags=["reports"])

@router.get("/nudge-performance")
def nudge_performance(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    nudge_type: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Get summary statistics for nudge/recommendation performance."""
    query = db.query(NudgeFeedbackMetrics)
    if nudge_type:
        query = query.filter(NudgeFeedbackMetrics.nudge_type == nudge_type)
    metrics = query.all()
    return {"metrics": [m.to_dict() for m in metrics]}

@router.get("/staff-performance")
def staff_performance(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    staff_id: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Get performance metrics for staff/recruiters (by enrollment_agent_id)."""
    query = db.query(StudentProfile)
    if staff_id:
        query = query.filter(StudentProfile.enrollment_agent_id == staff_id)
    students = query.all()
    # Aggregate by staff
    staff_stats = {}
    for student in students:
        agent_id = student.enrollment_agent_id or "unassigned"
        if agent_id not in staff_stats:
            staff_stats[agent_id] = {"students": 0, "at_risk": 0, "applications_completed": 0}
        staff_stats[agent_id]["students"] += 1
        if student.dropout_risk_score and student.dropout_risk_score > 0.7:
            staff_stats[agent_id]["at_risk"] += 1
        if student.application_status and student.application_status.lower() == "completed":
            staff_stats[agent_id]["applications_completed"] += 1
    return {"staff_performance": staff_stats}

@router.get("/unmatched-engagements")
def unmatched_engagements(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    db: Session = Depends(get_db)
):
    """Get a list of unmatched engagements (not linked to any nudge)."""
    query = db.query(EngagementHistory).outerjoin(
        NudgeAction, EngagementHistory.engagement_id == NudgeAction.nudge_id
    ).filter(NudgeAction.id.is_(None))
    if start_date:
        query = query.filter(EngagementHistory.timestamp >= start_date)
    if end_date:
        query = query.filter(EngagementHistory.timestamp <= end_date)
    engagements = query.all()
    return {"count": len(engagements), "engagements": [e.to_dict() for e in engagements]}

@router.get("/import-history")
def import_history(
    days: int = Query(30, description="Number of days to look back"),
    db: Session = Depends(get_db)
):
    """Get a summary of import history and status changes."""
    since = datetime.now() - timedelta(days=days)
    changes = db.query(StatusChange).filter(StatusChange.timestamp >= since).all()
    return {"import_history": [c.to_dict() for c in changes]} 