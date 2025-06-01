from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
from typing import List, Optional, Dict

from ..dependencies import get_db
from ..services.matching_service import MatchingService
from data.models import EngagementHistory, NudgeFeedbackMetrics

router = APIRouter(prefix="/matching", tags=["matching"])

@router.post("/match/{engagement_id}")
async def match_engagement(
    engagement_id: str,
    db: Session = Depends(get_db)
):
    """Attempt to match an engagement to a nudge."""
    engagement = db.query(EngagementHistory).filter(EngagementHistory.engagement_id == engagement_id).first()
    if not engagement:
        raise HTTPException(status_code=404, detail="Engagement not found")

    matching_service = MatchingService(db)
    matched, confidence = matching_service.match_engagement_to_nudge(engagement)
    
    return {
        "matched": matched,
        "confidence": confidence,
        "engagement_id": engagement_id,
        "student_id": engagement.student_id
    }

@router.get("/unmatched")
async def get_unmatched_engagements(
    start_date: Optional[datetime] = None,
    db: Session = Depends(get_db)
):
    """Get list of unmatched engagements."""
    matching_service = MatchingService(db)
    unmatched = matching_service.get_unmatched_engagements(start_date)
    
    return {
        "count": len(unmatched),
        "engagements": [e.to_dict() for e in unmatched]
    }

@router.get("/metrics")
async def get_feedback_metrics(
    nudge_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get feedback metrics for nudge types."""
    matching_service = MatchingService(db)
    metrics = matching_service.get_feedback_metrics(nudge_type)
    
    return {
        "metrics": [m.to_dict() for m in metrics]
    }

@router.get("/metrics/{nudge_type}")
async def get_feedback_metrics_by_type(
    nudge_type: str,
    db: Session = Depends(get_db)
):
    """Get feedback metrics for a specific nudge type."""
    matching_service = MatchingService(db)
    metrics = matching_service.get_feedback_metrics(nudge_type)
    
    if not metrics:
        raise HTTPException(status_code=404, detail=f"No metrics found for nudge type: {nudge_type}")
    
    return metrics[0].to_dict()

@router.get("/quality")
async def get_match_quality_stats(
    nudge_type: Optional[str] = None,
    db: Session = Depends(get_db)
) -> Dict[str, float]:
    """Get statistics about match quality."""
    matching_service = MatchingService(db)
    return matching_service.get_match_quality_stats(nudge_type) 