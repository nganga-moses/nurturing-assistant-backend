from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from database.session import get_db
from data.models import EngagementHistory, RecommendationFeedbackMetrics
from api.services import MatchingService

router = APIRouter(prefix="/matching", tags=["matching"])

@router.post("/match")
async def match_engagement(engagement: EngagementHistory):
    """Attempt to match an engagement to a recommendation."""
    matched, confidence = matching_service.match_engagement_to_recommendation(engagement)
    return {"matched": matched, "confidence": confidence}

@router.get("/unmatched")
def get_unmatched_engagements(
    start_date: datetime = None,
    end_date: datetime = None,
    db: Session = Depends(get_db)
):
    """Get list of engagements that haven't been matched to recommendations."""
    if not start_date:
        start_date = datetime.now() - timedelta(days=30)
    if not end_date:
        end_date = datetime.now()

    service = MatchingService(db)
    return service.get_students_without_actions()

@router.get("/metrics")
async def get_feedback_metrics(
    recommendation_type: Optional[str] = None,
):
    """Get feedback metrics for recommendation types."""
    metrics = matching_service.get_feedback_metrics(recommendation_type)
    return metrics

@router.get("/metrics/{recommendation_type}")
async def get_feedback_metrics_by_type(
    recommendation_type: str,
):
    """Get feedback metrics for a specific recommendation type."""
    metrics = matching_service.get_feedback_metrics(recommendation_type)
    if not metrics:
        raise HTTPException(status_code=404, detail=f"No metrics found for recommendation type: {recommendation_type}")
    return metrics

@router.get("/quality")
def get_matching_quality(
    start_date: datetime = None,
    end_date: datetime = None,
    db: Session = Depends(get_db)
):
    """Get matching quality metrics."""
    if not start_date:
        start_date = datetime.now() - timedelta(days=30)
    if not end_date:
        end_date = datetime.now()

    service = MatchingService(db)
    return service.get_recommendation_effectiveness() 