from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from database.session import get_db
from api.services.goal_analytics_service import GoalAnalyticsService
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/goal-analytics/metrics")
def get_goal_metrics(db: Session = Depends(get_db)):
    """Get goal metrics for all stages."""
    service = GoalAnalyticsService(db)
    return service.get_goal_metrics()

@router.get("/goal-analytics/stage/{stage_id}")
def get_stage_metrics(stage_id: str, db: Session = Depends(get_db)):
    """Get goal metrics for a specific stage."""
    service = GoalAnalyticsService(db)
    return service.get_stage_metrics(stage_id) 