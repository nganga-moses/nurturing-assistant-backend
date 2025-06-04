from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session
from api.services import RecommendationService, RecommendationTrackingService
from data.schemas.schemas import RecommendationResponse
from database.session import get_db
from api.auth.supabase import get_current_user_id

router = APIRouter(prefix="/recommendations", tags=["recommendations"])

recommendation_service = RecommendationService()
tracking_service = RecommendationTrackingService()

def get_recommendation_service():
    """Get the recommendation service instance."""
    return recommendation_service

def get_tracking_service():
    """Get the recommendation tracking service instance."""
    return tracking_service

# Recommendation endpoints
@router.get("", response_model=RecommendationResponse)
async def get_recommendations(
    student_id: str = Query(..., description="Student ID"),
    top_k: Optional[int] = Query(5, description="Number of recommendations to return"),
    funnel_stage: Optional[str] = Query(None, description="Filter by funnel stage"),
    risk_level: Optional[str] = Query(None, description="Filter by risk level"),
    service: RecommendationService = Depends(get_recommendation_service)
):
    """
    Get personalized engagement recommendations for a student.
    """
    try:
        recommendations = service.get_recommendations(
            student_id=student_id,
            top_k=top_k,
            funnel_stage=funnel_stage,
            risk_level=risk_level
        )
        return RecommendationResponse(
            student_id=student_id,
            recommendations=recommendations
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/me", response_model=List[Dict[str, Any]])
async def get_recommendations_for_current_user(
    current_user_id: str = Depends(get_current_user_id),
    service: RecommendationService = Depends(get_recommendation_service)
):
    """
    Get recommendations for the current user (recruiter).
    """
    try:
        recommendations = service.get_recommendations_for_current_user(current_user_id)
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Recommendation tracking endpoints
@router.post("/{recommendation_id}/track")
def track_recommendation_action(
    recommendation_id: int,
    action: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """
    Track a student's action on a recommendation.
    """
    tracking_service.track_recommendation_action(
        student_id=action["student_id"],
        recommendation_id=recommendation_id,
        action_type=action["action_type"]
    )
    return {"status": "success"}

@router.post("/{recommendation_id}/complete")
def track_recommendation_completion(
    recommendation_id: int,
    completion: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """
    Track whether a student completed the suggested action.
    """
    tracking_service.track_completion(
        student_id=completion["student_id"],
        recommendation_id=recommendation_id,
        completed=completion["completed"],
        dropoff_point=completion.get("dropoff_point")
    )
    return {"status": "success"}

@router.get("/feedback/metrics")
def get_feedback_metrics(
    recommendation_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Get feedback metrics for recommendation types.
    """
    return tracking_service.get_feedback_metrics(recommendation_type)

@router.get("/students/{student_id}/actions")
def get_student_actions(
    student_id: str,
    db: Session = Depends(get_db)
):
    """
    Get all actions for a specific student.
    """
    return tracking_service.get_student_actions(student_id) 