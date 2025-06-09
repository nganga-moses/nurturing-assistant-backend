from fastapi import APIRouter, HTTPException, Depends, Query, Request
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session
from api.services.recommendation_service import RecommendationService
from api.services.model_manager import ModelManager
from data.schemas.schemas import RecommendationResponse
from database.session import get_db
from api.auth.supabase import get_current_user_id

router = APIRouter(prefix="/recommendations", tags=["recommendations"])

def get_model_manager(request: Request) -> ModelManager:
    """Get the model manager from app state."""
    model_manager = getattr(request.app.state, 'model_manager', None)
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not available")
    return model_manager

def get_recommendation_service(request: Request) -> RecommendationService:
    """Get the recommendation service instance with model manager."""
    model_manager = get_model_manager(request)
    return RecommendationService(model_manager=model_manager)

# Recommendation endpoints
@router.get("", response_model=RecommendationResponse)
async def get_recommendations(
    student_id: str = Query(..., description="Student ID"),
    top_k: Optional[int] = Query(5, description="Number of recommendations to return"),
    funnel_stage: Optional[str] = Query(None, description="Filter by funnel stage"),
    risk_level: Optional[str] = Query(None, description="Filter by risk level"),
    db: Session = Depends(get_db),
    service: RecommendationService = Depends(get_recommendation_service)
):
    """
    Get personalized engagement recommendations for a student using trained AI model.
    
    This endpoint uses the hybrid recommender model to generate personalized
    engagement recommendations based on student preferences and similarity patterns.
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
        raise HTTPException(status_code=500, detail=f"Recommendation generation failed: {str(e)}")

@router.get("/{student_id}", response_model=RecommendationResponse)
async def get_student_recommendations(
    student_id: str,
    top_k: Optional[int] = Query(5, description="Number of recommendations to return"),
    funnel_stage: Optional[str] = Query(None, description="Filter by funnel stage"),
    risk_level: Optional[str] = Query(None, description="Filter by risk level"),
    db: Session = Depends(get_db),
    service: RecommendationService = Depends(get_recommendation_service)
):
    """
    Get personalized recommendations for a specific student.
    
    Args:
        student_id: The student's unique identifier
        top_k: Number of recommendations to return (default: 5)
        funnel_stage: Optional filter by funnel stage
        risk_level: Optional filter by risk level
    
    Returns:
        RecommendationResponse with personalized recommendations
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
        raise HTTPException(status_code=500, detail=f"Recommendation generation failed: {str(e)}")

@router.get("/me", response_model=List[Dict[str, Any]])
async def get_recommendations_for_current_user(
    current_user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
    service: RecommendationService = Depends(get_recommendation_service)
):
    """
    Get recommendations for the current user (recruiter).
    """
    try:
        recommendations = service.get_recommendations_for_current_user(db, current_user_id)
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get user recommendations: {str(e)}")

# Recommendation tracking endpoints
@router.post("/{recommendation_id}/track")
async def track_recommendation_action(
    recommendation_id: int,
    action: Dict[str, Any],
    db: Session = Depends(get_db),
    service: RecommendationService = Depends(get_recommendation_service)
):
    """
    Track a student's action on a recommendation.
    """
    try:
        service.track_recommendation_action(
            db=db,
            student_id=action["student_id"],
            recommendation_id=recommendation_id
        )
        return {"status": "success", "message": "Action tracked successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to track action: {str(e)}")

@router.post("/{recommendation_id}/complete")
async def track_recommendation_completion(
    recommendation_id: int,
    completion: Dict[str, Any],
    db: Session = Depends(get_db),
    service: RecommendationService = Depends(get_recommendation_service)
):
    """
    Track whether a student completed the suggested action.
    """
    try:
        # Note: This would need a tracking service with completion tracking
        # For now, we'll use the basic tracking
        service.track_recommendation_action(
            db=db,
            student_id=completion["student_id"],
            recommendation_id=recommendation_id
        )
        return {"status": "success", "message": "Completion tracked successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to track completion: {str(e)}")

@router.get("/feedback/metrics")
async def get_feedback_metrics(
    recommendation_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Get feedback metrics for recommendation types.
    """
    try:
        # This would need implementation in the tracking service
        # For now, return basic metrics
        return {
            "recommendation_type": recommendation_type or "all",
            "total_recommendations": 0,
            "engagement_rate": 0.0,
            "completion_rate": 0.0,
            "satisfaction_score": 0.0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

@router.get("/students/{student_id}/actions")
async def get_student_actions(
    student_id: str,
    db: Session = Depends(get_db)
):
    """
    Get all actions for a specific student.
    """
    try:
        # This would need implementation in the tracking service
        # For now, return empty list
        return []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get student actions: {str(e)}") 