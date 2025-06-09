from fastapi import APIRouter, HTTPException, Depends, Request
from sqlalchemy.orm import Session
from api.services.risk_assessment_service import RiskAssessmentService
from api.services.model_manager import ModelManager
from data.schemas.schemas import RiskAssessmentRequest, RiskAssessmentResponse
from database.session import get_db

router = APIRouter(prefix="/risk", tags=["risk"])

def get_model_manager(request: Request) -> ModelManager:
    """Get the model manager from app state."""
    model_manager = getattr(request.app.state, 'model_manager', None)
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not available")
    return model_manager

def get_risk_service(request: Request) -> RiskAssessmentService:
    """Get the risk assessment service instance with model manager."""
    model_manager = get_model_manager(request)
    return RiskAssessmentService(model_manager=model_manager)

@router.post("", response_model=RiskAssessmentResponse)
async def get_dropout_risk(
    request: RiskAssessmentRequest,
    db: Session = Depends(get_db),
    service: RiskAssessmentService = Depends(get_risk_service)
):
    """
    Get the predicted risk of a student dropping off the application funnel using trained AI model.
    
    This endpoint uses the hybrid recommender model to predict dropout risk,
    with intelligent fallback to heuristic calculations based on engagement
    patterns and funnel stage.
    """
    try:
        risk = service.get_dropout_risk(db=db, student_id=request.student_id)
        return RiskAssessmentResponse(
            student_id=request.student_id,
            risk_category=risk["risk_category"],
            risk_score=risk["risk_score"]
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk prediction failed: {str(e)}")

@router.get("/{student_id}", response_model=RiskAssessmentResponse)
async def get_student_risk(
    student_id: str,
    db: Session = Depends(get_db),
    service: RiskAssessmentService = Depends(get_risk_service)
):
    """
    Get dropout risk assessment for a specific student.
    
    Args:
        student_id: The student's unique identifier
    
    Returns:
        RiskAssessmentResponse with risk score, category, and confidence
    """
    try:
        risk = service.get_dropout_risk(db=db, student_id=student_id)
        return RiskAssessmentResponse(
            student_id=student_id,
            risk_category=risk["risk_category"],
            risk_score=risk["risk_score"]
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk prediction failed: {str(e)}") 