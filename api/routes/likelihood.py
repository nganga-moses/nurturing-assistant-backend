from fastapi import APIRouter, HTTPException, Depends, Request
from sqlalchemy.orm import Session
from api.services.likelihood_service import LikelihoodService
from api.services.model_manager import ModelManager
from data.schemas.schemas import LikelihoodRequest, LikelihoodResponse
from database.session import get_db

router = APIRouter(prefix="/likelihood", tags=["likelihood"])

def get_model_manager(request: Request) -> ModelManager:
    """Get the model manager from app state."""
    model_manager = getattr(request.app.state, 'model_manager', None)
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not available")
    return model_manager

def get_likelihood_service(request: Request) -> LikelihoodService:
    """Get the likelihood service instance with model manager."""
    model_manager = get_model_manager(request)
    return LikelihoodService(model_manager=model_manager)

@router.post("", response_model=LikelihoodResponse)
async def get_application_likelihood(
    request: LikelihoodRequest,
    db: Session = Depends(get_db),
    service: LikelihoodService = Depends(get_likelihood_service)
):
    """
    Get the predicted likelihood of application completion using trained AI model.
    
    This endpoint uses the hybrid recommender model to predict how likely
    a student is to complete their application process, optionally for
    a specific engagement activity.
    """
    try:
        likelihood = service.get_application_likelihood(
            db=db,
            student_id=request.student_id,
            engagement_id=getattr(request, 'engagement_id', None)
        )
        return LikelihoodResponse(
            student_id=request.student_id,
            likelihood=likelihood
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.get("/{student_id}", response_model=LikelihoodResponse)
async def get_student_likelihood(
    student_id: str,
    engagement_id: str = None,
    db: Session = Depends(get_db),
    service: LikelihoodService = Depends(get_likelihood_service)
):
    """
    Get application likelihood for a specific student.
    
    Args:
        student_id: The student's unique identifier
        engagement_id: Optional engagement activity identifier
    
    Returns:
        LikelihoodResponse with prediction percentage
    """
    try:
        likelihood = service.get_application_likelihood(
            db=db,
            student_id=student_id,
            engagement_id=engagement_id
        )
        return LikelihoodResponse(
            student_id=student_id,
            likelihood=likelihood
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}") 