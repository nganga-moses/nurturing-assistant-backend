from fastapi import APIRouter, HTTPException, Depends
from api.services import RiskAssessmentService
from data.schemas.schemas import RiskAssessmentRequest, RiskAssessmentResponse

router = APIRouter(prefix="/risk", tags=["risk"])

risk_service = RiskAssessmentService()

def get_risk_service():
    """Get the risk assessment service instance."""
    return risk_service

@router.post("", response_model=RiskAssessmentResponse)
async def get_dropout_risk(
    request: RiskAssessmentRequest,
    service: RiskAssessmentService = Depends(get_risk_service)
):
    """
    Get the predicted risk of a student dropping off the application funnel.
    """
    try:
        risk = service.get_dropout_risk(request.student_id)
        return RiskAssessmentResponse(
            student_id=request.student_id,
            risk_category=risk["risk_category"],
            risk_score=risk["risk_score"]
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 