from fastapi import APIRouter, HTTPException, Depends
from api.services import LikelihoodService
from data.schemas.schemas import LikelihoodRequest, LikelihoodResponse

router = APIRouter(prefix="/likelihood", tags=["likelihood"])

likelihood_service = LikelihoodService()

def get_likelihood_service():
    """Get the likelihood service instance."""
    return likelihood_service

@router.post("", response_model=LikelihoodResponse)
async def get_application_likelihood(
    request: LikelihoodRequest,
    service: LikelihoodService = Depends(get_likelihood_service)
):
    """
    Get the predicted likelihood of application completion.
    """
    try:
        likelihood = service.get_application_likelihood(request.student_id)
        return LikelihoodResponse(
            student_id=request.student_id,
            application_likelihood=likelihood
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 