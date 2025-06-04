from fastapi import APIRouter, HTTPException, Depends
from api.services import RecommendationSettingsService
from data.schemas.schemas import RecommendationSettingsRequest, RecommendationSettingsResponse

router = APIRouter(prefix="/recommendation-settings", tags=["recommendation-settings"])

settings_service = RecommendationSettingsService()

def get_settings_service():
    """Get the recommendation settings service instance."""
    return settings_service

@router.get("", response_model=RecommendationSettingsResponse)
async def get_recommendation_settings(
    service: RecommendationSettingsService = Depends(get_settings_service)
):
    """
    Get current recommendation settings.
    """
    try:
        settings = service.get_settings()
        return RecommendationSettingsResponse(**settings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("", response_model=RecommendationSettingsResponse)
async def update_recommendation_settings(
    request: RecommendationSettingsRequest,
    service: RecommendationSettingsService = Depends(get_settings_service)
):
    """
    Update recommendation settings.
    """
    try:
        settings = service.update_settings(request.dict(exclude_unset=True))
        return RecommendationSettingsResponse(**settings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 