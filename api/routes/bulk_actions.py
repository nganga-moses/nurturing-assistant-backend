from fastapi import APIRouter, HTTPException, Depends
from api.services import BulkActionService
from data.schemas.schemas import BulkActionRequest, BulkActionPreviewResponse, BulkActionApplyResponse

router = APIRouter(prefix="/bulk-actions", tags=["bulk-actions"])

bulk_action_service = BulkActionService()

def get_bulk_action_service():
    """Get the bulk action service instance."""
    return bulk_action_service

@router.post("/preview", response_model=BulkActionPreviewResponse)
async def preview_bulk_action(
    request: BulkActionRequest,
    service: BulkActionService = Depends(get_bulk_action_service)
):
    """
    Preview a bulk action.
    """
    try:
        preview = service.preview_bulk_action(request.action, request.segment)
        return BulkActionPreviewResponse(
            students=preview["students"],
            count=preview["count"],
            action_details=preview["action_details"]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/apply", response_model=BulkActionApplyResponse)
async def apply_bulk_action(
    request: BulkActionRequest,
    service: BulkActionService = Depends(get_bulk_action_service)
):
    """
    Apply a bulk action.
    """
    try:
        result = service.apply_bulk_action(request.action, request.segment)
        return BulkActionApplyResponse(
            success=result["success"],
            students_affected=result["students_affected"],
            action_details=result["action_details"]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 