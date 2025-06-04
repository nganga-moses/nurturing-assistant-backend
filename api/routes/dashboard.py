from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import Dict, Any
from api.services import DashboardService
from database.session import get_db

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

dashboard_service = DashboardService()

@router.get("/stats")
async def get_dashboard_stats(
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get statistics for the dashboard.
    """
    return dashboard_service.get_dashboard_stats(db) 