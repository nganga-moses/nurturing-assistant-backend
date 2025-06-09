from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from database.session import get_db
from api.services.funnel_stage_service import FunnelStageService
from data.models.funnel_stage import FunnelStage
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/funnel-stages", response_model=List[dict])
def get_funnel_stages(db: Session = Depends(get_db)):
    """Get all funnel stages."""
    service = FunnelStageService(db)
    return service.get_all_stages()

@router.post("/funnel-stages", response_model=dict)
def create_funnel_stage(stage_data: dict, db: Session = Depends(get_db)):
    """Create a new funnel stage."""
    service = FunnelStageService(db)
    stage = service.create_stage(stage_data)
    return stage

@router.put("/funnel-stages/{stage_id}", response_model=dict)
def update_funnel_stage(stage_id: str, stage_data: dict, db: Session = Depends(get_db)):
    """Update a funnel stage."""
    service = FunnelStageService(db)
    stage = service.update_stage(stage_id, stage_data)
    if not stage:
        raise HTTPException(status_code=404, detail="Funnel stage not found")
    return stage

@router.delete("/funnel-stages/{stage_id}")
def delete_funnel_stage(stage_id: str, db: Session = Depends(get_db)):
    """Delete a funnel stage."""
    service = FunnelStageService(db)
    if not service.delete_stage(stage_id):
        raise HTTPException(status_code=404, detail="Funnel stage not found")
    return {"message": "Funnel stage deleted successfully"}

@router.post("/funnel-stages/initialize")
def initialize_funnel_stages(db: Session = Depends(get_db)):
    """Initialize default funnel stages."""
    service = FunnelStageService(db)
    stages = service.initialize_default_stages()
    return {"message": "Default funnel stages initialized", "stages": stages} 