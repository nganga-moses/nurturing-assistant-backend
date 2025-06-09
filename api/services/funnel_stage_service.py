from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from data.models.funnel_stage import FunnelStage
from fastapi import HTTPException
import uuid
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

class FunnelStageService:
    def __init__(self, db: Session):
        self.db = db

    @lru_cache(maxsize=100)
    def get_stages_for_university(self) -> List[Dict]:
        """Get all active funnel stages, ordered by stage_order."""
        stages = (
            self.db.query(FunnelStage)
            .filter(
                FunnelStage.is_active == True
            )
            .order_by(FunnelStage.stage_order)
            .all()
        )
        return [{"id": str(s.id), "name": s.stage_name, "order": s.stage_order} for s in stages]

    def get_all_stages(self) -> List[FunnelStage]:
        """Get all funnel stages."""
        return self.db.query(FunnelStage).order_by(FunnelStage.stage_order).all()

    def get_stage_by_id(self, stage_id: str) -> Optional[FunnelStage]:
        """Get a funnel stage by ID."""
        return self.db.query(FunnelStage).filter(FunnelStage.id == stage_id).first()

    def create_stage(self, stage_data: dict) -> FunnelStage:
        """Create a new funnel stage."""
        stage = FunnelStage(**stage_data)
        self.db.add(stage)
        self.db.commit()
        self.db.refresh(stage)
        self.get_stages_for_university.cache_clear()  # Clear cache
        return stage

    def update_stage(self, stage_id: str, stage_data: dict) -> Optional[FunnelStage]:
        """Update a funnel stage."""
        stage = self.get_stage_by_id(stage_id)
        if stage:
            for key, value in stage_data.items():
                setattr(stage, key, value)
            self.db.commit()
            self.db.refresh(stage)
            self.get_stages_for_university.cache_clear()  # Clear cache
        return stage

    def delete_stage(self, stage_id: str) -> bool:
        """Delete a funnel stage."""
        stage = self.get_stage_by_id(stage_id)
        if stage:
            self.db.delete(stage)
            self.db.commit()
            self.get_stages_for_university.cache_clear()  # Clear cache
            return True
        return False

    def reorder_stages(self, stage_orders: List[Dict[str, int]]) -> bool:
        """Update the order of multiple stages at once."""
        try:
            for order_info in stage_orders:
                stage = self.db.query(FunnelStage).filter(
                    FunnelStage.id == order_info["stage_id"]
                ).first()
                if stage:
                    stage.stage_order = order_info["new_order"]
            
            self.db.commit()
            self.get_stages_for_university.cache_clear()  # Clear cache
            return True
        except Exception:
            self.db.rollback()
            return False

    def initialize_default_stages(self) -> List[FunnelStage]:
        """Initialize default funnel stages if none exist."""
        existing_stages = self.get_all_stages()
        if not existing_stages:
            default_stages = FunnelStage.get_default_stages()
            for stage_data in default_stages:
                self.create_stage(stage_data)
            return self.get_all_stages()
        return existing_stages 