from sqlalchemy import Column, String, Integer, Boolean, DateTime, ForeignKey, Enum
from sqlalchemy.sql import func
import uuid
from database.base import Base
import enum

class TrackingGoalType(str, enum.Enum):
    APPLICATION = "application"
    DEPOSIT = "deposit"
    ENROLLMENT = "enrollment"
    MATRICULATION = "matriculation"
    CUSTOM = "custom"

class FunnelStage(Base):
    __tablename__ = "funnel_stages"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    stage_name = Column(String(100), nullable=False)
    stage_order = Column(Integer, nullable=False)
    is_active = Column(Boolean, default=True)
    is_tracking_goal = Column(Boolean, default=False)
    tracking_goal_type = Column(Enum(TrackingGoalType), nullable=True)
    tracking_goal_description = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<FunnelStage {self.stage_name}>"

    @classmethod
    def get_default_stages(cls):
        """Get default funnel stages for new universities."""
        return [
            {
                "stage_name": "Awareness",
                "stage_order": 0,
                "is_tracking_goal": False,
                "tracking_goal_type": None
            },
            {
                "stage_name": "Interest",
                "stage_order": 1,
                "is_tracking_goal": False,
                "tracking_goal_type": None
            },
            {
                "stage_name": "Consideration",
                "stage_order": 2,
                "is_tracking_goal": False,
                "tracking_goal_type": None
            },
            {
                "stage_name": "Application",
                "stage_order": 3,
                "is_tracking_goal": True,
                "tracking_goal_type": TrackingGoalType.APPLICATION
            },
            {
                "stage_name": "Enrollment",
                "stage_order": 4,
                "is_tracking_goal": False,
                "tracking_goal_type": None
            }
        ] 