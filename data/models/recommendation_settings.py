from database.base import Base
from sqlalchemy import Column, Integer, String, DateTime, Boolean
from datetime import datetime

class RecommendationSettings(Base):
    """Model for storing recommendation generation settings."""
    __tablename__ = "recommendation_settings"
    
    id = Column(Integer, primary_key=True)
    mode = Column(String, default="scheduled")  # "scheduled" or "realtime"
    schedule_type = Column(String, nullable=True)  # "daily" or "weekly"
    schedule_day = Column(String, nullable=True)  # For weekly: "monday", "tuesday", etc.
    schedule_time = Column(String, default="00:00")  # 24-hour format "HH:MM"
    batch_size = Column(Integer, default=1000)
    last_run = Column(DateTime, nullable=True)
    next_run = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    def to_dict(self):
        return {
            "id": self.id,
            "mode": self.mode,
            "schedule_type": self.schedule_type,
            "schedule_day": self.schedule_day,
            "schedule_time": self.schedule_time,
            "batch_size": self.batch_size,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "next_run": self.next_run.isoformat() if self.next_run else None,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        } 