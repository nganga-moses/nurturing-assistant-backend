from .base import Base
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime

class NudgeAction(Base):
    """Model for tracking student interactions with nudges."""
    __tablename__ = "nudge_actions"
    
    id = Column(Integer, primary_key=True)
    student_id = Column(String, ForeignKey("student_profiles.student_id"))
    nudge_id = Column(Integer, ForeignKey("stored_recommendations.id"))
    action_type = Column(String)  # "acted", "ignored", "untouched"
    action_timestamp = Column(DateTime, default=datetime.now)
    time_to_action = Column(Integer)  # seconds between nudge and action
    action_completed = Column(Boolean, default=False)
    dropoff_point = Column(String, nullable=True)
    
    # Relationships
    student = relationship("StudentProfile", back_populates="nudge_actions")
    nudge = relationship("StoredRecommendation", back_populates="actions")

    def to_dict(self):
        return {
            "id": self.id,
            "student_id": self.student_id,
            "nudge_id": self.nudge_id,
            "action_type": self.action_type,
            "action_timestamp": self.action_timestamp.isoformat() if self.action_timestamp else None,
            "time_to_action": self.time_to_action,
            "action_completed": self.action_completed,
            "dropoff_point": self.dropoff_point
        } 