from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Boolean
from sqlalchemy.orm import relationship
from database.base import Base

class RecommendationAction(Base):
    """Model for tracking student interactions with recommendations."""
    __tablename__ = "recommendation_actions"

    id = Column(Integer, primary_key=True)
    student_id = Column(String, ForeignKey("student_profiles.student_id"))
    recommendation_id = Column(Integer, ForeignKey("stored_recommendations.id"))
    action_type = Column(String)  # e.g., "viewed", "acted", "completed"
    action_timestamp = Column(DateTime)
    time_to_action = Column(Integer)  # seconds between recommendation and action
    action_completed = Column(Integer, default=0)  # 0 or 1
    dropoff_point = Column(String, nullable=True)  # where student dropped off if applicable
    action_taken = Column(Boolean, default=False)
    action_date = Column(DateTime, nullable=True)

    student = relationship("StudentProfile", back_populates="recommendation_actions")
    recommendation = relationship("StoredRecommendation", back_populates="actions")

    def to_dict(self):
        return {
            "id": self.id,
            "student_id": self.student_id,
            "recommendation_id": self.recommendation_id,
            "action_type": self.action_type,
            "action_timestamp": self.action_timestamp.isoformat() if self.action_timestamp else None,
            "time_to_action": self.time_to_action,
            "action_completed": bool(self.action_completed),
            "dropoff_point": self.dropoff_point
        } 