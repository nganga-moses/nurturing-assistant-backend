from .base import Base
from sqlalchemy import Column, String, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship

class EngagementHistory(Base):
    __tablename__ = "engagement_history"

    engagement_id = Column(String, primary_key=True)
    student_id = Column(String, ForeignKey("student_profiles.student_id"))
    engagement_type = Column(String)
    engagement_content_id = Column(String, ForeignKey("engagement_content.content_id"))
    timestamp = Column(DateTime)
    engagement_response = Column(String)
    engagement_metrics = Column(JSON)
    funnel_stage_before = Column(String)
    funnel_stage_after = Column(String)

    # Relationships
    student = relationship("StudentProfile", back_populates="engagements")
    content = relationship("EngagementContent", back_populates="engagements")

    def to_dict(self):
        return {
            "engagement_id": self.engagement_id,
            "student_id": self.student_id,
            "engagement_type": self.engagement_type,
            "engagement_content_id": self.engagement_content_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "engagement_response": self.engagement_response,
            "engagement_metrics": self.engagement_metrics,
            "funnel_stage_before": self.funnel_stage_before,
            "funnel_stage_after": self.funnel_stage_after
        } 