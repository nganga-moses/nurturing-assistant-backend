from database.base import Base
from sqlalchemy import Column, String, Float, JSON
from sqlalchemy.orm import relationship

class EngagementContent(Base):
    __tablename__ = "engagement_content"

    content_id = Column(String, primary_key=True)
    engagement_type = Column(String)
    content_category = Column(String)
    content_description = Column(String)
    content_features = Column(JSON)
    success_rate = Column(Float)
    target_funnel_stage = Column(String)
    appropriate_for_risk_level = Column(String)

    # Relationships
    engagements = relationship("EngagementHistory", back_populates="content")

    def to_dict(self):
        return {
            "content_id": self.content_id,
            "engagement_type": self.engagement_type,
            "content_category": self.content_category,
            "content_description": self.content_description,
            "content_features": self.content_features,
            "success_rate": self.success_rate,
            "target_funnel_stage": self.target_funnel_stage,
            "appropriate_for_risk_level": self.appropriate_for_risk_level
        } 