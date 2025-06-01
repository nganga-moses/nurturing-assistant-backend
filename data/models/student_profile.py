from .base import Base
from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime

class StudentProfile(Base):
    __tablename__ = "student_profiles"

    student_id = Column(String, primary_key=True)
    demographic_features = Column(JSON)
    application_status = Column(String)
    funnel_stage = Column(String)
    first_interaction_date = Column(DateTime)
    last_interaction_date = Column(DateTime)
    interaction_count = Column(Integer)
    application_likelihood_score = Column(Float)
    dropout_risk_score = Column(Float)
    last_recommended_engagement_id = Column(String, nullable=True)
    last_recommended_engagement_date = Column(DateTime, nullable=True)
    last_recommendation_at = Column(DateTime)
    enrollment_agent_id = Column(String, nullable=True)

    # Relationships
    engagements = relationship("EngagementHistory", back_populates="student")
    stored_recommendations = relationship("StoredRecommendation", back_populates="student")
    nudge_actions = relationship("NudgeAction", back_populates="student")

    def to_dict(self):
        return {
            "student_id": self.student_id,
            "demographic_features": self.demographic_features,
            "application_status": self.application_status,
            "funnel_stage": self.funnel_stage,
            "first_interaction_date": self.first_interaction_date.isoformat() if self.first_interaction_date else None,
            "last_interaction_date": self.last_interaction_date.isoformat() if self.last_interaction_date else None,
            "interaction_count": self.interaction_count,
            "application_likelihood_score": self.application_likelihood_score,
            "dropout_risk_score": self.dropout_risk_score,
            "last_recommended_engagement_id": self.last_recommended_engagement_id,
            "last_recommended_engagement_date": self.last_recommended_engagement_date.isoformat() if self.last_recommended_engagement_date else None,
            "last_recommendation_at": self.last_recommendation_at.isoformat() if self.last_recommendation_at else None
        } 