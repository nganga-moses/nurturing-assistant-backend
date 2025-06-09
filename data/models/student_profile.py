from database.base import Base
from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from datetime import datetime

class StudentProfile(Base):
    __tablename__ = "student_profiles"

    student_id = Column(String, primary_key=True)
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    birthdate = Column(DateTime, nullable=True)
    recruiter_id = Column(String, nullable=True)
    demographic_features = Column(JSONB)
    application_status = Column(String)
    funnel_stage = Column(String)
    current_stage_id = Column(String(36), ForeignKey("funnel_stages.id"), nullable=True)
    first_interaction_date = Column(DateTime)
    last_interaction_date = Column(DateTime)
    interaction_count = Column(Integer)
    application_likelihood_score = Column(Float)
    dropout_risk_score = Column(Float)
    last_recommended_engagement_id = Column(String)
    last_recommended_engagement_date = Column(DateTime)
    last_recommendation_at = Column(DateTime)
    enrollment_agent_id = Column(String)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    gpa = Column(Float)
    sat_score = Column(Float)
    act_score = Column(Float)
    is_successful = Column(Boolean, default=False)

    # Relationships
    stored_recommendations = relationship("StoredRecommendation", back_populates="student")
    recommendation_actions = relationship("RecommendationAction", back_populates="student")
    status_changes = relationship("StatusChange", back_populates="student")
    engagements = relationship("EngagementHistory", back_populates="student")
    recommendations = relationship("Recommendation", back_populates="student", lazy="dynamic")
    current_stage = relationship("FunnelStage")

    def to_dict(self):
        return {
            "student_id": self.student_id,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "birthdate": self.birthdate.isoformat() if self.birthdate else None,
            "recruiter_id": self.recruiter_id or self.enrollment_agent_id,
            "demographic_features": self.demographic_features,
            "application_status": self.application_status,
            "funnel_stage": self.funnel_stage,
            "current_stage_id": self.current_stage_id,
            "first_interaction_date": self.first_interaction_date.isoformat() if self.first_interaction_date else None,
            "last_interaction_date": self.last_interaction_date.isoformat() if self.last_interaction_date else None,
            "interaction_count": self.interaction_count,
            "application_likelihood_score": self.application_likelihood_score,
            "dropout_risk_score": self.dropout_risk_score,
            "last_recommended_engagement_id": self.last_recommended_engagement_id,
            "last_recommended_engagement_date": self.last_recommended_engagement_date.isoformat() if self.last_recommended_engagement_date else None,
            "last_recommendation_at": self.last_recommendation_at.isoformat() if self.last_recommendation_at else None,
            "enrollment_agent_id": self.enrollment_agent_id,
            "gpa": self.gpa,
            "sat_score": self.sat_score,
            "act_score": self.act_score
        } 