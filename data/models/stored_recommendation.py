from .base import Base
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime

class StoredRecommendation(Base):
    """Model for storing scheduled recommendations."""
    __tablename__ = "stored_recommendations"
    
    id = Column(Integer, primary_key=True)
    student_id = Column(String, ForeignKey("student_profiles.student_id"))
    recommendations = Column(JSON)  # List of recommendation objects
    generated_at = Column(DateTime, default=datetime.now)
    expires_at = Column(DateTime)
    
    # Relationship to student
    student = relationship("StudentProfile", back_populates="stored_recommendations")
    actions = relationship("NudgeAction", back_populates="nudge") 