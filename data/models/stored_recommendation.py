from database.base import Base
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime

class StoredRecommendation(Base):
    """Model for storing scheduled recommendations and their associated actions."""
    __tablename__ = "stored_recommendations"
    
    id = Column(Integer, primary_key=True)
    student_id = Column(String, ForeignKey("student_profiles.student_id"))
    recommendations = Column(JSON)  # List of recommendation objects
    generated_at = Column(DateTime, default=datetime.now)
    expires_at = Column(DateTime)
    
    # Relationship to student
    student = relationship("StudentProfile", back_populates="stored_recommendations")
    actions = relationship("RecommendationAction", back_populates="recommendation")

    def to_dict(self):
        return {
            "id": self.id,
            "student_id": self.student_id,
            "recommendations": self.recommendations,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None
        } 