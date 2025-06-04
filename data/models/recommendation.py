from database.base import Base
from sqlalchemy import Column, Integer, String, ForeignKey, Float
from sqlalchemy.orm import relationship

class Recommendation(Base):
    """Model for storing individual recommendations."""
    __tablename__ = "recommendations"
    
    id = Column(Integer, primary_key=True)
    student_id = Column(String, ForeignKey("student_profiles.student_id"))
    recommendation_type = Column(String)
    content_id = Column(String)
    confidence_score = Column(Float)
    
    # Relationship to student
    student = relationship("StudentProfile", back_populates="recommendations", lazy="joined") 