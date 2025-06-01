from .base import Base
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime

class StatusChange(Base):
    __tablename__ = "status_changes"

    id = Column(Integer, primary_key=True)
    student_id = Column(String, ForeignKey("student_profiles.student_id"))
    field = Column(String)  # e.g., 'funnel_stage', 'application_status'
    old_value = Column(String)
    new_value = Column(String)
    batch_id = Column(String)  # ID of the batch update that caused this change
    timestamp = Column(DateTime, default=datetime.now)

    # Relationships
    student = relationship("StudentProfile")

    def to_dict(self):
        return {
            "id": self.id,
            "student_id": self.student_id,
            "field": self.field,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "batch_id": self.batch_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        } 