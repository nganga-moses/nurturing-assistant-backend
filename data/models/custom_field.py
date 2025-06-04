from database.base import Base
from sqlalchemy import Column, Integer, String, Boolean, DateTime
from datetime import datetime

class CustomField(Base):
    __tablename__ = "custom_fields"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)  # e.g., 'Parent Contacted'
    applies_to = Column(String, nullable=False)  # 'student', 'engagement', etc.
    field_type = Column(String, nullable=False)  # 'string', 'date', 'bool', etc.
    is_required = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now) 