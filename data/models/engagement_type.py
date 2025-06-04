from database.base import Base
from sqlalchemy import Column, Integer, String, Boolean, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime

class EngagementType(Base):
    __tablename__ = "engagement_types"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)  # e.g., 'Send Email', 'Schedule Meeting'
    description = Column(String)
    required_fields = Column(JSON, default=list)  # List of required fields for this type
    integration_id = Column(Integer, ForeignKey("integration_configs.id"), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    integration = relationship("IntegrationConfig") 