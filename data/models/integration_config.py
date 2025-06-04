from database.base import Base
from sqlalchemy import Column, Integer, String, Boolean, DateTime, JSON
from datetime import datetime

class IntegrationConfig(Base):
    __tablename__ = "integration_configs"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)  # e.g., 'Slate', 'Ellucian', 'Google Calendar'
    type = Column(String, nullable=False)  # 'CRM', 'Email', 'Calendar', etc.
    config = Column(JSON, default=dict)  # API keys, URLs, field mappings, etc.
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now) 