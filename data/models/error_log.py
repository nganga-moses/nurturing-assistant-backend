from database.base import Base
from sqlalchemy import Column, Integer, String, Boolean, DateTime
from datetime import datetime

class ErrorLog(Base):
    __tablename__ = "error_logs"
    id = Column(Integer, primary_key=True)
    error_type = Column(String, nullable=False)  # e.g., 'import', 'integration', etc.
    details = Column(String, nullable=False)
    row_number = Column(Integer, nullable=True)
    file_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime, nullable=True) 