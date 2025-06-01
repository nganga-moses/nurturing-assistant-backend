from .base import Base
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from datetime import datetime

class NudgeFeedbackMetrics(Base):
    """Model for tracking aggregate feedback metrics for different types of nudges."""
    __tablename__ = "nudge_feedback_metrics"
    
    id = Column(Integer, primary_key=True)
    nudge_type = Column(String)  # type of recommendation
    total_shown = Column(Integer, default=0)
    acted_count = Column(Integer, default=0)
    ignored_count = Column(Integer, default=0)
    untouched_count = Column(Integer, default=0)
    avg_time_to_action = Column(Float, default=0.0)
    completion_rate = Column(Float, default=0.0)
    dropoff_rates = Column(JSON, default=dict)
    last_updated = Column(DateTime, default=datetime.now)

    def to_dict(self):
        return {
            "id": self.id,
            "nudge_type": self.nudge_type,
            "total_shown": self.total_shown,
            "acted_count": self.acted_count,
            "ignored_count": self.ignored_count,
            "untouched_count": self.untouched_count,
            "avg_time_to_action": self.avg_time_to_action,
            "completion_rate": self.completion_rate,
            "dropoff_rates": self.dropoff_rates,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None
        } 