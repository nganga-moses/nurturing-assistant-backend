from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from database.base import Base
from datetime import datetime

class RecommendationFeedbackMetrics(Base):
    """Model for tracking aggregate feedback metrics for different types of recommendations."""
    __tablename__ = "recommendation_feedback_metrics"
    
    id = Column(Integer, primary_key=True)
    recommendation_type = Column(String)  # type of recommendation
    total_shown = Column(Integer, default=0)
    acted_count = Column(Integer, default=0)
    ignored_count = Column(Integer, default=0)
    untouched_count = Column(Integer, default=0)
    avg_time_to_action = Column(Float, default=0.0)
    avg_time_to_completion = Column(Float, default=0.0)
    completion_rate = Column(Float, default=0.0)
    satisfaction_score = Column(Float, default=0.0)
    dropoff_rates = Column(JSON, default=dict)
    last_updated = Column(DateTime, default=datetime.now)

    def to_dict(self):
        return {
            "id": self.id,
            "recommendation_type": self.recommendation_type,
            "total_shown": self.total_shown,
            "acted_count": self.acted_count,
            "ignored_count": self.ignored_count,
            "untouched_count": self.untouched_count,
            "avg_time_to_action": self.avg_time_to_action,
            "avg_time_to_completion": self.avg_time_to_completion,
            "completion_rate": self.completion_rate,
            "satisfaction_score": self.satisfaction_score,
            "dropoff_rates": self.dropoff_rates,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None
        } 