import uuid
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd

class RecommendationLogger:
    """
    Logs recommendation metadata for reporting and traceability.
    Stores logs in a DataFrame (can be extended to DB or file).
    """
    def __init__(self):
        self.logs = []

    def log_recommendation(self, recommendation: Dict[str, Any], status: str = "active", outcome: Optional[str] = None, rationale: Optional[str] = None):
        log_entry = {
            "recommendation_id": str(uuid.uuid4()),
            "student_id": recommendation.get("student_id"),
            "engagement_type": recommendation.get("engagement_type"),
            "content_id": recommendation.get("content_id"),
            "features_used": recommendation.get("features_used"),
            "model_version": recommendation.get("model_version"),
            "created_at": datetime.utcnow().isoformat(),
            "status": status,
            "outcome": outcome,
            "rationale": rationale
        }
        self.logs.append(log_entry)
        return log_entry

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.logs)

    def save_to_csv(self, path: str):
        df = self.to_dataframe()
        df.to_csv(path, index=False) 