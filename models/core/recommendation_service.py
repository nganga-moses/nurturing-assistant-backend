import os
import sys
import numpy as np
import joblib
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy.orm import Session
import uuid

# Add the parent directory to the path so we can import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.models.student_profile import StudentProfile
from data.models.engagement_history import EngagementHistory
from data.models.engagement_content import EngagementContent
from database.session import get_db

class RecommendationService:
    """Service for generating recommendations using the database and fallback logic."""
    
    def __init__(self, model_dir: str = None):
        """
        Initialize the recommendation service.
        
        Args:
            model_dir: Directory containing the trained model (unused)
        """
        self.model_dir = model_dir
        self.content_cache = {}
        self.last_update = None
        self.update_threshold = timedelta(hours=1)  # Update model if last update was more than 1 hour ago
        self.logs = []
    
    def log_recommendation(self, recommendation: Dict[str, Any], status: str = "active", 
                         outcome: Optional[str] = None, rationale: Optional[str] = None) -> Dict[str, Any]:
        """
        Log recommendation metadata for reporting and traceability.
        
        Args:
            recommendation: The recommendation to log
            status: Current status of the recommendation
            outcome: Outcome of the recommendation (if known)
            rationale: Rationale for the recommendation
            
        Returns:
            Log entry dictionary
        """
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
    
    def get_recommendation_logs(self) -> pd.DataFrame:
        """Get all recommendation logs as a DataFrame."""
        return pd.DataFrame(self.logs)
    
    def save_logs_to_csv(self, path: str) -> None:
        """Save recommendation logs to a CSV file."""
        df = self.get_recommendation_logs()
        df.to_csv(path, index=False)
    
    def get_recommendations(self, student_id: str, count: int = 3) -> List[Dict[str, Any]]:
        session = next(get_db())
        try:
            student = session.query(StudentProfile).filter_by(student_id=student_id).first()
            if not student:
                print(f"Student with ID {student_id} not found in database")
                return self._get_default_recommendations(count)
            
            # Get recent engagements
            recent_engagements = session.query(EngagementHistory).filter(
                EngagementHistory.student_id == student_id,
                EngagementHistory.timestamp >= datetime.now() - timedelta(days=7)
            ).all()
            
            # Use fallback logic to get recommendations
            recommendations = self._get_fallback_recommendations(student, count, session)
            
            # Adjust recommendations based on recent engagements
            recommendations = self._adjust_recommendations(recommendations, recent_engagements)
            return recommendations
        except Exception as e:
            print(f"Error getting recommendations: {str(e)}")
            return self._get_default_recommendations(count)
        finally:
            session.close()
    
    def _adjust_recommendations(self, recommendations: List[Dict[str, Any]], 
                              recent_engagements: List[EngagementHistory]) -> List[Dict[str, Any]]:
        """
        Adjust recommendations based on recent engagements.
        
        Args:
            recommendations: List of recommendations
            recent_engagements: List of recent engagements
            
        Returns:
            Adjusted recommendations
        """
        recent_content_ids = {e.engagement_content_id for e in recent_engagements}
        for rec in recommendations:
            if rec.get('engagement_id') in recent_content_ids:
                rec['expected_effectiveness'] *= 0.5
                rec['rationale'] += " (Recently engaged with this content)"
        recommendations.sort(key=lambda x: x.get('expected_effectiveness', 0), reverse=True)
        return recommendations
    
    def _get_default_recommendations(self, count: int) -> List[Dict[str, Any]]:
        """
        Get default recommendations when student is not found.
        
        Args:
            count: Number of recommendations to return
            
        Returns:
            List of default recommendations
        """
        default_recommendations = [
            {
                "engagement_id": "default_1",
                "engagement_type": "email",
                "content": "Introduction to university programs",
                "expected_effectiveness": 0.85,
                "rationale": "Default recommendation for new students"
            },
            {
                "engagement_id": "default_2",
                "engagement_type": "sms",
                "content": "Upcoming virtual open house",
                "expected_effectiveness": 0.75,
                "rationale": "Default recommendation for new students"
            },
            {
                "engagement_id": "default_3",
                "engagement_type": "email",
                "content": "Student success stories",
                "expected_effectiveness": 0.8,
                "rationale": "Default recommendation for new students"
            }
        ]
        return default_recommendations[:count]
    
    def _get_fallback_recommendations(self, student: StudentProfile, count: int, session: Session) -> List[Dict[str, Any]]:
        """
        Get rule-based recommendations based on student's funnel stage.
        
        Args:
            student: Student profile
            count: Number of recommendations to return
            session: Database session
            
        Returns:
            List of recommendations
        """
        if student is None:
            return self._get_default_recommendations(count)
        funnel_stage = student.funnel_stage
        content_items = session.query(EngagementContent).filter_by(target_funnel_stage=funnel_stage).all()
        if not content_items:
            content_items = session.query(EngagementContent).limit(10).all()
        if content_items and hasattr(content_items[0], 'success_rate'):
            content_items.sort(key=lambda x: x.success_rate if x.success_rate is not None else 0, reverse=True)
        content_items = content_items[:count]
        recommendations = []
        for item in content_items:
            recommendations.append({
                "engagement_id": item.content_id,
                "engagement_type": item.engagement_type,
                "content": item.content_description,
                "expected_effectiveness": item.success_rate if hasattr(item, 'success_rate') and item.success_rate is not None else 0.7,
                "rationale": f"Recommended based on student's funnel stage: {funnel_stage}"
            })
        while len(recommendations) < count:
            idx = len(recommendations)
            recommendations.append({
                "engagement_id": f"default_{idx}",
                "engagement_type": "email",
                "content": f"Default recommendation {idx+1} for {funnel_stage} stage",
                "expected_effectiveness": 0.5,
                "rationale": f"Default recommendation for {funnel_stage} stage"
            })
        return recommendations
