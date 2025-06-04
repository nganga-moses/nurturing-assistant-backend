import os
import sys
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from sqlalchemy import or_
from data.models.student_profile import StudentProfile
from data.models.engagement_history import EngagementHistory
from data.models.engagement_content import EngagementContent
from data.models.stored_recommendation import StoredRecommendation
from data.models.recommendation_settings import RecommendationSettings
from data.models.recommendation_action import RecommendationAction
from data.models.recommendation_feedback_metrics import RecommendationFeedbackMetrics
from data.models.recommendation import Recommendation
from database.session import get_db
from models.core.recommendation_service import RecommendationService as ModelRecommendationService
from api.services.matching_service import MatchingService
import logging

logger = logging.getLogger(__name__)

class RecommendationService:
    """Service for generating personalized engagement recommendations."""
    def __init__(self, model_dir=None, mode="scheduled"):
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "saved_models")
        self.mode = mode
        self.recommendation_service = ModelRecommendationService(model_dir=model_dir)
        self.db = next(get_db())

    def get_recommendations(self, student_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get personalized recommendations for a student."""
        try:
            # Get student profile
            student = StudentProfile.query.filter_by(student_id=student_id).first()
            if not student:
                logger.warning(f"Student {student_id} not found, using fallback recommendations")
                return self.recommendation_service.get_fallback_recommendations(limit)
            
            # Get recommendations from model
            recommendations = self.recommendation_service.get_recommendations(student, limit)
            
            # Track recommendations
            for rec in recommendations:
                self.track_recommendation_action(student_id, rec['id'])
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting recommendations for student {student_id}: {str(e)}")
            return self.recommendation_service.get_fallback_recommendations(limit)

    def track_recommendation_action(self, student_id: str, recommendation_id: int) -> None:
        """Track a recommendation action."""
        action = RecommendationAction(
            student_id=student_id,
            recommendation_id=recommendation_id,
            action_taken=False,
            action_date=None
        )
        self.db.add(action)
        self.db.commit()

    def update_recommendation_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update recommendation metrics."""
        feedback_metrics = RecommendationFeedbackMetrics(
            recommendation_type=metrics.get('recommendation_type', ''),
            total_shown=metrics.get('total_shown', 0),
            acted_count=metrics.get('acted_count', 0),
            ignored_count=metrics.get('ignored_count', 0),
            untouched_count=metrics.get('untouched_count', 0),
            avg_time_to_action=metrics.get('avg_time_to_action', 0.0),
            completion_rate=metrics.get('completion_rate', 0.0),
            satisfaction_score=metrics.get('satisfaction_score', 0.0),
            dropoff_rates=metrics.get('dropoff_rates', {})
        )
        
        self.db.add(feedback_metrics)
        self.db.commit()

    def get_recommendations_for_current_user(self, current_user_id: str) -> List[dict]:
        """Get recommendations for all students assigned to the current recruiter (user)."""
        try:
            recommendations = (
                self.db.query(Recommendation)
                .join(StudentProfile, Recommendation.student_id == StudentProfile.student_id)
                .filter(StudentProfile.recruiter_id == current_user_id)
                .all()
            )
            return [rec.to_dict() for rec in recommendations]
        except Exception as e:
            logger.error(f"Error getting recommendations for current user {current_user_id}: {str(e)}")
            return []

    def create_recommendation(self, student_id: str, recommendation_type: str, content_id: str, confidence_score: float) -> Recommendation:
        """Create and store a new recommendation for a student."""
        rec = Recommendation(
            student_id=student_id,
            recommendation_type=recommendation_type,
            content_id=content_id,
            confidence_score=confidence_score
        )
        self.db.add(rec)
        self.db.commit()
        self.db.refresh(rec)
        return rec
    # ... (rest of the RecommendationService methods from services.py) ... 