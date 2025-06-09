import os
import sys
import logging
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

logger = logging.getLogger(__name__)

class RecommendationService:
    """Service for generating personalized engagement recommendations using trained model."""
    
    def __init__(self, model_manager=None):
        """
        Initialize the recommendation service.
        
        Args:
            model_manager: ModelManager instance for predictions
        """
        self.model_manager = model_manager
        logger.info("RecommendationService initialized")

    def get_recommendations(self, student_id: str, top_k: int = 5, funnel_stage: str = None, risk_level: str = None) -> List[Dict[str, Any]]:
        """
        Get personalized recommendations for a student using trained model.
        
        Args:
            student_id: Student identifier
            top_k: Number of recommendations to return
            funnel_stage: Optional filter by funnel stage
            risk_level: Optional filter by risk level
            
        Returns:
            List of recommendation dictionaries
        """
        try:
            # Use real model recommendation if available
            if self.model_manager and self.model_manager.is_healthy:
                recommendations = self.model_manager.get_recommendations(student_id, top_k)
                # Enhance recommendations with metadata
                enhanced_recs = []
                for i, rec in enumerate(recommendations):
                    enhanced_rec = {
                        "id": rec.get("engagement_id", f"rec_{i}"),
                        "student_id": student_id,
                        "engagement_id": rec.get("engagement_id"),
                        "title": rec.get("title", f"Recommended Activity {i+1}"),
                        "description": rec.get("description", "Personalized engagement activity"),
                        "content_type": rec.get("content_type", "engagement"),
                        "confidence_score": rec.get("score", 0.8),
                        "relevance_score": rec.get("relevance", 0.7),
                        "predicted_impact": rec.get("predicted_impact", "medium"),
                        "recommendation_type": "model_based",
                        "priority": i + 1,
                        "created_at": datetime.now().isoformat(),
                        "metadata": rec.get("metadata", {})
                    }
                    
                    # Apply filters if specified
                    if funnel_stage and not self._matches_funnel_stage(enhanced_rec, funnel_stage):
                        continue
                    if risk_level and not self._matches_risk_level(enhanced_rec, risk_level):
                        continue
                        
                    enhanced_recs.append(enhanced_rec)
                
                logger.info(f"Generated {len(enhanced_recs)} model-based recommendations for student {student_id}")
                return enhanced_recs[:top_k]
            else:
                # Fallback to heuristic recommendations
                logger.warning("Model manager not available, using heuristic recommendations")
                return self._get_fallback_recommendations(student_id, top_k, funnel_stage, risk_level)
                
        except Exception as e:
            logger.error(f"Error getting recommendations for student {student_id}: {e}")
            return self._get_fallback_recommendations(student_id, top_k, funnel_stage, risk_level)

    def get_batch_recommendations(self, student_ids: List[str], top_k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get recommendations for multiple students efficiently.
        
        Args:
            student_ids: List of student identifiers
            top_k: Number of recommendations per student
            
        Returns:
            Dict mapping student_id to recommendations list
        """
        results = {}
        
        for student_id in student_ids:
            try:
                recommendations = self.get_recommendations(student_id, top_k)
                results[student_id] = recommendations
            except Exception as e:
                logger.error(f"Batch recommendations failed for {student_id}: {e}")
                results[student_id] = self._get_fallback_recommendations(student_id, top_k)
        
        return results

    def _get_fallback_recommendations(self, student_id: str, top_k: int = 5, funnel_stage: str = None, risk_level: str = None) -> List[Dict[str, Any]]:
        """
        Generate fallback recommendations using heuristic approach.
        
        Args:
            student_id: Student identifier
            top_k: Number of recommendations to return
            funnel_stage: Optional filter by funnel stage
            risk_level: Optional filter by risk level
            
        Returns:
            List of fallback recommendations
        """
        # Generate generic recommendations based on common engagement patterns
        fallback_recs = [
            {
                "id": f"fallback_{i}",
                "student_id": student_id,
                "engagement_id": f"eng_{i}",
                "title": f"Recommended Action {i+1}",
                "description": self._get_fallback_description(i, funnel_stage),
                "content_type": "engagement",
                "confidence_score": 0.5 - (i * 0.1),  # Decreasing confidence
                "relevance_score": 0.6 - (i * 0.05),
                "predicted_impact": "medium" if i < 2 else "low",
                "recommendation_type": "heuristic",
                "priority": i + 1,
                "created_at": datetime.now().isoformat(),
                "metadata": {
                    "fallback": True,
                    "funnel_stage": funnel_stage,
                    "risk_level": risk_level
                }
            }
            for i in range(top_k)
        ]
        
        logger.info(f"Generated {len(fallback_recs)} fallback recommendations for student {student_id}")
        return fallback_recs

    def _get_fallback_description(self, index: int, funnel_stage: str = None) -> str:
        """Generate contextual fallback descriptions."""
        stage_descriptions = {
            "awareness": [
                "Explore program overview materials",
                "Watch introductory videos",
                "Download program brochure",
                "Attend virtual information session",
                "Connect with current students"
            ],
            "interest": [
                "Schedule one-on-one consultation",
                "Take virtual campus tour",
                "Review admission requirements",
                "Explore scholarship opportunities",
                "Join program-specific webinar"
            ],
            "consideration": [
                "Submit preliminary application",
                "Request transcript evaluation",
                "Connect with admissions counselor",
                "Review financial aid options",
                "Schedule campus visit"
            ],
            "decision": [
                "Complete full application",
                "Submit required documents",
                "Apply for financial aid",
                "Schedule interview",
                "Connect with academic advisor"
            ]
        }
        
        if funnel_stage and funnel_stage.lower() in stage_descriptions:
            descriptions = stage_descriptions[funnel_stage.lower()]
            return descriptions[index % len(descriptions)]
        
        # Default descriptions
        default_descriptions = [
            "Engage with personalized content",
            "Complete recommended assessment",
            "Join relevant discussion forum",
            "Review program materials",
            "Connect with program representative"
        ]
        return default_descriptions[index % len(default_descriptions)]

    def _matches_funnel_stage(self, recommendation: Dict[str, Any], funnel_stage: str) -> bool:
        """Check if recommendation matches funnel stage filter."""
        # For now, accept all recommendations regardless of funnel stage
        # This could be enhanced with funnel-stage-specific logic
        return True

    def _matches_risk_level(self, recommendation: Dict[str, Any], risk_level: str) -> bool:
        """Check if recommendation matches risk level filter."""
        # For now, accept all recommendations regardless of risk level
        # This could be enhanced with risk-level-specific logic
        return True

    # Keep existing methods for backward compatibility
    def track_recommendation_action(self, db, student_id: str, recommendation_id: int) -> None:
        """Track a recommendation action."""
        action = RecommendationAction(
            student_id=student_id,
            recommendation_id=recommendation_id,
            action_taken=False,
            action_date=None
        )
        db.add(action)
        db.commit()

    def update_recommendation_metrics(self, db, metrics: Dict[str, Any]) -> None:
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
        
        db.add(feedback_metrics)
        db.commit()

    def get_recommendations_for_current_user(self, db, current_user_id: str) -> List[dict]:
        """Get recommendations for all students assigned to the current recruiter (user)."""
        try:
            recommendations = (
                db.query(Recommendation)
                .join(StudentProfile, Recommendation.student_id == StudentProfile.student_id)
                .filter(StudentProfile.recruiter_id == current_user_id)
                .all()
            )
            return [rec.to_dict() for rec in recommendations]
        except Exception as e:
            logger.error(f"Error getting recommendations for current user {current_user_id}: {str(e)}")
            return []

    def create_recommendation(self, db, student_id: str, recommendation_type: str, content_id: str, confidence_score: float) -> Recommendation:
        """Create and store a new recommendation for a student."""
        rec = Recommendation(
            student_id=student_id,
            recommendation_type=recommendation_type,
            content_id=content_id,
            confidence_score=confidence_score
        )
        db.add(rec)
        db.commit()
        db.refresh(rec)
        return rec
    # ... (rest of the RecommendationService methods from services.py) ... 