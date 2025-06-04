from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from typing import Dict, Tuple, Optional, List, Any
from difflib import SequenceMatcher
from data.models import (
    EngagementHistory,
    StoredRecommendation,
    RecommendationAction,
    RecommendationFeedbackMetrics,
    StudentProfile
)

class MatchingService:
    def __init__(self, db_session: Session):
        self.db = db_session
        self.type_similarity_threshold = 0.8
        self.category_similarity_threshold = 0.7
        self.time_window_hours = 24

    def match_engagement_to_recommendation(self, engagement: EngagementHistory) -> Tuple[bool, float]:
        """
        Attempt to match an engagement to a stored recommendation.
        
        Returns:
            Tuple[bool, float]: (matched, confidence_score)
        """
        stored_recommendation = (
            self.db.query(StoredRecommendation)
            .filter(StoredRecommendation.student_id == engagement.student_id)
            .first()
        )

        if not stored_recommendation:
            return False, 0.0

        # Check if this engagement matches any recommendation in the stored recommendation
        recommendations = stored_recommendation.recommendations or []
        best_match = None
        highest_confidence = 0.0

        for rec in recommendations:
            is_match, confidence = self._is_match(engagement, rec)
            if is_match and confidence > highest_confidence:
                best_match = rec
                highest_confidence = confidence

        if best_match:
            # Create or update recommendation action with confidence score
            self._create_or_update_recommendation_action(
                stored_recommendation.id,
                engagement.student_id,
                engagement,
                highest_confidence
            )
            # Update feedback metrics
            self._update_feedback_metrics(best_match.get('type', 'unknown'), highest_confidence)
            return True, highest_confidence

        return False, 0.0

    def _is_match(self, engagement: EngagementHistory, recommendation: dict) -> Tuple[bool, float]:
        """
        Determine if an engagement matches a recommendation and calculate confidence score.
        Returns (is_match: bool, confidence: float) tuple.
        """
        confidence_scores = []

        # Type matching with similarity score
        type_similarity = self._calculate_similarity(
            engagement.engagement_type,
            recommendation.get('type', '')
        )
        confidence_scores.append(type_similarity)
        if type_similarity < self.type_similarity_threshold:
            return False, 0.0

        # Content category matching with similarity score
        if engagement.content and recommendation.get('category'):
            category_similarity = self._calculate_similarity(
                engagement.content.content_category,
                recommendation.get('category', '')
            )
            confidence_scores.append(category_similarity)
            if category_similarity < self.category_similarity_threshold:
                return False, 0.0

        # Timing check with confidence decay
        if engagement.timestamp:
            time_diff = abs((engagement.timestamp - datetime.now()).total_seconds())
            time_confidence = max(0, 1 - (time_diff / (self.time_window_hours * 3600)))
            confidence_scores.append(time_confidence)

        # Calculate overall confidence score
        overall_confidence = sum(confidence_scores) / len(confidence_scores)
        return True, overall_confidence

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using SequenceMatcher."""
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

    def _create_or_update_recommendation_action(
        self,
        student_id: str,
        recommendation_id: int,
        action_type: str
    ) -> None:
        """Create or update a recommendation action record."""
        action = (
            self.db.query(RecommendationAction)
            .filter(
                RecommendationAction.recommendation_id == recommendation_id,
                RecommendationAction.student_id == student_id
            )
            .first()
        )

        if not action:
            action = RecommendationAction(
                recommendation_id=recommendation_id,
                student_id=student_id,
                action_type=action_type,
                action_timestamp=datetime.now(),
                action_completed=True
            )
            self.db.add(action)
        else:
            action.action_type = action_type
            action.action_timestamp = datetime.now()
            action.action_completed = True

        self.db.commit()

    def _update_feedback_metrics(self, recommendation_type: str, confidence_score: float) -> None:
        """Update feedback metrics for a recommendation type with confidence score."""
        metrics = (
            self.db.query(RecommendationFeedbackMetrics)
            .filter(RecommendationFeedbackMetrics.recommendation_type == recommendation_type)
            .first()
        )

        if not metrics:
            metrics = RecommendationFeedbackMetrics(
                recommendation_type=recommendation_type,
                total_shown=1,
                acted_count=1,
                completion_rate=1.0,
                average_confidence=confidence_score
            )
            self.db.add(metrics)
        else:
            metrics.total_shown += 1
            metrics.acted_count += 1
            metrics.completion_rate = metrics.acted_count / metrics.total_shown
            # Update average confidence using weighted average
            metrics.average_confidence = (
                (metrics.average_confidence * (metrics.total_shown - 1) + confidence_score)
                / metrics.total_shown
            )

        metrics.last_updated = datetime.now()
        self.db.commit()

    def get_unmatched_engagements(self) -> List[EngagementHistory]:
        """
        Get list of engagements that weren't matched to any recommendation.
        """
        query = (
            self.db.query(EngagementHistory)
            .outerjoin(
                RecommendationAction,
                EngagementHistory.student_id == RecommendationAction.student_id
            )
            .filter(RecommendationAction.id.is_(None))
        )

        return query.all()

    def get_recommendations(self, student_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Get personalized recommendations for a student."""
        # Get student profile
        student = self.db.query(StudentProfile).filter_by(student_id=student_id).first()
        if not student:
            raise ValueError(f"Student {student_id} not found")

        # Get student's engagement history
        engagement_history = self.db.query(EngagementHistory).filter_by(student_id=student_id).all()

        # Get student's previous recommendation actions
        previous_actions = self.db.query(RecommendationAction).filter_by(student_id=student_id).all()

        # Generate recommendations based on student profile and history
        recommendations = self._generate_recommendations(student, engagement_history, previous_actions)

        # Store recommendations
        stored_recommendation = StoredRecommendation(
            student_id=student_id,
            recommendations=recommendations,
            generated_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=7)
        )
        self.db.add(stored_recommendation)
        self.db.commit()

        return recommendations[:top_k]

    def _generate_recommendations(self, student: StudentProfile, 
                                engagement_history: List[EngagementHistory],
                                previous_actions: List[RecommendationAction]) -> List[Dict[str, Any]]:
        """Generate personalized recommendations based on student data."""
        recommendations = []
        
        # Get student's academic profile
        academic_profile = {
            "gpa": student.gpa,
            "sat_score": student.sat_score,
            "act_score": student.act_score,
            "funnel_stage": student.funnel_stage
        }
        
        # Get student's engagement history
        engagement_types = set(e.engagement_type for e in engagement_history)
        
        # Generate recommendations based on academic profile
        if academic_profile["gpa"] >= 3.5:
            recommendations.append({
                "type": "honors_program",
                "title": "Honors Program Application",
                "description": "Apply for our honors program based on your strong academic performance",
                "priority": "high"
            })
        
        if academic_profile["sat_score"] and academic_profile["sat_score"] >= 1400:
            recommendations.append({
                "type": "scholarship_opportunity",
                "title": "Merit Scholarship Application",
                "description": "Apply for our merit-based scholarship program",
                "priority": "high"
            })
        
        # Add recommendations based on funnel stage
        if academic_profile["funnel_stage"] == "prospect":
            recommendations.append({
                "type": "campus_visit",
                "title": "Schedule Campus Visit",
                "description": "Experience our campus firsthand",
                "priority": "medium"
            })
        elif academic_profile["funnel_stage"] == "applicant":
            recommendations.append({
                "type": "application_completion",
                "title": "Complete Application",
                "description": "Finish your application to move forward in the process",
                "priority": "high"
            })
        
        # Add recommendations based on engagement history
        if "campus_visit" not in engagement_types:
            recommendations.append({
                "type": "virtual_tour",
                "title": "Take Virtual Tour",
                "description": "Explore our campus virtually",
                "priority": "medium"
            })
        
        return recommendations

    def track_recommendation_action(self, student_id: str, recommendation_id: int, action_type: str):
        """Track a student's action on a recommendation."""
        self._create_or_update_recommendation_action(
            student_id=student_id,
            recommendation_id=recommendation_id,
            action_type=action_type
        )

    def get_students_without_actions(self) -> List[str]:
        """Get list of students who haven't taken any actions on recommendations."""
        students = (
            self.db.query(EngagementHistory.student_id)
            .outerjoin(
                RecommendationAction,
                EngagementHistory.student_id == RecommendationAction.student_id
            )
            .filter(RecommendationAction.id.is_(None))
            .distinct()
            .all()
        )
        return [s[0] for s in students]

    def get_feedback_metrics(self, recommendation_type: str = None) -> List[Dict[str, Any]]:
        """Get feedback metrics for all recommendation types or a specific type."""
        query = self.db.query(RecommendationFeedbackMetrics)
        if recommendation_type:
            query = query.filter(RecommendationFeedbackMetrics.recommendation_type == recommendation_type)
        return [m.to_dict() for m in query.all()]

    def get_recommendation_effectiveness(self, recommendation_type: str = None) -> Dict[str, float]:
        """Get effectiveness metrics for recommendations."""
        query = self.db.query(RecommendationFeedbackMetrics)
        if recommendation_type:
            query = query.filter(RecommendationFeedbackMetrics.recommendation_type == recommendation_type)
        
        metrics = query.all()
        if not metrics:
            return {
                "action_rate": 0.0,
                "completion_rate": 0.0,
                "average_time_to_action": 0.0
            }

        total_shown = sum(m.total_shown for m in metrics)
        total_acted = sum(m.acted_count for m in metrics)
        total_completed = sum(m.completion_rate * m.total_shown for m in metrics)
        total_time = sum(m.avg_time_to_action * m.total_shown for m in metrics)

        return {
            "action_rate": total_acted / total_shown if total_shown > 0 else 0.0,
            "completion_rate": total_completed / total_shown if total_shown > 0 else 0.0,
            "average_time_to_action": total_time / total_shown if total_shown > 0 else 0.0
        } 