from datetime import datetime
from sqlalchemy.orm import Session
from typing import Dict, Tuple, Optional
from difflib import SequenceMatcher
from data.models import (
    EngagementHistory,
    StoredRecommendation,
    NudgeAction,
    NudgeFeedbackMetrics
)

class MatchingService:
    def __init__(self, db_session: Session):
        self.db = db_session
        self.type_similarity_threshold = 0.8
        self.category_similarity_threshold = 0.7
        self.time_window_hours = 24

    def match_engagement_to_nudge(self, engagement: EngagementHistory) -> Tuple[bool, float]:
        """
        Attempt to match an engagement to a stored recommendation/nudge.
        Returns (matched: bool, confidence: float) tuple.
        """
        # Find the most recent stored recommendation for this student
        stored_nudge = (
            self.db.query(StoredRecommendation)
            .filter(
                StoredRecommendation.student_id == engagement.student_id,
                StoredRecommendation.expires_at > datetime.now()
            )
            .order_by(StoredRecommendation.generated_at.desc())
            .first()
        )

        if not stored_nudge:
            return False, 0.0

        # Check if this engagement matches any recommendation in the stored nudge
        recommendations = stored_nudge.recommendations or []
        best_match = None
        highest_confidence = 0.0

        for rec in recommendations:
            is_match, confidence = self._is_match(engagement, rec)
            if is_match and confidence > highest_confidence:
                best_match = rec
                highest_confidence = confidence

        if best_match:
            # Create or update nudge action with confidence score
            self._create_or_update_nudge_action(
                stored_nudge.id,
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

    def _create_or_update_nudge_action(
        self,
        nudge_id: int,
        student_id: str,
        engagement: EngagementHistory,
        confidence_score: float
    ) -> None:
        """Create or update a nudge action record with confidence score."""
        action = (
            self.db.query(NudgeAction)
            .filter(
                NudgeAction.nudge_id == nudge_id,
                NudgeAction.student_id == student_id
            )
            .first()
        )

        if not action:
            action = NudgeAction(
                nudge_id=nudge_id,
                student_id=student_id,
                action_type="acted",
                action_timestamp=engagement.timestamp,
                time_to_action=int((engagement.timestamp - datetime.now()).total_seconds()),
                action_completed=True,
                confidence_score=confidence_score
            )
            self.db.add(action)
        else:
            action.action_type = "acted"
            action.action_timestamp = engagement.timestamp
            action.time_to_action = int((engagement.timestamp - datetime.now()).total_seconds())
            action.action_completed = True
            action.confidence_score = confidence_score

        self.db.commit()

    def _update_feedback_metrics(self, nudge_type: str, confidence_score: float) -> None:
        """Update feedback metrics for a nudge type with confidence score."""
        metrics = (
            self.db.query(NudgeFeedbackMetrics)
            .filter(NudgeFeedbackMetrics.nudge_type == nudge_type)
            .first()
        )

        if not metrics:
            metrics = NudgeFeedbackMetrics(
                nudge_type=nudge_type,
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

    def get_unmatched_engagements(self, start_date: datetime = None) -> list:
        """
        Get list of engagements that weren't matched to any nudge.
        Optionally filter by start date.
        """
        query = (
            self.db.query(EngagementHistory)
            .outerjoin(
                NudgeAction,
                EngagementHistory.student_id == NudgeAction.student_id
            )
            .filter(NudgeAction.id.is_(None))
        )

        if start_date:
            query = query.filter(EngagementHistory.timestamp >= start_date)

        return query.all()

    def get_feedback_metrics(self, nudge_type: str = None) -> list:
        """
        Get feedback metrics for all nudge types or a specific type.
        """
        query = self.db.query(NudgeFeedbackMetrics)
        if nudge_type:
            query = query.filter(NudgeFeedbackMetrics.nudge_type == nudge_type)
        return query.all()

    def get_match_quality_stats(self, nudge_type: str = None) -> Dict[str, float]:
        """
        Get statistics about match quality for a nudge type.
        Returns average confidence, match rate, and other quality metrics.
        """
        query = self.db.query(NudgeFeedbackMetrics)
        if nudge_type:
            query = query.filter(NudgeFeedbackMetrics.nudge_type == nudge_type)
        
        metrics = query.all()
        if not metrics:
            return {
                "average_confidence": 0.0,
                "match_rate": 0.0,
                "high_confidence_rate": 0.0
            }

        total_metrics = len(metrics)
        total_confidence = sum(m.average_confidence for m in metrics)
        high_confidence_count = sum(1 for m in metrics if m.average_confidence > 0.8)

        return {
            "average_confidence": total_confidence / total_metrics if total_metrics > 0 else 0.0,
            "match_rate": sum(m.completion_rate for m in metrics) / total_metrics if total_metrics > 0 else 0.0,
            "high_confidence_rate": high_confidence_count / total_metrics if total_metrics > 0 else 0.0
        } 