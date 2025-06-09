import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from sqlalchemy import func, case, and_, or_, text
from sqlalchemy.orm import aliased, joinedload
from data.models.student_profile import StudentProfile
from data.models.engagement_history import EngagementHistory

logger = logging.getLogger(__name__)

class RiskAssessmentService:
    """Service for assessing dropout risk using trained model and optimized fallback logic."""

    def __init__(self, model_manager=None):
        """
        Initialize the risk assessment service.
        
        Args:
            model_manager: ModelManager instance for predictions
        """
        self.model_manager = model_manager
        # Cache stage multipliers for performance
        self.stage_multipliers = {
            "awareness": 1.2,
            "interest": 1.1,
            "consideration": 1.0,
            "decision": 0.9,
            "application": 0.8
        }
        self.stage_likelihood = {
            "awareness": 0.3,
            "interest": 0.5,
            "consideration": 0.7,
            "decision": 0.9
        }
        logger.info("RiskAssessmentService initialized")

    def get_dropout_risk(self, db, student_id: str) -> Dict[str, Any]:
        """
        Get dropout risk for a single student using trained model or optimized heuristics.
        
        Args:
            db: Database session
            student_id: Student identifier
            
        Returns:
            Dict with risk_score, risk_category, and confidence
        """
        try:
            # First, validate that student exists and get basic info
            result = db.query(
                StudentProfile.student_id,
                StudentProfile.funnel_stage,
                func.max(EngagementHistory.timestamp).label('last_engagement')
            ).outerjoin(
                EngagementHistory,
                StudentProfile.student_id == EngagementHistory.student_id
            ).filter(
                StudentProfile.student_id == student_id
            ).group_by(
                StudentProfile.student_id,
                StudentProfile.funnel_stage
            ).first()

            if not result:
                raise ValueError(f"Student with ID {student_id} not found")

            # Use real model prediction if available
            if self.model_manager and self.model_manager.is_healthy:
                risk_score = self.model_manager.predict_risk(student_id)
                confidence = 0.8  # Model predictions have higher confidence
                logger.info(f"Real model risk prediction for student {student_id}: {risk_score:.3f}")
            else:
                # Fallback to heuristic calculation
                logger.warning("Model manager not available, using heuristic risk calculation")
                risk_score = self._calculate_risk_score(
                    result.last_engagement,
                    result.funnel_stage
                )
                confidence = 0.6  # Heuristic predictions have lower confidence

            return {
                "risk_score": float(risk_score),
                "risk_category": self._score_to_category(risk_score),
                "confidence": confidence,
                "prediction_method": "model" if (self.model_manager and self.model_manager.is_healthy) else "heuristic"
            }
            
        except Exception as e:
            logger.error(f"Risk prediction failed for student {student_id}: {e}")
            # Fallback on any error
            result = db.query(
                StudentProfile.student_id,
                StudentProfile.funnel_stage,
                func.max(EngagementHistory.timestamp).label('last_engagement')
            ).outerjoin(
                EngagementHistory,
                StudentProfile.student_id == EngagementHistory.student_id
            ).filter(
                StudentProfile.student_id == student_id
            ).group_by(
                StudentProfile.student_id,
                StudentProfile.funnel_stage
            ).first()
            
            if result:
                risk_score = self._calculate_risk_score(
                    result.last_engagement,
                    result.funnel_stage
                )
                return {
                    "risk_score": float(risk_score),
                    "risk_category": self._score_to_category(risk_score),
                    "confidence": 0.4,
                    "prediction_method": "fallback"
                }
            else:
                return {
                    "risk_score": 0.5,
                    "risk_category": "medium",
                    "confidence": 0.2,
                    "prediction_method": "default"
                }

    def get_batch_risk_assessment(self, db, student_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get risk assessments for multiple students efficiently.
        
        Args:
            db: Database session
            student_ids: List of student identifiers
            
        Returns:
            Dict mapping student_id to risk assessment
        """
        results = {}
        
        for student_id in student_ids:
            try:
                risk_assessment = self.get_dropout_risk(db, student_id)
                results[student_id] = risk_assessment
            except Exception as e:
                logger.error(f"Batch risk assessment failed for {student_id}: {e}")
                results[student_id] = {
                    "risk_score": 0.5,
                    "risk_category": "medium",
                    "confidence": 0.1,
                    "prediction_method": "error_fallback"
                }
        
        return results

    def get_at_risk_students(self, db, risk_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Get all at-risk students using optimized bulk query."""
        # Single query with subquery for latest engagement per student
        latest_engagement_subq = db.query(
            EngagementHistory.student_id,
            func.max(EngagementHistory.timestamp).label('last_engagement')
        ).group_by(EngagementHistory.student_id).subquery()

        # Main query with join
        results = db.query(
            StudentProfile.student_id,
            StudentProfile.demographic_features,
            StudentProfile.funnel_stage,
            latest_engagement_subq.c.last_engagement
        ).outerjoin(
            latest_engagement_subq,
            StudentProfile.student_id == latest_engagement_subq.c.student_id
        ).all()

        # Process results in memory (faster than individual queries)
        at_risk_students = []
        for result in results:
            risk_score = self._calculate_risk_score(
                result.last_engagement,
                result.funnel_stage
            )

            if risk_score >= risk_threshold:
                at_risk_students.append({
                    "student_id": result.student_id,
                    "demographic_features": result.demographic_features,
                    "funnel_stage": result.funnel_stage,
                    "risk_score": risk_score,
                    "risk_category": self._score_to_category(risk_score)
                })

        # Sort by risk score descending
        at_risk_students.sort(key=lambda x: x["risk_score"], reverse=True)
        return at_risk_students

    def get_at_risk_count(self, db, risk_threshold: float = 0.7) -> int:
        """Get count of at-risk students using database-level calculation."""
        # Use database functions to calculate risk scores and count
        now = datetime.now()
        thirty_days_ago = now - timedelta(days=30)
        fourteen_days_ago = now - timedelta(days=14)
        seven_days_ago = now - timedelta(days=7)

        # Subquery for latest engagement per student
        latest_engagement_subq = db.query(
            EngagementHistory.student_id,
            func.max(EngagementHistory.timestamp).label('last_engagement')
        ).group_by(EngagementHistory.student_id).subquery()

        # Calculate risk score in database using CASE statements
        risk_score_case = case(
            # No engagement = 0.8 base risk
            (latest_engagement_subq.c.last_engagement.is_(None), 0.8),
            # > 30 days = 0.9 base risk
            (latest_engagement_subq.c.last_engagement < thirty_days_ago, 0.9),
            # > 14 days = 0.7 base risk
            (latest_engagement_subq.c.last_engagement < fourteen_days_ago, 0.7),
            # > 7 days = 0.5 base risk
            (latest_engagement_subq.c.last_engagement < seven_days_ago, 0.5),
            # <= 7 days = 0.2 base risk
            else_=0.2
        )

        # Apply stage multiplier
        stage_multiplier_case = case(
            (func.lower(StudentProfile.funnel_stage) == 'awareness', 1.2),
            (func.lower(StudentProfile.funnel_stage) == 'interest', 1.1),
            (func.lower(StudentProfile.funnel_stage) == 'consideration', 1.0),
            (func.lower(StudentProfile.funnel_stage) == 'decision', 0.9),
            (func.lower(StudentProfile.funnel_stage) == 'application', 0.8),
            else_=1.0
        )

        # Final risk score with minimum of 1.0
        final_risk_score = func.least(1.0, risk_score_case * stage_multiplier_case)

        # Count students with risk score >= threshold
        count = db.query(func.count()).select_from(
            db.query(StudentProfile).outerjoin(
                latest_engagement_subq,
                StudentProfile.student_id == latest_engagement_subq.c.student_id
            ).filter(final_risk_score >= risk_threshold).subquery()
        ).scalar()

        return count or 0

    def get_high_potential_students(self, db, likelihood_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Get high potential students using optimized query."""
        # Filter out application stage students in the query itself
        results = db.query(
            StudentProfile.student_id,
            StudentProfile.demographic_features,
            StudentProfile.funnel_stage
        ).filter(
            StudentProfile.funnel_stage != 'application'
        ).all()

        high_potential_students = []
        for result in results:
            likelihood_score = self.stage_likelihood.get(
                result.funnel_stage.lower(),
                0.5
            )

            if likelihood_score >= likelihood_threshold:
                high_potential_students.append({
                    "student_id": result.student_id,
                    "demographic_features": result.demographic_features,
                    "funnel_stage": result.funnel_stage,
                    "likelihood_score": likelihood_score
                })

        high_potential_students.sort(key=lambda x: x["likelihood_score"], reverse=True)
        return high_potential_students

    def get_high_potential_count(self, db, likelihood_threshold: float = 0.7) -> int:
        """Get count of high potential students using database calculation."""
        # Calculate likelihood in database
        likelihood_case = case(
            (func.lower(StudentProfile.funnel_stage) == 'awareness', 0.3),
            (func.lower(StudentProfile.funnel_stage) == 'interest', 0.5),
            (func.lower(StudentProfile.funnel_stage) == 'consideration', 0.7),
            (func.lower(StudentProfile.funnel_stage) == 'decision', 0.9),
            else_=0.5
        )

        count = db.query(func.count()).select_from(StudentProfile).filter(
            and_(
                StudentProfile.funnel_stage != 'application',
                likelihood_case >= likelihood_threshold
            )
        ).scalar()

        return count or 0

    def _calculate_risk_score(self, last_engagement: Optional[datetime], funnel_stage: str) -> float:
        """Calculate risk score for a student."""
        if not last_engagement:
            base_risk = 0.8
        else:
            days_since = (datetime.now() - last_engagement).days
            if days_since > 30:
                base_risk = 0.9
            elif days_since > 14:
                base_risk = 0.7
            elif days_since > 7:
                base_risk = 0.5
            else:
                base_risk = 0.2

        multiplier = self.stage_multipliers.get(funnel_stage.lower(), 1.0)
        return min(1.0, base_risk * multiplier)

    def _score_to_category(self, score: float) -> str:
        """Convert risk score to category."""
        if score >= 0.7:
            return "high"
        elif score >= 0.4:
            return "medium"
        else:
            return "low"