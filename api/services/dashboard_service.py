from typing import Dict, Any
from sqlalchemy import func, case
from database.session import get_db
from data.models.student_profile import StudentProfile
from api.services.risk_assessment_service import RiskAssessmentService


class DashboardService:
    """Optimized service for generating dashboard statistics."""

    def __init__(self):
        self.risk_service = RiskAssessmentService()

    def get_dashboard_stats(self, db) -> Dict[str, Any]:
        """
        Get statistics for the dashboard using optimized queries.
        Returns:
            Dictionary of dashboard statistics
        """
        # Single optimized query to get all stats at once
        stats_query = db.query(
            func.count(StudentProfile.student_id).label('total_students'),
            func.sum(
                case(
                    (StudentProfile.application_status == "Completed", 1),
                    else_=0
                )
            ).label('completed_applications'),
            StudentProfile.funnel_stage
        ).group_by(StudentProfile.funnel_stage).all()

        # Process results
        total_students = 0
        completed_applications = 0
        stage_distribution = {}

        # Get totals and stage distribution in one pass
        for row in stats_query:
            if hasattr(row, 'total_students') and row.total_students:
                total_students += row.total_students
                if hasattr(row, 'completed_applications') and row.completed_applications:
                    completed_applications += row.completed_applications

            if row.funnel_stage:
                # For stage distribution, we need a separate count query per stage
                stage_count = db.query(func.count(StudentProfile.student_id)).filter(
                    StudentProfile.funnel_stage == row.funnel_stage
                ).scalar()
                stage_distribution[row.funnel_stage] = stage_count

        # If the above approach is complex, use this simpler optimized version:
        # Get total count with single query (safer approach)
        total_students = db.query(func.count()).select_from(StudentProfile).scalar()

        # Get completed applications count with single query
        completed_applications = db.query(func.count()).select_from(StudentProfile).filter(
            StudentProfile.application_status == "Completed"
        ).scalar()

        # Calculate application rate
        application_rate = (completed_applications / total_students * 100) if total_students > 0 else 0

        # Get stage distribution with single query
        stage_results = db.query(
            StudentProfile.funnel_stage,
            func.count().label('count')
        ).group_by(StudentProfile.funnel_stage).all()

        stage_distribution = {
            stage: count for stage, count in stage_results if stage is not None
        }

        # Get at-risk count using optimized method
        at_risk_count = self.risk_service.get_at_risk_count(db)

        return {
            "total_students": total_students,
            "application_rate": round(application_rate, 2),
            "at_risk_count": at_risk_count,
            "stage_distribution": stage_distribution
        }

    def get_dashboard_stats_cached(self, db, cache_duration_minutes: int = 5) -> Dict[str, Any]:
        """
        Get dashboard stats with optional caching.
        For high-traffic scenarios, implement Redis caching here.
        """
        # TODO: Implement Redis caching
        # cache_key = "dashboard_stats"
        # cached_result = redis_client.get(cache_key)
        # if cached_result:
        #     return json.loads(cached_result)

        stats = self.get_dashboard_stats(db)

        # TODO: Cache the result
        # redis_client.setex(cache_key, cache_duration_minutes * 60, json.dumps(stats))

        return stats