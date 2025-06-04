from typing import Dict, Any
from database.session import get_db
from data.models.student_profile import StudentProfile
from api.services.risk_assessment_service import RiskAssessmentService

class DashboardService:
    """Service for generating dashboard statistics."""
    def __init__(self):
        self.risk_service = RiskAssessmentService()

    def get_dashboard_stats(self, db) -> Dict[str, Any]:
        """
        Get statistics for the dashboard.
        Returns:
            Dictionary of dashboard statistics
        """
        # Query all students
        students = db.query(StudentProfile).all()
        # Calculate total students
        total_students = len(students)
        # Calculate application rate
        completed_applications = sum(1 for student in students if student.application_status == "Completed")
        application_rate = (completed_applications / total_students * 100) if total_students > 0 else 0
        # Calculate at-risk count
        at_risk_students = self.risk_service.get_at_risk_students(db)
        at_risk_count = len(at_risk_students)
        # Calculate funnel stage distribution
        stage_distribution = {}
        for student in students:
            stage = student.funnel_stage
            if stage in stage_distribution:
                stage_distribution[stage] += 1
            else:
                stage_distribution[stage] = 1
        return {
            "total_students": total_students,
            "application_rate": application_rate,
            "at_risk_count": at_risk_count,
            "stage_distribution": stage_distribution
        }
    # ... (rest of DashboardService methods from services.py) ... 