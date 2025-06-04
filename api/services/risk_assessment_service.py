from datetime import datetime
from typing import Dict, List, Any
from data.models.student_profile import StudentProfile
from data.models.engagement_history import EngagementHistory

class RiskAssessmentService:
    """Service for assessing dropout risk."""
    def __init__(self):
        self.model = None

    def get_dropout_risk(self, db, student_id: str) -> Dict[str, Any]:
        student = db.query(StudentProfile).filter_by(student_id=student_id).first()
        if not student:
            raise ValueError(f"Student with ID {student_id} not found")
        last_engagement = db.query(EngagementHistory)\
            .filter_by(student_id=student_id)\
            .order_by(EngagementHistory.timestamp.desc())\
            .first()
        if not last_engagement:
            risk_score = 0.8
        else:
            days_since = (datetime.now() - last_engagement.timestamp).days
            if days_since > 30:
                risk_score = 0.9
            elif days_since > 14:
                risk_score = 0.7
            elif days_since > 7:
                risk_score = 0.5
            else:
                risk_score = 0.2
            stage_multiplier = {
                "awareness": 1.2,
                "interest": 1.1,
                "consideration": 1.0,
                "decision": 0.9,
                "application": 0.8
            }
            risk_score = min(1.0, risk_score * stage_multiplier.get(student.funnel_stage.lower(), 1.0))
        return {
            "risk_score": risk_score,
            "risk_category": self._score_to_category(risk_score)
        }

    def get_at_risk_students(self, db, risk_threshold: float = 0.7) -> List[Dict[str, Any]]:
        students = db.query(StudentProfile).all()
        at_risk_students = []
        for student in students:
            last_engagement = db.query(EngagementHistory)\
                .filter_by(student_id=student.student_id)\
                .order_by(EngagementHistory.timestamp.desc())\
                .first()
            if not last_engagement:
                risk_score = 0.8
            else:
                days_since = (datetime.now() - last_engagement.timestamp).days
                if days_since > 30:
                    risk_score = 0.9
                elif days_since > 14:
                    risk_score = 0.7
                elif days_since > 7:
                    risk_score = 0.5
                else:
                    risk_score = 0.2
                stage_multiplier = {
                    "awareness": 1.2,
                    "interest": 1.1,
                    "consideration": 1.0,
                    "decision": 0.9,
                    "application": 0.8
                }
                risk_score = min(1.0, risk_score * stage_multiplier.get(student.funnel_stage.lower(), 1.0))
            if risk_score >= risk_threshold:
                at_risk_students.append({
                    "student_id": student.student_id,
                    "demographic_features": student.demographic_features,
                    "funnel_stage": student.funnel_stage,
                    "risk_score": risk_score,
                    "risk_category": self._score_to_category(risk_score)
                })
        at_risk_students.sort(key=lambda x: x["risk_score"], reverse=True)
        return at_risk_students

    def get_high_potential_students(self, db, likelihood_threshold: float = 0.7) -> List[Dict[str, Any]]:
        students = db.query(StudentProfile).all()
        high_potential_students = []
        for student in students:
            if student.funnel_stage == "application":
                continue
            stage_likelihood = {
                "awareness": 0.3,
                "interest": 0.5,
                "consideration": 0.7,
                "decision": 0.9
            }
            likelihood_score = stage_likelihood.get(student.funnel_stage.lower(), 0.5)
            if likelihood_score >= likelihood_threshold:
                high_potential_students.append({
                    "student_id": student.student_id,
                    "demographic_features": student.demographic_features,
                    "funnel_stage": student.funnel_stage,
                    "likelihood_score": likelihood_score
                })
        high_potential_students.sort(key=lambda x: x["likelihood_score"], reverse=True)
        return high_potential_students

    def _score_to_category(self, score: float) -> str:
        if score >= 0.7:
            return "high"
        elif score >= 0.4:
            return "medium"
        else:
            return "low" 