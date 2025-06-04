from typing import Dict, Any
from data.models.student_profile import StudentProfile

class LikelihoodService:
    """Service for predicting application likelihood."""
    def __init__(self, model_path="models/student_engagement_model"):
        self.model = None
        self.vocabularies = {}

    def get_application_likelihood(self, db, student_id: str) -> float:
        if self.model is None:
            return self._get_mock_likelihood(db, student_id)
        student_features = self._get_student_features(db, student_id)
        student_embedding = self.model.get_student_embeddings(student_features)
        likelihood = self.model.predict_likelihood(student_embedding)
        return float(likelihood.numpy()[0][0] * 100)

    def _get_mock_likelihood(self, db, student_id: str) -> float:
        student = db.query(StudentProfile).filter_by(student_id=student_id).first()
        if not student:
            raise ValueError(f"Student with ID {student_id} not found")
        stage_likelihood = {
            "awareness": 20.0,
            "interest": 40.0,
            "consideration": 60.0,
            "decision": 80.0,
            "application": 95.0
        }
        return stage_likelihood.get(student.funnel_stage.lower(), 50.0)

    def _get_student_features(self, db, student_id: str) -> Dict[str, Any]:
        student = db.query(StudentProfile).filter_by(student_id=student_id).first()
        if not student:
            raise ValueError(f"Student with ID {student_id} not found")
        features = {
            "student_id": student_id
        }
        return features 