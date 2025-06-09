import logging
from typing import Dict, Any, Optional
from data.models.student_profile import StudentProfile

logger = logging.getLogger(__name__)

class LikelihoodService:
    """Service for predicting application likelihood using trained model."""
    
    def __init__(self, model_manager=None):
        """
        Initialize the likelihood service.
        
        Args:
            model_manager: ModelManager instance for predictions
        """
        self.model_manager = model_manager
        logger.info("LikelihoodService initialized")

    def get_application_likelihood(self, db, student_id: str, engagement_id: str = None) -> float:
        """
        Get application likelihood for a student-engagement pair.
        
        Args:
            db: Database session
            student_id: Student identifier
            engagement_id: Optional engagement identifier
            
        Returns:
            float: Likelihood score as percentage (0-100)
        """
        try:
            # First, validate that student exists
            student = db.query(StudentProfile).filter_by(student_id=student_id).first()
            if not student:
                raise ValueError(f"Student with ID {student_id} not found")
            
            # Use real model prediction if available
            if self.model_manager and self.model_manager.is_healthy:
                likelihood = self.model_manager.predict_likelihood(student_id, engagement_id)
                # Convert from 0-1 scale to 0-100 percentage
                percentage = float(likelihood * 100)
                logger.info(f"Real model prediction for student {student_id}: {percentage:.2f}%")
                return percentage
            else:
                # Fallback to enhanced mock prediction
                logger.warning("Model manager not available, using fallback prediction")
                return self._get_fallback_likelihood(student, engagement_id)
                
        except Exception as e:
            logger.error(f"Likelihood prediction failed for student {student_id}: {e}")
            # Fallback on any error
            student = db.query(StudentProfile).filter_by(student_id=student_id).first()
            if student:
                return self._get_fallback_likelihood(student, engagement_id)
            else:
                return 50.0  # Default neutral likelihood

    def _get_fallback_likelihood(self, student: StudentProfile, engagement_id: str = None) -> float:
        """
        Generate fallback likelihood based on student profile.
        
        Args:
            student: StudentProfile object
            engagement_id: Optional engagement identifier
            
        Returns:
            float: Fallback likelihood percentage
        """
        # Base likelihood from funnel stage
        stage_likelihood = {
            "awareness": 20.0,
            "interest": 40.0,
            "consideration": 60.0,
            "decision": 80.0,
            "application": 95.0
        }
        
        base_likelihood = stage_likelihood.get(student.funnel_stage.lower(), 50.0)
        
        # Adjust based on academic performance if available
        if hasattr(student, 'gpa') and student.gpa:
            try:
                gpa = float(student.gpa)
                # Higher GPA increases likelihood
                gpa_adjustment = (gpa - 2.5) * 10  # Scale GPA (2.5-4.0) to adjustment (-5 to +15)
                base_likelihood += gpa_adjustment
            except (ValueError, TypeError):
                pass
        
        # Adjust based on engagement history if available
        if hasattr(student, 'engagement_count') and student.engagement_count:
            try:
                engagement_count = int(student.engagement_count)
                # More engagements increase likelihood
                engagement_adjustment = min(engagement_count * 2, 20)  # Cap at +20%
                base_likelihood += engagement_adjustment
            except (ValueError, TypeError):
                pass
        
        # Ensure likelihood stays within bounds
        likelihood = max(0.0, min(100.0, base_likelihood))
        
        logger.info(f"Fallback prediction for student {student.student_id}: {likelihood:.2f}%")
        return likelihood

    def get_batch_likelihood(self, db, student_engagement_pairs: list) -> Dict[str, float]:
        """
        Get likelihood predictions for multiple student-engagement pairs.
        
        Args:
            db: Database session
            student_engagement_pairs: List of (student_id, engagement_id) tuples
            
        Returns:
            Dict mapping "{student_id}_{engagement_id}" to likelihood percentage
        """
        results = {}
        
        for student_id, engagement_id in student_engagement_pairs:
            try:
                likelihood = self.get_application_likelihood(db, student_id, engagement_id)
                key = f"{student_id}_{engagement_id or 'general'}"
                results[key] = likelihood
            except Exception as e:
                logger.error(f"Batch prediction failed for {student_id}, {engagement_id}: {e}")
                key = f"{student_id}_{engagement_id or 'general'}"
                results[key] = 50.0  # Default on error
        
        return results 