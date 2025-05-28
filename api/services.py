import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import os
import json
import joblib
import sys

# Use relative imports instead of absolute imports
from data.models import StudentProfile, EngagementHistory, EngagementContent, get_session

# Import our new recommendation service
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.recommendation_service import RecommendationService as ModelRecommendationService


class RecommendationService:
    """Service for generating personalized engagement recommendations."""
    
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "saved_models")
        
        # Initialize our new recommendation service
        self.recommendation_service = ModelRecommendationService(model_dir=model_dir)
        self.session = get_session()
    
    def get_recommendations(self, student_id: str, top_k: int = 5, 
                            funnel_stage: Optional[str] = None, 
                            risk_level: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get personalized engagement recommendations for a student.
        
        Args:
            student_id: Unique identifier for the student
            top_k: Number of recommendations to return
            funnel_stage: Optional filter for specific funnel stage
            risk_level: Optional risk level to consider
            
        Returns:
            List of recommended engagements with scores
        """
        try:
            # Use our recommendation service to get recommendations
            recommendations = self.recommendation_service.get_recommendations(student_id, count=top_k)
            
            # Format the recommendations to match the expected format
            formatted_recommendations = []
            for rec in recommendations:
                formatted_recommendations.append({
                    "content_id": rec.get("engagement_id", ""),
                    "engagement_type": rec.get("engagement_type", ""),
                    "content_category": "engagement",  # Default category
                    "content_description": rec.get("content", ""),
                    "success_rate": rec.get("expected_effectiveness", 0.5),
                    "target_funnel_stage": funnel_stage or "any",
                    "score": rec.get("expected_effectiveness", 0.5),
                    "rationale": rec.get("rationale", "")
                })
            
            return formatted_recommendations
            
        except Exception as e:
            print(f"Error getting recommendations: {str(e)}")
            # Fall back to mock recommendations if there's an error
            return self._get_mock_recommendations(student_id, top_k)
        
    def _get_recommendations_with_nn(self, student_embedding, student_id: str, top_k: int = 5,
                                  funnel_stage: Optional[str] = None, risk_level: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get recommendations using the nearest neighbors model.
        
        Args:
            student_embedding: Embedding vector for the student
            student_id: Unique identifier for the student
            top_k: Number of recommendations to return
            funnel_stage: Optional filter for specific funnel stage
            risk_level: Optional risk level to consider
            
        Returns:
            List of recommended engagements with scores
        """
        # Reshape embedding for nearest neighbors query
        student_embedding = student_embedding.reshape(1, -1)
        
        # Query nearest neighbors
        distances, indices = self.nn_data['nn_model'].kneighbors(student_embedding, n_neighbors=min(top_k*2, len(self.nn_data['engagement_ids'])))
        
        # Convert distances to similarity scores (1 - distance)
        scores = 1 - distances.flatten()
        
        # Get engagement IDs
        engagement_ids = [self.nn_data['engagement_ids'][idx] for idx in indices.flatten()]
        
        # Get engagement details from database
        recommendations = []
        added_ids = set()  # To avoid duplicates
        
        for i, (engagement_id, score) in enumerate(zip(engagement_ids, scores)):
            # Skip if we already have this engagement
            if engagement_id in added_ids:
                continue
                
            # Get engagement from database
            engagement = self.session.query(EngagementContent).filter_by(engagement_id=engagement_id).first()
            
            if not engagement:
                continue
                
            # Check if student has already had this engagement
            existing = self.session.query(EngagementHistory).filter_by(
                student_id=student_id, engagement_content_id=engagement_id
            ).first()
            
            if existing:
                continue
                
            # Apply filters
            if funnel_stage and engagement.target_funnel_stage != funnel_stage:
                score = score * 0.5  # Reduce score for non-matching funnel stage
            
            if risk_level and engagement.appropriate_for_risk_level != risk_level:
                score = score * 0.5  # Reduce score for non-matching risk level
                
            # Add to recommendations
            recommendations.append({
                "content_id": engagement.content_id,
                "engagement_type": engagement.engagement_type,
                "content_category": engagement.content_category,
                "content_description": engagement.content_description,
                "success_rate": engagement.success_rate,
                "target_funnel_stage": engagement.target_funnel_stage,
                "score": float(score)
            })
            
            added_ids.add(engagement_id)
            
            # Stop if we have enough recommendations
            if len(recommendations) >= top_k:
                break
                
        return recommendations
    
    def _get_mock_recommendations(self, student_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Get mock recommendations when model is not available.
        
        Args:
            student_id: Unique identifier for the student
            top_k: Number of recommendations to return
            
        Returns:
            List of mock recommended engagements
        """
        # Get student from database
        student = self.session.query(StudentProfile).filter_by(student_id=student_id).first()
        
        if not student:
            raise ValueError(f"Student with ID {student_id} not found")
            
        # Get some random engagements from database
        engagements = self.session.query(EngagementContent).limit(top_k).all()
        
        # Create mock recommendations
        recommendations = []
        for i, engagement in enumerate(engagements):
            recommendations.append({
                "content_id": engagement.content_id,
                "engagement_type": engagement.engagement_type,
                "content_category": engagement.content_category,
                "content_description": engagement.content_description,
                "success_rate": engagement.success_rate,
                "target_funnel_stage": engagement.target_funnel_stage,
                "score": 0.9 - (i * 0.1)  # Decreasing scores
            })
            
        return recommendations
    
    def _get_student_features(self, student_id: str) -> Dict[str, Any]:
        """
        Get features for a student.
        
        Args:
            student_id: Unique identifier for the student
            
        Returns:
            Dictionary of student features
        """
        # Query student from database
        student = self.session.query(StudentProfile).filter_by(student_id=student_id).first()
        
        if not student:
            raise ValueError(f"Student with ID {student_id} not found")
        
        # Create features dictionary
        features = {
            "student_id": student_id
            # Additional features could be added here
        }
        
        return features


class LikelihoodService:
    """Service for predicting application likelihood."""
    
    def __init__(self, model_path="models/student_engagement_model"):
        # For development, allow service to initialize without model
        self.model = None
        self.vocabularies = {}
            
        self.session = get_session()
    
    def get_application_likelihood(self, student_id: str) -> float:
        """
        Get the predicted likelihood of application completion.
        
        Args:
            student_id: Unique identifier for the student
            
        Returns:
            Percentage likelihood of application completion
        """
        # If model is not loaded yet, return mock likelihood
        if self.model is None:
            return self._get_mock_likelihood(student_id)
            
        # Get student features
        student_features = self._get_student_features(student_id)
        
        # Get student embedding
        student_embedding = self.model.get_student_embeddings(student_features)
        
        # Predict likelihood
        likelihood = self.model.predict_likelihood(student_embedding)
        
        # Convert to percentage
        return float(likelihood.numpy()[0][0] * 100)
        
    def _get_mock_likelihood(self, student_id: str) -> float:
        """
        Get mock likelihood when model is not available.
        
        Args:
            student_id: Unique identifier for the student
            
        Returns:
            Mock percentage likelihood of application completion
        """
        # Get student from database
        student = self.session.query(StudentProfile).filter_by(student_id=student_id).first()
        
        if not student:
            raise ValueError(f"Student with ID {student_id} not found")
            
        # Generate a likelihood based on funnel stage
        stage_likelihood = {
            "awareness": 20.0,
            "interest": 40.0,
            "consideration": 60.0,
            "decision": 80.0,
            "application": 95.0
        }
        
        # Return likelihood based on funnel stage, or a default value
        return stage_likelihood.get(student.funnel_stage.lower(), 50.0)
    
    def _get_student_features(self, student_id: str) -> Dict[str, Any]:
        """
        Get features for a student.
        
        Args:
            student_id: Unique identifier for the student
            
        Returns:
            Dictionary of student features
        """
        # Query student from database
        student = self.session.query(StudentProfile).filter_by(student_id=student_id).first()
        
        if not student:
            raise ValueError(f"Student with ID {student_id} not found")
        
        # Create features dictionary
        features = {
            "student_id": student_id
            # Additional features could be added here
        }
        
        return features


class RiskAssessmentService:
    """Service for assessing dropout risk."""
    
    def __init__(self):
        # For development, allow service to initialize without model
        self.model = None
        self.session = get_session()
    
    def get_dropout_risk(self, student_id: str) -> Dict[str, Any]:
        """
        Get dropout risk assessment for a specific student.
        
        Args:
            student_id: Student ID
            
        Returns:
            Dictionary containing risk assessment
        """
        # Get student
        student = self.session.query(StudentProfile).filter_by(student_id=student_id).first()
        if not student:
            raise ValueError(f"Student with ID {student_id} not found")
            
        # Get last engagement
        last_engagement = self.session.query(EngagementHistory)\
            .filter_by(student_id=student_id)\
            .order_by(EngagementHistory.timestamp.desc())\
            .first()
            
        # Calculate risk score
        if not last_engagement:
            risk_score = 0.8
        else:
            # Calculate days since last engagement
            days_since = (datetime.now() - last_engagement.timestamp).days
            
            # Higher days since = higher risk
            if days_since > 30:
                risk_score = 0.9
            elif days_since > 14:
                risk_score = 0.7
            elif days_since > 7:
                risk_score = 0.5
            else:
                risk_score = 0.2
                
            # Adjust based on funnel stage (earlier stages have higher risk)
            stage_multiplier = {
                "awareness": 1.2,
                "interest": 1.1,
                "consideration": 1.0,
                "decision": 0.9,
                "application": 0.8
            }
            
            # Apply multiplier (capped at 1.0)
            risk_score = min(1.0, risk_score * stage_multiplier.get(student.funnel_stage.lower(), 1.0))
        
        return {
            "risk_score": risk_score,
            "risk_category": self._score_to_category(risk_score)
        }
    
    def get_at_risk_students(self, risk_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Get a list of students at high risk of dropping off.
        
        Args:
            risk_threshold: Minimum risk score to include
            
        Returns:
            List of high-risk students with their risk scores
        """
        # Get all students
        students = self.session.query(StudentProfile).all()
        
        # Assess risk for each student using mock logic
        at_risk_students = []
        
        for student in students:
            # Get last engagement
            last_engagement = self.session.query(EngagementHistory)\
                .filter_by(student_id=student.student_id)\
                .order_by(EngagementHistory.timestamp.desc())\
                .first()
                
            # Determine risk based on last engagement and funnel stage
            if not last_engagement:
                risk_score = 0.8
            else:
                # Calculate days since last engagement
                days_since = (datetime.now() - last_engagement.timestamp).days
                
                # Higher days since = higher risk
                if days_since > 30:
                    risk_score = 0.9
                elif days_since > 14:
                    risk_score = 0.7
                elif days_since > 7:
                    risk_score = 0.5
                else:
                    risk_score = 0.2
                    
                # Adjust based on funnel stage (earlier stages have higher risk)
                stage_multiplier = {
                    "awareness": 1.2,
                    "interest": 1.1,
                    "consideration": 1.0,
                    "decision": 0.9,
                    "application": 0.8
                }
                
                # Apply multiplier (capped at 1.0)
                risk_score = min(1.0, risk_score * stage_multiplier.get(student.funnel_stage.lower(), 1.0))
            
            # Add to list if above threshold
            if risk_score >= risk_threshold:
                at_risk_students.append({
                    "student_id": student.student_id,
                    "demographic_features": student.demographic_features,
                    "funnel_stage": student.funnel_stage,
                    "risk_score": risk_score,
                    "risk_category": self._score_to_category(risk_score)
                })
        
        # Sort by risk score (descending)
        at_risk_students.sort(key=lambda x: x["risk_score"], reverse=True)
        
        return at_risk_students
    
    def get_high_potential_students(self, likelihood_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Get a list of students with high potential for application completion.
        
        Args:
            likelihood_threshold: Minimum application likelihood score to include
            
        Returns:
            List of high-potential students with their likelihood scores
        """
        # Get all students
        students = self.session.query(StudentProfile).all()
        
        # Assess potential for each student
        high_potential_students = []
        
        for student in students:
            # Skip students who are already in the application stage
            if student.funnel_stage == "application":
                continue
                
            # Calculate likelihood based on funnel stage and engagement
            stage_likelihood = {
                "awareness": 0.3,
                "interest": 0.5,
                "consideration": 0.7,
                "decision": 0.9
            }
            
            likelihood_score = stage_likelihood.get(student.funnel_stage.lower(), 0.5)
            
            # Add to list if above threshold
            if likelihood_score >= likelihood_threshold:
                high_potential_students.append({
                    "student_id": student.student_id,
                    "demographic_features": student.demographic_features,
                    "funnel_stage": student.funnel_stage,
                    "application_likelihood": likelihood_score,
                    "next_best_actions": self._get_next_best_actions(student.student_id, student.funnel_stage)
                })
        
        # Sort by likelihood score (descending)
        high_potential_students.sort(key=lambda x: x["application_likelihood"], reverse=True)
        
        return high_potential_students
    
    def _get_next_best_actions(self, student_id: str, current_stage: str) -> List[str]:
        """Get recommended next actions based on student's current stage"""
        stage_actions = {
            "awareness": ["Send program brochure", "Invite to virtual tour", "Share student testimonials"],
            "interest": ["Invite to campus event", "Send financial aid information", "Schedule advisor call"],
            "consideration": ["Send application checklist", "Offer application workshop", "Share department highlights"],
            "decision": ["Provide application assistance", "Send deadline reminder", "Offer fee waiver"]
        }
        return stage_actions.get(current_stage.lower(), ["Send general information"])
    
    def _score_to_category(self, score: float) -> str:
        """
        Convert risk score to risk category.
        
        Args:
            score: Risk score (0-1)
            
        Returns:
            Risk category (high, medium, low)
        """
        if score > 0.7:
            return "high"
        elif score > 0.4:
            return "medium"
        else:
            return "low"


class DashboardService:
    """Service for generating dashboard statistics."""
    
    def __init__(self):
        self.session = get_session()
        self.risk_service = RiskAssessmentService()
    
    def get_dashboard_stats(self) -> Dict[str, Any]:
        """
        Get statistics for the dashboard.
        
        Returns:
            Dictionary of dashboard statistics
        """
        # Query all students
        students = self.session.query(StudentProfile).all()
        
        # Calculate total students
        total_students = len(students)
        
        # Calculate application rate
        completed_applications = sum(1 for student in students if student.application_status == "Completed")
        application_rate = (completed_applications / total_students * 100) if total_students > 0 else 0
        
        # Calculate at-risk count
        at_risk_students = self.risk_service.get_at_risk_students()
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


class BulkActionService:
    """Service for managing bulk actions."""
    
    def __init__(self):
        self.session = get_session()
        self.recommendation_service = RecommendationService()
        self.risk_service = RiskAssessmentService()
    
    def preview_bulk_action(self, action: str, segment: str) -> Dict[str, Any]:
        """
        Preview a bulk action.
        
        Args:
            action: Type of action (email_campaign, sms_campaign, event_invitation, high_risk_intervention)
            segment: Target segment (all_active, high_likelihood, at_risk, awareness_stage, etc.)
            
        Returns:
            Preview of the bulk action
        """
        # Get students based on segment
        students = self._get_students_by_segment(segment)
        
        # Get action details
        action_details = self._get_action_details(action)
        
        # Create preview
        preview = []
        
        for student in students:
            # Get specific action for this student
            specific_action = self._get_specific_action(student, action)
            
            preview.append({
                "id": student.student_id,
                "name": f"Student {student.student_id}",  # In a real system, this would be the student's name
                "email": f"student{student.student_id}@example.com",  # In a real system, this would be the student's email
                "funnel_stage": student.funnel_stage,
                "specificAction": specific_action
            })
        
        return {
            "students": preview,
            "count": len(preview),
            "action_details": action_details
        }
    
    def apply_bulk_action(self, action: str, segment: str) -> Dict[str, Any]:
        """
        Apply a bulk action.
        
        Args:
            action: Type of action (email_campaign, sms_campaign, event_invitation, high_risk_intervention)
            segment: Target segment (all_active, high_likelihood, at_risk, awareness_stage, etc.)
            
        Returns:
            Result of the bulk action
        """
        # Get preview
        preview = self.preview_bulk_action(action, segment)
        
        # In a real system, this would actually apply the action
        # For now, we'll just return the preview as if it was applied
        
        return {
            "success": True,
            "students_affected": preview["count"],
            "action_details": preview["action_details"]
        }
    
    def _get_students_by_segment(self, segment: str) -> List[StudentProfile]:
        """
        Get students based on segment.
        
        Args:
            segment: Target segment
            
        Returns:
            List of students
        """
        if segment == "all_active":
            return self.session.query(StudentProfile).all()
        elif segment == "high_likelihood":
            return self.session.query(StudentProfile).filter(
                StudentProfile.application_likelihood_score >= 0.7
            ).all()
        elif segment == "at_risk":
            # Get at-risk students
            at_risk_students = self.risk_service.get_at_risk_students()
            student_ids = [student["student_id"] for student in at_risk_students]
            
            return self.session.query(StudentProfile).filter(
                StudentProfile.student_id.in_(student_ids)
            ).all()
        elif segment == "awareness_stage":
            return self.session.query(StudentProfile).filter_by(funnel_stage="Awareness").all()
        elif segment == "interest_stage":
            return self.session.query(StudentProfile).filter_by(funnel_stage="Interest").all()
        elif segment == "consideration_stage":
            return self.session.query(StudentProfile).filter_by(funnel_stage="Consideration").all()
        elif segment == "decision_stage":
            return self.session.query(StudentProfile).filter_by(funnel_stage="Decision").all()
        else:
            raise ValueError(f"Invalid segment: {segment}")
    
    def _get_action_details(self, action: str) -> Dict[str, Any]:
        """
        Get details for an action.
        
        Args:
            action: Type of action
            
        Returns:
            Action details
        """
        if action == "email_campaign":
            return {
                "type": "email_campaign",
                "name": "Email Campaign",
                "description": "Send personalized emails to students",
                "template": "email_template.html"
            }
        elif action == "sms_campaign":
            return {
                "type": "sms_campaign",
                "name": "SMS Campaign",
                "description": "Send personalized SMS messages to students",
                "template": "sms_template.txt"
            }
        elif action == "event_invitation":
            return {
                "type": "event_invitation",
                "name": "Event Invitation",
                "description": "Invite students to an upcoming event",
                "event_date": (datetime.now() + datetime.timedelta(days=7)).isoformat(),
                "event_location": "University Campus"
            }
        elif action == "high_risk_intervention":
            return {
                "type": "high_risk_intervention",
                "name": "High Risk Intervention",
                "description": "Targeted intervention for high-risk students",
                "intervention_type": "Personal Call"
            }
        else:
            raise ValueError(f"Invalid action: {action}")
    
    def _get_specific_action(self, student: StudentProfile, action: str) -> str:
        """
        Get specific action for a student.
        
        Args:
            student: Student profile
            action: Type of action
            
        Returns:
            Specific action description
        """
        if action == "email_campaign":
            return f"Send '{student.funnel_stage} Stage' email"
        elif action == "sms_campaign":
            return f"Send '{student.funnel_stage} Stage' SMS"
        elif action == "event_invitation":
            return f"Invite to '{student.demographic_features.get('intended_major', 'General')}' information session"
        elif action == "high_risk_intervention":
            return f"Schedule personal call with admissions counselor"
        else:
            return "Unknown action"
