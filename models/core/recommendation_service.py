import os
import sys
import numpy as np
import joblib
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy.orm import Session

# Add the parent directory to the path so we can import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.models import StudentProfile, EngagementHistory, EngagementContent, get_session
from models.simple_recommender import SimpleRecommender

class RecommendationService:
    """Service for generating recommendations using the trained model."""
    
    def __init__(self, model_dir: str = None):
        """
        Initialize the recommendation service.
        
        Args:
            model_dir: Directory containing the trained model
        """
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
        
        self.model_dir = model_dir
        self.recommender = None
        self.content_cache = {}
        self.last_update = None
        self.update_threshold = timedelta(hours=1)  # Update model if last update was more than 1 hour ago
        
        # Try to load the model
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the trained model."""
        try:
            # Initialize the simple recommender
            self.recommender = SimpleRecommender(model_dir=self.model_dir)
            self.last_update = datetime.now()
            print("Recommender initialized successfully")
        except Exception as e:
            print(f"Error initializing recommender: {str(e)}")
            print("Recommendations will use fallback logic")
    
    def _check_and_update_model(self, session: Session) -> None:
        """
        Check if model needs updating and update if necessary.
        
        Args:
            session: Database session
        """
        if self.last_update is None or datetime.now() - self.last_update > self.update_threshold:
            print("Updating model with new data...")
            
            # Get all data
            students_df = pd.read_sql(session.query(StudentProfile).statement, session.bind)
            content_df = pd.read_sql(session.query(EngagementContent).statement, session.bind)
            engagements_df = pd.read_sql(session.query(EngagementHistory).statement, session.bind)
            
            # Update the model
            if self.recommender is None:
                self.recommender = SimpleRecommender(model_dir=self.model_dir)
            
            self.recommender.train(students_df, content_df, engagements_df)
            self.last_update = datetime.now()
            print("Model updated successfully")
    
    def get_recommendations(self, student_id: str, count: int = 3) -> List[Dict[str, Any]]:
        """
        Get recommendations for a student.
        
        Args:
            student_id: ID of the student
            count: Number of recommendations to return
            
        Returns:
            List of recommendations
        """
        session = get_session()
        
        try:
            # Get student profile
            student = session.query(StudentProfile).filter_by(student_id=student_id).first()
            
            if not student:
                print(f"Student with ID {student_id} not found in database")
                return self._get_default_recommendations(count)
            
            # Check and update model if necessary
            self._check_and_update_model(session)
            
            # Get recent engagements
            recent_engagements = session.query(EngagementHistory).filter(
                EngagementHistory.student_id == student_id,
                EngagementHistory.timestamp >= datetime.now() - timedelta(days=7)
            ).all()
            
            # Use the simple recommender to get recommendations
            if self.recommender is not None:
                recommendations = self.recommender.get_recommendations(student_id, count)
                
                # Adjust recommendations based on recent engagements
                recommendations = self._adjust_recommendations(recommendations, recent_engagements)
                
                return recommendations
            else:
                # Fall back to rule-based recommendations
                return self._get_fallback_recommendations(student, count, session)
        except Exception as e:
            print(f"Error getting recommendations: {str(e)}")
            # Fall back to rule-based recommendations if there's an error
            if student:
                return self._get_fallback_recommendations(student, count, session)
            else:
                return self._get_default_recommendations(count)
        finally:
            session.close()
    
    def _adjust_recommendations(self, recommendations: List[Dict[str, Any]], 
                              recent_engagements: List[EngagementHistory]) -> List[Dict[str, Any]]:
        """
        Adjust recommendations based on recent engagements.
        
        Args:
            recommendations: List of recommendations
            recent_engagements: List of recent engagements
            
        Returns:
            Adjusted recommendations
        """
        # Get content IDs from recent engagements
        recent_content_ids = {e.engagement_content_id for e in recent_engagements}
        
        # Adjust scores for recently engaged content
        for rec in recommendations:
            if rec.get('engagement_id') in recent_content_ids:
                # Reduce score for recently engaged content
                rec['expected_effectiveness'] *= 0.5
                rec['rationale'] += " (Recently engaged with this content)"
        
        # Sort by expected effectiveness
        recommendations.sort(key=lambda x: x.get('expected_effectiveness', 0), reverse=True)
        
        return recommendations
    
    def _get_default_recommendations(self, count: int) -> List[Dict[str, Any]]:
        """
        Get default recommendations when student is not found.
        
        Args:
            count: Number of recommendations to return
            
        Returns:
            List of default recommendations
        """
        # Define default recommendations for awareness stage
        default_recommendations = [
            {
                "engagement_id": "default_1",
                "engagement_type": "email",
                "content": "Introduction to university programs",
                "expected_effectiveness": 0.85,
                "rationale": "Default recommendation for new students"
            },
            {
                "engagement_id": "default_2",
                "engagement_type": "sms",
                "content": "Upcoming virtual open house",
                "expected_effectiveness": 0.75,
                "rationale": "Default recommendation for new students"
            },
            {
                "engagement_id": "default_3",
                "engagement_type": "email",
                "content": "Student success stories",
                "expected_effectiveness": 0.8,
                "rationale": "Default recommendation for new students"
            }
        ]
        
        # Return requested number of recommendations
        return default_recommendations[:count]
    
    def _get_fallback_recommendations(self, student: StudentProfile, count: int, session: Session) -> List[Dict[str, Any]]:
        """
        Get rule-based recommendations based on student's funnel stage.
        
        Args:
            student: Student profile
            count: Number of recommendations to return
            session: Database session
            
        Returns:
            List of recommendations
        """
        if student is None:
            return self._get_default_recommendations(count)
            
        funnel_stage = student.funnel_stage
        
        # Get content items for the student's funnel stage
        content_items = session.query(EngagementContent).filter_by(target_funnel_stage=funnel_stage).all()
        
        # If no content items found for the funnel stage, get any content items
        if not content_items:
            content_items = session.query(EngagementContent).limit(10).all()
        
        # Sort by success rate if available
        if content_items and hasattr(content_items[0], 'success_rate'):
            content_items.sort(key=lambda x: x.success_rate if x.success_rate is not None else 0, reverse=True)
        
        # Limit to requested count
        content_items = content_items[:count]
        
        # Convert to recommendations
        recommendations = []
        for item in content_items:
            recommendations.append({
                "engagement_id": item.content_id,
                "engagement_type": item.engagement_type,
                "content": item.content_description,
                "expected_effectiveness": item.success_rate if hasattr(item, 'success_rate') and item.success_rate is not None else 0.7,
                "rationale": f"Recommended based on student's funnel stage: {funnel_stage}"
            })
        
        # If we still don't have enough recommendations, add some defaults
        while len(recommendations) < count:
            idx = len(recommendations)
            recommendations.append({
                "engagement_id": f"default_{idx}",
                "engagement_type": "email",
                "content": f"Default recommendation {idx+1} for {funnel_stage} stage",
                "expected_effectiveness": 0.5,
                "rationale": f"Default recommendation for {funnel_stage} stage"
            })
        
        return recommendations
