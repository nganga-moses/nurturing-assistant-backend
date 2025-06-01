from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import pandas as pd
import tensorflow as tf
import logging

# Configure logging
logger = logging.getLogger(__name__)

class BaseRecommender(ABC):
    """Base class for all recommendation models."""
    
    @abstractmethod
    def train(
        self,
        student_data: pd.DataFrame,
        content_data: pd.DataFrame,
        engagement_data: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Train the recommendation model.
        
        Args:
            student_data: DataFrame with student information
            content_data: DataFrame with content information
            engagement_data: Optional DataFrame with engagement history
        """
        pass
    
    @abstractmethod
    def get_recommendations(
        self,
        student_id: str,
        count: int = 3,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recommendations for a student.
        
        Args:
            student_id: ID of the student
            count: Number of recommendations to return
            context: Optional context information
            
        Returns:
            List of recommendations
        """
        pass
    
    @abstractmethod
    def save(self, model_dir: str) -> None:
        """
        Save the model.
        
        Args:
            model_dir: Directory to save the model
        """
        pass
    
    @abstractmethod
    def load(self, model_dir: str) -> None:
        """
        Load the model.
        
        Args:
            model_dir: Directory to load the model from
        """
        pass
    
    def get_student_features(self, student_id: str) -> Dict[str, Any]:
        """
        Get features for a student.
        
        Args:
            student_id: ID of the student
            
        Returns:
            Dictionary of student features
        """
        raise NotImplementedError("Student feature extraction not implemented")
    
    def get_content_features(self, content_id: str) -> Dict[str, Any]:
        """
        Get features for content.
        
        Args:
            content_id: ID of the content
            
        Returns:
            Dictionary of content features
        """
        raise NotImplementedError("Content feature extraction not implemented")
    
    def update_student_embedding(self, student_id: str, embedding: tf.Tensor) -> None:
        """
        Update a student's embedding.
        
        Args:
            student_id: ID of the student
            embedding: New embedding vector
        """
        raise NotImplementedError("Student embedding update not implemented")
    
    def update_content_embedding(self, content_id: str, embedding: tf.Tensor) -> None:
        """
        Update content embedding.
        
        Args:
            content_id: ID of the content
            embedding: New embedding vector
        """
        raise NotImplementedError("Content embedding update not implemented")
    
    def get_similar_students(
        self,
        student_id: str,
        count: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get similar students.
        
        Args:
            student_id: ID of the student
            count: Number of similar students to return
            
        Returns:
            List of similar students
        """
        raise NotImplementedError("Similar student retrieval not implemented")
    
    def get_similar_content(
        self,
        content_id: str,
        count: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get similar content.
        
        Args:
            content_id: ID of the content
            count: Number of similar content items to return
            
        Returns:
            List of similar content items
        """
        raise NotImplementedError("Similar content retrieval not implemented")
    
    def get_explanations(
        self,
        student_id: str,
        content_id: str
    ) -> Dict[str, Any]:
        """
        Get explanations for recommendations.
        
        Args:
            student_id: ID of the student
            content_id: ID of the content
            
        Returns:
            Dictionary of explanations
        """
        raise NotImplementedError("Recommendation explanation not implemented")
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get model metrics.
        
        Returns:
            Dictionary of metrics
        """
        raise NotImplementedError("Metric calculation not implemented")
    
    def validate_data(
        self,
        student_data: pd.DataFrame,
        content_data: pd.DataFrame,
        engagement_data: Optional[pd.DataFrame] = None
    ) -> bool:
        """
        Validate input data.
        
        Args:
            student_data: DataFrame with student information
            content_data: DataFrame with content information
            engagement_data: Optional DataFrame with engagement history
            
        Returns:
            True if data is valid, False otherwise
        """
        try:
            # Check required columns
            required_student_cols = ['student_id']
            required_content_cols = ['content_id']
            
            if not all(col in student_data.columns for col in required_student_cols):
                logger.error("Missing required columns in student data")
                return False
            
            if not all(col in content_data.columns for col in required_content_cols):
                logger.error("Missing required columns in content data")
                return False
            
            # Check for empty dataframes
            if student_data.empty:
                logger.error("Student data is empty")
                return False
            
            if content_data.empty:
                logger.error("Content data is empty")
                return False
            
            # Check for duplicate IDs
            if student_data['student_id'].duplicated().any():
                logger.error("Duplicate student IDs found")
                return False
            
            if content_data['content_id'].duplicated().any():
                logger.error("Duplicate content IDs found")
                return False
            
            # Check engagement data if provided
            if engagement_data is not None:
                required_engagement_cols = ['student_id', 'content_id']
                if not all(col in engagement_data.columns for col in required_engagement_cols):
                    logger.error("Missing required columns in engagement data")
                    return False
                
                # Check for invalid references
                invalid_students = ~engagement_data['student_id'].isin(student_data['student_id'])
                invalid_content = ~engagement_data['content_id'].isin(content_data['content_id'])
                
                if invalid_students.any():
                    logger.error("Invalid student IDs in engagement data")
                    return False
                
                if invalid_content.any():
                    logger.error("Invalid content IDs in engagement data")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating data: {str(e)}")
            return False
    
    def preprocess_data(
        self,
        student_data: pd.DataFrame,
        content_data: pd.DataFrame,
        engagement_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Preprocess input data.
        
        Args:
            student_data: DataFrame with student information
            content_data: DataFrame with content information
            engagement_data: Optional DataFrame with engagement history
            
        Returns:
            Dictionary of preprocessed DataFrames
        """
        try:
            # Make copies to avoid modifying original data
            student_df = student_data.copy()
            content_df = content_data.copy()
            engagement_df = engagement_data.copy() if engagement_data is not None else None
            
            # Clean student data
            student_df = student_df.fillna({
                'funnel_stage': 'awareness',
                'demographic_features': {}
            })
            
            # Clean content data
            content_df = content_df.fillna({
                'engagement_type': 'email',
                'content_category': 'general',
                'target_funnel_stage': 'awareness'
            })
            
            # Clean engagement data if provided
            if engagement_df is not None:
                engagement_df = engagement_df.fillna({
                    'engagement_type': 'email',
                    'engagement_status': 'pending'
                })
            
            return {
                'student_data': student_df,
                'content_data': content_df,
                'engagement_data': engagement_df
            }
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            return {
                'student_data': student_data,
                'content_data': content_data,
                'engagement_data': engagement_data
            } 