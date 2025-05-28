"""
Simple recommendation model for the Student Engagement Recommender System.
This module provides a simpler implementation that doesn't rely on TensorFlow Recommenders.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
import tensorflow_recommenders as tfrs
from .base_recommender import BaseRecommender

class SimpleRecommender:
    """A simple content-based recommendation model for student engagements."""
    
    def __init__(self, model_dir: str = None):
        """
        Initialize the simple recommender.
        
        Args:
            model_dir: Directory to save/load the model
        """
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
        
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "simple_recommender.joblib")
        self.model = None
        self.content_data = None
        self.student_data = None
        
        # Try to load the model if it exists
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the model if it exists."""
        if os.path.exists(self.model_path):
            try:
                data = joblib.load(self.model_path)
                self.model = data.get('model')
                self.content_data = data.get('content_data')
                self.student_data = data.get('student_data')
                print(f"Loaded model from {self.model_path}")
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                self.model = None
                self.content_data = None
                self.student_data = None
    
    def train(self, student_data: pd.DataFrame, content_data: pd.DataFrame, engagement_data: pd.DataFrame = None) -> None:
        """
        Train the recommendation model.
        
        Args:
            student_data: DataFrame with student information
            content_data: DataFrame with content information
            engagement_data: Optional DataFrame with engagement history
        """
        print("Training simple recommendation model...")
        
        # Store the data
        self.student_data = student_data
        self.content_data = content_data
        
        # Create features for content items
        content_features = []
        for _, content in content_data.iterrows():
            # Create a feature string from content attributes
            feature_string = f"{content.get('engagement_type', '')} {content.get('content_category', '')} {content.get('content_description', '')} {content.get('target_funnel_stage', '')}"
            # Ensure we have some meaningful content
            if len(feature_string.strip()) < 3:
                feature_string = "default content item"
            content_features.append(feature_string)
        
        # Add a default feature if the list is empty
        if not content_features:
            content_features = ["default content item"]
        
        # Create TF-IDF vectors for content items
        vectorizer = TfidfVectorizer(stop_words=None)  # Don't filter stop words to ensure we have content
        content_vectors = vectorizer.fit_transform(content_features)
        
        # Create nearest neighbors model
        # Ensure we have at least 1 neighbor
        n_neighbors = max(1, min(20, len(content_data)))
        nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
        nn_model.fit(content_vectors)
        
        # Store the model
        self.model = {
            'vectorizer': vectorizer,
            'nn_model': nn_model,
            'content_vectors': content_vectors
        }
        
        # Save the model
        self._save_model()
        
        print("Model training completed successfully")
    
    def _save_model(self) -> None:
        """Save the model to disk."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        joblib.dump({
            'model': self.model,
            'content_data': self.content_data,
            'student_data': self.student_data
        }, self.model_path)
        
        print(f"Model saved to {self.model_path}")
    
    def get_recommendations(self, student_id: str, count: int = 3) -> List[Dict[str, Any]]:
        """
        Get recommendations for a student.
        
        Args:
            student_id: ID of the student
            count: Number of recommendations to return
            
        Returns:
            List of recommendations
        """
        if self.model is None or self.content_data is None or self.student_data is None:
            print("Model not trained yet, using rule-based recommendations")
            return self._get_rule_based_recommendations(student_id, count)
        
        try:
            # Get student data
            student = self.student_data[self.student_data['student_id'] == student_id].iloc[0] if student_id in self.student_data['student_id'].values else None
            
            if student is None:
                print(f"Student with ID {student_id} not found, using rule-based recommendations")
                return self._get_rule_based_recommendations(student_id, count)
            
            # Get student's funnel stage
            funnel_stage = student.get('funnel_stage', 'awareness')
            
            # Filter content by funnel stage if possible
            filtered_content = self.content_data[self.content_data['target_funnel_stage'] == funnel_stage] if 'target_funnel_stage' in self.content_data.columns else self.content_data
            
            # If no content matches the funnel stage, use all content
            if len(filtered_content) == 0:
                filtered_content = self.content_data
            
            # Get content features
            content_features = []
            for _, content in filtered_content.iterrows():
                feature_string = f"{content.get('engagement_type', '')} {content.get('content_category', '')} {content.get('content_description', '')} {content.get('target_funnel_stage', '')}"
                content_features.append(feature_string)
            
            # Create TF-IDF vectors for filtered content
            content_vectors = self.model['vectorizer'].transform(content_features)
            
            # Create a query vector based on student's funnel stage and preferences
            query_string = f"{funnel_stage} {student.get('demographic_features', {}).get('intended_major', '')}"
            query_vector = self.model['vectorizer'].transform([query_string])
            
            # Calculate similarity
            similarities = cosine_similarity(query_vector, content_vectors).flatten()
            
            # Get top recommendations
            top_indices = similarities.argsort()[-count:][::-1]
            
            # Create recommendations
            recommendations = []
            for idx in top_indices:
                content = filtered_content.iloc[idx]
                recommendations.append({
                    "engagement_id": content.get('content_id', f"rec_{idx}"),
                    "engagement_type": content.get('engagement_type', 'email'),
                    "content": content.get('content_description', 'Personalized engagement'),
                    "expected_effectiveness": float(similarities[idx]),
                    "rationale": f"Recommended based on student's funnel stage: {funnel_stage}"
                })
            
            return recommendations
            
        except Exception as e:
            print(f"Error getting recommendations: {str(e)}")
            return self._get_rule_based_recommendations(student_id, count)
    
    def _get_rule_based_recommendations(self, student_id: str, count: int = 3) -> List[Dict[str, Any]]:
        """
        Get rule-based recommendations when model is not available.
        
        Args:
            student_id: ID of the student
            count: Number of recommendations to return
            
        Returns:
            List of recommendations
        """
        # Define some default recommendations for each funnel stage
        funnel_stages = ['awareness', 'interest', 'consideration', 'decision', 'application']
        
        # Get student's funnel stage if available
        funnel_stage = 'awareness'  # Default
        if self.student_data is not None and student_id in self.student_data['student_id'].values:
            student = self.student_data[self.student_data['student_id'] == student_id].iloc[0]
            funnel_stage = student.get('funnel_stage', 'awareness')
        
        # Define default recommendations for each stage
        default_recommendations = {
            'awareness': [
                {"engagement_type": "email", "content": "Introduction to university programs", "effectiveness": 0.85},
                {"engagement_type": "sms", "content": "Upcoming virtual open house", "effectiveness": 0.75},
                {"engagement_type": "email", "content": "Student success stories", "effectiveness": 0.8}
            ],
            'interest': [
                {"engagement_type": "email", "content": "Program details and curriculum", "effectiveness": 0.9},
                {"engagement_type": "call", "content": "Speak with an admissions counselor", "effectiveness": 0.85},
                {"engagement_type": "email", "content": "Campus life showcase", "effectiveness": 0.8}
            ],
            'consideration': [
                {"engagement_type": "email", "content": "Financial aid opportunities", "effectiveness": 0.9},
                {"engagement_type": "sms", "content": "Application deadline reminder", "effectiveness": 0.95},
                {"engagement_type": "email", "content": "Career outcomes for graduates", "effectiveness": 0.85}
            ],
            'decision': [
                {"engagement_type": "call", "content": "Personal follow-up from admissions", "effectiveness": 0.95},
                {"engagement_type": "email", "content": "Scholarship opportunities", "effectiveness": 0.9},
                {"engagement_type": "sms", "content": "Campus visit invitation", "effectiveness": 0.85}
            ],
            'application': [
                {"engagement_type": "email", "content": "Application completion assistance", "effectiveness": 0.95},
                {"engagement_type": "sms", "content": "Final deadline reminder", "effectiveness": 0.9},
                {"engagement_type": "call", "content": "Application review and next steps", "effectiveness": 0.95}
            ]
        }
        
        # Get recommendations for the student's funnel stage
        stage_recommendations = default_recommendations.get(funnel_stage, default_recommendations['awareness'])
        
        # Format the recommendations
        recommendations = []
        for i, rec in enumerate(stage_recommendations[:count]):
            recommendations.append({
                "engagement_id": f"default_{funnel_stage}_{i}",
                "engagement_type": rec["engagement_type"],
                "content": rec["content"],
                "expected_effectiveness": rec["effectiveness"],
                "rationale": f"Default recommendation for {funnel_stage} stage"
            })
        
        return recommendations

class ContentBasedRecommender(BaseRecommender):
    """Content-based recommender using TensorFlow Recommenders."""
    
    def __init__(
        self,
        embedding_dimension: int = 64,
        learning_rate: float = 0.1,
        model_dir: str = "models/saved_models"
    ):
        """
        Initialize the content-based recommender.
        
        Args:
            embedding_dimension: Dimension of embeddings
            learning_rate: Learning rate for optimizer
            model_dir: Directory to save/load models
        """
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.learning_rate = learning_rate
        self.model_dir = model_dir
        
        # Model components
        self.student_model = None
        self.content_model = None
        self.task = None
        self.model = None
        
        # Data components
        self.student_ids = None
        self.content_ids = None
        self.content_features = None
        
    def _create_student_model(self) -> tf.keras.Model:
        """Create the student tower model."""
        return tf.keras.Sequential([
            # Student features processing
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.embedding_dimension)
        ])
    
    def _create_content_model(self) -> tf.keras.Model:
        """Create the content tower model."""
        return tf.keras.Sequential([
            # Content features processing
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.embedding_dimension)
        ])
    
    def _create_retrieval_model(self) -> tfrs.Model:
        """Create the retrieval model."""
        # Create the two towers
        self.student_model = self._create_student_model()
        self.content_model = self._create_content_model()
        
        # Create the retrieval task
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=self.content_model
            )
        )
        
        # Create the model
        return tfrs.Model(
            student_model=self.student_model,
            content_model=self.content_model,
            task=self.task
        )
    
    def prepare_data(
        self,
        students_df: pd.DataFrame,
        content_df: pd.DataFrame,
        engagements_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Prepare data for training.
        
        Args:
            students_df: DataFrame of student profiles
            content_df: DataFrame of content items
            engagements_df: DataFrame of engagement history
            
        Returns:
            Dictionary containing prepared data
        """
        # Extract unique IDs
        self.student_ids = students_df['student_id'].unique()
        self.content_ids = content_df['content_id'].unique()
        
        # Process student features
        student_features = self._process_student_features(students_df)
        
        # Process content features
        content_features = self._process_content_features(content_df)
        
        # Create TensorFlow datasets
        student_dataset = tf.data.Dataset.from_tensor_slices({
            'student_id': students_df['student_id'].values,
            'student_features': student_features
        })
        
        content_dataset = tf.data.Dataset.from_tensor_slices({
            'content_id': content_df['content_id'].values,
            'content_features': content_features
        })
        
        # Create interaction dataset
        interaction_dataset = tf.data.Dataset.from_tensor_slices({
            'student_id': engagements_df['student_id'].values,
            'content_id': engagements_df['engagement_content_id'].values,
            'engagement_response': engagements_df['engagement_response'].values
        })
        
        return {
            'student_dataset': student_dataset,
            'content_dataset': content_dataset,
            'interaction_dataset': interaction_dataset
        }
    
    def _process_student_features(self, students_df: pd.DataFrame) -> np.ndarray:
        """Process student features for the model."""
        # Extract and normalize numerical features
        features = []
        for _, row in students_df.iterrows():
            # Process demographic features
            demo_features = row['demographic_features']
            if isinstance(demo_features, str):
                demo_features = eval(demo_features)
            
            # Convert to numerical features
            feature_vector = [
                float(demo_features.get('age', 0)),
                float(demo_features.get('gpa', 0)),
                float(demo_features.get('test_score', 0))
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def _process_content_features(self, content_df: pd.DataFrame) -> np.ndarray:
        """Process content features for the model."""
        # Extract and normalize content features
        features = []
        for _, row in content_df.iterrows():
            # Process content features
            content_features = row['content_features']
            if isinstance(content_features, str):
                content_features = eval(content_features)
            
            # Convert to numerical features
            feature_vector = [
                float(content_features.get('difficulty', 0)),
                float(content_features.get('duration', 0)),
                float(content_features.get('engagement_score', 0))
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def train(
        self,
        students_df: pd.DataFrame,
        content_df: pd.DataFrame,
        engagements_df: pd.DataFrame,
        epochs: int = 5,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Train the model.
        
        Args:
            students_df: DataFrame of student profiles
            content_df: DataFrame of content items
            engagements_df: DataFrame of engagement history
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary of training metrics
        """
        # Prepare data
        data_dict = self.prepare_data(students_df, content_df, engagements_df)
        
        # Create model
        self.model = self._create_retrieval_model()
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adagrad(learning_rate=self.learning_rate)
        )
        
        # Train model
        history = self.model.fit(
            data_dict['interaction_dataset'].batch(batch_size),
            epochs=epochs,
            verbose=1
        )
        
        return history.history
    
    def get_recommendations(
        self,
        student_id: str,
        n_recommendations: int = 5
    ) -> List[str]:
        """
        Get recommendations for a student.
        
        Args:
            student_id: ID of the student
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of recommended content IDs
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create index for efficient retrieval
        index = tfrs.layers.factorized_top_k.BruteForce(self.student_model)
        index.index_from_dataset(
            tf.data.Dataset.zip((
                tf.data.Dataset.from_tensor_slices(self.content_ids),
                tf.data.Dataset.from_tensor_slices(
                    self.content_model(tf.constant(self.content_ids))
                )
            ))
        )
        
        # Get recommendations
        _, content_ids = index(tf.constant([student_id]))
        
        return content_ids[0, :n_recommendations].numpy().tolist()
    
    def save(self) -> None:
        """Save the model."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        # Save model weights
        self.model.save_weights(f"{self.model_dir}/content_based_model")
        
        # Save vocabularies
        np.save(f"{self.model_dir}/student_ids.npy", self.student_ids)
        np.save(f"{self.model_dir}/content_ids.npy", self.content_ids)
    
    def load(self) -> None:
        """Load the model."""
        # Load vocabularies
        self.student_ids = np.load(f"{self.model_dir}/student_ids.npy")
        self.content_ids = np.load(f"{self.model_dir}/content_ids.npy")
        
        # Create and load model
        self.model = self._create_retrieval_model()
        self.model.load_weights(f"{self.model_dir}/content_based_model")
