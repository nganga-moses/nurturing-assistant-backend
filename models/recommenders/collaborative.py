import tensorflow as tf
import tensorflow_recommenders as tfrs
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from .base_recommender import BaseRecommender
from .utils import (
    create_student_tower,
    create_candidate_tower,
    create_retrieval_model,
    prepare_training_data,
    calculate_metrics
)
import os

# Configure logging
logger = logging.getLogger(__name__)

class CollaborativeFilteringModel(BaseRecommender):
    """Collaborative filtering model using TensorFlow Recommenders."""
    
    def __init__(
        self,
        embedding_dimension: int = 64,
        learning_rate: float = 0.1,
        model_dir: str = "models/saved_models"
    ):
        """
        Initialize the collaborative filtering model.
        
        Args:
            embedding_dimension: Dimension of embeddings
            learning_rate: Learning rate for optimizer
            model_dir: Directory to save/load models
        """
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.learning_rate = learning_rate
        self.model_dir = model_dir
        self.model = None
        self.student_ids = None
        self.content_ids = None
        self.model_version = "v2.0"
    
    def train(
        self,
        student_data: pd.DataFrame,
        content_data: pd.DataFrame,
        engagement_data: Optional[pd.DataFrame] = None,
        epochs: int = 5,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Train the model.
        
        Args:
            student_data: DataFrame of student profiles
            content_data: DataFrame of content information
            engagement_data: Optional DataFrame of engagement history
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary of training metrics
        """
        logger.info("Preparing data for training...")
        data_dict = prepare_training_data(student_data, content_data, engagement_data)
        
        # Store IDs for later use
        self.student_ids = data_dict['student_ids']
        self.content_ids = data_dict['content_ids']
        
        # Create models
        logger.info("Creating models...")
        student_model = create_student_tower(
            self.student_ids,
            self.embedding_dimension
        )
        candidate_model = create_candidate_tower(
            self.content_ids,
            self.embedding_dimension
        )
        
        # Create retrieval model
        self.model = create_retrieval_model(
            student_model,
            candidate_model,
            data_dict['candidate_ids']
        )
        
        # Compile model
        logger.info("Compiling model...")
        self.model.compile(
            optimizer=tf.keras.optimizers.Adagrad(learning_rate=self.learning_rate)
        )
        
        # Train model
        logger.info("Training model...")
        history = self.model.fit(
            data_dict['interaction_dataset'].batch(batch_size),
            epochs=epochs,
            verbose=1
        )
        
        return history.history
    
    def get_recommendations(
        self,
        student_id: str,
        count: int = 5,
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
        if self.model is None:
            logger.warning("Model not trained yet")
            return []
        
        try:
            # Get student embedding
            student_embedding = self.model.student_model(
                tf.constant([student_id])
            )
            
            # Get top-k candidates
            candidates = self.model.candidate_model(
                tf.constant(self.content_ids)
            )
            
            # Calculate scores
            scores = tf.matmul(student_embedding, candidates, transpose_b=True)
            
            # Get top-k indices
            top_k = tf.math.top_k(scores, k=count)
            
            # Create recommendations
            recommendations = []
            for idx, score in zip(top_k.indices[0], top_k.values[0]):
                content_id = self.content_ids[idx]
                recommendations.append({
                    'content_id': content_id,
                    'score': float(score),
                    'rank': len(recommendations) + 1
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            return []
    
    def save(self, model_dir: str) -> None:
        """
        Save the model.
        
        Args:
            model_dir: Directory to save the model to
        """
        try:
            # Create directories
            os.makedirs(os.path.join(model_dir, "saved", "model_weights"), exist_ok=True)
            os.makedirs(os.path.join(model_dir, "saved_models", "collaborative"), exist_ok=True)
            
            # Save metadata
            metadata = {
                'embedding_dimension': self.embedding_dimension,
                'learning_rate': self.learning_rate,
                'student_ids': self.student_ids,
                'content_ids': self.content_ids,
                'model_version': self.model_version
            }
            
            metadata_path = os.path.join(model_dir, "saved_models", "collaborative", "collaborative_metadata.npy")
            np.save(metadata_path, metadata)
            
            # Save model weights
            weights_path = os.path.join(model_dir, "saved", "model_weights", "collaborative_model")
            self.model.save_weights(weights_path)
            
            logger.info(f"Model saved to {model_dir}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load(self, model_dir: str) -> None:
        """
        Load the model.
        
        Args:
            model_dir: Directory to load the model from
        """
        try:
            # Load metadata
            metadata_path = os.path.join(model_dir, "saved_models", "collaborative", "collaborative_metadata.npy")
            metadata = np.load(metadata_path, allow_pickle=True).item()
            
            # Set attributes
            self.embedding_dimension = metadata['embedding_dimension']
            self.learning_rate = metadata['learning_rate']
            self.student_ids = metadata['student_ids']
            self.content_ids = metadata['content_ids']
            self.model_version = metadata['model_version']
            
            # Create models
            student_model = create_student_tower(
                self.student_ids,
                self.embedding_dimension
            )
            candidate_model = create_candidate_tower(
                self.content_ids,
                self.embedding_dimension
            )
            
            # Create retrieval model
            self.model = create_retrieval_model(
                student_model,
                candidate_model,
                self.content_ids
            )
            
            # Load weights
            weights_path = os.path.join(model_dir, "saved", "model_weights", "collaborative_model")
            self.model.load_weights(weights_path)
            
            logger.info(f"Model loaded from {model_dir}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.model = None
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get model metrics.
        
        Returns:
            Dictionary of metrics
        """
        if self.model is None:
            logger.warning("No model to get metrics from")
            return {}
        
        try:
            # Create test dataset
            test_data = tf.data.Dataset.from_tensor_slices({
                'student_id': self.student_ids,
                'candidate_id': self.content_ids
            }).batch(32)
            
            # Calculate metrics
            return calculate_metrics(self.model, test_data)
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {} 