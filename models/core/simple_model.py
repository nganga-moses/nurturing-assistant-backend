import tensorflow as tf
import keras
import numpy as np
from typing import Dict, List, Optional, Text, Tuple, Any, Union

@keras.saving.register_keras_serializable(package="Custom")
class SimpleRecommenderModel(tf.keras.Model):
    """Simple, serializable recommender model."""
    
    def __init__(self, embedding_dimension: int = 128):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        
        # Simple layers that will definitely serialize
        self.student_dense = tf.keras.layers.Dense(64, activation='relu', name='student_dense')
        self.engagement_dense = tf.keras.layers.Dense(64, activation='relu', name='engagement_dense')
        
        # Output heads
        self.ranking_head = tf.keras.layers.Dense(1, activation='sigmoid', name='ranking_head')
        self.likelihood_head = tf.keras.layers.Dense(1, activation='sigmoid', name='likelihood_head')
        self.risk_head = tf.keras.layers.Dense(1, activation='sigmoid', name='risk_head')
    
    def call(self, inputs):
        """Forward pass through the model."""
        student_features = inputs['student_features']
        engagement_features = inputs['engagement_features']
        
        # Process features
        student_processed = self.student_dense(student_features)
        engagement_processed = self.engagement_dense(engagement_features)
        
        # Combine features
        combined = tf.concat([student_processed, engagement_processed], axis=-1)
        
        # Generate predictions
        ranking_score = self.ranking_head(combined)
        likelihood_score = self.likelihood_head(combined)
        risk_score = self.risk_head(combined)
        
        return {
            'ranking_head': ranking_score,
            'likelihood_head': likelihood_score,
            'risk_head': risk_score
        }
    
    def get_config(self):
        """Return model configuration for serialization."""
        return {
            'embedding_dimension': self.embedding_dimension
        }
    
    @classmethod
    def from_config(cls, config):
        """Create model from configuration."""
        return cls(**config)

class SimpleModelTrainer:
    """Simple trainer for the basic model."""
    
    def __init__(self, model_dir: str = "models/saved_models"):
        self.model_dir = model_dir
        self.model = None
        
    def build_model(self, embedding_dimension: int = 128):
        """Build the simple model."""
        self.model = SimpleRecommenderModel(embedding_dimension)
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss={
                'ranking_head': 'binary_crossentropy',
                'likelihood_head': 'binary_crossentropy', 
                'risk_head': 'binary_crossentropy'
            },
            metrics={
                'ranking_head': ['accuracy'],
                'likelihood_head': ['accuracy'],
                'risk_head': ['accuracy']
            }
        )
        
        return self.model
    
    def train_simple(self, train_data, epochs=3):
        """Train the model with simple approach."""
        if self.model is None:
            self.build_model()
            
        # Build the model with a dummy forward pass
        dummy_input = {
            'student_features': tf.zeros((1, 10)),
            'engagement_features': tf.zeros((1, 10))
        }
        _ = self.model(dummy_input)
        
        # Train
        history = self.model.fit(
            train_data,
            epochs=epochs,
            verbose=1
        )
        
        return history
    
    def save_model(self):
        """Save the trained model."""
        import os
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, "simple_recommender_model.keras")
        
        print(f"💾 Saving model to {model_path}")
        self.model.save(model_path)
        print("✅ Model saved successfully!")
        
        return model_path 