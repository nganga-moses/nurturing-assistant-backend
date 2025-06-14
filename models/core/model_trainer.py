import os
import json
import tensorflow as tf
# from tensorflow import keras # Removed direct import
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from sklearn.neighbors import NearestNeighbors
import joblib
from models.core.recommender_model import RecommenderModel, StudentTower, EngagementTower
from models.core.data_quality_monitor import DataQualityMonitor
from models.core.data_generator import DataGenerator

class ModelTrainer:
    """Class for training and evaluating the student engagement model."""
    
    def __init__(self, data_dict, embedding_dimension=128):
        self.train_dataset = data_dict['train_dataset']
        self.test_dataset = data_dict['test_dataset']
        self.vocabularies = data_dict['vocabularies']
        self.dataframes = data_dict['dataframes']
        self.embedding_dimension = embedding_dimension
        
        # Create engagement corpus dataset
        self.engagement_corpus = tf.data.Dataset.from_tensor_slices({
            "engagement_id": self.vocabularies['engagement_ids'],
            "content_id": [self.dataframes['engagements'].loc[
                self.dataframes['engagements']['engagement_id'] == eid, 'engagement_content_id'
            ].iloc[0] if eid in self.dataframes['engagements']['engagement_id'].values else "" 
              for eid in self.vocabularies['engagement_ids']]
        })
        
        # Create model
        self.model = RecommenderModel(
            embedding_dimension=embedding_dimension,
            dropout_rate=0.2,
            l2_reg=0.01
        )
        
        # Define learning rate schedule
        initial_learning_rate = 0.001
        decay_steps = 1000
        decay_rate = 0.9
        self.learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True
        )
        
        # Define optimizer with learning rate schedule
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_schedule)
        
        # Initialize vocabularies for hashing
        self.student_vocab = set(self.vocabularies['student_ids'])
        self.engagement_vocab = set(self.vocabularies['engagement_ids'])
    
    def train(self, epochs=10, batch_size=32):
        """Train the model."""
        # Create model directory if it doesn't exist
        os.makedirs('models/saved_models', exist_ok=True)
        
        # Define callbacks
        callbacks = [
            # Early stopping
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True,
                verbose=1
            ),
            # Learning rate reduction on plateau
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6,
                verbose=1
            )
            # Removed model checkpointing to avoid overwrite parameter issue
        ]
        
        # Compile model with loss functions for each head
        self.model.compile(
            optimizer=self.optimizer,
            loss={
                'ranking_score': tf.keras.losses.BinaryCrossentropy(),
                'likelihood_score': tf.keras.losses.BinaryCrossentropy(),
                'risk_score': tf.keras.losses.BinaryCrossentropy()
            },
            metrics={
                'ranking_score': ['accuracy', tf.keras.metrics.AUC()],
                'likelihood_score': ['accuracy', tf.keras.metrics.AUC()],
                'risk_score': ['accuracy', tf.keras.metrics.AUC()]
            }
        )
        
        # Train model
        history = self.model.fit(
            self.train_dataset,
            epochs=epochs,
            validation_data=self.test_dataset,
            callbacks=callbacks,
            batch_size=batch_size
        )
        
        # Vector stores will be handled by ModelManager after model loading
        # Simplified approach doesn't require vector store updates during training
        
        return history
    
    def evaluate(self):
        """Evaluate the model."""
        return self.model.evaluate(self.test_dataset, return_dict=True)
    
    def save_model(self, model_dir: str):
        """Save the trained model and create nearest neighbors index."""
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Save the full model with .keras extension
        model_path = os.path.join(model_dir, "recommender_model.keras")
        self.model.save(model_path)
        print(f"✅ Model saved to {model_path}")
        
        # Create and save nearest neighbors index for engagements
        engagement = {
            "engagement_id": tf.constant([str(i) for i in range(len(self.dataframes["engagements"]))]),
            "content_id": tf.constant([str(i) for i in range(len(self.dataframes["engagements"]))]),
            "engagement_features": tf.zeros((len(self.dataframes["engagements"]), 10))
        }
        embeddings = self.model.engagement_tower(engagement)
        
        # Create nearest neighbors index
        nn_model = NearestNeighbors(n_neighbors=10, metric='cosine')
        nn_model.fit(embeddings.numpy())
        
        # Save the nearest neighbors model
        joblib.dump(nn_model, os.path.join(model_dir, "engagement_nn_model.joblib"))
    
    def load_model(self, model_dir="models"):
        """Load a trained model."""
        try:
            # Load model
            self.model = tf.saved_model.load(os.path.join(model_dir, "saved", "model_weights", "student_engagement_model"))
            
            # Load vocabularies
            vocabularies_path = os.path.join(model_dir, "saved", "vocabularies", "model_vocabularies.json")
            with open(vocabularies_path, "r") as f:
                vocabularies = json.load(f)
                self.student_vocab = set(vocabularies['student_ids'])
                self.engagement_vocab = set(vocabularies['engagement_ids'])
            
            return self.model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None 