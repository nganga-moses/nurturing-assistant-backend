import os
import json
import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from sklearn.neighbors import NearestNeighbors
import joblib
from models.core.recommender_model import RecommenderModel
from models.core.data_quality_monitor import DataQualityMonitor
from models.core.data_generator import DataGenerator

class ModelTrainer:
    """Class for training and evaluating the student engagement model."""
    
    def __init__(self, data_dict, embedding_dimension=64):
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
            student_ids=self.vocabularies['student_ids'],
            engagement_ids=self.vocabularies['engagement_ids'],
            embedding_dimension=self.embedding_dimension
        )
        
        # Define optimizer
        self.optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.1)
        
        # Initialize vocabularies for hashing
        self.student_vocab = set(self.vocabularies['student_ids'])
        self.engagement_vocab = set(self.vocabularies['engagement_ids'])
    
    def train(self, epochs=5):
        """Train the model."""
        # Compile model
        self.model.compile(optimizer=self.optimizer)
        
        # Train model
        history = self.model.fit(
            self.train_dataset,
            epochs=epochs,
            validation_data=self.test_dataset
        )
        
        return history
    
    def evaluate(self):
        """Evaluate the model."""
        return self.model.evaluate(self.test_dataset, return_dict=True)
    
    def save_model(self, model_dir: str = "models") -> None:
        """
        Save the trained model.
        
        Args:
            model_dir: Directory to save the model
        """
        # Ensure the model directory exists
        os.makedirs(os.path.join(model_dir, "saved", "model_weights"), exist_ok=True)
        os.makedirs(os.path.join(model_dir, "saved", "vocabularies"), exist_ok=True)
        os.makedirs(os.path.join(model_dir, "saved_models", "collaborative"), exist_ok=True)
        
        # Save model weights
        student_tower_weights_path = os.path.join(model_dir, "saved", "model_weights", "student_tower_weights.weights.h5")
        engagement_tower_weights_path = os.path.join(model_dir, "saved", "model_weights", "engagement_tower_weights.weights.h5")
        
        self.model.student_tower.save_weights(student_tower_weights_path)
        self.model.engagement_tower.save_weights(engagement_tower_weights_path)
        
        # Save vocabularies
        vocabularies_path = os.path.join(model_dir, "saved", "vocabularies", "model_vocabularies.json")
        with open(vocabularies_path, "w") as f:
            json.dump({
                'student_ids': list(self.student_vocab),
                'engagement_ids': list(self.engagement_vocab)
            }, f)
        
        # Create and save nearest neighbors model for fast retrieval
        print("Creating nearest neighbors model...")
        
        # Get engagement embeddings
        engagement_embeddings = []
        engagement_ids = []
        
        # Process engagement corpus in batches
        for engagement in self.engagement_corpus.batch(128):
            embeddings = self.model.engagement_tower(engagement)
            engagement_embeddings.extend(embeddings.numpy())
            engagement_ids.extend(engagement["engagement_id"].numpy())
        
        # Create nearest neighbors model
        nn_model = NearestNeighbors(n_neighbors=20, algorithm='auto', metric='cosine')
        nn_model.fit(engagement_embeddings)
        
        # Save nearest neighbors model and engagement IDs
        nn_model_path = os.path.join(model_dir, "saved_models", "collaborative", "nearest_neighbors.joblib")
        joblib.dump({
            'nn_model': nn_model,
            'engagement_ids': engagement_ids,
            'engagement_embeddings': engagement_embeddings
        }, nn_model_path)
        
        print(f"Model saved to {model_dir}")
    
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