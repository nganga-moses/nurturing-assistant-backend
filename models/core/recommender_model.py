import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np
from typing import Dict, List, Optional, Text, Tuple, Any, Union
import os
import json
import joblib
from sklearn.neighbors import NearestNeighbors
from data.processing.vector_store import VectorStore
from data.processing.engagement_handler import DynamicEngagementHandler
from data.processing.quality_monitor import DataQualityMonitor


@tf.keras.utils.register_keras_serializable()
class StudentTower(tf.keras.Model):
    """Tower for processing student features."""
    
    def __init__(self, embedding_dimension: int, student_vocab: Dict[str, tf.keras.layers.StringLookup]):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.student_vocab = student_vocab
        self.vector_store = VectorStore(embedding_dimension)
        self.student_embeddings = {}
        self.embedding_updates = {}
        
        # Student feature processing
        self.student_feature_processing = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.1)
        ])
        
        # Student embedding layer
        self.student_embedding = tf.keras.layers.Dense(
            embedding_dimension,
            activation=None,
            name='student_embedding'
        )
        
    def call(self, inputs, training=False):
        if isinstance(inputs, dict):
            student_id = inputs['student_id']
            student_features = inputs['student_features']
        else:
            student_features = inputs
            
        # Process student features
        processed_features = self.student_feature_processing(student_features)
        
        # Generate student embedding
        student_embedding = self.student_embedding(processed_features)
        
        return student_embedding
    
    def update_embeddings(self, student_ids, student_features):
        """Update embeddings for a batch of students"""
        for ids_batch, features_batch in zip(student_ids, student_features):
            embeddings = self(features_batch, training=False)
            for i, student_id in enumerate(ids_batch.numpy()):
                student_id = student_id.decode('utf-8') if isinstance(student_id, bytes) else student_id
                if student_id not in self.vector_store:
                    # Initialize with a zero vector if not present
                    self.vector_store.store_vector(student_id, np.zeros(self.embedding_dimension))
                self.vector_store.update_vector(student_id, embeddings[i].numpy())
    
    def save_vector_store(self, path: str):
        """Save the vector store to disk"""
        self.vector_store.save(path)
    
    def load_vector_store(self, path: str):
        """Load the vector store from disk"""
        self.vector_store.load(path)


@tf.keras.utils.register_keras_serializable()
class EngagementTower(tf.keras.Model):
    """Tower for processing engagement features."""
    
    def __init__(self, embedding_dimension: int, engagement_vocab: Dict[str, tf.keras.layers.StringLookup]):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.engagement_vocab = engagement_vocab
        self.vector_store = VectorStore(embedding_dimension)
        self.engagement_embeddings = {}
        self.embedding_updates = {}
        
        # Engagement feature processing
        self.engagement_feature_processing = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dropout(0.1)
        ])
        
        # Engagement embedding layer
        self.engagement_embedding = tf.keras.layers.Dense(
            embedding_dimension,
            activation=None,
            name='engagement_embedding'
        )
        
    def call(self, inputs, training=False):
        if isinstance(inputs, dict):
            engagement_id = inputs['engagement_id']
            engagement_features = inputs['engagement_features']
        else:
            engagement_features = inputs
            
        # Process engagement features
        processed_features = self.engagement_feature_processing(engagement_features)
        
        # Generate engagement embedding
        engagement_embedding = self.engagement_embedding(processed_features)
        
        return engagement_embedding
    
    def update_embeddings(self, engagement_ids, engagement_features):
        """Update embeddings for a batch of engagements"""
        for ids_batch, features_batch in zip(engagement_ids, engagement_features):
            embeddings = self(features_batch, training=False)
            for i, engagement_id in enumerate(ids_batch.numpy()):
                engagement_id = engagement_id.decode('utf-8') if isinstance(engagement_id, bytes) else engagement_id
                if engagement_id not in self.vector_store:
                    # Initialize with a zero vector if not present
                    self.vector_store.store_vector(engagement_id, np.zeros(self.embedding_dimension))
                self.vector_store.update_vector(engagement_id, embeddings[i].numpy())
    
    def save_vector_store(self, path: str):
        """Save the vector store to disk"""
        self.vector_store.save(path)
    
    def load_vector_store(self, path: str):
        """Load the vector store from disk"""
        self.vector_store.load(path)


@tf.keras.utils.register_keras_serializable()
class RecommenderModel(tf.keras.Model):
    """Hybrid recommender model combining collaborative and content-based approaches."""
    
    def __init__(
        self,
        student_tower: StudentTower,
        engagement_tower: EngagementTower,
        embedding_dimension: int = 128,  # Increased from 64
        dropout_rate: float = 0.2,  # Added dropout
        l2_reg: float = 0.01  # Added L2 regularization
    ):
        super().__init__()
        self.student_tower = student_tower
        self.engagement_tower = engagement_tower
        self.embedding_dimension = embedding_dimension
        
        # Add more layers for better feature extraction
        self.student_dense1 = tf.keras.layers.Dense(
            256, 
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
        )
        self.student_dense2 = tf.keras.layers.Dense(
            128,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
        )
        self.student_dropout = tf.keras.layers.Dropout(dropout_rate)
        
        self.engagement_dense1 = tf.keras.layers.Dense(
            256,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
        )
        self.engagement_dense2 = tf.keras.layers.Dense(
            128,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
        )
        self.engagement_dropout = tf.keras.layers.Dropout(dropout_rate)
        
        # Output heads with regularization
        self.ranking_head = tf.keras.layers.Dense(
            1,
            activation='sigmoid',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
        )
        self.likelihood_head = tf.keras.layers.Dense(
            1,
            activation='sigmoid',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
        )
        self.risk_head = tf.keras.layers.Dense(
            1,
            activation='sigmoid',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
        )
        
        # Initialize vector stores
        self.student_vector_store = VectorStore(embedding_dimension)
        self.engagement_vector_store = VectorStore(embedding_dimension)
        
        # Initialize nearest neighbors models
        self.student_nn = NearestNeighbors(n_neighbors=10, metric='cosine')
        self.engagement_nn = NearestNeighbors(n_neighbors=10, metric='cosine')
        
        # Track if vector stores are initialized
        self.student_vectors_initialized = False
        self.engagement_vectors_initialized = False

    def call(self, inputs, training=False):
        """Forward pass through the model."""
        # Get student and engagement features
        student_features = inputs['student_features']
        engagement_features = inputs['engagement_features']
        
        # Get embeddings from towers
        student_embeddings = self.student_tower(student_features)
        engagement_embeddings = self.engagement_tower(engagement_features)
        
        # Apply additional layers with dropout during training
        student_embeddings = self.student_dense1(student_embeddings)
        student_embeddings = self.student_dropout(student_embeddings, training=training)
        student_embeddings = self.student_dense2(student_embeddings)
        
        engagement_embeddings = self.engagement_dense1(engagement_embeddings)
        engagement_embeddings = self.engagement_dropout(engagement_embeddings, training=training)
        engagement_embeddings = self.engagement_dense2(engagement_embeddings)
        
        # Compute similarity scores
        similarity_scores = tf.reduce_sum(
            student_embeddings * engagement_embeddings,
            axis=1,
            keepdims=True
        )
        
        # Generate predictions from each head
        ranking_score = self.ranking_head(similarity_scores)
        likelihood_score = self.likelihood_head(similarity_scores)
        risk_score = self.risk_head(similarity_scores)
        
        return {
            'ranking_score': ranking_score,
            'likelihood_score': likelihood_score,
            'risk_score': risk_score
        }
    
    def update_vector_stores(self, student_data, engagement_data):
        """Update vector stores for both towers after training"""
        self.student_tower.update_embeddings(
            student_data['student_id'],
            student_data['student_features']
        )
        self.engagement_tower.update_embeddings(
            engagement_data['engagement_id'],
            engagement_data['engagement_features']
        )
    
    def save(self, model_dir: str):
        """Save the model to disk."""
        # Save the model architecture and weights
        super().save(os.path.join(model_dir, "recommender_model.keras"))
        
        # Save the vector store
        self.student_tower.vector_store.save(os.path.join(model_dir, "student_vectors"))
        self.engagement_tower.vector_store.save(os.path.join(model_dir, "engagement_vectors"))
    
    def load(self, model_dir: str):
        """Load the model and vector stores"""
        # Load the full model
        super().load(os.path.join(model_dir, "recommender_model"))
        
        # Load vector stores
        self.student_tower.load_vector_store(os.path.join(model_dir, "student_vectors"))
        self.engagement_tower.load_vector_store(os.path.join(model_dir, "engagement_vectors"))


class ModelTrainer:
    """Class for training and evaluating the student engagement model."""
    
    def __init__(self, data_dict, embedding_dimension=64):
        self.train_dataset = data_dict['train_dataset']
        self.test_dataset = data_dict['test_dataset']
        self.vocabularies = data_dict['vocabularies']
        self.dataframes = data_dict['dataframes']
        self.embedding_dimension = embedding_dimension
        
        # Create model
        self.model = RecommenderModel(
            student_tower=StudentTower(embedding_dimension, self.vocabularies['student_vocab']),
            engagement_tower=EngagementTower(embedding_dimension, self.vocabularies['engagement_vocab'])
        )
        
        # Define optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    def train(self, epochs=5):
        """Train the model."""
        # Compile model with loss functions for each head
        self.model.compile(
            optimizer=self.optimizer,
            loss={
                'ranking_head': tf.keras.losses.BinaryCrossentropy(),
                'likelihood_head': tf.keras.losses.BinaryCrossentropy(),
                'risk_head': tf.keras.losses.BinaryCrossentropy()
            },
            metrics={
                'ranking_head': ['accuracy'],
                'likelihood_head': ['accuracy'],
                'risk_head': ['accuracy']
            }
        )
        
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
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model weights
        student_tower_weights_path = os.path.join(model_dir, "model_weights", "student_tower_weights.weights.h5")
        engagement_tower_weights_path = os.path.join(model_dir, "model_weights", "engagement_tower_weights.weights.h5")
        
        os.makedirs(os.path.dirname(student_tower_weights_path), exist_ok=True)
        os.makedirs(os.path.dirname(engagement_tower_weights_path), exist_ok=True)
        
        self.model.student_tower.save_weights(student_tower_weights_path)
        self.model.engagement_tower.save_weights(engagement_tower_weights_path)
        
        # Save vocabularies
        with open(os.path.join(model_dir, "vocabularies.json"), "w") as f:
            json.dump(self.vocabularies, f)
        
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
        joblib.dump({
            'nn_model': nn_model,
            'engagement_ids': engagement_ids,
            'engagement_embeddings': engagement_embeddings
        }, os.path.join(model_dir, "nearest_neighbors.joblib"))
        
        print(f"Model saved to {model_dir}")
    
    def load_model(self, model_dir="models"):
        """Load a trained model."""
        # Load model
        self.model = tf.saved_model.load(os.path.join(model_dir, "recommender_model"))
        
        # Load vocabularies
        with open(os.path.join(model_dir, "vocabularies.json"), "r") as f:
            self.vocabularies = json.load(f)
        
        return self.model
