import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np
from typing import Dict, List, Optional, Text, Tuple, Any
import os
import json
import joblib
from sklearn.neighbors import NearestNeighbors
from data.processing.vector_store import VectorStore
from data.processing.engagement_handler import DynamicEngagementHandler
from data.processing.quality_monitor import DataQualityMonitor


class StudentTower(tf.keras.Model):
    """Tower for processing student features."""
    
    def __init__(self, student_ids: List[str], embedding_dimension: int):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.student_embedding = tf.keras.layers.Embedding(
            len(student_ids) + 1,  # +1 for unknown
            embedding_dimension,
            name="student_embedding"
        )
        self.student_hashing = tf.keras.layers.Hashing(
            num_bins=len(student_ids) + 1,
            name="student_hashing"
        )
        self.feature_processing = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(embedding_dimension, activation='relu')
        ])
    
    def build(self, input_shape):
        """Build the model."""
        self.feature_processing.build((None, 10))  # 10 student features
        super().build(input_shape)
    
    def call(self, inputs: Dict[str, Any], training: bool = False) -> tf.Tensor:
        """Process inputs and return student embedding."""
        # Extract inputs
        student_id = inputs["student_id"]
        student_features = inputs.get("student_features", tf.zeros([tf.shape(student_id)[0], 10]))
        
        # Convert string ID to integer index using hashing
        student_index = self.student_hashing(student_id)
        
        # Get base student embedding
        base_embedding = self.student_embedding(student_index)
        
        # Process student features
        feature_embedding = self.feature_processing(student_features)
        
        # Combine embeddings
        combined_embedding = base_embedding + feature_embedding
        
        # Normalize
        return tf.nn.l2_normalize(combined_embedding, axis=-1)
    
    def get_student_embedding(self, student_id: str) -> tf.Tensor:
        """Get the current embedding for a student."""
        if student_id in self.student_embeddings:
            return self.student_embeddings[student_id]
        return None
    
    def get_embedding_updates(self, student_id: str) -> int:
        """Get the number of times a student's embedding has been updated."""
        return self.embedding_updates.get(student_id, 0)
    
    def save_vector_store(self, path: str):
        """Save vector store to disk."""
        tf.saved_model.save(self.vector_store, path)
    
    def load_vector_store(self, path: str):
        """Load vector store from disk."""
        self.vector_store = tf.saved_model.load(path)

    def update_student_embeddings(self, student_ids: List[str], embeddings: List[tf.Tensor]) -> None:
        """Update student embeddings for a batch of student IDs."""
        for student_id, embedding in zip(student_ids, embeddings):
            if student_id not in self.student_embeddings:
                self.student_embeddings[student_id] = embedding
                self.embedding_updates[student_id] = 1
            else:
                # Update with moving average
                alpha = 0.1  # Learning rate for updates
                self.student_embeddings[student_id] = (
                    (1 - alpha) * self.student_embeddings[student_id] + 
                    alpha * embedding
                )
                self.embedding_updates[student_id] += 1


class EngagementTower(tf.keras.Model):
    """Tower for processing engagement features."""
    
    def __init__(self, engagement_ids: List[str], embedding_dimension: int):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.engagement_embedding = tf.keras.layers.Embedding(
            len(engagement_ids) + 1,  # +1 for unknown
            embedding_dimension,
            name="engagement_embedding"
        )
        self.engagement_hashing = tf.keras.layers.Hashing(
            num_bins=len(engagement_ids) + 1,
            name="engagement_hashing"
        )
        self.feature_processing = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(embedding_dimension, activation='relu')
        ])
    
    def build(self, input_shape):
        """Build the model."""
        self.feature_processing.build((None, 10))  # 10 engagement features
        super().build(input_shape)
    
    def call(self, inputs: Dict[str, Any], training: bool = False) -> tf.Tensor:
        """Process inputs and return engagement embedding."""
        # Extract inputs
        engagement_id = inputs["engagement_id"]
        engagement_features = inputs.get("engagement_features", tf.zeros([tf.shape(engagement_id)[0], 10]))
        
        # If features are a dictionary, convert to tensor
        if isinstance(engagement_features, dict):
            feature_names = ["type", "duration", "difficulty", "prerequisites", "popularity", "success_rate",
                            "engagement_level", "feedback_score", "completion_rate", "interaction_frequency"]
            engagement_features = tf.stack([engagement_features.get(k, tf.zeros([tf.shape(engagement_id)[0]])) 
                                          for k in feature_names], axis=1)
        
        # Convert string ID to integer index using hashing
        engagement_index = self.engagement_hashing(engagement_id)
        
        # Get base engagement embedding
        base_embedding = self.engagement_embedding(engagement_index)
        
        # Process engagement features
        feature_embedding = self.feature_processing(engagement_features)
        
        # Combine embeddings
        combined_embedding = base_embedding + feature_embedding
        
        # Normalize
        return tf.nn.l2_normalize(combined_embedding, axis=-1)
    
    def get_engagement_embedding(self, engagement_id: str) -> tf.Tensor:
        """Get the current embedding for an engagement."""
        if engagement_id in self.engagement_embeddings:
            return self.engagement_embeddings[engagement_id]
        return None
    
    def get_embedding_updates(self, engagement_id: str) -> int:
        """Get the number of times an engagement's embedding has been updated."""
        return self.embedding_updates.get(engagement_id, 0)
    
    def save_vector_store(self, path: str):
        """Save vector store to disk."""
        tf.saved_model.save(self.vector_store, path)
    
    def load_vector_store(self, path: str):
        """Load vector store from disk."""
        self.vector_store = tf.saved_model.load(path)

    def update_engagement_embeddings(self, engagement_ids: List[str], embeddings: List[tf.Tensor]) -> None:
        """Update engagement embeddings for a batch of engagement IDs."""
        for engagement_id, embedding in zip(engagement_ids, embeddings):
            if engagement_id not in self.engagement_embeddings:
                self.engagement_embeddings[engagement_id] = embedding
                self.embedding_updates[engagement_id] = 1
            else:
                # Update with moving average
                alpha = 0.1  # Learning rate for updates
                self.engagement_embeddings[engagement_id] = (
                    (1 - alpha) * self.engagement_embeddings[engagement_id] + 
                    alpha * embedding
                )
                self.embedding_updates[engagement_id] += 1


class RecommenderModel(tf.keras.Model):
    """Model for recommending engagements to students."""
    
    def __init__(self, student_ids: List[str], engagement_ids: List[str], embedding_dimension: int = 64):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.student_tower = StudentTower(student_ids, embedding_dimension)
        self.engagement_tower = EngagementTower(engagement_ids, embedding_dimension)
        self.ranking_head = tf.keras.layers.Dense(1, activation='sigmoid', name="ranking_head")
        self.likelihood_head = tf.keras.layers.Dense(1, activation='sigmoid', name="likelihood_head")
        self.risk_head = tf.keras.layers.Dense(1, activation='sigmoid', name="risk_head")
    
    def build(self, input_shape):
        """Build the model."""
        # Build student tower
        self.student_tower.build(input_shape)
        
        # Build engagement tower
        self.engagement_tower.build(input_shape)
        
        # Build prediction heads
        self.ranking_head.build((None, self.embedding_dimension))
        self.likelihood_head.build((None, self.embedding_dimension))
        self.risk_head.build((None, self.embedding_dimension))
        
        super().build(input_shape)
    
    def call(self, inputs: Dict[str, Any], training: bool = False) -> Dict[str, tf.Tensor]:
        """Process inputs and return predictions."""
        # Get embeddings
        student_embedding = self.student_tower(inputs)
        engagement_embedding = self.engagement_tower(inputs)
        
        # Combine embeddings
        combined_embedding = student_embedding * engagement_embedding
        
        # Generate predictions
        ranking_score = self.ranking_head(combined_embedding)
        likelihood_score = self.likelihood_head(combined_embedding)
        risk_score = self.risk_head(combined_embedding)
        
        return {
            "ranking_score": ranking_score,
            "likelihood_score": likelihood_score,
            "risk_score": risk_score
        }
    
    def save(self, model_dir: str):
        """Save model to disk."""
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
        # Save student vector store
        self.student_tower.save_vector_store(os.path.join(model_dir, "student_vector_store"))
        
        # Save engagement vector store
        self.engagement_tower.save_vector_store(os.path.join(model_dir, "engagement_vector_store"))
        
        # Save quality metrics
        metrics = {
            "ranking_rmse": self.ranking_head.rmse.result().numpy(),
            "likelihood_auc": self.likelihood_head.auc.result().numpy(),
            "risk_auc": self.risk_head.auc.result().numpy()
        }
        with open(os.path.join(model_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f)
    
    def load(self, model_dir: str):
        """Load model from disk."""
        # Load student vector store
        self.student_tower.load_vector_store(os.path.join(model_dir, "student_vector_store"))
        
        # Load engagement vector store
        self.engagement_tower.load_vector_store(os.path.join(model_dir, "engagement_vector_store"))
        
        # Load quality metrics
        with open(os.path.join(model_dir, "metrics.json"), "r") as f:
            metrics = json.load(f)
            self.ranking_head.rmse.update_state(metrics["ranking_rmse"])
            self.likelihood_head.auc.update_state(metrics["likelihood_auc"])
            self.risk_head.auc.update_state(metrics["risk_auc"])

    def update_embeddings(self, student_ids: List[str], engagement_ids: List[str]):
        """Update student and engagement embeddings for all IDs."""
        # Update student embeddings
        student_embeddings = []
        for student_id in student_ids:
            emb = self.student_tower({
                "student_id": tf.convert_to_tensor([student_id]),
                "student_features": {k: tf.convert_to_tensor([0.0]) for k in [
                    "age", "gender", "ethnicity", "location", "gpa", "test_scores", "courses", "major",
                    "attendance", "participation", "feedback", "study_habits", "social_activity", "stress_level"
                ]}
            })
            student_embeddings.append(emb[0])
        self.student_tower.update_student_embeddings(student_embeddings, student_ids)

        # Update engagement embeddings
        engagement_embeddings = []
        for engagement_id in engagement_ids:
            emb = self.engagement_tower({
                "engagement_id": tf.convert_to_tensor([engagement_id]),
                "engagement_features": {k: tf.convert_to_tensor([0.0]) for k in [
                    "type", "duration", "difficulty", "prerequisites", "popularity", "success_rate"
                ]}
            })
            engagement_embeddings.append(emb[0])
        self.engagement_tower.update_engagement_embeddings(engagement_embeddings, engagement_ids)


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
        self.model = tf.saved_model.load(os.path.join(model_dir, "student_engagement_model"))
        
        # Load vocabularies
        with open(os.path.join(model_dir, "vocabularies.json"), "r") as f:
            self.vocabularies = json.load(f)
        
        return self.model
