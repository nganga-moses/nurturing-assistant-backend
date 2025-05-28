import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np
from typing import Dict, List, Optional, Text, Tuple
import os
import json
import joblib
from sklearn.neighbors import NearestNeighbors
from data.vector_store import VectorStore
from data.engagement_handler import DynamicEngagementHandler
from data.quality_monitor import DataQualityMonitor


class StudentTower(tf.keras.Model):
    """Model for encoding student features with collaborative filtering."""
    
    def __init__(self, student_ids: List[str], embedding_dimension: int = 64):
        super().__init__()
        
        # Embedding layer for student IDs
        self.student_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=student_ids, mask_token=None),
            tf.keras.layers.Embedding(len(student_ids) + 1, embedding_dimension)
        ])
        
        # Demographic feature processing
        self.demographic_processor = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(embedding_dimension)
        ])
        
        # Academic score processing
        self.academic_processor = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(embedding_dimension)
        ])
        
        # Engagement pattern processing
        self.engagement_processor = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(embedding_dimension)
        ])
        
        # Similar student detection
        self.similarity_layer = tf.keras.layers.Dense(
            embedding_dimension,
            activation='tanh'
        )
        
        # Final MLP layers for student representation
        self.dense_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(embedding_dimension),
            # Use Lambda layer for L2 normalization
            tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))
        ])
        
        # Store student embeddings for similarity computation
        self.student_embeddings = None
        
        # Initialize vector store for student embeddings
        self.vector_store = VectorStore(dimension=embedding_dimension, index_type="IP")
    
    def process_demographic_features(self, features):
        """Process demographic features into embeddings."""
        # Extract and process demographic features
        location = features.get("location", "")
        age = features.get("age", 0)
        intended_major = features.get("intended_major", "")
        
        # Combine features
        demographic_features = tf.concat([
            tf.strings.to_hash_bucket_fast(location, 100),
            tf.cast(age, tf.float32),
            tf.strings.to_hash_bucket_fast(intended_major, 100)
        ], axis=1)
        
        return self.demographic_processor(demographic_features)
    
    def process_academic_scores(self, features):
        """Process academic scores into embeddings."""
        # Extract and process academic scores
        gpa = features.get("high_school_gpa", 0.0)
        sat = features.get("academic_scores", {}).get("SAT", 0)
        act = features.get("academic_scores", {}).get("ACT", 0)
        
        # Combine scores
        academic_scores = tf.stack([
            tf.cast(gpa, tf.float32),
            tf.cast(sat, tf.float32),
            tf.cast(act, tf.float32)
        ], axis=1)
        
        return self.academic_processor(academic_scores)
    
    def process_engagement_patterns(self, features):
        """Process engagement patterns into embeddings."""
        # Extract engagement features
        interaction_count = features.get("interaction_count", 0)
        funnel_stage = features.get("funnel_stage", "")
        application_likelihood = features.get("application_likelihood_score", 0.0)
        dropout_risk = features.get("dropout_risk_score", 0.0)
        
        # Combine features
        engagement_features = tf.stack([
            tf.cast(interaction_count, tf.float32),
            tf.strings.to_hash_bucket_fast(funnel_stage, 10),
            tf.cast(application_likelihood, tf.float32),
            tf.cast(dropout_risk, tf.float32)
        ], axis=1)
        
        return self.engagement_processor(engagement_features)
    
    def find_similar_students(self, student_embedding, all_student_embeddings, k=5):
        """Find k most similar students using cosine similarity."""
        # Calculate cosine similarity
        similarities = tf.matmul(
            student_embedding,
            all_student_embeddings,
            transpose_b=True
        )
        
        # Get top k similar students
        top_k_values, top_k_indices = tf.nn.top_k(similarities, k=k)
        
        return top_k_values, top_k_indices
    
    def call(self, inputs, training=False):
        # Extract student ID and features
        student_id = inputs["student_id"]
        student_features = inputs.get("student_features", {})
        
        # Get base embedding for student ID
        student_embedding = self.student_embedding(student_id)
        
        # Process different feature types
        demographic_embedding = self.process_demographic_features(student_features)
        academic_embedding = self.process_academic_scores(student_features)
        engagement_embedding = self.process_engagement_patterns(student_features)
        
        # Combine all embeddings
        combined_embedding = tf.concat([
            student_embedding,
            demographic_embedding,
            academic_embedding,
            engagement_embedding
        ], axis=1)
        
        # Find similar students if we have stored embeddings
        if self.student_embeddings is not None and training:
            similar_student_values, similar_student_indices = self.find_similar_students(
                combined_embedding,
                self.student_embeddings
            )
            
            # Get embeddings of similar students
            similar_student_embeddings = tf.gather(
                self.student_embeddings,
                similar_student_indices
            )
            
            # Weight similar student embeddings by similarity
            weighted_similar_embeddings = tf.reduce_sum(
                similar_student_embeddings * tf.expand_dims(similar_student_values, -1),
                axis=1
            )
            
            # Combine with original embedding
            combined_embedding = tf.concat([
                combined_embedding,
                weighted_similar_embeddings
            ], axis=1)
        
        # Pass through final dense layers
        return self.dense_layers(combined_embedding)
    
    def update_student_embeddings(self, student_embeddings: np.ndarray, student_ids: List[str]):
        """Update stored student embeddings in vector store."""
        self.student_embeddings = student_embeddings
        self.vector_store.add_embeddings(
            ids=student_ids,
            embeddings=student_embeddings
        )
    
    def find_similar_students(self, student_embedding: np.ndarray, k: int = 5) -> Tuple[List[str], List[float]]:
        """Find k most similar students using vector store."""
        similar_ids, scores, _ = self.vector_store.search(
            student_embedding.reshape(1, -1),
            k=k
        )
        return similar_ids, scores


class EngagementTower(tf.keras.Model):
    """Model for encoding engagement features."""
    
    def __init__(self, engagement_ids: List[str], embedding_dimension: int = 64):
        super().__init__()
        
        # Initialize dynamic engagement handler
        self.engagement_handler = DynamicEngagementHandler(embedding_dimension=embedding_dimension)
        
        # MLP layers for engagement representation
        self.dense_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(embedding_dimension),
            # Use Lambda layer for L2 normalization
            tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))
        ])
    
    def add_new_engagement_type(self, new_engagement: Dict):
        """Add new engagement type without retraining."""
        self.engagement_handler.add_new_engagement_type(new_engagement)
    
    def call(self, inputs):
        # Extract engagement ID and features
        engagement_id = inputs["engagement_id"]
        engagement_features = inputs.get("engagement_features", {})
        
        # Get embedding from engagement handler
        if engagement_id in self.engagement_handler.vector_store.id_to_embedding:
            embedding = self.engagement_handler.vector_store.id_to_embedding[engagement_id]
        else:
            # Create new embedding
            embedding = self.engagement_handler.create_new_embedding({
                "type": engagement_id,
                **engagement_features
            })
        
        # Pass through dense layers
        return self.dense_layers(embedding)


class RankingModel(tf.keras.Model):
    """Model for ranking engagements."""
    
    def __init__(self, embedding_dimension: int = 64):
        super().__init__()
        
        # Deep & Cross Network for feature interactions
        self.cross_layer = tf.keras.layers.Dense(256, activation="relu")
        
        # MLP layers
        self.dense_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1)
        ])
    
    def call(self, student_embedding, engagement_embedding, collaborative_features):
        # Combine student and engagement embeddings
        combined_embedding = tf.concat([student_embedding, engagement_embedding, collaborative_features], axis=1)
        
        # Apply cross layer
        cross_output = self.cross_layer(combined_embedding)
        
        # Pass through dense layers
        return self.dense_layers(cross_output)


class LikelihoodModel(tf.keras.Model):
    """Model for predicting application likelihood."""
    
    def __init__(self, embedding_dimension: int = 64):
        super().__init__()
        
        # MLP layers
        self.dense_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])
    
    def call(self, student_embedding, engagement_embedding, collaborative_features):
        # Combine student and engagement embeddings
        combined_embedding = tf.concat([student_embedding, engagement_embedding, collaborative_features], axis=1)
        
        # Pass through dense layers
        return self.dense_layers(combined_embedding)


class RiskModel(tf.keras.Model):
    """Model for predicting dropout risk."""
    
    def __init__(self, embedding_dimension: int = 64):
        super().__init__()
        
        # MLP layers
        self.dense_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])
    
    def call(self, student_embedding, engagement_embedding, collaborative_features):
        # Combine student and engagement embeddings
        combined_embedding = tf.concat([student_embedding, engagement_embedding, collaborative_features], axis=1)
        
        # Pass through dense layers
        return self.dense_layers(combined_embedding)


class StudentEngagementModel(tf.keras.Model):
    """Model for student engagement recommendations with collaborative filtering."""
    
    def __init__(
        self,
        student_ids: List[str],
        engagement_ids: List[str],
        embedding_dimension: int = 64
    ):
        super().__init__()
        
        # Student and engagement towers
        self.student_tower = StudentTower(student_ids, embedding_dimension)
        self.engagement_tower = EngagementTower(engagement_ids, embedding_dimension)
        
        # Task-specific models
        self.ranking_model = RankingModel(embedding_dimension)
        self.likelihood_model = LikelihoodModel(embedding_dimension)
        self.risk_model = RiskModel(embedding_dimension)
        
        # Collaborative feature processing
        self.collaborative_processor = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(embedding_dimension)
        ])
        
        # Task-specific collaborative processors
        self.ranking_collaborative = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu")
        ])
        
        self.likelihood_collaborative = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu")
        ])
        
        self.risk_collaborative = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu")
        ])
        
        # Initialize quality monitor
        self.quality_monitor = DataQualityMonitor()
    
    def process_collaborative_features(self, student_embedding, engagement_embedding):
        """Process collaborative features for all tasks."""
        # Combine student and engagement embeddings
        combined = tf.concat([student_embedding, engagement_embedding], axis=1)
        
        # Process through collaborative processor
        collaborative_features = self.collaborative_processor(combined)
        
        # Process for each task
        ranking_features = self.ranking_collaborative(collaborative_features)
        likelihood_features = self.likelihood_collaborative(collaborative_features)
        risk_features = self.risk_collaborative(collaborative_features)
        
        return ranking_features, likelihood_features, risk_features
    
    def call(self, inputs, training=False):
        # Extract inputs
        student_id = inputs["student_id"]
        student_features = inputs.get("student_features", {})
        engagement_id = inputs["engagement_id"]
        engagement_features = inputs.get("engagement_features", {})
        
        # Get student and engagement embeddings
        student_embedding = self.student_tower({
            "student_id": student_id,
            "student_features": student_features
        }, training=training)
        
        engagement_embedding = self.engagement_tower({
            "engagement_id": engagement_id,
            "engagement_features": engagement_features
        })
        
        # Process collaborative features
        ranking_collab, likelihood_collab, risk_collab = self.process_collaborative_features(
            student_embedding,
            engagement_embedding
        )
        
        # Get task-specific predictions
        ranking_score = self.ranking_model(
            student_embedding,
            engagement_embedding,
            ranking_collab
        )
        
        likelihood_score = self.likelihood_model(
            student_embedding,
            engagement_embedding,
            likelihood_collab
        )
        
        risk_score = self.risk_model(
            student_embedding,
            engagement_embedding,
            risk_collab
        )
        
        return {
            "ranking_score": ranking_score,
            "likelihood_score": likelihood_score,
            "risk_score": risk_score
        }
    
    def add_new_engagement_type(self, new_engagement: Dict):
        """Add new engagement type without retraining."""
        self.engagement_tower.add_new_engagement_type(new_engagement)
    
    def update_student_embeddings(self, student_embeddings: np.ndarray, student_ids: List[str]):
        """Update stored student embeddings in the student tower."""
        self.student_tower.update_student_embeddings(student_embeddings, student_ids)
    
    def save(self, model_dir: str = "models"):
        """Save the model and its components."""
        # Save model weights
        super().save_weights(os.path.join(model_dir, "model_weights"))
        
        # Save engagement handler
        self.engagement_tower.engagement_handler.save(
            os.path.join(model_dir, "engagement_handler")
        )
        
        # Save student vector store
        self.student_tower.vector_store.save(
            os.path.join(model_dir, "student_vectors")
        )
        
        # Save quality metrics
        self.quality_monitor.save_metrics()
    
    def load(self, model_dir: str = "models"):
        """Load the model and its components."""
        # Load model weights
        super().load_weights(os.path.join(model_dir, "model_weights"))
        
        # Load engagement handler
        self.engagement_tower.engagement_handler.load(
            os.path.join(model_dir, "engagement_handler")
        )
        
        # Load student vector store
        self.student_tower.vector_store.load(
            os.path.join(model_dir, "student_vectors")
        )
        
        # Load quality metrics
        self.quality_monitor.load_metrics()


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
        self.model = StudentEngagementModel(
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
        
        # Save the model weights
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
