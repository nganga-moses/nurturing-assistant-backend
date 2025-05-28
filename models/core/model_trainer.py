import os
import json
import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from sklearn.neighbors import NearestNeighbors
import joblib

class ModelTrainer:
    """Consolidated model trainer for all recommendation models."""
    
    def __init__(
        self,
        model: tf.keras.Model,
        data_dict: Dict,
        embedding_dimension: int = 64,
        model_dir: str = "models"
    ):
        """
        Initialize the model trainer.
        
        Args:
            model: The model to train
            data_dict: Dictionary containing training data
            embedding_dimension: Dimension of embeddings
            model_dir: Directory to save models
        """
        self.model = model
        self.train_dataset = data_dict['train_dataset']
        self.test_dataset = data_dict['test_dataset']
        self.vocabularies = data_dict['vocabularies']
        self.dataframes = data_dict['dataframes']
        self.embedding_dimension = embedding_dimension
        self.model_dir = model_dir
        
        # Create engagement corpus dataset
        self.engagement_corpus = tf.data.Dataset.from_tensor_slices({
            "engagement_id": self.vocabularies['engagement_ids'],
            "content_id": [self.dataframes['engagements'].loc[
                self.dataframes['engagements']['engagement_id'] == eid, 'engagement_content_id'
            ].iloc[0] if eid in self.dataframes['engagements']['engagement_id'].values else "" 
              for eid in self.vocabularies['engagement_ids']]
        })
        
        # Define optimizer
        self.optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.1)
        
        # Define metrics
        self.metrics = {
            'ranking': tf.keras.metrics.RootMeanSquaredError(name='ranking_rmse'),
            'likelihood': [
                tf.keras.metrics.AUC(name='likelihood_auc'),
                tf.keras.metrics.BinaryAccuracy(name='likelihood_accuracy')
            ],
            'risk': [
                tf.keras.metrics.AUC(name='risk_auc'),
                tf.keras.metrics.BinaryAccuracy(name='risk_accuracy')
            ]
        }
    
    def train(self, epochs: int = 5, batch_size: int = 32) -> Dict[str, float]:
        """
        Train the model.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary of training metrics
        """
        print(f"Training model for {epochs} epochs...")
        
        # Compile model
        self.model.compile(optimizer=self.optimizer)
        
        # Training loop
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Reset metrics
            self._reset_metrics()
            
            # Training step
            for batch in self.train_dataset:
                with tf.GradientTape() as tape:
                    # Forward pass
                    predictions = self.model(batch, training=True)
                    
                    # Calculate losses
                    losses = self._calculate_losses(batch, predictions)
                    total_loss = sum(losses.values())
                
                # Calculate gradients and update weights
                gradients = tape.gradient(total_loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                
                # Update metrics
                self._update_metrics(batch, predictions)
            
            # Print metrics
            self._print_metrics()
            
            # Update student embeddings for next epoch
            self._update_student_embeddings()
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        self._reset_metrics()
        
        for batch in self.test_dataset:
            predictions = self.model(batch, training=False)
            self._update_metrics(batch, predictions)
        
        # Return final metrics
        return self._get_metrics()
    
    def _reset_metrics(self) -> None:
        """Reset all metrics."""
        for metric in self.metrics.values():
            if isinstance(metric, list):
                for m in metric:
                    m.reset_states()
            else:
                metric.reset_states()
    
    def _calculate_losses(self, batch: Dict, predictions: Dict) -> Dict[str, tf.Tensor]:
        """Calculate losses for each task."""
        return {
            'ranking': tf.keras.losses.mean_squared_error(
                batch['engagement_response'],
                predictions['ranking_score']
            ),
            'likelihood': tf.keras.losses.binary_crossentropy(
                batch['application_completion'],
                predictions['likelihood_score']
            ),
            'risk': tf.keras.losses.binary_crossentropy(
                batch['dropout_indicator'],
                predictions['risk_score']
            )
        }
    
    def _update_metrics(self, batch: Dict, predictions: Dict) -> None:
        """Update metrics with batch predictions."""
        self.metrics['ranking'].update_state(
            batch['engagement_response'],
            predictions['ranking_score']
        )
        
        for metric in self.metrics['likelihood']:
            metric.update_state(
                batch['application_completion'],
                predictions['likelihood_score']
            )
        
        for metric in self.metrics['risk']:
            metric.update_state(
                batch['dropout_indicator'],
                predictions['risk_score']
            )
    
    def _print_metrics(self) -> None:
        """Print current metrics."""
        print(f"Ranking RMSE: {self.metrics['ranking'].result():.4f}")
        print(f"Likelihood AUC: {self.metrics['likelihood'][0].result():.4f}")
        print(f"Likelihood Accuracy: {self.metrics['likelihood'][1].result():.4f}")
        print(f"Risk AUC: {self.metrics['risk'][0].result():.4f}")
        print(f"Risk Accuracy: {self.metrics['risk'][1].result():.4f}")
    
    def _get_metrics(self) -> Dict[str, float]:
        """Get current metric values."""
        return {
            'ranking_rmse': self.metrics['ranking'].result().numpy(),
            'likelihood_auc': self.metrics['likelihood'][0].result().numpy(),
            'likelihood_accuracy': self.metrics['likelihood'][1].result().numpy(),
            'risk_auc': self.metrics['risk'][0].result().numpy(),
            'risk_accuracy': self.metrics['risk'][1].result().numpy()
        }
    
    def _update_student_embeddings(self) -> None:
        """Update student embeddings for next epoch."""
        student_embeddings = []
        for batch in self.train_dataset:
            student_embedding = self.model.student_tower({
                "student_id": batch["student_id"],
                "student_features": batch.get("student_features", {})
            }, training=False)
            student_embeddings.append(student_embedding)
        
        # Concatenate all embeddings
        all_student_embeddings = tf.concat(student_embeddings, axis=0)
        
        # Update model with new embeddings
        self.model.update_student_embeddings(all_student_embeddings)
    
    def save_model(self) -> None:
        """Save the trained model and associated data."""
        # Ensure the model directory exists
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Save model weights
        student_tower_weights_path = os.path.join(self.model_dir, "model_weights", "student_tower_weights.weights.h5")
        engagement_tower_weights_path = os.path.join(self.model_dir, "model_weights", "engagement_tower_weights.weights.h5")
        
        os.makedirs(os.path.dirname(student_tower_weights_path), exist_ok=True)
        os.makedirs(os.path.dirname(engagement_tower_weights_path), exist_ok=True)
        
        self.model.student_tower.save_weights(student_tower_weights_path)
        self.model.engagement_tower.save_weights(engagement_tower_weights_path)
        
        # Save vocabularies
        with open(os.path.join(self.model_dir, "vocabularies.json"), "w") as f:
            json.dump(self.vocabularies, f)
        
        # Create and save nearest neighbors model
        self._save_nearest_neighbors()
        
        print(f"Model saved to {self.model_dir}")
    
    def _save_nearest_neighbors(self) -> None:
        """Create and save nearest neighbors model for fast retrieval."""
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
        }, os.path.join(self.model_dir, "nearest_neighbors.joblib"))
    
    def load_model(self) -> tf.keras.Model:
        """Load a trained model."""
        # Load model
        self.model = tf.saved_model.load(os.path.join(self.model_dir, "student_engagement_model"))
        
        # Load vocabularies
        with open(os.path.join(self.model_dir, "vocabularies.json"), "r") as f:
            self.vocabularies = json.load(f)
        
        return self.model 