"""Data generator for recommender model training."""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Any, Tuple

class DataGenerator:
    """Generator for training data."""
    
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
    
    def generate_batch(self, student_ids: List[str], engagement_ids: List[str]) -> Dict[str, tf.Tensor]:
        """Generate a batch of training data."""
        # Sample random student and engagement IDs
        batch_student_ids = np.random.choice(student_ids, size=self.batch_size)
        batch_engagement_ids = np.random.choice(engagement_ids, size=self.batch_size)
        
        # Generate random features
        student_features = {
            "age": np.random.uniform(15, 25, size=self.batch_size),
            "gender": np.random.uniform(0, 1, size=self.batch_size),
            "ethnicity": np.random.uniform(0, 1, size=self.batch_size),
            "location": np.random.uniform(0, 1, size=self.batch_size),
            "gpa": np.random.uniform(0, 4, size=self.batch_size),
            "test_scores": np.random.uniform(0, 100, size=self.batch_size),
            "courses": np.random.uniform(0, 10, size=self.batch_size),
            "major": np.random.uniform(0, 1, size=self.batch_size),
            "attendance": np.random.uniform(0, 1, size=self.batch_size),
            "participation": np.random.uniform(0, 1, size=self.batch_size),
            "feedback": np.random.uniform(0, 1, size=self.batch_size),
            "study_habits": np.random.uniform(0, 1, size=self.batch_size),
            "social_activity": np.random.uniform(0, 1, size=self.batch_size),
            "stress_level": np.random.uniform(0, 1, size=self.batch_size)
        }
        
        engagement_features = {
            "type": np.random.uniform(0, 1, size=self.batch_size),
            "duration": np.random.uniform(0, 100, size=self.batch_size),
            "difficulty": np.random.uniform(0, 1, size=self.batch_size),
            "prerequisites": np.random.uniform(0, 1, size=self.batch_size),
            "popularity": np.random.uniform(0, 1, size=self.batch_size),
            "success_rate": np.random.uniform(0, 1, size=self.batch_size)
        }
        
        # Generate random labels
        ranking_labels = np.random.uniform(0, 10, size=self.batch_size)
        likelihood_labels = np.random.uniform(0, 1, size=self.batch_size)
        risk_labels = np.random.uniform(0, 1, size=self.batch_size)
        
        # Convert to tensors
        return {
            "student_id": tf.convert_to_tensor(batch_student_ids),
            "engagement_id": tf.convert_to_tensor(batch_engagement_ids),
            "student_features": {k: tf.convert_to_tensor(v) for k, v in student_features.items()},
            "engagement_features": {k: tf.convert_to_tensor(v) for k, v in engagement_features.items()},
            "ranking_label": tf.convert_to_tensor(ranking_labels),
            "likelihood_label": tf.convert_to_tensor(likelihood_labels),
            "risk_label": tf.convert_to_tensor(risk_labels)
        }
    
    def generate_dataset(self, student_ids: List[str], engagement_ids: List[str], num_batches: int) -> tf.data.Dataset:
        """Generate a dataset of training data."""
        # Create dataset from generator
        dataset = tf.data.Dataset.from_generator(
            lambda: (self.generate_batch(student_ids, engagement_ids) for _ in range(num_batches)),
            output_signature={
                "student_id": tf.TensorSpec(shape=(self.batch_size,), dtype=tf.string),
                "engagement_id": tf.TensorSpec(shape=(self.batch_size,), dtype=tf.string),
                "student_features": {
                    k: tf.TensorSpec(shape=(self.batch_size,), dtype=tf.float32)
                    for k in ["age", "gender", "ethnicity", "location", "gpa", "test_scores",
                             "courses", "major", "attendance", "participation", "feedback",
                             "study_habits", "social_activity", "stress_level"]
                },
                "engagement_features": {
                    k: tf.TensorSpec(shape=(self.batch_size,), dtype=tf.float32)
                    for k in ["type", "duration", "difficulty", "prerequisites", "popularity", "success_rate"]
                },
                "ranking_label": tf.TensorSpec(shape=(self.batch_size,), dtype=tf.float32),
                "likelihood_label": tf.TensorSpec(shape=(self.batch_size,), dtype=tf.float32),
                "risk_label": tf.TensorSpec(shape=(self.batch_size,), dtype=tf.float32)
            }
        )
        
        return dataset 