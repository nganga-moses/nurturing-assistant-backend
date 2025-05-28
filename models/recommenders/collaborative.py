import tensorflow as tf
import tensorflow_recommenders as tfrs
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from .base_recommender import BaseRecommender
from ..core.engagement_content_preprocessor import EngagementContentPreprocessor

class CandidateModel(tf.keras.Model):
    def __init__(self, engagement_type_ids, content_ids, embedding_dimension):
        super().__init__()
        self.engagement_type_lookup = tf.keras.layers.StringLookup(vocabulary=engagement_type_ids, mask_token=None)
        self.engagement_type_embedding = tf.keras.layers.Embedding(len(engagement_type_ids) + 1, embedding_dimension // 2)
        self.content_lookup = tf.keras.layers.StringLookup(vocabulary=content_ids, mask_token=None)
        self.content_embedding = tf.keras.layers.Embedding(len(content_ids) + 1, embedding_dimension)
        self.embedding_dimension = embedding_dimension
    def call(self, candidate_id):
        # candidate_id: shape (batch,), e.g., 'email|C1'
        split = tf.strings.split(candidate_id, sep='|').to_tensor()
        engagement_type = split[:, 0]
        content_id = split[:, 1]
        engagement_type_idx = self.engagement_type_lookup(engagement_type)
        content_idx = self.content_lookup(content_id)
        engagement_type_emb = self.engagement_type_embedding(engagement_type_idx)
        content_emb = self.content_embedding(content_idx)
        # If content_id == 'NO_CONTENT', zero out content embedding
        has_content = tf.cast(tf.not_equal(content_id, 'NO_CONTENT'), tf.float32)
        content_emb = tf.where(tf.expand_dims(has_content, -1) > 0, content_emb, tf.zeros_like(content_emb))
        return tf.concat([engagement_type_emb, content_emb], axis=1)

class TFRSCollaborativeModel(tfrs.models.Model):
    def __init__(self, student_model, candidate_model, task):
        super().__init__()
        self.student_model = student_model
        self.candidate_model = candidate_model
        self.task = task
    def compute_loss(self, features: Dict[str, tf.Tensor], training=False) -> tf.Tensor:
        return self.task(
            self.student_model(features['student_id']),
            self.candidate_model(features['candidate_id'])
        )

class CollaborativeFilteringModel(BaseRecommender):
    """Collaborative filtering model using TensorFlow Recommenders with engagement type and content towers."""
    
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
        self.preprocessor = EngagementContentPreprocessor()
        self.student_model = None
        self.candidate_model = None
        self.task = None
        self.model = None
        self.student_ids = None
        self.engagement_type_ids = None
        self.content_ids = None
        self.feature_usage = None
        self.model_version = "v2.0"
        
    def _create_student_model(self) -> tf.keras.Model:
        """Create the student tower model."""
        # Output dimension must match engagement_type + content embedding dims
        combined_dim = (self.embedding_dimension // 2) + self.embedding_dimension
        return tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=self.student_ids, mask_token=None),
            tf.keras.layers.Embedding(len(self.student_ids) + 1, combined_dim),
            tf.keras.layers.Dense(combined_dim, activation='relu'),
            tf.keras.layers.Dense(combined_dim)
        ])
    
    def _create_candidate_model(self) -> tf.keras.Model:
        """Create the candidate model for engagement types and content."""
        candidate_id_input = tf.keras.Input(shape=(), dtype=tf.string, name="candidate_id")
        split = tf.keras.layers.Lambda(lambda x: tf.strings.split(x, sep='|'), output_shape=(None,))(candidate_id_input)
        split_dense = tf.keras.layers.Lambda(lambda x: x.to_tensor(default_value='NO_CONTENT'), output_shape=(2,))(split)
        engagement_type = tf.keras.layers.Lambda(lambda x: x[:, 0], output_shape=())(split_dense)
        content_id = tf.keras.layers.Lambda(lambda x: x[:, 1], output_shape=())(split_dense)
        engagement_type_lookup = tf.keras.layers.StringLookup(vocabulary=self.engagement_type_ids, mask_token=None)
        engagement_type_embedding = tf.keras.layers.Embedding(len(self.engagement_type_ids) + 1, self.embedding_dimension // 2)
        engagement_type_idx = engagement_type_lookup(engagement_type)
        engagement_type_emb = engagement_type_embedding(engagement_type_idx)
        content_lookup = tf.keras.layers.StringLookup(vocabulary=self.content_ids, mask_token=None)
        content_embedding = tf.keras.layers.Embedding(len(self.content_ids) + 1, self.embedding_dimension)
        content_idx = content_lookup(content_id)
        content_emb = content_embedding(content_idx)
        has_content = tf.keras.layers.Lambda(lambda x: tf.cast(tf.not_equal(x, 'NO_CONTENT'), tf.float32), output_shape=())(content_id)
        has_content_expanded = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, -1), output_shape=(1,))(has_content)
        content_emb_masked = tf.keras.layers.Multiply()([has_content_expanded, content_emb])
        combined_emb = tf.keras.layers.Concatenate(axis=1)([engagement_type_emb, content_emb_masked])
        return tf.keras.Model(inputs=candidate_id_input, outputs=combined_emb, name="candidate_model")
    
    def _make_candidate_ids(self):
        """Create a list of candidate IDs."""
        candidate_ids = []
        for engagement_type in self.engagement_type_ids:
            for content_id in self.content_ids:
                candidate_ids.append(f"{engagement_type}|{content_id}")
        return candidate_ids
    
    def _create_retrieval_model(self, candidate_ids) -> tfrs.models.Model:
        """Create the retrieval model with both engagement type and content."""
        self.student_model = self._create_student_model()
        self.candidate_model = self._create_candidate_model()
        
        # Debug: Print candidate_ids
        print(f"DEBUG candidate_ids (first 10): {candidate_ids[:10]}")
        print(f"DEBUG total candidate_ids: {len(candidate_ids)}")

        # Debug: Print sample candidate embedding output
        if len(candidate_ids) > 0:
            sample_cid = candidate_ids[0]
            sample_emb = self.candidate_model(tf.constant([sample_cid])).numpy()
            print(f"DEBUG sample candidate_id: {sample_cid}")
            print(f"DEBUG sample candidate embedding shape: {sample_emb.shape}")
            print(f"DEBUG sample candidate embedding: {sample_emb}")

        # Create a BruteForce retrieval layer using the candidate model directly
        retrieval_layer = tfrs.layers.factorized_top_k.BruteForce(
            self.student_model,
            k=100
        )
        retrieval_layer.index_from_dataset(
            tf.data.Dataset.from_tensor_slices(candidate_ids).map(
                lambda x: (x, tf.squeeze(self.candidate_model(tf.expand_dims(x, 0)), axis=0))
            )
        )

        # Create the retrieval task with the candidate model directly
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=self.candidate_model
            )
        )
        
        return TFRSCollaborativeModel(
            student_model=self.student_model,
            candidate_model=self.candidate_model,
            task=self.task
        )
    
    def prepare_data(
        self,
        students_df: pd.DataFrame,
        engagements_df: pd.DataFrame,
        content_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Prepare data for training.
        
        Args:
            students_df: DataFrame of student profiles
            engagements_df: DataFrame of engagement history
            content_df: DataFrame of engagement content
            
        Returns:
            Dictionary containing prepared data
        """
        processed = self.preprocessor.preprocess(students_df, engagements_df, content_df)
        self.student_ids = processed['students_df']['student_id'].unique()
        self.engagement_type_ids = processed['engagement_types']
        self.content_ids = processed['content_ids']
        merged = processed['merged_df']

        # Debug: Print vocabularies
        print(f"DEBUG student_ids: {self.student_ids}")
        print(f"DEBUG engagement_type_ids: {self.engagement_type_ids}")
        print(f"DEBUG content_ids: {self.content_ids}")

        # Debug: Check merged DataFrame for NaNs or missing values
        print("DEBUG merged DataFrame info:")
        print(merged.info())
        print("DEBUG merged DataFrame head:")
        print(merged.head())
        print("DEBUG merged DataFrame NaNs:")
        print(merged.isnull().sum())
        
        # For training, create candidate_id column
        merged['candidate_id'] = merged['engagement_type'] + '|' + merged['content_id']
        
        # Prepare TensorFlow datasets
        student_dataset = tf.data.Dataset.from_tensor_slices({
            'student_id': merged['student_id'].values
        })
        interaction_dataset = tf.data.Dataset.from_tensor_slices({
            'student_id': merged['student_id'].values,
            'candidate_id': merged['candidate_id'].values
        })
        
        return {
            'student_dataset': student_dataset,
            'interaction_dataset': interaction_dataset,
            'merged_df': merged
        }
    
    def train(
        self,
        students_df: pd.DataFrame,
        engagements_df: pd.DataFrame,
        content_df: pd.DataFrame,
        epochs: int = 5,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Train the model.
        
        Args:
            students_df: DataFrame of student profiles
            engagements_df: DataFrame of engagement history
            content_df: DataFrame of engagement content
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary of training metrics
        """
        # Prepare data
        data_dict = self.prepare_data(students_df, engagements_df, content_df)
        
        # Create model
        candidate_ids = self._make_candidate_ids()
        self.model = self._create_retrieval_model(candidate_ids)
        
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
    ) -> List[Dict[str, Any]]:
        """
        Get recommendations for a student.
        
        Args:
            student_id: ID of the student
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of recommended engagements with both type and content
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        candidate_ids = self._make_candidate_ids()
        
        # Build index for retrieval
        index = tfrs.layers.factorized_top_k.BruteForce(self.student_model)
        index.index_from_dataset(
            tf.data.Dataset.from_tensor_slices(candidate_ids).map(lambda x: (x, self.candidate_model(tf.expand_dims(x, 0))))
        )
        
        # Get recommendations
        _, top_candidate_ids = index(tf.constant([student_id]))
        
        # Format recommendations
        recommendations = []
        for cid in top_candidate_ids[0, :n_recommendations].numpy():
            cid_str = cid.decode() if isinstance(cid, bytes) else str(cid)
            engagement_type, content_id = cid_str.split('|')
            features_used = ["student", "engagement_type"]
            if content_id != 'NO_CONTENT':
                features_used.append("content")
            recommendations.append({
                'student_id': student_id,
                'engagement_type': engagement_type,
                'content_id': content_id,
                'features_used': features_used,
                'model_version': self.model_version
            })
        
        return recommendations
    
    def save(self) -> None:
        """Save the model."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        # Save model weights
        self.model.save_weights(f"{self.model_dir}/collaborative_model")
        
        # Save vocabularies
        np.save(f"{self.model_dir}/student_ids.npy", self.student_ids)
        np.save(f"{self.model_dir}/engagement_type_ids.npy", self.engagement_type_ids)
        np.save(f"{self.model_dir}/content_ids.npy", self.content_ids)
    
    def load(self) -> None:
        """Load the model."""
        # Load vocabularies
        self.student_ids = np.load(f"{self.model_dir}/student_ids.npy")
        self.engagement_type_ids = np.load(f"{self.model_dir}/engagement_type_ids.npy")
        self.content_ids = np.load(f"{self.model_dir}/content_ids.npy")
        
        # Create and load model
        candidate_ids = self._make_candidate_ids()
        self.model = self._create_retrieval_model(candidate_ids)
        self.model.load_weights(f"{self.model_dir}/collaborative_model") 