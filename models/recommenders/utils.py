import tensorflow as tf
import tensorflow_recommenders as tfrs
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_student_tower(
    student_ids: List[str],
    embedding_dimension: int,
    name: str = "student_tower"
) -> tf.keras.Model:
    """
    Create a student tower model.
    
    Args:
        student_ids: List of student IDs
        embedding_dimension: Dimension of embeddings
        name: Name of the model
        
    Returns:
        Student tower model
    """
    return tf.keras.Sequential([
        tf.keras.layers.StringLookup(vocabulary=student_ids, mask_token=None),
        tf.keras.layers.Embedding(len(student_ids) + 1, embedding_dimension),
        tf.keras.layers.Dense(embedding_dimension, activation='relu'),
        tf.keras.layers.Dense(embedding_dimension)
    ], name=name)

def create_candidate_tower(
    candidate_ids: List[str],
    embedding_dimension: int,
    name: str = "candidate_tower"
) -> tf.keras.Model:
    """
    Create a candidate tower model.
    
    Args:
        candidate_ids: List of candidate IDs
        embedding_dimension: Dimension of embeddings
        name: Name of the model
        
    Returns:
        Candidate tower model
    """
    return tf.keras.Sequential([
        tf.keras.layers.StringLookup(vocabulary=candidate_ids, mask_token=None),
        tf.keras.layers.Embedding(len(candidate_ids) + 1, embedding_dimension),
        tf.keras.layers.Dense(embedding_dimension, activation='relu'),
        tf.keras.layers.Dense(embedding_dimension)
    ], name=name)

def create_retrieval_model(
    student_model: tf.keras.Model,
    candidate_model: tf.keras.Model,
    candidate_ids: List[str],
    k: int = 100
) -> tfrs.models.Model:
    """
    Create a retrieval model.
    
    Args:
        student_model: Student tower model
        candidate_model: Candidate tower model
        candidate_ids: List of candidate IDs
        k: Number of candidates to retrieve
        
    Returns:
        Retrieval model
    """
    # Create retrieval layer
    retrieval_layer = tfrs.layers.factorized_top_k.BruteForce(
        student_model,
        k=k
    )
    
    # Index candidates
    retrieval_layer.index_from_dataset(
        tf.data.Dataset.from_tensor_slices(candidate_ids).map(
            lambda x: (x, tf.squeeze(candidate_model(tf.expand_dims(x, 0)), axis=0))
        )
    )
    
    # Create retrieval task
    task = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            candidates=candidate_model
        )
    )
    
    # Create model
    return tfrs.models.Model(
        student_model=student_model,
        candidate_model=candidate_model,
        task=task
    )

def prepare_training_data(
    student_data: pd.DataFrame,
    content_data: pd.DataFrame,
    engagement_data: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Prepare data for training.
    
    Args:
        student_data: DataFrame of student profiles
        content_data: DataFrame of content information
        engagement_data: Optional DataFrame of engagement history
        
    Returns:
        Dictionary containing prepared data
    """
    # Extract unique IDs
    student_ids = student_data['student_id'].unique()
    content_ids = content_data['content_id'].unique()
    
    # Create candidate IDs
    candidate_ids = []
    for content_id in content_ids:
        candidate_ids.append(content_id)
    
    # Create TensorFlow datasets
    student_dataset = tf.data.Dataset.from_tensor_slices({
        'student_id': student_data['student_id'].values
    })
    
    if engagement_data is not None:
        interaction_dataset = tf.data.Dataset.from_tensor_slices({
            'student_id': engagement_data['student_id'].values,
            'candidate_id': engagement_data['content_id'].values
        })
    else:
        # Create synthetic interactions if no engagement data
        interaction_dataset = tf.data.Dataset.from_tensor_slices({
            'student_id': np.repeat(student_ids, len(content_ids)),
            'candidate_id': np.tile(content_ids, len(student_ids))
        })
    
    return {
        'student_dataset': student_dataset,
        'interaction_dataset': interaction_dataset,
        'student_ids': student_ids,
        'content_ids': content_ids,
        'candidate_ids': candidate_ids
    }

def calculate_metrics(
    model: tf.keras.Model,
    test_data: tf.data.Dataset,
    k: int = 10
) -> Dict[str, float]:
    """
    Calculate model metrics.
    
    Args:
        model: Trained model
        test_data: Test dataset
        k: Number of top-k recommendations
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Calculate top-k accuracy
    top_k_accuracy = tf.keras.metrics.TopKCategoricalAccuracy(k=k)
    
    for batch in test_data:
        predictions = model(batch)
        top_k_accuracy.update_state(batch['label'], predictions)
    
    metrics['top_k_accuracy'] = float(top_k_accuracy.result())
    
    return metrics 