"""
Synthetic data generation for the recommendation model training.
This module provides functions to generate synthetic data when real data is not available.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Dict, List, Any
from datetime import datetime, timedelta

def generate_synthetic_data() -> Dict:
    """
    Generate synthetic data for training the recommendation model.
    
    Returns:
        Dictionary containing prepared data
    """
    print("Generating synthetic data for model training...")
    
    # Define constants
    num_students = 100
    num_engagements = 500
    num_content_items = 50
    embedding_dimension = 64
    
    # Generate student IDs
    student_ids = [f"S{i:04d}" for i in range(1, num_students + 1)]
    
    # Generate engagement IDs
    engagement_ids = [f"E{i:04d}" for i in range(1, num_engagements + 1)]
    
    # Generate content IDs
    content_ids = [f"C{i:04d}" for i in range(1, num_content_items + 1)]
    
    # Generate synthetic interactions
    interactions = []
    funnel_stages = ['awareness', 'interest', 'consideration', 'decision', 'application']
    
    for _ in range(num_engagements):
        student_id = np.random.choice(student_ids)
        content_id = np.random.choice(content_ids)
        engagement_id = np.random.choice(engagement_ids)
        
        # Random funnel stages
        funnel_stage_before_idx = np.random.randint(0, len(funnel_stages))
        funnel_stage_after_idx = min(funnel_stage_before_idx + np.random.randint(0, 2), len(funnel_stages) - 1)
        
        funnel_stage_before = funnel_stages[funnel_stage_before_idx]
        funnel_stage_after = funnel_stages[funnel_stage_after_idx]
        funnel_stage = funnel_stages[min(funnel_stage_after_idx, funnel_stage_before_idx)]
        
        # Random scores
        dropout_risk_score = np.random.uniform(0.0, 1.0)
        application_likelihood_score = np.random.uniform(0.0, 1.0)
        effectiveness_score = 1.0 if funnel_stage_after_idx > funnel_stage_before_idx else 0.5
        
        interactions.append({
            'student_id': student_id,
            'engagement_id': engagement_id,
            'content_id': content_id,
            'funnel_stage_before': funnel_stage_before,
            'funnel_stage_after': funnel_stage_after,
            'funnel_stage': funnel_stage,
            'dropout_risk_score': dropout_risk_score,
            'application_likelihood_score': application_likelihood_score,
            'effectiveness_score': effectiveness_score
        })
    
    # Create DataFrame
    interactions_df = pd.DataFrame(interactions)
    
    # Split data into train and test sets (80/20 split)
    np.random.seed(42)
    mask = np.random.rand(len(interactions_df)) < 0.8
    train_interactions = interactions_df[mask]
    test_interactions = interactions_df[~mask]
    
    print(f"Created {len(train_interactions)} synthetic training examples and {len(test_interactions)} synthetic test examples")
    
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices({
        "student_id": train_interactions['student_id'].values,
        "engagement_id": train_interactions['engagement_id'].values,
        "content_id": train_interactions['content_id'].values,
        "effectiveness_score": train_interactions['effectiveness_score'].values,
        "application_likelihood": train_interactions['application_likelihood_score'].values,
        "dropout_risk": train_interactions['dropout_risk_score'].values
    }).shuffle(10000).batch(128)
    
    test_dataset = tf.data.Dataset.from_tensor_slices({
        "student_id": test_interactions['student_id'].values,
        "engagement_id": test_interactions['engagement_id'].values,
        "content_id": test_interactions['content_id'].values,
        "effectiveness_score": test_interactions['effectiveness_score'].values,
        "application_likelihood": test_interactions['application_likelihood_score'].values,
        "dropout_risk": test_interactions['dropout_risk_score'].values
    }).batch(128)
    
    # Create empty DataFrames for the original data
    students_df = pd.DataFrame({
        'student_id': student_ids,
        'funnel_stage': np.random.choice(funnel_stages, size=len(student_ids)),
        'dropout_risk_score': np.random.uniform(0.0, 1.0, size=len(student_ids)),
        'application_likelihood_score': np.random.uniform(0.0, 1.0, size=len(student_ids))
    })
    
    engagements_df = pd.DataFrame({
        'engagement_id': engagement_ids,
        'student_id': np.random.choice(student_ids, size=len(engagement_ids)),
        'engagement_content_id': np.random.choice(content_ids, size=len(engagement_ids))
    })
    
    content_df = pd.DataFrame({
        'content_id': content_ids,
        'engagement_type': np.random.choice(['email', 'sms', 'call', 'event'], size=len(content_ids))
    })
    
    # Create vocabularies dictionary
    vocabularies = {
        'student_ids': student_ids,
        'engagement_ids': engagement_ids,
        'content_ids': content_ids
    }
    
    result = {
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'vocabularies': vocabularies,
        'dataframes': {
            'students': students_df,
            'engagements': engagements_df,
            'content': content_df
        }
    }
    print("Generated synthetic data keys:", result.keys())
    return result
