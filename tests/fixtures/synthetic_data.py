"""
Synthetic data generation for the recommendation model training.
This module provides functions to generate synthetic data when real data is not available.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Dict, List, Any
from datetime import datetime, timedelta
import json

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

def create_sample_data(num_students: int = 100, num_engagements: int = 100, batch_size: int = 32) -> Dict[str, Any]:
    """Create synthetic data for testing."""
    # Generate student and engagement IDs
    student_ids = [f"student{i}" for i in range(num_students)]
    engagement_ids = [f"engagement{i}" for i in range(num_engagements)]
    
    # Generate random features for students and engagements
    student_features = {
        "age": np.random.uniform(15, 25, size=num_students),
        "gender": np.random.uniform(0, 1, size=num_students),
        "ethnicity": np.random.uniform(0, 1, size=num_students),
        "location": np.random.uniform(0, 1, size=num_students),
        "gpa": np.random.uniform(0, 4, size=num_students),
        "test_scores": np.random.uniform(0, 100, size=num_students),
        "courses": np.random.uniform(0, 10, size=num_students),
        "major": np.random.uniform(0, 1, size=num_students),
        "attendance": np.random.uniform(0, 1, size=num_students),
        "participation": np.random.uniform(0, 1, size=num_students)
    }
    student_features_df = pd.DataFrame(student_features)
    
    engagement_features = {
        "type": np.random.uniform(0, 1, size=num_engagements),
        "duration": np.random.uniform(0, 100, size=num_engagements),
        "difficulty": np.random.uniform(0, 1, size=num_engagements),
        "prerequisites": np.random.uniform(0, 1, size=num_engagements),
        "popularity": np.random.uniform(0, 1, size=num_engagements),
        "success_rate": np.random.uniform(0, 1, size=num_engagements),
        "engagement_level": np.random.uniform(0, 1, size=num_engagements),
        "feedback_score": np.random.uniform(0, 1, size=num_engagements),
        "completion_rate": np.random.uniform(0, 1, size=num_engagements),
        "interaction_frequency": np.random.uniform(0, 1, size=num_engagements)
    }
    engagement_features_df = pd.DataFrame(engagement_features)
    
    # Generate random labels
    ranking_labels = np.random.uniform(0, 10, size=num_students)
    likelihood_labels = np.random.uniform(0, 1, size=num_students)
    risk_labels = np.random.uniform(0, 1, size=num_students)
    
    # Generate random timestamps
    timestamps = [datetime.now() - timedelta(days=np.random.randint(0, 365)) for _ in range(num_students)]
    
    # Generate random funnel stages
    funnel_stages = ['awareness', 'interest', 'consideration', 'decision', 'application']
    funnel_stage_before = [np.random.choice(funnel_stages) for _ in range(num_students)]
    
    # Generate random engagement metrics
    engagement_metrics = [{'duration': np.random.randint(30, 180), 'attendance': np.random.randint(10, 100)} for _ in range(num_students)]
    
    # Generate random engagement types
    engagement_types = ['academic', 'social', 'campus_visit', 'info_session'] * (num_students // 4)
    engagement_types.extend(['academic'] * (num_students % 4))  # Ensure the list is of length num_students
    
    # Create a DataFrame for the dataset
    dataset_df = pd.DataFrame({
        "student_id": student_ids,
        "engagement_id": engagement_ids,
        "ranking_label": ranking_labels,
        "likelihood_label": likelihood_labels,
        "risk_label": risk_labels,
        "timestamp": timestamps,  # Add timestamp column
        "funnel_stage_before": funnel_stage_before,  # Add funnel stage column
        "engagement_metrics": engagement_metrics,  # Add engagement metrics column
        "engagement_type": engagement_types  # Add engagement type column
    })
    
    # Split the dataset into training and validation sets
    train_size = int(0.8 * num_students)
    train_data = dataset_df.iloc[:train_size]
    val_data = dataset_df.iloc[train_size:]
    
    return train_data, val_data, student_ids, engagement_ids
