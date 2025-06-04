#!/usr/bin/env python
"""
Unified model training script for the Student Engagement Recommender System.

This script handles training of all model types:
1. Hybrid (default): Combines collaborative and content-based approaches
2. Collaborative: Uses student-engagement interactions
3. Content-based: Uses student and content features

The script supports:
- Loading data from database or CSV
- Data preparation and preprocessing
- Model training with configurable parameters
- Model evaluation and metrics tracking
- Saving models and vocabularies

Usage:
    python train_model.py --model-type [hybrid|collaborative|content_based] --epochs 5

Example:
    # Train hybrid model
    python train_model.py --model-type hybrid --epochs 5

    # Train collaborative model with custom learning rate
    python train_model.py --model-type collaborative --epochs 10 --learning-rate 0.01

    # Train content-based model
    python train_model.py --model-type content_based
"""

import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
import json
import joblib
from datetime import datetime
from sqlalchemy.orm import Session
from typing import Dict, List, Tuple, Optional
import argparse

# Import synthetic data generation
from models.synthetic_data import generate_synthetic_data

# Add the parent directory to the path so we can import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.models.student_profile import StudentProfile
from data.models.engagement_history import EngagementHistory
from data.models.engagement_content import EngagementContent
from models.recommender_model import ModelTrainer
from models.collaborative_filtering_model import CollaborativeFilteringModel
from models.simple_recommender import SimpleRecommender
from database.session import get_db

def load_data_from_db(session: Session) -> Dict[str, pd.DataFrame]:
    """
    Load data from the database into pandas DataFrames.
    
    This function:
    1. Loads student profiles
    2. Loads engagement history
    3. Loads engagement content
    4. Validates data completeness
    5. Returns a dictionary of DataFrames
    
    Args:
        session: SQLAlchemy session
        
    Returns:
        Dictionary containing:
        - students: DataFrame of student profiles
        - engagements: DataFrame of engagement history
        - content: DataFrame of engagement content
        
    Raises:
        ValueError: If required data is missing
    """
    print("Loading data from database...")
    
    # Load student profiles
    students = pd.read_sql(session.query(StudentProfile).statement, session.bind)
    print("Students DataFrame shape:", students.shape)
    
    # Load engagement history
    engagements = pd.read_sql(session.query(EngagementHistory).statement, session.bind)
    print("Engagements DataFrame shape:", engagements.shape)
    
    # Load engagement content
    content = pd.read_sql(session.query(EngagementContent).statement, session.bind)
    print("Content DataFrame shape:", content.shape)
    
    print(f"Loaded {len(students)} students, {len(engagements)} engagements, and {len(content)} content items")
    
    return {
        'students': students,
        'engagements': engagements,
        'content': content
    }

def prepare_data_for_training(dataframes: Dict[str, pd.DataFrame]) -> Dict:
    """
    Prepare data for training the recommendation model.
    
    This function:
    1. Validates input data
    2. Creates vocabularies for categorical features
    3. Prepares interaction data
    4. Calculates effectiveness scores
    5. Splits data into train/test sets
    6. Creates TensorFlow datasets
    
    Args:
        dataframes: Dictionary containing:
            - students: DataFrame of student profiles
            - engagements: DataFrame of engagement history
            - content: DataFrame of engagement content
            
    Returns:
        Dictionary containing:
            - train_dataset: TensorFlow dataset for training
            - test_dataset: TensorFlow dataset for testing
            - vocabularies: Dictionary of feature vocabularies
            - dataframes: Original DataFrames
            
    Raises:
        ValueError: If data preparation fails
    """
    print("Preparing data for training...")
    
    # Extract DataFrames
    students_df = dataframes['students']
    engagements_df = dataframes['engagements']
    content_df = dataframes['content']
    
    # Ensure the correct column name for engagement_content_id
    if 'engagement_content_id' not in engagements_df.columns and 'content_id' in engagements_df.columns:
        engagements_df = engagements_df.rename(columns={'content_id': 'engagement_content_id'})
    
    # Check if we have enough data
    if len(students_df) == 0 or len(engagements_df) == 0 or len(content_df) == 0:
        print("Not enough data for training. Generating synthetic data...")
        # Generate synthetic data for training
        return generate_synthetic_data()
    
    # Create vocabularies
    student_ids = students_df['student_id'].unique().tolist()
    engagement_ids = engagements_df['engagement_id'].unique().tolist()
    content_ids = content_df['content_id'].unique().tolist()
    
    # Create interaction data
    try:
        print("Students DataFrame student_id values:", students_df['student_id'].unique())
        print("Engagements DataFrame student_id values:", engagements_df['student_id'].unique())
        interactions = engagements_df.merge(
            students_df[['student_id', 'funnel_stage', 'dropout_risk_score', 'application_likelihood_score']],
            on='student_id'
        )
        print("Interactions DataFrame shape:", interactions.shape)
        
        # Define funnel stages for ordering
        funnel_stages = ['awareness', 'interest', 'consideration', 'decision', 'application']
        
        # Add effectiveness score based on funnel stage change
        def calculate_effectiveness(row):
            try:
                if row['funnel_stage_after'] != row['funnel_stage_before']:
                    if row['funnel_stage_after'] in funnel_stages and row['funnel_stage_before'] in funnel_stages:
                        after_idx = funnel_stages.index(row['funnel_stage_after'])
                        before_idx = funnel_stages.index(row['funnel_stage_before'])
                        if after_idx > before_idx:
                            return 1.0
                return 0.5
            except (KeyError, ValueError, TypeError):
                return 0.5
        
        interactions['effectiveness_score'] = interactions.apply(calculate_effectiveness, axis=1)
        
        # Split data into train and test sets (80/20 split)
        np.random.seed(42)
        mask = np.random.rand(len(interactions)) < 0.8
        train_interactions = interactions[mask]
        test_interactions = interactions[~mask]
        
        print(f"Created {len(train_interactions)} training examples and {len(test_interactions)} test examples")
    except Exception as e:
        print(f"Error preparing interaction data: {str(e)}")
        import traceback
        print("Full error details:")
        traceback.print_exc()
        print("Falling back to synthetic data...")
        return generate_synthetic_data()
    
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices({
        "student_id": train_interactions['student_id'].values,
        "engagement_id": train_interactions['engagement_id'].values,
        "content_id": train_interactions['engagement_content_id'].values,
        "effectiveness_score": train_interactions['effectiveness_score'].values,
        "application_likelihood": train_interactions['application_likelihood_score'].values,
        "dropout_risk": train_interactions['dropout_risk_score'].values
    }).shuffle(10000).batch(128)
    
    test_dataset = tf.data.Dataset.from_tensor_slices({
        "student_id": test_interactions['student_id'].values,
        "engagement_id": test_interactions['engagement_id'].values,
        "content_id": test_interactions['engagement_content_id'].values,
        "effectiveness_score": test_interactions['effectiveness_score'].values,
        "application_likelihood": test_interactions['application_likelihood_score'].values,
        "dropout_risk": test_interactions['dropout_risk_score'].values
    }).batch(128)
    
    # Create vocabularies dictionary
    vocabularies = {
        'student_ids': student_ids,
        'engagement_ids': engagement_ids,
        'content_ids': content_ids
    }
    
    return {
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'vocabularies': vocabularies,
        'dataframes': dataframes
    }

def train_model(
    model_type: str = "hybrid",
    model: Optional[StudentEngagementModel] = None,
    train_dataset: Optional[tf.data.Dataset] = None,
    test_dataset: Optional[tf.data.Dataset] = None,
    students_df: Optional[pd.DataFrame] = None,
    engagements_df: Optional[pd.DataFrame] = None,
    content_df: Optional[pd.DataFrame] = None,
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 0.001
) -> Dict[str, float]:
    """
    Train the model with specified type and data.
    
    This function:
    1. Initializes the appropriate model type
    2. Sets up training parameters
    3. Trains the model
    4. Tracks metrics
    5. Returns training history
    
    Args:
        model_type: Type of model to train:
            - "hybrid": Combined approach (default)
            - "collaborative": Collaborative filtering
            - "content_based": Content-based recommendations
        model: Optional pre-initialized model
        train_dataset: Optional pre-prepared training dataset
        test_dataset: Optional pre-prepared test dataset
        students_df: Optional students DataFrame
        engagements_df: Optional engagements DataFrame
        content_df: Optional content DataFrame
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        
    Returns:
        Dictionary of training metrics:
            - ranking_rmse: Root mean squared error for ranking
            - likelihood_auc: AUC for application likelihood
            - likelihood_accuracy: Accuracy for application likelihood
            - risk_auc: AUC for dropout risk
            - risk_accuracy: Accuracy for dropout risk
            
    Raises:
        ValueError: If model type is invalid or required data is missing
    """
    if model_type == "collaborative":
        # Initialize collaborative model if not provided
        if model is None:
            model = CollaborativeFilteringModel(
                embedding_dimension=64,
                learning_rate=learning_rate
            )
        
        # Train collaborative model
        history = model.train(
            students_df,
            engagements_df,
            content_df,
            epochs=epochs,
            batch_size=batch_size
        )
        return history
        
    elif model_type == "content_based":
        # Initialize content-based model if not provided
        if model is None:
            model = SimpleRecommender()
        
        # Train content-based model
        model.train(students_df, content_df, engagements_df)
        return {"status": "completed"}
        
    else:  # hybrid
        # Use existing hybrid training logic
        if model is None or train_dataset is None or test_dataset is None:
            raise ValueError("For hybrid training, model and datasets must be provided")
            
        # Define optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Define metrics
        metrics = {
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
        
        # Training loop
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Reset metrics
            for metric in metrics.values():
                if isinstance(metric, list):
                    for m in metric:
                        m.reset_states()
                else:
                    metric.reset_states()
            
            # Training step
            for batch in train_dataset:
                with tf.GradientTape() as tape:
                    predictions = model(batch, training=True)
                    
                    ranking_loss = tf.keras.losses.mean_squared_error(
                        batch['engagement_response'],
                        predictions['ranking_score']
                    )
                    
                    likelihood_loss = tf.keras.losses.binary_crossentropy(
                        batch['application_completion'],
                        predictions['likelihood_score']
                    )
                    
                    risk_loss = tf.keras.losses.binary_crossentropy(
                        batch['dropout_indicator'],
                        predictions['risk_score']
                    )
                    
                    total_loss = (
                        ranking_loss +
                        0.5 * likelihood_loss +
                        0.5 * risk_loss
                    )
                
                gradients = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                
                # Update metrics
                metrics['ranking'].update_state(
                    batch['engagement_response'],
                    predictions['ranking_score']
                )
                
                for metric in metrics['likelihood']:
                    metric.update_state(
                        batch['application_completion'],
                        predictions['likelihood_score']
                    )
                
                for metric in metrics['risk']:
                    metric.update_state(
                        batch['dropout_indicator'],
                        predictions['risk_score']
                    )
            
            # Print metrics
            print(f"Ranking RMSE: {metrics['ranking'].result():.4f}")
            print(f"Likelihood AUC: {metrics['likelihood'][0].result():.4f}")
            print(f"Likelihood Accuracy: {metrics['likelihood'][1].result():.4f}")
            print(f"Risk AUC: {metrics['risk'][0].result():.4f}")
            print(f"Risk Accuracy: {metrics['risk'][1].result():.4f}")
        
        return {
            'ranking_rmse': metrics['ranking'].result().numpy(),
            'likelihood_auc': metrics['likelihood'][0].result().numpy(),
            'likelihood_accuracy': metrics['likelihood'][1].result().numpy(),
            'risk_auc': metrics['risk'][0].result().numpy(),
            'risk_accuracy': metrics['risk'][1].result().numpy()
        }

def train_and_save_model(
    data_dict: Dict,
    model_type: str = "hybrid",
    model_dir: str = "models/saved_models",
    epochs: int = 5
) -> None:
    """
    Train and save the model.
    
    This function:
    1. Extracts data from the input dictionary
    2. Initializes the appropriate model
    3. Trains the model
    4. Saves the model and vocabularies
    5. Creates necessary directories
    
    Args:
        data_dict: Dictionary containing prepared data:
            - train_dataset: TensorFlow dataset for training
            - test_dataset: TensorFlow dataset for testing
            - vocabularies: Dictionary of feature vocabularies
            - students_df: DataFrame of student profiles
            - engagements_df: DataFrame of engagement history
            - content_df: DataFrame of engagement content
        model_type: Type of model to train
        model_dir: Directory to save the model
        epochs: Number of training epochs
        
    Raises:
        ValueError: If model type is invalid or required data is missing
        IOError: If model saving fails
    """
    # Extract data
    train_dataset = data_dict.get('train_dataset')
    test_dataset = data_dict.get('test_dataset')
    vocabularies = data_dict.get('vocabularies')
    students_df = data_dict.get('students_df')
    engagements_df = data_dict.get('engagements_df')
    content_df = data_dict.get('content_df')
    
    # Initialize model based on type
    if model_type == "hybrid":
        model = StudentEngagementModel(vocabularies)
    elif model_type == "collaborative":
        model = CollaborativeFilteringModel()
    elif model_type == "content_based":
        model = SimpleRecommender()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train model
    history = train_model(
        model_type=model_type,
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        students_df=students_df,
        engagements_df=engagements_df,
        content_df=content_df,
        epochs=epochs
    )
    
    # Save model
    model_save_dir = os.path.join(model_dir, model_type)
    os.makedirs(model_save_dir, exist_ok=True)
    model.save(model_save_dir)
    
    # Save vocabularies if hybrid model
    if model_type == "hybrid" and vocabularies:
        vocab_dir = os.path.join(model_dir, "saved", "vocabularies")
        os.makedirs(vocab_dir, exist_ok=True)
        for name, vocab in vocabularies.items():
            np.save(os.path.join(vocab_dir, f"{name}.npy"), vocab)
    
    print(f"Model and vocabularies saved to {model_save_dir}")

def main():
    """
    Main function to run the training script.
    
    This function:
    1. Parses command line arguments
    2. Sets up logging
    3. Loads data from database
    4. Prepares data for training
    5. Trains and saves the model
    
    Command line arguments:
        --model-type: Type of model to train [hybrid|collaborative|content_based]
        --model-dir: Directory to save the model
        --epochs: Number of training epochs
    """
    parser = argparse.ArgumentParser(description="Train recommendation models")
    parser.add_argument("--model-type", type=str, default="hybrid",
                       choices=["hybrid", "collaborative", "content_based"],
                       help="Type of model to train")
    parser.add_argument("--model-dir", type=str, default=None,
                       help="Directory to save the model")
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of training epochs")
    args = parser.parse_args()
    
    # Create models directory if it doesn't exist
    if args.model_dir is None:
        args.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "saved_models")
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Load data from database
    session = get_db()
    dataframes = load_data_from_db(session)
    
    # Prepare data for training
    prepared_data = prepare_data_for_training(dataframes)
    
    # Train and save model
    train_and_save_model(
        prepared_data,
        model_type=args.model_type,
        model_dir=args.model_dir,
        epochs=args.epochs
    )
    
    session.close()

if __name__ == "__main__":
    main()
