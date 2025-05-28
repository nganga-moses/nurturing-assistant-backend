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
from typing import Dict, List, Tuple

# Import synthetic data generation
from models.synthetic_data import generate_synthetic_data

# Add the parent directory to the path so we can import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.models import StudentProfile, EngagementHistory, EngagementContent, get_session
from models.recommender_model import ModelTrainer

def load_data_from_db(session: Session) -> Dict[str, pd.DataFrame]:
    """
    Load data from the database into pandas DataFrames.
    
    Args:
        session: SQLAlchemy session
        
    Returns:
        Dictionary of DataFrames
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
    
    Args:
        dataframes: Dictionary of DataFrames
        
    Returns:
        Dictionary containing prepared data
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
    model: StudentEngagementModel,
    train_dataset: tf.data.Dataset,
    test_dataset: tf.data.Dataset,
    epochs: int = 5,
    batch_size: int = 32
) -> Dict[str, float]:
    """Train the model with collaborative features."""
    
    # Define optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
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
                # Forward pass
                predictions = model(batch, training=True)
                
                # Calculate losses
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
                
                # Combine losses with weights
                total_loss = (
                    ranking_loss +
                    0.5 * likelihood_loss +
                    0.5 * risk_loss
                )
            
            # Calculate gradients and update weights
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
        
        # Update student embeddings for next epoch
        student_embeddings = []
        for batch in train_dataset:
            student_embedding = model.student_tower({
                "student_id": batch["student_id"],
                "student_features": batch.get("student_features", {})
            }, training=False)
            student_embeddings.append(student_embedding)
        
        # Concatenate all embeddings
        all_student_embeddings = tf.concat(student_embeddings, axis=0)
        
        # Update model with new embeddings
        model.update_student_embeddings(all_student_embeddings)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    
    # Reset metrics
    for metric in metrics.values():
        if isinstance(metric, list):
            for m in metric:
                m.reset_states()
        else:
            metric.reset_states()
    
    # Evaluation step
    for batch in test_dataset:
        predictions = model(batch, training=False)
        
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
    
    # Return final metrics
    return {
        'ranking_rmse': metrics['ranking'].result().numpy(),
        'likelihood_auc': metrics['likelihood'][0].result().numpy(),
        'likelihood_accuracy': metrics['likelihood'][1].result().numpy(),
        'risk_auc': metrics['risk'][0].result().numpy(),
        'risk_accuracy': metrics['risk'][1].result().numpy()
    }

def train_and_save_model(data_dict: Dict, model_dir: str = "models", epochs: int = 5) -> None:
    """
    Train and save the recommendation model.
    
    Args:
        data_dict: Dictionary containing prepared data
        model_dir: Directory to save the model
        epochs: Number of training epochs
    """
    print(f"Training model for {epochs} epochs...")
    
    # Create model trainer
    trainer = ModelTrainer(data_dict)
    
    # Train model
    trainer.train(epochs=epochs)
    
    # Evaluate model
    evaluation_results = trainer.evaluate()
    print(f"Evaluation results: {evaluation_results}")
    
    # Save model
    trainer.save_model(model_dir=model_dir)
    
    print(f"Model saved to {model_dir}")

def main(model_dir=None, epochs=5):
    # Create models directory if it doesn't exist
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Get database session
    session = get_session()
    
    try:
        # Load data from database
        dataframes = load_data_from_db(session)
        
        # Prepare data for training
        data_dict = prepare_data_for_training(dataframes)
        
        # Train and save model
        train_and_save_model(data_dict, model_dir=model_dir, epochs=epochs)
        
        print("Model training completed successfully!")
        
    except Exception as e:
        print(f"Error training model: {str(e)}")
        raise
    finally:
        session.close()

if __name__ == "__main__":
    main()
