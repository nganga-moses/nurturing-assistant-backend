import pandas as pd
import tensorflow as tf
from models.core.recommender_model import RecommenderModel
from models.core.data_generator import DataGenerator
import json
import numpy as np
import pytest
from tests.fixtures.synthetic_data import create_sample_data

def create_sample_data(num_samples=100):
    """Create sample data for testing."""
    # Generate student IDs
    student_ids = [f"student{i}" for i in range(num_samples)]
    engagement_ids = [f"engagement{i}" for i in range(num_samples)]
    
    # Create sample student data
    students_data = []
    for i in range(num_samples):
        students_data.append({
            'student_id': student_ids[i],
            'demographic_features': json.dumps({
                'age': 20.0,
                'gender': 1.0,
                'ethnicity': 1.0,
                'location': 1.0,
                'gpa': 3.5,
                'test_scores': 1200.0,
                'courses': 5.0,
                'major': 1.0,
                'attendance': 0.9,
                'participation': 0.8,
                'feedback': 0.7,
                'study_habits': 0.8,
                'social_activity': 0.6,
                'stress_level': 0.4
            }),
            'application_status': 'active',
            'funnel_stage': 'consideration',
            'first_interaction_date': '2024-01-01',
            'last_interaction_date': '2024-03-01',
            'interaction_count': 10,
            'application_likelihood_score': 0.8,
            'dropout_risk_score': 0.2
        })
    
    # Create sample engagement data
    engagements_data = []
    for i in range(num_samples):
        engagements_data.append({
            'engagement_id': engagement_ids[i],
            'student_id': student_ids[i],
            'engagement_type': 'email',
            'engagement_content_id': f'content_{i}',
            'timestamp': '2024-03-01',
            'engagement_response': 'opened',
            'engagement_metrics': json.dumps({
                'type': 1.0,
                'duration': 60.0,
                'difficulty': 0.7,
                'prerequisites': 0.5,
                'popularity': 0.8,
                'success_rate': 0.9
            }),
            'funnel_stage_before': 'awareness',
            'funnel_stage_after': 'interest'
        })
    
    # Create DataFrames
    students_df = pd.DataFrame(students_data)
    engagements_df = pd.DataFrame(engagements_data)
    
    # Create TensorFlow datasets with labels
    train_data = tf.data.Dataset.from_tensor_slices({
        'student_id': students_df['student_id'].values,
        'engagement_id': engagements_df['engagement_id'].values,
        'student_features': {
            k: tf.constant([json.loads(x)[k] for x in students_df['demographic_features']], dtype=tf.float32)
            for k in json.loads(students_df['demographic_features'].iloc[0]).keys()
        },
        'engagement_features': {
            k: tf.constant([json.loads(x)[k] for x in engagements_df['engagement_metrics']], dtype=tf.float32)
            for k in json.loads(engagements_df['engagement_metrics'].iloc[0]).keys()
        },
        # Add labels for all three tasks
        'ranking_label': tf.constant(np.random.uniform(0, 1, num_samples), dtype=tf.float32),  # Random ranking scores
        'likelihood_label': tf.constant(np.random.binomial(1, 0.7, num_samples), dtype=tf.float32),  # Binary labels for likelihood
        'risk_label': tf.constant(np.random.binomial(1, 0.3, num_samples), dtype=tf.float32)  # Binary labels for risk
    }).batch(32)
    
    val_data = train_data.take(1)  # Use a small portion for validation
    
    return train_data, val_data, student_ids, engagement_ids

def main():
    print("Creating sample data...")
    train_data, val_data, student_ids, engagement_ids = create_sample_data(num_samples=100)

    print("Initializing model and trainer...")
    data_dict = {
        "train_dataset": train_data,
        "test_dataset": val_data,
        "vocabularies": {
            "student_ids": student_ids,
            "engagement_ids": engagement_ids
        },
        "dataframes": {
            "engagements": pd.DataFrame({
                "engagement_id": engagement_ids,
                "engagement_content_id": [f"content_{i}" for i in range(len(engagement_ids))]
            })
        }
    }
    
    # Initialize model
    model = RecommenderModel(
        student_ids=student_ids,
        engagement_ids=engagement_ids,
        embedding_dimension=64
    )
    
    print("Training model...")
    # Train for a few epochs
    for epoch in range(2):
        total_loss = 0
        num_batches = 0
        
        for batch in train_data:
            with tf.GradientTape() as tape:
                predictions = model(batch, training=True)
                
                # Calculate losses for each task
                ranking_loss = tf.reduce_mean(tf.square(predictions['ranking_score'] - batch['ranking_label']))
                likelihood_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                    tf.expand_dims(batch['likelihood_label'], axis=-1), predictions['likelihood_score']))
                risk_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                    tf.expand_dims(batch['risk_label'], axis=-1), predictions['risk_score']))
                
                # Combine losses with weights
                loss = ranking_loss + 0.5 * likelihood_loss + 0.5 * risk_loss
                
                total_loss += loss
                num_batches += 1
            
            # Calculate and apply gradients
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # Print average loss for the epoch
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")

    print("Testing predictions...")
    # Use DataGenerator for proper batching of test data
    data_generator = DataGenerator(batch_size=1)
    test_dataset = data_generator.generate_dataset(
        student_ids=["student1"],
        engagement_ids=["engagement1"],
        num_batches=1
    )
    
    # Make predictions using the batched dataset
    for batch in test_dataset:
        predictions = model(batch, training=False)
        print("Prediction results:")
        for task, score in predictions.items():
            print(f"{task}: {score.numpy()[0][0]:.4f}")

def test_model_initialization():
    """Test model initialization with sample data."""
    train_data, val_data, student_ids, engagement_ids = create_sample_data(num_samples=100)
    
    model = RecommenderModel(
        student_ids=student_ids,
        engagement_ids=engagement_ids,
        embedding_dimension=64
    )
    
    assert model is not None
    assert len(model.trainable_variables) > 0

def test_model_training(tf_dataset, model_config):
    """Test model training with TensorFlow dataset."""
    model = RecommenderModel(
        student_ids=[f"student{i}" for i in range(100)],
        engagement_ids=[f"engagement{i}" for i in range(100)],
        embedding_dimension=model_config['embedding_dimension']
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=model_config['learning_rate'])
    
    for epoch in range(model_config['epochs']):
        total_loss = 0
        num_batches = 0
        
        for batch in tf_dataset:
            with tf.GradientTape() as tape:
                predictions = model(batch, training=True)
                
                # Calculate losses for each task
                ranking_loss = tf.reduce_mean(tf.square(predictions['ranking_score'] - batch['ranking_label']))
                likelihood_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                    tf.expand_dims(batch['likelihood_label'], axis=-1), predictions['likelihood_score']))
                risk_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                    tf.expand_dims(batch['risk_label'], axis=-1), predictions['risk_score']))
                
                # Combine losses with weights
                loss = ranking_loss + 0.5 * likelihood_loss + 0.5 * risk_loss
                
                total_loss += loss
                num_batches += 1
            
            # Calculate and apply gradients
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # Print average loss for the epoch
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")
        
        # Assert that loss is decreasing
        if epoch > 0:
            assert avg_loss < prev_loss
        prev_loss = avg_loss

def test_model_prediction(tf_dataset):
    """Test model prediction functionality."""
    model = RecommenderModel(
        student_ids=[f"student{i}" for i in range(100)],
        engagement_ids=[f"engagement{i}" for i in range(100)],
        embedding_dimension=64
    )
    
    # Get a single batch for testing
    test_batch = next(iter(tf_dataset))
    
    # Make predictions
    predictions = model(test_batch, training=False)
    
    # Check prediction structure
    assert 'ranking_score' in predictions
    assert 'likelihood_score' in predictions
    assert 'risk_score' in predictions
    
    # Check prediction shapes
    assert predictions['ranking_score'].shape == (32, 1)
    assert predictions['likelihood_score'].shape == (32, 1)
    assert predictions['risk_score'].shape == (32, 1)
    
    # Check prediction ranges
    assert tf.reduce_all(predictions['ranking_score'] >= 0) and tf.reduce_all(predictions['ranking_score'] <= 1)
    assert tf.reduce_all(predictions['likelihood_score'] >= 0) and tf.reduce_all(predictions['likelihood_score'] <= 1)
    assert tf.reduce_all(predictions['risk_score'] >= 0) and tf.reduce_all(predictions['risk_score'] <= 1)

if __name__ == "__main__":
    main() 