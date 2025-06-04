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
    student_ids = [f"student{i}" for i in range(num_samples)]
    engagement_ids = [f"engagement{i}" for i in range(num_samples)]
    student_features = tf.convert_to_tensor(np.random.rand(num_samples, 10))
    engagement_features = tf.convert_to_tensor(np.random.rand(num_samples, 10))
    labels = np.random.randint(0, 2, size=(num_samples, 1))
    risk_labels = np.random.randint(0, 2, size=(num_samples, 1))
    return student_features, engagement_features, student_ids, engagement_ids, labels, risk_labels

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
                predictions = model(batch[0], training=True)
                
                # Calculate losses for each task
                ranking_loss = tf.reduce_mean(tf.square(predictions['ranking_score'] - batch[1]['ranking_label']))
                likelihood_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                    tf.expand_dims(batch[1]['likelihood_label'], axis=-1), predictions['likelihood_score']))
                risk_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                    tf.expand_dims(batch[1]['risk_label'], axis=-1), predictions['risk_score']))
                
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
        predictions = model(batch[0], training=False)
        print("Prediction results:")
        for task, score in predictions.items():
            print(f"{task}: {score.numpy()[0][0]:.4f}")

def test_model_initialization():
    """Test model initialization with sample data."""
    student_features, engagement_features, student_ids, engagement_ids, labels, risk_labels = create_sample_data(num_samples=100)
    
    model = RecommenderModel(
        student_ids=student_ids,
        engagement_ids=engagement_ids,
        embedding_dimension=64
    )
    
    # Build the model with a sample input shape
    sample_input = {
        "student_id": tf.convert_to_tensor([student_ids[0]]),
        "engagement_id": tf.convert_to_tensor([engagement_ids[0]]),
        "student_features": tf.convert_to_tensor([student_features[0]]),
        "engagement_features": tf.convert_to_tensor([engagement_features[0]])
    }
    model.build(sample_input)
    
    assert model is not None
    assert len(model.trainable_variables) > 0

def test_model_training(tf_dataset, model_config):
    """Test model training with TensorFlow dataset."""
    model = RecommenderModel(
        student_ids=[f"student{i}" for i in range(100)],
        engagement_ids=[f"engagement{i}" for i in range(100)],
        embedding_dimension=model_config['embedding_dimension']
    )
    
    # Build the model with a sample input shape
    sample_input = {
        "student_id": tf.convert_to_tensor([f"student{i}" for i in range(100)]),
        "engagement_id": tf.convert_to_tensor([f"engagement{i}" for i in range(100)]),
        "student_features": tf.convert_to_tensor(np.random.rand(100, 10)),
        "engagement_features": tf.convert_to_tensor(np.random.rand(100, 10))
    }
    model.build(sample_input)
    
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
    
    # Build the model with a sample input shape
    sample_input = {
        "student_id": tf.convert_to_tensor([f"student{i}" for i in range(100)]),
        "engagement_id": tf.convert_to_tensor([f"engagement{i}" for i in range(100)]),
        "student_features": tf.convert_to_tensor(np.random.rand(100, 10)),
        "engagement_features": tf.convert_to_tensor(np.random.rand(100, 10))
    }
    model.build(sample_input)
    
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

@pytest.fixture
def tf_dataset():
    """Create a TensorFlow dataset for testing."""
    # Generate student and engagement IDs
    student_ids = [f"student{i}" for i in range(100)]
    engagement_ids = [f"engagement{i}" for i in range(100)]
    
    # Generate random features for students and engagements
    student_features = {
        "age": np.random.uniform(15, 25, size=100),
        "gender": np.random.uniform(0, 1, size=100),
        "ethnicity": np.random.uniform(0, 1, size=100),
        "location": np.random.uniform(0, 1, size=100),
        "gpa": np.random.uniform(0, 4, size=100),
        "test_scores": np.random.uniform(0, 100, size=100),
        "courses": np.random.uniform(0, 10, size=100),
        "major": np.random.uniform(0, 1, size=100),
        "attendance": np.random.uniform(0, 1, size=100),
        "participation": np.random.uniform(0, 1, size=100)
    }
    student_features_tensor = tf.stack([tf.convert_to_tensor(student_features[k], dtype=tf.float32) for k in [
        "age", "gender", "ethnicity", "location", "gpa", "test_scores", "courses", "major", "attendance", "participation"
    ]], axis=1)
    
    engagement_features = {
        "type": np.random.uniform(0, 1, size=100),
        "duration": np.random.uniform(0, 100, size=100),
        "difficulty": np.random.uniform(0, 1, size=100),
        "prerequisites": np.random.uniform(0, 1, size=100),
        "popularity": np.random.uniform(0, 1, size=100),
        "success_rate": np.random.uniform(0, 1, size=100),
        "engagement_level": np.random.uniform(0, 1, size=100),
        "feedback_score": np.random.uniform(0, 1, size=100),
        "completion_rate": np.random.uniform(0, 1, size=100),
        "interaction_frequency": np.random.uniform(0, 1, size=100)
    }
    engagement_features_tensor = tf.stack([tf.convert_to_tensor(engagement_features[k], dtype=tf.float32) for k in [
        "type", "duration", "difficulty", "prerequisites", "popularity", "success_rate",
        "engagement_level", "feedback_score", "completion_rate", "interaction_frequency"
    ]], axis=1)
    
    # Generate random labels
    ranking_labels = np.random.uniform(0, 10, size=100)
    likelihood_labels = np.random.uniform(0, 1, size=100)
    risk_labels = np.random.uniform(0, 1, size=100)
    
    # Create a TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices({
        "student_id": tf.convert_to_tensor(student_ids),
        "engagement_id": tf.convert_to_tensor(engagement_ids),
        "student_features": student_features_tensor,
        "engagement_features": engagement_features_tensor,
        "ranking_label": tf.convert_to_tensor(ranking_labels, dtype=tf.float32),
        "likelihood_label": tf.convert_to_tensor(likelihood_labels, dtype=tf.float32),
        "risk_label": tf.convert_to_tensor(risk_labels, dtype=tf.float32)
    }).batch(32)
    
    return dataset

if __name__ == "__main__":
    main() 