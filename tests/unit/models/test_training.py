import pandas as pd
import tensorflow as tf
import pytest
from models.core.recommender_model import StudentEngagementModel
from models.core.cross_validation import CrossValidationTrainer
from data.processing.data_processor import DataProcessor
from data.processing.feature_engineering import AdvancedFeatureEngineering
from tests.fixtures.synthetic_data import create_sample_data
import json

def main():
    print("Loading sample data...")
    # Load sample data
    students_df = pd.read_csv('data/sample_students.csv')
    engagements_df = pd.read_csv('data/sample_engagements.csv')
    content_df = pd.read_csv('data/sample_content.csv')

    # Add engagement_features column by parsing engagement_metrics
    engagements_df['engagement_features'] = engagements_df['engagement_metrics'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    
    print("Processing data...")
    # Initialize data processor
    data_processor = DataProcessor()
    
    # Create TensorFlow datasets
    data_dict = data_processor.prepare_data_for_training(batch_size=32)
    
    print("Initializing model...")
    # Create model
    model = StudentEngagementModel(
        student_ids=data_dict['vocabularies']['student_ids'],
        engagement_ids=data_dict['vocabularies']['engagement_ids'],
        embedding_dimension=64
    )
    
    print("Setting up cross-validation trainer...")
    # Initialize cross-validation trainer
    cv_trainer = CrossValidationTrainer(
        model=model,
        data=engagements_df,
        n_splits=3  # Using 3 splits for testing
    )
    
    print("Starting training...")
    # Train model with cross-validation
    cv_results = cv_trainer.train_with_cross_validation(
        epochs=2,  # Using 2 epochs for testing
        batch_size=32
    )
    
    print("\nTraining Results:")
    print("Average Scores:", cv_results['average_scores'])
    print("\nFold Scores:")
    for i, fold_score in enumerate(cv_results['fold_scores']):
        print(f"Fold {i+1}:", fold_score)

def test_data_preparation():
    """Test data preparation for training."""
    # Create sample data
    train_data, val_data, student_ids, engagement_ids = create_sample_data(num_samples=100)
    
    # Initialize data processor
    data_processor = DataProcessor()
    
    # Create TensorFlow datasets
    data_dict = data_processor.prepare_data_for_training(batch_size=32)
    
    # Check data dictionary structure
    assert 'train_dataset' in data_dict
    assert 'test_dataset' in data_dict
    assert 'vocabularies' in data_dict
    assert 'student_ids' in data_dict['vocabularies']
    assert 'engagement_ids' in data_dict['vocabularies']

def test_model_initialization():
    """Test model initialization."""
    # Create sample data
    train_data, val_data, student_ids, engagement_ids = create_sample_data(num_samples=100)
    
    # Create model
    model = StudentEngagementModel(
        student_ids=student_ids,
        engagement_ids=engagement_ids,
        embedding_dimension=64
    )
    
    assert model is not None
    assert len(model.trainable_variables) > 0

def test_cross_validation(tf_dataset, model_config):
    """Test cross-validation training."""
    # Create model
    model = StudentEngagementModel(
        student_ids=[f"student{i}" for i in range(100)],
        engagement_ids=[f"engagement{i}" for i in range(100)],
        embedding_dimension=model_config['embedding_dimension']
    )
    
    # Initialize cross-validation trainer
    cv_trainer = CrossValidationTrainer(
        model=model,
        data=tf_dataset,
        n_splits=3
    )
    
    # Train model with cross-validation
    cv_results = cv_trainer.train_with_cross_validation(
        epochs=model_config['epochs'],
        batch_size=model_config['batch_size']
    )
    
    # Check results structure
    assert 'average_scores' in cv_results
    assert 'fold_scores' in cv_results
    assert len(cv_results['fold_scores']) == 3
    
    # Check metrics
    for fold_score in cv_results['fold_scores']:
        assert 'loss' in fold_score
        assert 'accuracy' in fold_score
        assert 'auc' in fold_score

if __name__ == "__main__":
    main() 