import pandas as pd
import tensorflow as tf
from models.recommender_model import StudentEngagementModel
from models.cross_validation import CrossValidationTrainer
from data.data_processor import DataProcessor
from data.feature_engineering import AdvancedFeatureEngineering
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

if __name__ == "__main__":
    main() 