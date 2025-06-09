#!/usr/bin/env python
"""
Script for initial data ingestion and model training.

This script is meant to be run ONCE during initial setup to:
1. Load initial data from CSV files
2. Store it in the database
3. Train the initial recommendation models

The script supports:
- Loading data from multiple CSV files
- Data validation and preprocessing
- Database initialization and population
- Initial model training
- Model evaluation and saving

Usage:
    # Basic usage with students and engagements CSVs
    python ingest_and_train.py \
        --students-csv data/students.csv \
        --engagements-csv data/engagements.csv

    # With custom model parameters
    python ingest_and_train.py \
        --students-csv data/students.csv \
        --engagements-csv data/engagements.csv \
        --epochs 10 \
        --batch-size 64

For incremental updates, use:
- ingest_students.py: For new/updated student data
- ingest_engagements.py: For new engagement data
"""

import os
import sys
import argparse
import pandas as pd
from datetime import datetime
import logging
from sqlalchemy.orm import Session
from typing import Dict
import tensorflow as tf

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.core.recommender_model import RecommenderModel, StudentTower, EngagementTower
from models.core.model_trainer import ModelTrainer
from models.core.data_processor import DataProcessor
from data.models.student_profile import StudentProfile
from data.models.engagement_history import EngagementHistory
from data.models.stored_recommendation import StoredRecommendation
from api.services.matching_service import MatchingService
from database.session import get_db

# Import from the new models structure
from models.core.data_preprocessor import DataPreprocessor
from data.processing.validator import DataValidator, DataImputation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Ingest CSV data and train the recommendation model")
    parser.add_argument("--students-csv", type=str, required=True, help="Path to students CSV file")
    parser.add_argument("--engagements-csv", type=str, required=True, help="Path to engagements CSV file")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save the model (default: models/saved_models)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--embedding-dim", type=int, default=128, help="Dimension of embeddings")
    return parser.parse_args()

def load_data_from_csv(students_path, engagements_path):
    """Load student and engagement data from CSV files."""
    logging.info(f"Loading students from {students_path}")
    students_df = pd.read_csv(students_path)
    logging.info(f"Loaded {len(students_df)} students")
    
    logging.info(f"Loading engagements from {engagements_path}")
    # Parse timestamp field as datetime when loading engagements
    engagements_df = pd.read_csv(
        engagements_path,
        parse_dates=['timestamp'],  # This ensures timestamp is parsed as datetime
        date_format='ISO8601'       # Handle ISO format dates
    )
    logging.info(f"Loaded {len(engagements_df)} engagements")
    
    # Create empty content DataFrame for now
    content_df = pd.DataFrame(columns=['content_id', 'content_type', 'title'])
    
    return {
        'students': students_df,
        'engagements': engagements_df,
        'content': content_df
    }

def prepare_data_for_training(dataframes: dict) -> dict:
    """Prepare data for training."""
    logger.info("Preparing data for training...")
    
    # Initialize data processor
    data_processor = DataProcessor()
    
    # Process data
    processed_data = data_processor.prepare_data(
        dataframes['students'],
        dataframes['engagements'],
        dataframes['content']
    )
    
    # Create vocabularies
    vocabularies = {
        'student_vocab': dataframes['students']['student_id'].unique().tolist(),
        'engagement_vocab': dataframes['engagements']['engagement_id'].unique().tolist()
    }
    
    # Split data into train and test sets
    train_size = int(0.8 * len(processed_data['interaction_dataset']))
    train_dataset = processed_data['interaction_dataset'].take(train_size)
    test_dataset = processed_data['interaction_dataset'].skip(train_size)
    
    return {
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'vocabularies': vocabularies,
        'dataframes': dataframes
    }

def train_and_save_model(
    prepared_data: dict,
    model_dir: str,
    epochs: int = 10,
    batch_size: int = 32,
    embedding_dimension: int = 128
) -> None:
    """Train and save the model."""
    logger.info("Initializing model trainer...")
    trainer = ModelTrainer(prepared_data, embedding_dimension=embedding_dimension)
    
    logger.info("Starting model training...")
    history = trainer.train(epochs=epochs, batch_size=batch_size)
    
    # Save training history
    history_path = os.path.join(model_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        pd.DataFrame(history.history).to_json(f)
    
    logger.info(f"Training history saved to {history_path}")
    logger.info("Training completed successfully!")

def parse_and_format_date(date_str):
    """Parse a date string and return it in 'YYYY-MM-DD HH:MM:SS' format, or None if invalid."""
    if pd.isna(date_str) or not isinstance(date_str, str):
        return None
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%Y/%m/%d"):  # Try common formats
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            continue
    return None

def store_data_in_database(
    session: Session,
    dataframes: Dict[str, pd.DataFrame]
) -> None:
    """
    Store data in the database from the loaded DataFrames.
    This function expects students and engagements DataFrames to have flat columns.
    It reconstructs the JSON fields as needed.
    """
    students_df = dataframes['students']
    engagements_df = dataframes['engagements']

    # --- Parse and format date fields in students_df ---
    for col in ["birthdate", "first_interaction_date", "last_interaction_date"]:
        if col in students_df.columns:
            students_df[col] = students_df[col].apply(parse_and_format_date)

    # --- Parse and format date fields in engagements_df ---
    for col in ["date", "created_at", "updated_at", "interaction_date"]:
        if col in engagements_df.columns:
            engagements_df[col] = engagements_df[col].apply(parse_and_format_date)

    # --- Students: Build demographic_features JSON ---
    students_df['demographic_features'] = students_df.apply(
        lambda row: {
            "location": row.get("location"),
            "intended_major": row.get("intended_major"),
            "email": row.get("email"),
            "phone": row.get("phone")
        }, axis=1
    )

    # --- Engagements: Build engagement_metrics JSON ---
    metric_cols = ["open_time", "click_through", "time_spent"]
    if all(col in engagements_df.columns for col in metric_cols):
        engagements_df['engagement_metrics'] = engagements_df.apply(
            lambda row: {
                "open_time": row.get("open_time"),
                "click_through": row.get("click_through"),
                "time_spent": row.get("time_spent")
            }, axis=1
        )
    else:
        logger.warning("Missing engagement metric columns. engagement_metrics will not be built.")
        engagements_df['engagement_metrics'] = [{}] * len(engagements_df)

    # Now store in DB
    student_objects = []
    for _, row in students_df.iterrows():
        # Sanitize numeric fields
        def safe_int(val, default=0):
            try:
                if pd.isna(val):
                    return default
                return int(float(val))
            except Exception:
                return default
        def safe_float(val, default=0.0):
            try:
                if pd.isna(val):
                    return default
                return float(val)
            except Exception:
                return default

        interaction_count = safe_int(row.get('interaction_count'), 0)
        gpa = safe_float(row.get('gpa'), 0.0)
        sat_score = safe_int(row.get('sat_score'), 0)
        act_score = safe_int(row.get('act_score'), 0)

        # Check if student already exists
        existing_student = session.query(StudentProfile).filter_by(student_id=row['student_id']).first()
        
        if existing_student:
            # Update existing student
            for key, value in row.items():
                if key == 'interaction_count':
                    value = interaction_count
                elif key == 'gpa':
                    value = gpa
                elif key == 'sat_score':
                    value = sat_score
                elif key == 'act_score':
                    value = act_score
                if hasattr(existing_student, key):
                    setattr(existing_student, key, value)
            existing_student.updated_at = datetime.now()
            student_objects.append(existing_student)
        else:
            # Create new student
            student = StudentProfile(
                student_id=row['student_id'],
                first_name=row['first_name'],
                last_name=row['last_name'],
                birthdate=row['birthdate'],
                recruiter_id=row['recruiter_id'],
                demographic_features=row['demographic_features'],
                application_status=row['application_status'],
                funnel_stage=row['funnel_stage'],
                first_interaction_date=row['first_interaction_date'],
                last_interaction_date=row['last_interaction_date'],
                interaction_count=interaction_count,
                application_likelihood_score=None,
                dropout_risk_score=None,
                enrollment_agent_id=row['recruiter_id'],  # Using recruiter_id as enrollment_agent_id
                created_at=datetime.now(),
                updated_at=datetime.now(),
                gpa=gpa,
                sat_score=sat_score,
                act_score=act_score
            )
            student_objects.append(student)

    # Store students
    session.bulk_save_objects(student_objects)
    session.commit()

    # Store engagements
    engagement_objects = []
    for _, row in engagements_df.iterrows():
        # Check if engagement already exists
        existing_engagement = session.query(EngagementHistory).filter_by(engagement_id=row['engagement_id']).first()
        
        if existing_engagement:
            # Update existing engagement
            for key, value in row.items():
                if hasattr(existing_engagement, key):
                    setattr(existing_engagement, key, value)
            engagement_objects.append(existing_engagement)
        else:
            # Create new engagement
            engagement = EngagementHistory(
                engagement_id=row['engagement_id'],
                student_id=row['student_id'],
                engagement_type=row['engagement_type'],
                engagement_content_id=row.get('engagement_content_id'),
                timestamp=row['timestamp'],
                engagement_response=row['engagement_response'],
                engagement_metrics=row['engagement_metrics'],
                funnel_stage_before=row['funnel_stage_before'],
                funnel_stage_after=row['funnel_stage_after']
            )
            engagement_objects.append(engagement)

    session.bulk_save_objects(engagement_objects)
    session.commit()

def main():
    """
    Main function to run the initial data ingestion and training script.

    This function:
    1. Parses command line arguments
    2. Sets up logging
    3. Loads data from CSV files
    4. Stores data
    5. Trains initial models

    Command line arguments:
        --students-csv: Path to students CSV file
        --engagements-csv: Path to engagements CSV file
        --epochs: Number of training epochs
        --model-dir: Directory to save models
    """
    # Ensure PostgreSQL is used by default
    if "DATABASE_URL" not in os.environ:
        os.environ["DATABASE_URL"] = "postgresql://postgres:postgres@localhost:5432/student_engagement"

    args = parse_args()
    print("=" * 80)
    print(f"Starting data ingestion and model training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    try:
        # Get database session
        session = next(get_db())
        try:
            # Load data from CSV (only students and engagements)
            dataframes = load_data_from_csv(args.students_csv, args.engagements_csv)
            # Store data in database
            store_data_in_database(session, dataframes)

            # --- Post-import batch matching step ---
            # This step matches engagements to recommendations after import.
            print("Running post-import batch matching for all engagements...")
            matching_service = MatchingService(session)
            all_engagements = session.query(EngagementHistory).all()
            matched_count = 0
            # Note: This loop iterates through all engagements which might be slow for large datasets.
            # Consider optimizing this if performance is an issue.
            for engagement in all_engagements:
                # Assuming match_engagement_to_recommendation updates the DB directly
                matched, confidence = matching_service.match_engagement_to_recommendation(engagement)
                if matched:
                    matched_count += 1
            print(f"Batch matching complete. Matched {matched_count} engagements.")
            # --- End batch matching ---

            # Prepare data for training using the DataPreprocessor class
            # Since content.csv is no longer used for setup, pass an empty DataFrame for content_data
            # DataPreprocessor expects student, engagement, and content dataframes
            # Pass None or an empty DataFrame for content_data if required, but content is not used
            import pandas as pd
            empty_content_df = pd.DataFrame()
            data_preprocessor = DataPreprocessor(
                student_data=dataframes['students'],
                engagement_data=dataframes['engagements'],
                content_data=empty_content_df,
                db=session
            )
            prepared_data = data_preprocessor.prepare_data()

            # Train and save model using the ModelTrainer class
            model_dir_path = args.output_dir
            if model_dir_path is None:
                # Set default model save directory relative to the project root
                model_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "saved_models")

            # Ensure the model directory exists
            os.makedirs(model_dir_path, exist_ok=True)

            # Train and save model
            train_and_save_model(
                prepared_data,
                model_dir_path,
                epochs=args.epochs,
                batch_size=args.batch_size,
                embedding_dimension=args.embedding_dim
            )

            print("\n" + "=" * 80)
            print("Data ingestion and model training completed successfully!")
            print("=" * 80)
        finally:
            session.close()
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"Error during data ingestion and model training: {str(e)}")
        # Log the full traceback for better debugging
        import traceback
        logger.error("Full traceback of the error:")
        traceback.print_exc()
        print("=" * 80)
        sys.exit(1)

if __name__ == "__main__":
    main()