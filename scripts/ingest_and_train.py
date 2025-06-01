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
    # Basic usage with all required files
    python ingest_and_train.py \
        --students-csv data/students.csv \
        --engagements-csv data/engagements.csv \
        --content-csv data/content.csv

    # With custom model parameters
    python ingest_and_train.py \
        --students-csv data/students.csv \
        --engagements-csv data/engagements.csv \
        --content-csv data/content.csv \
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

# Add the parent directory to the path so we can import from other modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.models.models import (
    StudentProfile, 
    EngagementHistory, 
    EngagementContent, 
    get_session, 
    init_db
)
from models.train_model import train_and_save_model, prepare_data_for_training
from models.simple_recommender import SimpleRecommender
from api.services.matching_service import MatchingService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Ingest CSV data and train the recommendation model")
    parser.add_argument("--students-csv", type=str, required=True, help="Path to students CSV file")
    parser.add_argument("--engagements-csv", type=str, required=True, help="Path to engagements CSV file")
    parser.add_argument("--content-csv", type=str, required=True, help="Path to content CSV file")
    parser.add_argument("--output-dir", type=str, default=None, 
                        help="Directory to save the model (default: models/saved_models)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    return parser.parse_args()

def load_data_from_csv(
    students_csv: str,
    engagements_csv: str,
    content_csv: str
) -> Dict[str, pd.DataFrame]:
    """
    Load initial data from CSV files.
    
    This function:
    1. Reads all required CSV files
    2. Validates data structure
    3. Handles missing values
    4. Converts data types
    5. Returns a dictionary of DataFrames
    
    Args:
        students_csv: Path to students CSV file
        engagements_csv: Path to engagements CSV file
        content_csv: Path to content CSV file
        
    Returns:
        Dictionary containing:
        - students: DataFrame of student profiles
        - engagements: DataFrame of engagement history
        - content: DataFrame of engagement content
        
    Raises:
        FileNotFoundError: If any CSV file is missing
        ValueError: If data validation fails
    """
    logger.info("Loading CSV data...")
    
    try:
        students_df = pd.read_csv(students_csv)
        print("Students CSV shape:", students_df.shape)
        
        engagements_df = pd.read_csv(engagements_csv)
        print("Engagements CSV shape:", engagements_df.shape)
        
        content_df = pd.read_csv(content_csv)
        print("Content CSV shape:", content_df.shape)
        
        logger.info(f"Loaded {len(students_df)} students, {len(engagements_df)} engagements, and {len(content_df)} content items")
        
        return {
            'students': students_df,
            'engagements': engagements_df,
            'content': content_df
        }
    except Exception as e:
        logger.error(f"Error loading CSV data: {str(e)}")
        raise

def initialize_database(session: Session) -> None:
    """
    Initialize the database schema.
    
    This function:
    1. Creates all required tables
    2. Sets up indexes
    3. Configures constraints
    4. Handles existing tables
    
    Args:
        session: SQLAlchemy session
        
    Raises:
        SQLAlchemyError: If database initialization fails
    """
    logger.info("Initializing database...")
    
    try:
        # Initialize database
        init_db()
        
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise

def store_data_in_database(
    session: Session,
    dataframes: Dict[str, pd.DataFrame]
) -> None:
    """
    Store data in the database.
    
    This function:
    1. Validates data before storage
    2. Handles data insertion
    3. Manages transactions
    4. Updates related records
    
    Args:
        session: SQLAlchemy session
        dataframes: Dictionary containing:
            - students: DataFrame of student profiles
            - engagements: DataFrame of engagement history
            - content: DataFrame of engagement content
            
    Raises:
        ValueError: If data is invalid
        SQLAlchemyError: If database operations fail
    """
    logger.info("Ingesting data into database...")
    
    try:
        # Ingest students
        for _, row in dataframes['students'].iterrows():
            student = StudentProfile(
                student_id=row['student_id'],
                demographic_features=row['demographic_features'] if isinstance(row['demographic_features'], dict) else eval(row['demographic_features']),
                application_status=row['application_status'],
                funnel_stage=row['funnel_stage'],
                first_interaction_date=pd.to_datetime(row['first_interaction_date']),
                last_interaction_date=pd.to_datetime(row['last_interaction_date']),
                interaction_count=row['interaction_count'],
                application_likelihood_score=row['application_likelihood_score'],
                dropout_risk_score=row['dropout_risk_score']
            )
            session.add(student)
        
        # Ingest content
        for _, content in dataframes['content'].iterrows():
            content_obj = EngagementContent(
                content_id=content['content_id'],
                engagement_type=content['content_type'],
                content_category="general",
                content_description=content['content_description'],
                content_features=content['engagement_metrics'],
                success_rate=content['success_rate'],
                target_funnel_stage=content['target_funnel_stage'],
                appropriate_for_risk_level="all"
            )
            session.add(content_obj)
        
        # Ingest engagements
        logger.info("Engagements DataFrame columns: %s", dataframes['engagements'].columns.tolist())
        for _, engagement in dataframes['engagements'].iterrows():
            engagement_obj = EngagementHistory(
                engagement_id=engagement['engagement_id'],
                student_id=engagement['student_id'],
                engagement_type=engagement['engagement_type'],
                engagement_content_id=engagement['engagement_content_id'],
                timestamp=pd.to_datetime(engagement['timestamp']),
                engagement_response=engagement['engagement_response'],
                engagement_metrics=engagement['engagement_metrics'] if isinstance(engagement['engagement_metrics'], dict) else eval(engagement['engagement_metrics']),
                funnel_stage_before=engagement['funnel_stage_before'],
                funnel_stage_after=engagement['funnel_stage_after']
            )
            session.add(engagement_obj)
        
        session.commit()
        logger.info("Data ingestion completed successfully")
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error ingesting data: {str(e)}")
        raise

def main():
    """
    Main function to run the initial data ingestion and training script.
    
    This function:
    1. Parses command line arguments
    2. Sets up logging
    3. Loads data from CSV files
    4. Initializes database
    5. Stores data
    6. Trains initial models
    
    Command line arguments:
        --students-csv: Path to students CSV file
        --engagements-csv: Path to engagements CSV file
        --content-csv: Path to content CSV file
        --epochs: Number of training epochs
        --batch-size: Batch size for training
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
        session = get_session()
        try:
            # Load data from CSV
            dataframes = load_data_from_csv(args.students_csv, args.engagements_csv, args.content_csv)
            # Initialize database
            initialize_database(session)
            # Store data in database
            store_data_in_database(session, dataframes)
            # --- Post-import batch matching step ---
            print("Running post-import batch matching for all engagements...")
            matching_service = MatchingService(session)
            all_engagements = session.query(EngagementHistory).all()
            matched_count = 0
            for engagement in all_engagements:
                matched, confidence = matching_service.match_engagement_to_nudge(engagement)
                if matched:
                    matched_count += 1
            print(f"Batch matching complete. Matched {matched_count} engagements.")
            # --- End batch matching ---
            # Train and save model
            model_dir = args.output_dir
            if model_dir is None:
                model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "saved_models")
            os.makedirs(model_dir, exist_ok=True)
            prepared_data = prepare_data_for_training(dataframes)
            train_and_save_model(prepared_data, model_dir=model_dir, epochs=args.epochs)
            print("\n" + "=" * 80)
            print("Data ingestion and model training completed successfully!")
            print("=" * 80)
        finally:
            session.close()
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"Error during data ingestion and model training: {str(e)}")
        print("=" * 80)
        sys.exit(1)

if __name__ == "__main__":
    main() 