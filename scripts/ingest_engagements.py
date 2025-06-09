#!/usr/bin/env python
"""
Script for incremental engagement data updates.

This script is meant to be run PERIODICALLY (e.g., daily) to:
1. Get new engagements from CSV or database (last N days)
2. Update student states based on new engagements
3. Retrain/update the recommendation model if needed

The script supports:
- Loading new engagements from CSV files
- Loading recent engagements from database
- Updating student states and engagement history
- Triggering model retraining when needed
- Logging all changes and updates
- CRM data validation and status tracking

Usage:
    # Update from CSV file
    python ingest_engagements.py --engagements-csv data/new_engagements.csv

    # Update from database (last 7 days)
    python ingest_engagements.py --days 7

    # Update and force model retraining
    python ingest_engagements.py --days 7 --force-retrain

For initial setup, use:
- ingest_and_train.py: For initial data ingestion and model training

For student data updates, use:
- ingest_students.py: For new/updated student data
"""

import os
import sys
import argparse
import pandas as pd
from datetime import datetime, timedelta
import logging
from sqlalchemy.orm import Session
from sqlalchemy import desc

# Add the parent directory to the path so we can import from other modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.models.student_profile import StudentProfile
from data.models.engagement_history import EngagementHistory
from data.models.stored_recommendation import StoredRecommendation
from data.models.engagement_content import EngagementContent
from data.models.get_session import get_session
from models.recommendation_service import RecommendationService
from models.simple_recommender import SimpleRecommender
from utils.status_tracker import StatusTracker
from batch_processing.status_tracker import BatchStatusTracker
from api.services.matching_service import MatchingService
from database.session import get_db
from data.models.funnel_stage import FunnelStage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Ingest new engagements and update the recommendation model")
    parser.add_argument("--engagements-csv", type=str, help="Path to new engagements CSV file")
    parser.add_argument("--days", type=int, default=7, help="Number of days to look back for new engagements")
    parser.add_argument("--output-dir", type=str, default=None, 
                        help="Directory to save the updated model (default: models/saved_models)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--force-retrain", action="store_true", help="Force model retraining")
    return parser.parse_args()

def get_new_engagements_from_db(session: Session, days: int) -> pd.DataFrame:
    """
    Get new engagements from the database.
    
    Args:
        session: Database session
        days: Number of days to look back
        
    Returns:
        DataFrame containing new engagements
    """
    cutoff_date = datetime.now() - timedelta(days=days)
    
    # Query new engagements
    engagements = session.query(EngagementHistory).filter(
        EngagementHistory.timestamp >= cutoff_date
    ).order_by(desc(EngagementHistory.timestamp)).all()
    
    # Convert to DataFrame
    if engagements:
        return pd.DataFrame([{
            'engagement_id': e.engagement_id,
            'student_id': e.student_id,
            'engagement_type': e.engagement_type,
            'engagement_content_id': e.engagement_content_id,
            'timestamp': e.timestamp,
            'engagement_response': e.engagement_response,
            'engagement_metrics': e.engagement_metrics,
            'funnel_stage_before': e.funnel_stage_before,
            'funnel_stage_after': e.funnel_stage_after
        } for e in engagements])
    return pd.DataFrame()

def validate_crm_data(df: pd.DataFrame) -> bool:
    """
    Validate the CRM data before processing
    
    Args:
        df: DataFrame containing CRM data
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    required_columns = [
        'student_id',
        'engagement_type',
        'engagement_content_id',
        'timestamp',
        'engagement_response',
        'engagement_metrics',
        'funnel_stage_before',
        'funnel_stage_after'
    ]
    
    # Check required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
        
    # Check data types
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except Exception as e:
        logger.error(f"Error converting timestamp: {e}")
        return False
        
    return True

def load_csv_engagements(csv_path: str) -> pd.DataFrame:
    """
    Load engagements from CSV file.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame containing engagements
    """
    try:
        df = pd.read_csv(csv_path)
        if not validate_crm_data(df):
            raise ValueError("CRM data validation failed")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV data: {str(e)}")
        raise

def update_student_states(session: Session, engagements_df: pd.DataFrame) -> None:
    """
    Update student states based on new engagements.
    
    Args:
        session: Database session
        engagements_df: DataFrame containing new engagements
    """
    status_tracker = BatchStatusTracker()
    
    for _, engagement in engagements_df.iterrows():
        student = session.query(StudentProfile).filter_by(student_id=engagement['student_id']).first()
        if student:
            # Update last interaction date
            student.last_interaction_date = engagement['timestamp']
            
            # Update interaction count
            student.interaction_count += 1
            
            # Update funnel stage if changed
            if engagement['funnel_stage_after'] != engagement['funnel_stage_before']:
                student.funnel_stage = engagement['funnel_stage_after']
                # Get the corresponding FunnelStage record
                stage = session.query(FunnelStage).filter_by(stage_name=engagement['funnel_stage_after']).first()
                if stage:
                    student.current_stage_id = stage.id
                
            # Track status changes
            status_tracker.track_student_status(student)
    
    session.commit()

def should_retrain_model(session: Session, days: int = 7) -> bool:
    """
    Determine if the model should be retrained.
    
    This function:
    1. Checks number of new engagements
    2. Checks engagement diversity
    3. Checks time since last training
    4. Returns decision
    
    Args:
        session: SQLAlchemy session
        days: Number of days to consider
        
    Returns:
        Boolean indicating if retraining is needed
    """
    # Implementation of the function
    pass

def main():
    """
    Main function to run the engagement ingestion script.
    
    This function:
    1. Parses command line arguments
    2. Sets up logging
    3. Loads new engagements
    4. Updates student states
    5. Triggers model retraining if needed
    
    Command line arguments:
        --engagements-csv: Path to CSV file with new engagements
        --days: Number of days to look back in database
        --force-retrain: Force model retraining
    """
    # Ensure PostgreSQL is used by default
    if "DATABASE_URL" not in os.environ:
        os.environ["DATABASE_URL"] = "postgresql://postgres:postgres@localhost:5432/student_engagement"
    args = parse_args()
    print("=" * 80)
    print(f"Starting engagement ingestion and model update at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    try:
        # Initialize database
        # This step is handled by Alembic migrations, so no explicit init_db() call is needed here.
        # Get database session
        session = get_db()
        try:
            # Get new engagements
            if args.engagements_csv:
                engagements_df = load_csv_engagements(args.engagements_csv)
            else:
                engagements_df = get_new_engagements_from_db(session, args.days)
            if engagements_df.empty:
                print("No new engagements found")
                return
            print(f"Found {len(engagements_df)} new engagements")
            # Update student states
            update_student_states(session, engagements_df)
            # --- Post-import batch matching step ---
            print("Running post-import batch matching for all engagements...")
            matching_service = MatchingService(session)
            all_engagements = session.query(EngagementHistory).all()
            matched_count = 0
            for engagement in all_engagements:
                matched, confidence = matching_service.match_engagement_to_recommendation(engagement)
                if matched:
                    matched_count += 1
            print(f"Batch matching complete. Matched {matched_count} engagements.")
            # --- End batch matching ---
            # Get all data for model update
            students_df = pd.read_sql(session.query(StudentProfile).statement, session.bind)
            content_df = pd.read_sql(session.query(EngagementContent).statement, session.bind)
            all_engagements_df = pd.read_sql(session.query(EngagementHistory).statement, session.bind)
            model_dir = args.output_dir
            if model_dir is None:
                model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "saved_models", "content_based")
            os.makedirs(model_dir, exist_ok=True)
            recommender = SimpleRecommender(model_dir=model_dir)
            recommender.train(students_df, content_df, all_engagements_df)
            print("\n" + "=" * 80)
            print("Engagement ingestion and model update completed successfully!")
            print("=" * 80)
        finally:
            session.close()
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"Error during engagement ingestion and model update: {str(e)}")
        print("=" * 80)
        sys.exit(1)

if __name__ == "__main__":
    main() 