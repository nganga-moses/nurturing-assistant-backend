#!/usr/bin/env python
"""
Script to ingest new student engagements and update the recommendation model.
This script can be run periodically to update the model with new engagement data.
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

from data.models import (
    StudentProfile, 
    EngagementHistory, 
    EngagementContent, 
    get_session, 
    init_db
)
from models.recommendation_service import RecommendationService
from models.simple_recommender import SimpleRecommender

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

def load_csv_engagements(csv_path: str) -> pd.DataFrame:
    """
    Load engagements from CSV file.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame containing engagements
    """
    try:
        return pd.read_csv(csv_path)
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
    
    session.commit()

def main():
    args = parse_args()
    
    print("=" * 80)
    print(f"Starting engagement ingestion and model update at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    try:
        # Initialize database
        init_db()
        
        # Get database session
        session = get_session()
        
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
            
            # Get all data for model update
            students_df = pd.read_sql(session.query(StudentProfile).statement, session.bind)
            content_df = pd.read_sql(session.query(EngagementContent).statement, session.bind)
            all_engagements_df = pd.read_sql(session.query(EngagementHistory).statement, session.bind)
            
            # Initialize recommendation service
            model_dir = args.output_dir
            if model_dir is None:
                model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "saved_models")
            
            # Ensure model directory exists
            os.makedirs(model_dir, exist_ok=True)
            
            # Update the model
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