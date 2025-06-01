#!/usr/bin/env python
"""
Script to handle incremental updates to student data.
This script can be run periodically to:
1. Ingest new students from CSV or database
2. Update existing student data (demographics, funnel stage, risk scores, etc.)
3. Trigger model retraining if significant changes are detected
"""

import os
import sys
import argparse
import pandas as pd
from datetime import datetime, timedelta
import logging
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import Dict, List, Tuple

# Add the parent directory to the path so we can import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.models.models import (
    StudentProfile, 
    EngagementHistory, 
    EngagementContent, 
    get_session, 
    init_db
)
from models.recommendation_service import RecommendationService
from models.simple_recommender import SimpleRecommender
from utils.status_tracker import StatusTracker
from api.services.matching_service import MatchingService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Handle incremental student data updates")
    parser.add_argument("--students-csv", type=str, help="Path to new/updated students CSV file")
    parser.add_argument("--days", type=int, default=7, 
                       help="Number of days to look back for updated students in database")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Directory to save the updated model (default: models/saved_models)")
    parser.add_argument("--retrain-threshold", type=int, default=100,
                       help="Number of new/updated students that triggers model retraining")
    return parser.parse_args()

def load_csv_students(csv_path: str) -> pd.DataFrame:
    """
    Load student data from CSV file.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame containing student data
    """
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Error loading CSV data: {str(e)}")
        raise

def get_updated_students_from_db(session: Session, days: int) -> pd.DataFrame:
    """
    Get recently updated students from the database.
    
    Args:
        session: Database session
        days: Number of days to look back
        
    Returns:
        DataFrame containing updated students
    """
    cutoff_date = datetime.now() - timedelta(days=days)
    
    # Query recently updated students
    students = session.query(StudentProfile).filter(
        StudentProfile.last_interaction_date >= cutoff_date
    ).order_by(desc(StudentProfile.last_interaction_date)).all()
    
    # Convert to DataFrame
    if students:
        return pd.DataFrame([{
            'student_id': s.student_id,
            'demographic_features': s.demographic_features,
            'application_status': s.application_status,
            'funnel_stage': s.funnel_stage,
            'first_interaction_date': s.first_interaction_date,
            'last_interaction_date': s.last_interaction_date,
            'interaction_count': s.interaction_count,
            'application_likelihood_score': s.application_likelihood_score,
            'dropout_risk_score': s.dropout_risk_score
        } for s in students])
    return pd.DataFrame()

def calculate_risk_score(student: StudentProfile, engagements: List[EngagementHistory]) -> float:
    """
    Calculate dropout risk score for a student based on:
    1. Engagement patterns and frequency
    2. Funnel stage progression
    3. Content interaction quality
    4. Time-based factors
    5. Response patterns
    
    Args:
        student: StudentProfile object
        engagements: List of student's engagements
        
    Returns:
        float: Risk score between 0 and 1
    """
    if not engagements:
        return 0.5  # Default risk score
        
    # 1. Engagement Pattern Analysis
    engagement_scores = []
    for engagement in engagements:
        # Base score from engagement type
        type_scores = {
            'view': 0.3,
            'click': 0.4,
            'form_submit': 0.6,
            'application_start': 0.7,
            'application_complete': 0.8
        }
        base_score = type_scores.get(engagement.engagement_type, 0.3)
        
        # Adjust based on response
        if engagement.engagement_response:
            base_score *= 1.2
            
        engagement_scores.append(base_score)
    
    # 2. Time-based Analysis
    last_engagement = max(engagements, key=lambda x: x.timestamp)
    days_since_last = (datetime.now() - last_engagement.timestamp).days
    time_score = max(0, 1 - (days_since_last / 30))  # Decay over 30 days
    
    # 3. Funnel Stage Analysis
    funnel_stages = ['awareness', 'interest', 'consideration', 'decision', 'application']
    current_stage = funnel_stages.index(student.funnel_stage)
    stage_score = current_stage / len(funnel_stages)
    
    # Calculate final risk score
    engagement_score = sum(engagement_scores) / len(engagement_scores) if engagement_scores else 0
    risk_score = (engagement_score * 0.4 + time_score * 0.3 + stage_score * 0.3)
    
    return min(max(risk_score, 0), 1)  # Ensure score is between 0 and 1

def calculate_application_likelihood(student: StudentProfile, engagements: List[EngagementHistory]) -> float:
    """
    Calculate application likelihood score for a student based on:
    1. Engagement patterns and frequency
    2. Funnel stage progression
    3. Content interaction quality
    4. Time-based factors
    5. Response patterns
    
    Args:
        student: StudentProfile object
        engagements: List of student's engagements
        
    Returns:
        float: Likelihood score between 0 and 1
    """
    if not engagements:
        return 0.0
        
    # 1. Engagement Pattern Analysis
    engagement_scores = []
    for engagement in engagements:
        # Base score from engagement type
        type_scores = {
            'view': 0.2,
            'click': 0.3,
            'form_submit': 0.5,
            'application_start': 0.7,
            'application_complete': 0.9
        }
        base_score = type_scores.get(engagement.engagement_type, 0.2)
        
        # Adjust based on response
        if engagement.engagement_response:
            base_score *= 1.2
            
        engagement_scores.append(base_score)
    
    # 2. Time-based Analysis
    last_engagement = max(engagements, key=lambda x: x.timestamp)
    days_since_last = (datetime.now() - last_engagement.timestamp).days
    time_score = max(0, 1 - (days_since_last / 30))  # Decay over 30 days
    
    # 3. Funnel Stage Analysis
    funnel_stages = ['awareness', 'interest', 'consideration', 'decision', 'application']
    current_stage = funnel_stages.index(student.funnel_stage)
    stage_score = current_stage / len(funnel_stages)
    
    # Calculate final likelihood score
    engagement_score = sum(engagement_scores) / len(engagement_scores) if engagement_scores else 0
    likelihood_score = (engagement_score * 0.4 + time_score * 0.3 + stage_score * 0.3)
    
    return min(max(likelihood_score, 0), 1)  # Ensure score is between 0 and 1

def update_student_profiles(session: Session, students_df: pd.DataFrame) -> None:
    """
    Update student profiles with new data and calculated scores.
    
    Args:
        session: Database session
        students_df: DataFrame containing student data
    """
    status_tracker = BatchStatusTracker()
    
    for _, row in students_df.iterrows():
        student = session.query(StudentProfile).filter_by(student_id=row['student_id']).first()
        if student:
            # Update basic information
            student.funnel_stage = row['funnel_stage']
            student.last_interaction_date = pd.to_datetime(row['last_interaction_date'])
            student.interaction_count = row['interaction_count']
            
            # Get student's engagements
            engagements = session.query(EngagementHistory).filter_by(
                student_id=student.student_id
            ).all()
            
            # Calculate and update scores
            student.dropout_risk_score = calculate_risk_score(student, engagements)
            student.application_likelihood_score = calculate_application_likelihood(student, engagements)
            
            # Track status changes
            status_tracker.track_student_status(student)
    
    session.commit()

def update_student_data(session: Session, students_df: pd.DataFrame) -> Tuple[int, int]:
    """
    Update student data in the database.
    
    Args:
        session: Database session
        students_df: DataFrame containing student data
        
    Returns:
        Tuple of (new_students_count, updated_students_count)
    """
    new_students = 0
    updated_students = 0
    
    for _, student in students_df.iterrows():
        existing_student = session.query(StudentProfile).filter_by(
            student_id=student['student_id']
        ).first()
        
        if existing_student:
            # Update existing student
            existing_student.demographic_features = student['demographic_features']
            existing_student.application_status = student['application_status']
            existing_student.funnel_stage = student['funnel_stage']
            existing_student.last_interaction_date = student['last_interaction_date']
            existing_student.interaction_count = student['interaction_count']
            existing_student.application_likelihood_score = student['application_likelihood_score']
            existing_student.dropout_risk_score = student['dropout_risk_score']
            updated_students += 1
        else:
            # Create new student
            new_student = StudentProfile(
                student_id=student['student_id'],
                demographic_features=student['demographic_features'],
                application_status=student['application_status'],
                funnel_stage=student['funnel_stage'],
                first_interaction_date=student['first_interaction_date'],
                last_interaction_date=student['last_interaction_date'],
                interaction_count=student['interaction_count'],
                application_likelihood_score=student['application_likelihood_score'],
                dropout_risk_score=student['dropout_risk_score']
            )
            session.add(new_student)
            new_students += 1
    
    session.commit()
    return new_students, updated_students

def should_retrain_model(new_students: int, updated_students: int, threshold: int) -> bool:
    """
    Determine if model should be retrained based on number of changes.
    
    Args:
        new_students: Number of new students
        updated_students: Number of updated students
        threshold: Threshold for triggering retraining
        
    Returns:
        Boolean indicating if model should be retrained
    """
    total_changes = new_students + updated_students
    return total_changes >= threshold

def main():
    # Ensure PostgreSQL is used by default
    if "DATABASE_URL" not in os.environ:
        os.environ["DATABASE_URL"] = "postgresql://postgres:postgres@localhost:5432/student_engagement"
    args = parse_args()
    print("=" * 80)
    print(f"Starting student data update at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    try:
        # Initialize database
        init_db()
        # Get database session
        session = get_session()
        try:
            # Get student data
            if args.students_csv:
                students_df = load_csv_students(args.students_csv)
            else:
                students_df = get_updated_students_from_db(session, args.days)
            if students_df.empty:
                print("No student updates found")
                return
            print(f"Found {len(students_df)} students to process")
            # Update student data
            new_students, updated_students = update_student_data(session, students_df)
            print(f"Processed {new_students} new students and {updated_students} updated students")
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
            # Check if model should be retrained
            if should_retrain_model(new_students, updated_students, args.retrain_threshold):
                print("Significant changes detected. Retraining model...")
                # Get all data for model update
                students_df = pd.read_sql(session.query(StudentProfile).statement, session.bind)
                content_df = pd.read_sql(session.query(EngagementContent).statement, session.bind)
                engagements_df = pd.read_sql(session.query(EngagementHistory).statement, session.bind)
                model_dir = args.output_dir
                if model_dir is None:
                    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "saved_models")
                os.makedirs(model_dir, exist_ok=True)
                recommender = SimpleRecommender(model_dir=model_dir)
                recommender.train(students_df, content_df, engagements_df)
                print("Model retraining completed")
            else:
                print("No model retraining needed")
            print("\n" + "=" * 80)
            print("Student data update completed successfully!")
            print("=" * 80)
        finally:
            session.close()
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"Error during student data update: {str(e)}")
        print("=" * 80)
        sys.exit(1)

if __name__ == "__main__":
    main() 