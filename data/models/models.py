# DEPRECATED: All models have been modularized into individual files in this directory.
# This file is retained for legacy reference and utility/sample data code only.

# Utility/sample data code may remain below if needed.

from sqlalchemy import create_engine, sessionmaker
from datetime import datetime, timedelta
import os
import logging

# Database connection and session management
def get_engine(db_url=None):
    if db_url is None:
        db_url = os.environ.get("DATABASE_URL", "sqlite:///./student_engagement.db")
    return create_engine(db_url)


def get_session():
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def init_db():
    """Initialize the database with tables and sample data."""
    try:
        # Create tables
        from .base import Base
        Base.metadata.create_all(get_engine())
        
        # Get session
        session = get_session()
        
        # Only insert sample data if GENERATE_SAMPLE_DATA is set to 'true'
        if os.environ.get("GENERATE_SAMPLE_DATA", "false").lower() != "true":
            return
        
        # Check if we already have data
        from .student_profile import StudentProfile
        if session.query(StudentProfile).count() > 0:
            logging.info("Database already contains data, skipping initialization")
            session.close()
            return
            
        # Add sample students
        sample_students = [
            StudentProfile(
                student_id="S001",
                demographic_features={
                    "first_name": "John",
                    "last_name": "Doe",
                    "age": 18,
                    "location": "New York, NY",
                    "high_school_gpa": 3.9,
                    "intended_major": "Computer Science"
                },
                application_status="in_progress",
                funnel_stage="consideration",
                first_interaction_date=datetime.now() - timedelta(days=30),
                last_interaction_date=datetime.now() - timedelta(days=2),
                interaction_count=10,
                application_likelihood_score=0.85,
                dropout_risk_score=0.4
            ),
            StudentProfile(
                student_id="S1001",
                demographic_features={
                    "first_name": "Emma",
                    "last_name": "Johnson",
                    "age": 18,
                    "location": "Seattle, WA",
                    "high_school_gpa": 3.8,
                    "intended_major": "Computer Science"
                },
                application_status="in_progress",
                funnel_stage="consideration",
                first_interaction_date=datetime.now() - timedelta(days=45),
                last_interaction_date=datetime.now() - timedelta(days=5),
                interaction_count=12,
                application_likelihood_score=0.92,
                dropout_risk_score=0.3
            ),
            StudentProfile(
                student_id="S1042",
                demographic_features={
                    "first_name": "Michael",
                    "last_name": "Chen",
                    "age": 17,
                    "location": "Portland, OR",
                    "high_school_gpa": 4.0,
                    "intended_major": "Engineering"
                },
                application_status="not_started",
                funnel_stage="interest",
                first_interaction_date=datetime.now() - timedelta(days=30),
                last_interaction_date=datetime.now() - timedelta(days=2),
                interaction_count=8,
                application_likelihood_score=0.88,
                dropout_risk_score=0.2
            ),
            StudentProfile(
                student_id="S1078",
                demographic_features={
                    "first_name": "Sophia",
                    "last_name": "Garcia",
                    "age": 18,
                    "location": "San Francisco, CA",
                    "high_school_gpa": 3.9,
                    "intended_major": "Biology"
                },
                application_status="in_progress",
                funnel_stage="decision",
                first_interaction_date=datetime.now() - timedelta(days=60),
                last_interaction_date=datetime.now() - timedelta(days=1),
                interaction_count=15,
                application_likelihood_score=0.95,
                dropout_risk_score=0.1
            ),
            StudentProfile(
                student_id="S1103",
                demographic_features={
                    "first_name": "David",
                    "last_name": "Wilson",
                    "age": 19,
                    "location": "Chicago, IL",
                    "high_school_gpa": 3.7,
                    "intended_major": "Business"
                },
                application_status="not_started",
                funnel_stage="interest",
                first_interaction_date=datetime.now() - timedelta(days=20),
                last_interaction_date=datetime.now() - timedelta(days=15),
                interaction_count=5,
                application_likelihood_score=0.82,
                dropout_risk_score=0.6
            ),
            StudentProfile(
                student_id="S1156",
                demographic_features={
                    "first_name": "Olivia",
                    "last_name": "Martinez",
                    "age": 17,
                    "location": "Austin, TX",
                    "high_school_gpa": 3.95,
                    "intended_major": "Psychology"
                },
                application_status="not_started",
                funnel_stage="awareness",
                first_interaction_date=datetime.now() - timedelta(days=10),
                last_interaction_date=datetime.now() - timedelta(days=10),
                interaction_count=2,
                application_likelihood_score=0.75,
                dropout_risk_score=0.8
            )
        ]
        
        # Add students to database
        session.add_all(sample_students)
        session.commit()
        
        # Add sample engagement history
        from .engagement_history import EngagementHistory
        for student in sample_students:
            # Add engagement history based on interaction count
            for i in range(student.interaction_count):
                engagement = EngagementHistory(
                    engagement_id=f"E_{student.student_id}_{i+1}",
                    student_id=student.student_id,
                    engagement_content_id=f"CONTENT_{i+1}",
                    engagement_type="view",
                    timestamp=student.last_interaction_date - timedelta(days=i*2)
                )
                session.add(engagement)
        
        session.commit()
        logging.info("Database initialized with sample data")
        
    except Exception as e:
        logging.error(f"Error initializing database: {str(e)}")
        raise
    finally:
        session.close()
