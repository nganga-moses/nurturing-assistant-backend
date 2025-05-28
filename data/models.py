from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey, JSON, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime, timedelta
import json
import os
import logging

# Create the SQLAlchemy base
Base = declarative_base()


class StudentProfile(Base):
    __tablename__ = "student_profiles"

    student_id = Column(String, primary_key=True)
    demographic_features = Column(JSON)
    application_status = Column(String)
    funnel_stage = Column(String)
    first_interaction_date = Column(DateTime)
    last_interaction_date = Column(DateTime)
    interaction_count = Column(Integer)
    application_likelihood_score = Column(Float)
    dropout_risk_score = Column(Float)
    last_recommended_engagement_id = Column(String, nullable=True)
    last_recommended_engagement_date = Column(DateTime, nullable=True)
    
    # Relationships
    engagements = relationship("EngagementHistory", back_populates="student")

    def to_dict(self):
        return {
            "student_id": self.student_id,
            "demographic_features": self.demographic_features,
            "application_status": self.application_status,
            "funnel_stage": self.funnel_stage,
            "first_interaction_date": self.first_interaction_date.isoformat() if self.first_interaction_date else None,
            "last_interaction_date": self.last_interaction_date.isoformat() if self.last_interaction_date else None,
            "interaction_count": self.interaction_count,
            "application_likelihood_score": self.application_likelihood_score,
            "dropout_risk_score": self.dropout_risk_score,
            "last_recommended_engagement_id": self.last_recommended_engagement_id,
            "last_recommended_engagement_date": self.last_recommended_engagement_date.isoformat() if self.last_recommended_engagement_date else None
        }


class EngagementHistory(Base):
    __tablename__ = "engagement_history"

    engagement_id = Column(String, primary_key=True)
    student_id = Column(String, ForeignKey("student_profiles.student_id"))
    engagement_type = Column(String)
    engagement_content_id = Column(String, ForeignKey("engagement_content.content_id"))
    timestamp = Column(DateTime)
    engagement_response = Column(String)
    engagement_metrics = Column(JSON)
    funnel_stage_before = Column(String)
    funnel_stage_after = Column(String)
    
    # Relationships
    student = relationship("StudentProfile", back_populates="engagements")
    content = relationship("EngagementContent", back_populates="engagements")

    def to_dict(self):
        return {
            "engagement_id": self.engagement_id,
            "student_id": self.student_id,
            "engagement_type": self.engagement_type,
            "engagement_content_id": self.engagement_content_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "engagement_response": self.engagement_response,
            "engagement_metrics": self.engagement_metrics,
            "funnel_stage_before": self.funnel_stage_before,
            "funnel_stage_after": self.funnel_stage_after
        }


class EngagementContent(Base):
    __tablename__ = "engagement_content"

    content_id = Column(String, primary_key=True)
    engagement_type = Column(String)
    content_category = Column(String)
    content_description = Column(String)
    content_features = Column(JSON)
    success_rate = Column(Float)
    target_funnel_stage = Column(String)
    appropriate_for_risk_level = Column(String)
    
    # Relationships
    engagements = relationship("EngagementHistory", back_populates="content")

    def to_dict(self):
        return {
            "content_id": self.content_id,
            "engagement_type": self.engagement_type,
            "content_category": self.content_category,
            "content_description": self.content_description,
            "content_features": self.content_features,
            "success_rate": self.success_rate,
            "target_funnel_stage": self.target_funnel_stage,
            "appropriate_for_risk_level": self.appropriate_for_risk_level
        }


class StatusChange(Base):
    __tablename__ = "status_changes"

    id = Column(Integer, primary_key=True)
    student_id = Column(String, ForeignKey("student_profiles.student_id"))
    field = Column(String)  # e.g., 'funnel_stage', 'application_status'
    old_value = Column(String)
    new_value = Column(String)
    batch_id = Column(String)  # ID of the batch update that caused this change
    timestamp = Column(DateTime, default=datetime.now)
    
    # Relationships
    student = relationship("StudentProfile")

    def to_dict(self):
        return {
            "id": self.id,
            "student_id": self.student_id,
            "field": self.field,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "batch_id": self.batch_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }


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
        Base.metadata.create_all(get_engine())
        
        # Get session
        session = get_session()
        
        # Only insert sample data if GENERATE_SAMPLE_DATA is set to 'true'
        if os.environ.get("GENERATE_SAMPLE_DATA", "false").lower() != "true":
            return
        
        # Check if we already have data
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
