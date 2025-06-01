import pytest
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from data.models import (
    StudentProfile,
    EngagementHistory,
    StoredRecommendation,
    NudgeAction,
    NudgeFeedbackMetrics
)
from api.services.matching_service import MatchingService

@pytest.fixture
def sample_student(db_session: Session):
    student = StudentProfile(
        student_id="TEST001",
        demographic_features={"name": "Test Student"},
        application_status="in_progress",
        funnel_stage="consideration",
        first_interaction_date=datetime.now() - timedelta(days=30),
        last_interaction_date=datetime.now(),
        interaction_count=1,
        application_likelihood_score=0.8,
        dropout_risk_score=0.3
    )
    db_session.add(student)
    db_session.commit()
    return student

@pytest.fixture
def sample_nudge(db_session: Session, sample_student):
    nudge = StoredRecommendation(
        student_id=sample_student.student_id,
        recommendations=[{
            "type": "email",
            "category": "welcome",
            "content": "Welcome to our program!"
        }],
        generated_at=datetime.now(),
        expires_at=datetime.now() + timedelta(days=7)
    )
    db_session.add(nudge)
    db_session.commit()
    return nudge

@pytest.fixture
def sample_engagement(db_session: Session, sample_student):
    engagement = EngagementHistory(
        engagement_id="TEST_ENG_001",
        student_id=sample_student.student_id,
        engagement_type="email",
        engagement_content_id="CONTENT_001",
        timestamp=datetime.now(),
        engagement_response="viewed",
        engagement_metrics={"duration": 30}
    )
    db_session.add(engagement)
    db_session.commit()
    return engagement

def test_match_engagement_to_nudge(db_session: Session, sample_student, sample_nudge, sample_engagement):
    matching_service = MatchingService(db_session)
    matched, confidence = matching_service.match_engagement_to_nudge(sample_engagement)
    
    assert matched is True
    assert 0.0 <= confidence <= 1.0
    
    # Check that a nudge action was created with confidence score
    action = db_session.query(NudgeAction).filter(
        NudgeAction.student_id == sample_student.student_id,
        NudgeAction.nudge_id == sample_nudge.id
    ).first()
    
    assert action is not None
    assert action.action_type == "acted"
    assert action.action_completed is True
    assert action.confidence_score == confidence

def test_fuzzy_matching(db_session: Session, sample_student, sample_nudge):
    matching_service = MatchingService(db_session)
    
    # Test with slightly different engagement type
    engagement = EngagementHistory(
        engagement_id="TEST_ENG_002",
        student_id=sample_student.student_id,
        engagement_type="e-mail",  # Slightly different from "email"
        engagement_content_id="CONTENT_001",
        timestamp=datetime.now(),
        engagement_response="viewed",
        engagement_metrics={"duration": 30}
    )
    db_session.add(engagement)
    db_session.commit()
    
    matched, confidence = matching_service.match_engagement_to_nudge(engagement)
    assert matched is True
    assert confidence > 0.8  # Should have high confidence despite slight difference

def test_get_unmatched_engagements(db_session: Session, sample_engagement):
    matching_service = MatchingService(db_session)
    unmatched = matching_service.get_unmatched_engagements()
    
    assert len(unmatched) == 1
    assert unmatched[0].engagement_id == sample_engagement.engagement_id

def test_get_feedback_metrics(db_session: Session, sample_student, sample_nudge, sample_engagement):
    matching_service = MatchingService(db_session)
    
    # First match an engagement to create metrics
    matched, confidence = matching_service.match_engagement_to_nudge(sample_engagement)
    
    # Get metrics
    metrics = matching_service.get_feedback_metrics("email")
    
    assert len(metrics) == 1
    assert metrics[0].nudge_type == "email"
    assert metrics[0].total_shown == 1
    assert metrics[0].acted_count == 1
    assert metrics[0].completion_rate == 1.0
    assert metrics[0].average_confidence == confidence

def test_get_match_quality_stats(db_session: Session, sample_student, sample_nudge, sample_engagement):
    matching_service = MatchingService(db_session)
    
    # First match an engagement to create metrics
    matched, confidence = matching_service.match_engagement_to_nudge(sample_engagement)
    
    # Get quality stats
    stats = matching_service.get_match_quality_stats("email")
    
    assert "average_confidence" in stats
    assert "match_rate" in stats
    assert "high_confidence_rate" in stats
    assert stats["average_confidence"] == confidence
    assert stats["match_rate"] == 1.0
    assert stats["high_confidence_rate"] == 1.0 if confidence > 0.8 else 0.0

def test_time_based_confidence(db_session: Session, sample_student, sample_nudge):
    matching_service = MatchingService(db_session)
    
    # Create engagement with different timestamps
    timestamps = [
        datetime.now(),  # Current time
        datetime.now() - timedelta(hours=12),  # 12 hours ago
        datetime.now() - timedelta(hours=23),  # 23 hours ago
        datetime.now() - timedelta(hours=25)   # 25 hours ago (should not match)
    ]
    
    confidences = []
    for i, timestamp in enumerate(timestamps):
        engagement = EngagementHistory(
            engagement_id=f"TEST_ENG_{i}",
            student_id=sample_student.student_id,
            engagement_type="email",
            engagement_content_id="CONTENT_001",
            timestamp=timestamp,
            engagement_response="viewed",
            engagement_metrics={"duration": 30}
        )
        db_session.add(engagement)
        db_session.commit()
        
        matched, confidence = matching_service.match_engagement_to_nudge(engagement)
        if matched:
            confidences.append(confidence)
    
    # Should have 3 matches (excluding the 25-hour old one)
    assert len(confidences) == 3
    
    # Confidence should decrease with time
    assert confidences[0] >= confidences[1] >= confidences[2] 