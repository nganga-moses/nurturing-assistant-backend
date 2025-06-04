import pytest
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from database.session import get_db
from data.models import (
    StudentProfile,
    EngagementHistory,
    StoredRecommendation,
    RecommendationAction,
    RecommendationFeedbackMetrics
)
from api.services import MatchingService

@pytest.fixture
def db_session():
    """Create a test database session."""
    session = next(get_db())
    yield session
    session.close()

@pytest.fixture
def matching_service(db_session):
    """Create a matching service instance."""
    return MatchingService(db_session)

@pytest.fixture
def sample_student(db_session):
    """Create a sample student."""
    student = StudentProfile(
        student_id="test123",
        demographic_features={},
        application_status="active",
        funnel_stage="application"
    )
    db_session.add(student)
    db_session.commit()
    return student

@pytest.fixture
def sample_recommendation(db_session: Session, sample_student):
    """Create a sample recommendation."""
    recommendation = StoredRecommendation(
        student_id=sample_student.student_id,
        recommendations=[{
            "type": "email",
            "category": "welcome",
            "content": "Welcome to our program!"
        }],
        generated_at=datetime.now(),
        expires_at=datetime.now() + timedelta(days=7)
    )
    db_session.add(recommendation)
    db_session.commit()
    return recommendation

def test_get_recommendations(matching_service, db_session, sample_student):
    """Test getting recommendations for a student."""
    recommendations = matching_service.get_recommendations(
        student_id=sample_student.student_id,
        top_k=5
    )

    assert len(recommendations) <= 5
    assert all(isinstance(r, dict) for r in recommendations)

def test_track_recommendation_action(matching_service, db_session, sample_student, sample_recommendation):
    """Test tracking a recommendation action."""
    matching_service.track_recommendation_action(
        student_id=sample_student.student_id,
        recommendation_id=sample_recommendation.id,
        action_type="viewed"
    )

    action = db_session.query(RecommendationAction).filter(
        RecommendationAction.student_id == sample_student.student_id,
        RecommendationAction.recommendation_id == sample_recommendation.id
    ).first()

    assert action is not None
    assert action.action_type == "viewed"
    assert action.action_timestamp is not None

def test_get_feedback_metrics(matching_service, db_session, sample_student, sample_recommendation):
    """Test getting feedback metrics."""
    # Track some actions
    matching_service.track_recommendation_action(
        student_id=sample_student.student_id,
        recommendation_id=sample_recommendation.id,
        action_type="viewed"
    )

    metrics = matching_service.get_feedback_metrics()
    assert len(metrics) > 0
    assert any(m["recommendation_type"] == "test" for m in metrics)

def test_get_recommendation_effectiveness(matching_service, db_session, sample_student, sample_recommendation):
    """Test getting recommendation effectiveness metrics."""
    # Track some actions
    matching_service.track_recommendation_action(
        student_id=sample_student.student_id,
        recommendation_id=sample_recommendation.id,
        action_type="viewed"
    )

    effectiveness = matching_service.get_recommendation_effectiveness()
    assert "action_rate" in effectiveness
    assert "completion_rate" in effectiveness
    assert "average_time_to_action" in effectiveness

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

def test_match_engagement_to_recommendation(db_session: Session, sample_student, sample_recommendation, sample_engagement):
    matching_service = MatchingService(db_session)
    matched, confidence = matching_service.match_engagement_to_recommendation(sample_engagement, db=db_session)
    assert matched
    assert confidence > 0

    # Check that a recommendation action was created with confidence score
    action = db_session.query(RecommendationAction).filter(
        RecommendationAction.recommendation_id == sample_recommendation.id
    ).first()
    assert action is not None
    assert action.confidence_score == confidence

def test_fuzzy_matching(db_session: Session, sample_student, sample_recommendation):
    matching_service = MatchingService(db_session)
    engagement = EngagementHistory(
        student_id=sample_student.student_id,
        engagement_type="test",
        content="Test recommendation with slight variation",
        timestamp=datetime.now()
    )
    db_session.add(engagement)
    db_session.commit()

    matched, confidence = matching_service.match_engagement_to_recommendation(engagement, db=db_session)
    assert matched
    assert confidence > 0

def test_get_match_quality_stats(db_session: Session, sample_student, sample_recommendation, sample_engagement):
    matching_service = MatchingService(db_session)
    matched, confidence = matching_service.match_engagement_to_recommendation(sample_engagement, db=db_session)
    assert matched

    stats = matching_service.get_match_quality_stats()
    assert "total_matches" in stats
    assert "average_confidence" in stats

def test_time_based_confidence(db_session: Session, sample_student, sample_recommendation):
    matching_service = MatchingService(db_session)
    engagement = EngagementHistory(
        student_id=sample_student.student_id,
        engagement_type="test",
        content="Test recommendation",
        timestamp=datetime.now() - timedelta(hours=25)  # Outside time window
    )
    db_session.add(engagement)
    db_session.commit()

    matched, confidence = matching_service.match_engagement_to_recommendation(engagement, db=db_session)
    assert matched
    assert confidence < 1.0  # Should have reduced confidence due to time 