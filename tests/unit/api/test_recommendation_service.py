import pytest
from datetime import datetime, timedelta
from data.models.models import StudentProfile, StoredRecommendation
from api.services import RecommendationService

@pytest.fixture
def recommendation_service(db_session):
    return RecommendationService(mode="scheduled", session=db_session)

@pytest.fixture
def sample_student(db_session):
    student = StudentProfile(
        student_id="test_student",
        demographic_features={},
        application_status="in_progress",
        funnel_stage="consideration",
        last_recommendation_at=None
    )
    db_session.add(student)
    db_session.commit()
    return student

@pytest.fixture
def sample_recommendations(db_session, sample_student):
    # Create some recommendations with different expiration times
    now = datetime.now()
    recommendations = [
        StoredRecommendation(
            student_id=sample_student.student_id,
            recommendations=[{"id": 1, "score": 0.8}],
            generated_at=now - timedelta(days=1),
            expires_at=now + timedelta(hours=1)  # Expires soon
        ),
        StoredRecommendation(
            student_id=sample_student.student_id,
            recommendations=[{"id": 2, "score": 0.9}],
            generated_at=now - timedelta(days=1),
            expires_at=now + timedelta(days=1)  # Expires later
        )
    ]
    db_session.add_all(recommendations)
    db_session.commit()
    return recommendations

def test_purge_expired_recommendations(recommendation_service, db_session, sample_recommendations):
    """Test that expired recommendations are properly purged."""
    # Initial count
    initial_count = db_session.query(StoredRecommendation).count()
    assert initial_count == 2
    
    # Purge recommendations
    recommendation_service.purge_expired_recommendations(buffer_minutes=30)
    
    # Check that only non-expired recommendations remain
    remaining_count = db_session.query(StoredRecommendation).count()
    assert remaining_count == 1
    
    # Verify the correct recommendation was kept
    remaining = db_session.query(StoredRecommendation).first()
    assert remaining.recommendations[0]["id"] == 2

def test_generate_scheduled_recommendations(recommendation_service, db_session, sample_student):
    """Test scheduled recommendation generation with purging."""
    # Generate recommendations
    result = recommendation_service.generate_scheduled_recommendations(batch_size=1)
    
    # Check results
    assert result["total_attempted"] == 1
    assert result["successful_generations"] == 1
    assert result["failed_generations"] == 0
    
    # Verify recommendations were generated
    recommendations = db_session.query(StoredRecommendation).all()
    assert len(recommendations) == 1
    
    # Verify student was updated
    db_session.refresh(sample_student)
    assert sample_student.last_recommendation_at is not None

def test_generate_scheduled_recommendations_with_purge(recommendation_service, db_session, sample_student, sample_recommendations):
    """Test that old recommendations are purged before generating new ones."""
    # Initial count
    initial_count = db_session.query(StoredRecommendation).count()
    assert initial_count == 2
    
    # Generate new recommendations
    recommendation_service.generate_scheduled_recommendations(batch_size=1)
    
    # Check that old recommendations were purged and new ones were generated
    final_count = db_session.query(StoredRecommendation).count()
    assert final_count == 1
    
    # Verify the new recommendation
    new_recommendation = db_session.query(StoredRecommendation).first()
    assert new_recommendation.student_id == sample_student.student_id
    assert new_recommendation.generated_at > sample_recommendations[0].generated_at

def test_error_handling_in_generation(recommendation_service, db_session):
    """Test error handling during recommendation generation."""
    # Create an invalid student (missing required fields)
    invalid_student = StudentProfile(student_id="invalid_student")
    db_session.add(invalid_student)
    db_session.commit()
    
    # Attempt to generate recommendations
    result = recommendation_service.generate_scheduled_recommendations(batch_size=1)
    
    # Verify error handling
    assert result["total_attempted"] == 1
    assert result["successful_generations"] == 0
    assert result["failed_generations"] == 1

def test_purge_with_no_recommendations(recommendation_service, db_session):
    """Test purging when no recommendations exist."""
    # Purge should not raise an error
    recommendation_service.purge_expired_recommendations()
    
    # Count should still be 0
    assert db_session.query(StoredRecommendation).count() == 0 