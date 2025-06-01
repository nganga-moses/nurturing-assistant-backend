import pytest
from datetime import datetime, timedelta
from api.services import NudgeTrackingService
from data.models.models import StudentProfile, StoredRecommendation, NudgeAction, NudgeFeedbackMetrics

@pytest.fixture
def db_session():
    """Create a test database session."""
    from data.models.models import get_session, init_db
    session = get_session()
    init_db()
    yield session
    session.close()

@pytest.fixture
def tracking_service(db_session):
    """Create a tracking service instance."""
    return NudgeTrackingService()

@pytest.fixture
def sample_student(db_session):
    """Create a sample student."""
    student = StudentProfile(
        student_id="TEST001",
        demographic_features={"name": "Test Student"},
        application_status="in_progress",
        funnel_stage="consideration"
    )
    db_session.add(student)
    db_session.commit()
    return student

@pytest.fixture
def sample_nudge(db_session, sample_student):
    """Create a sample nudge."""
    nudge = StoredRecommendation(
        student_id=sample_student.student_id,
        recommendations=[{
            "type": "application_reminder",
            "content": "Complete your application"
        }],
        generated_at=datetime.now() - timedelta(hours=1),
        expires_at=datetime.now() + timedelta(days=1)
    )
    db_session.add(nudge)
    db_session.commit()
    return nudge

def test_track_nudge_action(tracking_service, db_session, sample_student, sample_nudge):
    """Test tracking a nudge action."""
    # Track action
    tracking_service.track_nudge_action(
        student_id=sample_student.student_id,
        nudge_id=sample_nudge.id,
        action_type="acted"
    )
    
    # Verify action was recorded
    action = db_session.query(NudgeAction).filter_by(
        student_id=sample_student.student_id,
        nudge_id=sample_nudge.id
    ).first()
    
    assert action is not None
    assert action.action_type == "acted"
    assert action.time_to_action is not None
    assert action.action_completed is False

def test_track_completion(tracking_service, db_session, sample_student, sample_nudge):
    """Test tracking action completion."""
    # First track the action
    tracking_service.track_nudge_action(
        student_id=sample_student.student_id,
        nudge_id=sample_nudge.id,
        action_type="acted"
    )
    
    # Then track completion
    tracking_service.track_completion(
        student_id=sample_student.student_id,
        nudge_id=sample_nudge.id,
        completed=True
    )
    
    # Verify completion was recorded
    action = db_session.query(NudgeAction).filter_by(
        student_id=sample_student.student_id,
        nudge_id=sample_nudge.id
    ).first()
    
    assert action.action_completed is True

def test_get_feedback_metrics(tracking_service, db_session, sample_student, sample_nudge):
    """Test getting feedback metrics."""
    # Track some actions
    tracking_service.track_nudge_action(
        student_id=sample_student.student_id,
        nudge_id=sample_nudge.id,
        action_type="acted"
    )
    
    tracking_service.track_completion(
        student_id=sample_student.student_id,
        nudge_id=sample_nudge.id,
        completed=True
    )
    
    # Get metrics
    metrics = tracking_service.get_feedback_metrics("application_reminder")
    
    assert len(metrics) == 1
    metric = metrics[0]
    assert metric["nudge_type"] == "application_reminder"
    assert metric["total_shown"] == 1
    assert metric["acted_count"] == 1
    assert metric["completion_rate"] == 1.0

def test_get_student_actions(tracking_service, db_session, sample_student, sample_nudge):
    """Test getting student actions."""
    # Track some actions
    tracking_service.track_nudge_action(
        student_id=sample_student.student_id,
        nudge_id=sample_nudge.id,
        action_type="acted"
    )
    
    tracking_service.track_completion(
        student_id=sample_student.student_id,
        nudge_id=sample_nudge.id,
        completed=True
    )
    
    # Get actions
    actions = tracking_service.get_student_actions(sample_student.student_id)
    
    assert len(actions) == 1
    action = actions[0]
    assert action["student_id"] == sample_student.student_id
    assert action["nudge_id"] == sample_nudge.id
    assert action["action_type"] == "acted"
    assert action["action_completed"] is True 