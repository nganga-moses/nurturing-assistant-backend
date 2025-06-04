import pytest
from fastapi.testclient import TestClient
from api.main import app
from data.models.student_profile import StudentProfile
from data.models.recommendation import Recommendation
from database.session import get_db

client = TestClient(app)

@pytest.fixture
def db_session():
    db = next(get_db())
    yield db
    db.rollback()
    db.close()

@pytest.fixture
def recruiter_and_students(db_session):
    # Create a recruiter user
    recruiter_id = "recruiter_1"
    # Create students assigned to the recruiter
    students = []
    for i in range(2):
        student = StudentProfile(
            student_id=f"student_{i}",
            recruiter_id=recruiter_id,
            demographic_features={},
            application_status="in_progress",
            funnel_stage="consideration"
        )
        db_session.add(student)
        students.append(student)
    db_session.commit()
    return recruiter_id, students

@pytest.fixture
def recommendations(db_session, recruiter_and_students):
    recruiter_id, students = recruiter_and_students
    recs = []
    for student in students:
        rec = Recommendation(
            student_id=student.student_id,
            recommendation_type="email",
            content_id="content_1",
            confidence_score=0.9
        )
        db_session.add(rec)
        recs.append(rec)
    db_session.commit()
    return recs

def override_get_current_user_id():
    return "recruiter_1"

app.dependency_overrides["get_current_user_id"] = override_get_current_user_id

def test_get_recommendations_for_current_user(recommendations):
    response = client.get("/recommendations/me")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2
    for rec in data:
        assert rec["recommendation_type"] == "email"
        assert rec["confidence_score"] == 0.9 