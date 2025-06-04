from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from data.models.user import User
from data.models.student_profile import StudentProfile
from database.session import get_db
from api.auth.supabase import check_vp

router = APIRouter(prefix="/vp", tags=["vp"])

def vp_required():
    # Placeholder for actual role-based dependency
    # Replace with real implementation
    pass

@router.get("/recruiters")
def list_recruiters_with_stats(
    db: Session = Depends(get_db),
    user=Depends(check_vp)
) -> List[Dict[str, Any]]:
    """
    List all recruiters and their stats. VP/admin only.
    """
    recruiters = db.query(User).filter(User.role == "recruiter").all()
    results = []
    for recruiter in recruiters:
        students = db.query(StudentProfile).filter(StudentProfile.enrollment_agent_id == recruiter.crm_user_id).all()
        total_students = len(students)
        completed_applications = sum(1 for s in students if s.application_status and s.application_status.lower() == "completed")
        at_risk = sum(1 for s in students if s.dropout_risk_score and s.dropout_risk_score > 0.7)
        results.append({
            **recruiter.to_dict(),
            "total_students": total_students,
            "completed_applications": completed_applications,
            "at_risk_students": at_risk,
            "completion_rate": (completed_applications / total_students * 100) if total_students > 0 else 0
        })
    return results

@router.get("/analytics")
def org_analytics(
    db: Session = Depends(get_db),
    user=Depends(check_vp)
) -> Dict[str, Any]:
    """
    Get organization-wide analytics. VP/admin only.
    """
    students = db.query(StudentProfile).all()
    total_students = len(students)
    completed_applications = sum(1 for s in students if s.application_status and s.application_status.lower() == "completed")
    at_risk = sum(1 for s in students if s.dropout_risk_score and s.dropout_risk_score > 0.7)
    return {
        "total_students": total_students,
        "completed_applications": completed_applications,
        "at_risk_students": at_risk,
        "completion_rate": (completed_applications / total_students * 100) if total_students > 0 else 0
    } 