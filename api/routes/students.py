from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
import logging
from data.models.student_profile import StudentProfile
from data.models.engagement_history import EngagementHistory
from database.session import get_db
from api.services.risk_assessment_service import RiskAssessmentService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/students", tags=["students"])

risk_service = RiskAssessmentService()

@router.get("")
async def get_students(
    funnelStage: Optional[str] = Query(None, description="Filter by funnel stage"),
    riskLevel: Optional[str] = Query(None, description="Filter by risk level (high, medium, low)"),
    searchQuery: Optional[str] = Query(None, description="Search by student ID, first name, or last name"),
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """
    Get a list of students with optional filtering.
    """
    try:
        query = db.query(StudentProfile)
        if funnelStage and funnelStage.strip():
            query = query.filter(StudentProfile.funnel_stage.ilike(f"%{funnelStage.strip()}%"))
        if riskLevel and riskLevel.strip():
            risk_ranges = {"high": (0.7, 1.0), "medium": (0.4, 0.7), "low": (0.0, 0.4)}
            min_score, max_score = risk_ranges.get(riskLevel.strip().lower(), (0.0, 1.0))
            query = query.filter(StudentProfile.dropout_risk_score.between(min_score, max_score))
        if searchQuery and searchQuery.strip():
            search_term = f"%{searchQuery.strip()}%"
            query = query.filter(
                StudentProfile.student_id.ilike(search_term)
            )
        students = query.all()
        return [
            {
                "student_id": student.student_id,
                "demographic_features": student.demographic_features,
                "application_status": student.application_status,
                "funnel_stage": student.funnel_stage,
                "last_interaction_date": student.last_interaction_date,
                "dropout_risk_score": student.dropout_risk_score,
                "application_likelihood_score": student.application_likelihood_score
            }
            for student in students
        ]
    except Exception as e:
        logger.error(f"Error getting students: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/at-risk")
async def get_at_risk_students(
    risk_threshold: float = Query(0.7, description="Minimum risk score to include"),
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """
    Get a list of students at high risk of dropping off.
    """
    try:
        return risk_service.get_at_risk_students(db, risk_threshold)
    except Exception as e:
        logger.error(f"Error getting at-risk students: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/high-potential")
async def get_high_potential_students(
    likelihood_threshold: float = Query(0.7, description="Minimum application likelihood score to include"),
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """
    Get a list of students with high potential for application completion.
    """
    try:
        return risk_service.get_high_potential_students(db, likelihood_threshold)
    except Exception as e:
        logger.error(f"Error getting high-potential students: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{student_id}")
async def get_student(
    student_id: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get details for a specific student.
    """
    try:
        student = db.query(StudentProfile).filter_by(student_id=student_id).first()
        if not student:
            raise HTTPException(status_code=404, detail=f"Student with ID {student_id} not found")
        return {
            "student_id": student.student_id,
            "demographic_features": student.demographic_features,
            "application_status": student.application_status,
            "funnel_stage": student.funnel_stage,
            "last_interaction_date": student.last_interaction_date,
            "dropout_risk_score": student.dropout_risk_score,
            "application_likelihood_score": student.application_likelihood_score
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting student: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{student_id}/engagements")
async def get_student_engagements(
    student_id: str,
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """
    Get engagement history for a specific student.
    """
    try:
        student = db.query(StudentProfile).filter_by(student_id=student_id).first()
        if not student:
            raise HTTPException(status_code=404, detail=f"Student with ID {student_id} not found")
        engagements = db.query(EngagementHistory).filter_by(student_id=student_id).all()
        return [engagement.to_dict() for engagement in engagements]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting student engagements: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 