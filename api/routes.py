from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Any, Optional
import logging
from sqlalchemy import or_
from sqlalchemy.orm import Session

from api.services import (
    RecommendationService, 
    LikelihoodService, 
    RiskAssessmentService,
    DashboardService,
    BulkActionService
)
from data.schemas import (
    RecommendationRequest,
    RecommendationResponse,
    LikelihoodRequest,
    LikelihoodResponse,
    RiskAssessmentRequest,
    RiskAssessmentResponse,
    AtRiskStudentsRequest,
    AtRiskStudentsResponse,
    BulkActionRequest,
    BulkActionPreviewResponse,
    BulkActionApplyResponse,
    DashboardStatsResponse
)
from data.models import StudentProfile, EngagementHistory, get_session

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api")

# Root endpoint
@router.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to the Student Engagement API"}

# Database dependency
def get_db():
    """Get database session."""
    db = get_session()
    try:
        yield db
    finally:
        db.close()

# Initialize services
recommendation_service = RecommendationService()
likelihood_service = LikelihoodService()
risk_service = RiskAssessmentService()
dashboard_service = DashboardService()
bulk_action_service = BulkActionService()

# Dependency to get recommendation service
def get_recommendation_service():
    return recommendation_service

# Dependency to get likelihood service
def get_likelihood_service():
    return likelihood_service

# Dependency to get risk service
def get_risk_service():
    return risk_service

# Dependency to get dashboard service
def get_dashboard_service():
    return dashboard_service

# Dependency to get bulk action service
def get_bulk_action_service():
    return bulk_action_service

# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

# Student endpoints
@router.get("/students")
async def get_students(
    funnelStage: Optional[str] = Query(None, description="Filter by funnel stage"),
    riskLevel: Optional[str] = Query(None, description="Filter by risk level (high, medium, low)"),
    searchQuery: Optional[str] = Query(None, description="Search by student ID, first name, or last name"),
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """
    Get a list of students with optional filtering.
    
    Args:
        funnelStage: Filter by funnel stage
        riskLevel: Filter by risk level (high, medium, low)
        searchQuery: Search by student ID, first name, or last name
        
    Returns:
        List of students matching the filters
    """
    try:
        # Start with base query
        query = db.query(StudentProfile)
        
        # Apply filters if provided
        if funnelStage and funnelStage.strip():
            query = query.filter(StudentProfile.funnel_stage.ilike(f"%{funnelStage.strip()}%"))
            
        if riskLevel and riskLevel.strip():
            # Convert risk level to score range
            risk_ranges = {
                "high": (0.7, 1.0),
                "medium": (0.4, 0.7),
                "low": (0.0, 0.4)
            }
            min_score, max_score = risk_ranges.get(riskLevel.strip().lower(), (0.0, 1.0))
            query = query.filter(StudentProfile.dropout_risk_score.between(min_score, max_score))
            
        if searchQuery and searchQuery.strip():
            search_term = f"%{searchQuery.strip()}%"
            query = query.filter(
                or_(
                    StudentProfile.student_id.ilike(search_term),
                    StudentProfile.demographic_features["first_name"].astext.ilike(search_term),
                    StudentProfile.demographic_features["last_name"].astext.ilike(search_term)
                )
            )
            
        # Execute query and convert to list of dicts
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

@router.get("/students/at-risk")
async def get_at_risk_students(
    risk_threshold: float = Query(0.7, description="Minimum risk score to include"),
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """
    Get a list of students at high risk of dropping off.
    
    Args:
        risk_threshold: Minimum risk score to include (default: 0.7)
        
    Returns:
        List of high-risk students with their risk scores
    """
    try:
        return risk_service.get_at_risk_students(risk_threshold)
    except Exception as e:
        logger.error(f"Error getting at-risk students: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/students/high-potential")
async def get_high_potential_students(
    likelihood_threshold: float = Query(0.7, description="Minimum application likelihood score to include"),
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """
    Get a list of students with high potential for application completion.
    
    Args:
        likelihood_threshold: Minimum application likelihood score to include (default: 0.7)
        
    Returns:
        List of high-potential students with their likelihood scores
    """
    try:
        return risk_service.get_high_potential_students(likelihood_threshold)
    except Exception as e:
        logger.error(f"Error getting high-potential students: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/students/{student_id}")
async def get_student(
    student_id: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get details for a specific student.
    
    Args:
        student_id: Unique identifier for the student
        
    Returns:
        Student details
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

@router.get("/students/{student_id}/engagements")
async def get_student_engagements(
    student_id: str,
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """
    Get engagement history for a specific student.
    
    Args:
        student_id: Unique identifier for the student
        
    Returns:
        List of engagements
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

# Dashboard endpoint
@router.get("/dashboard/stats")
async def get_dashboard_stats(
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get statistics for the dashboard.
    
    Returns:
        Dashboard statistics
    """
    try:
        # Get total students
        total_students = db.query(StudentProfile).count()
        
        # Get application rate
        completed_applications = db.query(StudentProfile).filter_by(application_status="completed").count()
        application_rate = (completed_applications / total_students * 100) if total_students > 0 else 0
        
        # Get at-risk count
        at_risk_students = risk_service.get_at_risk_students()
        at_risk_count = len(at_risk_students)
        
        # Get stage distribution
        stage_distribution = {}
        for student in db.query(StudentProfile).all():
            stage = student.funnel_stage.lower()
            stage_distribution[stage] = stage_distribution.get(stage, 0) + 1
        
        # Get engagement effectiveness over time
        engagement_effectiveness = [
            {"date": "2025-04-01", "effectiveness": 0.68},
            {"date": "2025-04-08", "effectiveness": 0.72},
            {"date": "2025-04-15", "effectiveness": 0.75},
            {"date": "2025-04-22", "effectiveness": 0.71},
            {"date": "2025-04-29", "effectiveness": 0.79},
            {"date": "2025-05-05", "effectiveness": 0.82}
        ]
        
        return {
            "total_students": total_students,
            "application_rate": application_rate,
            "at_risk_count": at_risk_count,
            "stage_distribution": stage_distribution,
            "engagement_effectiveness": engagement_effectiveness
        }
    except Exception as e:
        logger.error(f"Error getting dashboard stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(
    student_id: str = Query(..., description="Student ID"),
    top_k: Optional[int] = Query(5, description="Number of recommendations to return"),
    funnel_stage: Optional[str] = Query(None, description="Filter by funnel stage"),
    risk_level: Optional[str] = Query(None, description="Filter by risk level"),
    service: RecommendationService = Depends(get_recommendation_service)
):
    """Get personalized engagement recommendations for a student."""
    try:
        recommendations = service.get_recommendations(
            student_id=student_id,
            top_k=top_k,
            funnel_stage=funnel_stage,
            risk_level=risk_level
        )
        
        return RecommendationResponse(
            student_id=student_id,
            recommendations=recommendations
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/likelihood", response_model=LikelihoodResponse)
async def get_application_likelihood(
    request: LikelihoodRequest,
    service: LikelihoodService = Depends(get_likelihood_service)
):
    """Get the predicted likelihood of application completion."""
    try:
        likelihood = service.get_application_likelihood(request.student_id)
        
        return LikelihoodResponse(
            student_id=request.student_id,
            application_likelihood=likelihood
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/risk", response_model=RiskAssessmentResponse)
async def get_dropout_risk(
    request: RiskAssessmentRequest,
    service: RiskAssessmentService = Depends(get_risk_service)
):
    """Get the predicted risk of a student dropping off the application funnel."""
    try:
        risk = service.get_dropout_risk(request.student_id)
        
        return RiskAssessmentResponse(
            student_id=request.student_id,
            risk_category=risk["risk_category"],
            risk_score=risk["risk_score"]
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/bulk-actions/preview", response_model=BulkActionPreviewResponse)
async def preview_bulk_action(
    request: BulkActionRequest,
    service: BulkActionService = Depends(get_bulk_action_service)
):
    """Preview a bulk action."""
    try:
        preview = service.preview_bulk_action(request.action, request.segment)
        
        return BulkActionPreviewResponse(
            students=preview["students"],
            count=preview["count"],
            action_details=preview["action_details"]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/bulk-actions/apply", response_model=BulkActionApplyResponse)
async def apply_bulk_action(
    request: BulkActionRequest,
    service: BulkActionService = Depends(get_bulk_action_service)
):
    """Apply a bulk action."""
    try:
        result = service.apply_bulk_action(request.action, request.segment)
        
        return BulkActionApplyResponse(
            success=result["success"],
            students_affected=result["students_affected"],
            action_details=result["action_details"]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 