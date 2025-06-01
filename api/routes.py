from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Any, Optional, datetime
import logging
from sqlalchemy import or_
from sqlalchemy.orm import Session

from api.services import (
    RecommendationService, 
    LikelihoodService, 
    RiskAssessmentService,
    DashboardService,
    BulkActionService,
    RecommendationSettingsService,
    NudgeTrackingService
)
from data.schemas.schemas import (
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
    DashboardStatsResponse,
    RecommendationSettingsRequest,
    RecommendationSettingsResponse
)
from data.models.schemas import (
    StudentProfileCreate,
    StudentProfileUpdate,
    EngagementCreate,
    EngagementUpdate,
    ContentCreate,
    ContentUpdate
)
from data.models.models import StudentProfile, EngagementHistory, get_session

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
settings_service = RecommendationSettingsService()
tracking_service = NudgeTrackingService()

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

# Dependency to get settings service
def get_settings_service():
    return settings_service

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

@router.get("/recommendation-settings", response_model=RecommendationSettingsResponse)
async def get_recommendation_settings(
    service: RecommendationSettingsService = Depends(get_settings_service)
):
    """Get current recommendation settings."""
    try:
        settings = service.get_settings()
        return RecommendationSettingsResponse(**settings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/recommendation-settings", response_model=RecommendationSettingsResponse)
async def update_recommendation_settings(
    request: RecommendationSettingsRequest,
    service: RecommendationSettingsService = Depends(get_settings_service)
):
    """Update recommendation settings."""
    try:
        # Validate settings
        if request.mode == "scheduled":
            if not request.schedule_type:
                raise HTTPException(
                    status_code=400,
                    detail="schedule_type is required for scheduled mode"
                )
            if request.schedule_type == "weekly" and not request.schedule_day:
                raise HTTPException(
                    status_code=400,
                    detail="schedule_day is required for weekly schedule"
                )
            if not request.schedule_time:
                raise HTTPException(
                    status_code=400,
                    detail="schedule_time is required for scheduled mode"
                )
        
        # Update settings
        settings = service.update_settings(request.dict(exclude_unset=True))
        return RecommendationSettingsResponse(**settings)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/nudges/{nudge_id}/track")
def track_nudge_action(
    nudge_id: int,
    action: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Track a student's action on a nudge."""
    tracking_service.track_nudge_action(
        student_id=action["student_id"],
        nudge_id=nudge_id,
        action_type=action["action_type"]
    )
    return {"status": "success"}

@router.post("/api/nudges/{nudge_id}/complete")
def track_nudge_completion(
    nudge_id: int,
    completion: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Track whether a student completed the suggested action."""
    tracking_service.track_completion(
        student_id=completion["student_id"],
        nudge_id=nudge_id,
        completed=completion["completed"],
        dropoff_point=completion.get("dropoff_point")
    )
    return {"status": "success"}

@router.get("/api/feedback/metrics")
def get_feedback_metrics(
    nudge_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get feedback metrics for nudge types."""
    return tracking_service.get_feedback_metrics(nudge_type)

@router.get("/api/students/{student_id}/actions")
def get_student_actions(
    student_id: str,
    db: Session = Depends(get_db)
):
    """Get all actions for a specific student."""
    return tracking_service.get_student_actions(student_id)

@router.get("/api/reports/nudge-performance")
def get_nudge_performance_report(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    nudge_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get performance report for nudges."""
    try:
        # Get feedback metrics
        metrics = tracking_service.get_feedback_metrics(nudge_type)
        
        # Calculate overall statistics
        total_shown = sum(m["total_shown"] for m in metrics)
        total_acted = sum(m["acted_count"] for m in metrics)
        total_completed = sum(m["acted_count"] * m["completion_rate"] for m in metrics)
        
        # Calculate engagement and completion rates
        engagement_rate = total_acted / total_shown if total_shown > 0 else 0
        completion_rate = total_completed / total_acted if total_acted > 0 else 0
        
        return {
            "overall_metrics": {
                "total_shown": total_shown,
                "total_acted": total_acted,
                "total_completed": total_completed,
                "engagement_rate": engagement_rate,
                "completion_rate": completion_rate
            },
            "nudge_type_metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/reports/agent-performance")
def get_agent_performance_report(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    agent_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get performance report for enrollment agents."""
    try:
        # Get all students for the agent(s)
        query = db.query(StudentProfile)
        if agent_id:
            query = query.filter(StudentProfile.enrollment_agent_id == agent_id)
        students = query.all()
        
        # Calculate metrics for each agent
        agent_metrics = {}
        for student in students:
            agent_id = student.enrollment_agent_id
            if not agent_id:
                continue
                
            if agent_id not in agent_metrics:
                agent_metrics[agent_id] = {
                    "total_students": 0,
                    "active_students": 0,
                    "completed_applications": 0,
                    "avg_engagement_rate": 0.0,
                    "avg_completion_rate": 0.0
                }
            
            metrics = agent_metrics[agent_id]
            metrics["total_students"] += 1
            
            # Count active students (those with recent interactions)
            if student.last_interaction_date and (datetime.now() - student.last_interaction_date).days <= 30:
                metrics["active_students"] += 1
            
            # Count completed applications
            if student.application_status == "completed":
                metrics["completed_applications"] += 1
            
            # Get student's nudge actions
            actions = tracking_service.get_student_actions(student.student_id)
            if actions:
                acted_count = sum(1 for a in actions if a["action_type"] == "acted")
                completed_count = sum(1 for a in actions if a["action_completed"])
                
                # Update engagement and completion rates
                metrics["avg_engagement_rate"] = (metrics["avg_engagement_rate"] * (metrics["total_students"] - 1) + 
                                                (acted_count / len(actions))) / metrics["total_students"]
                metrics["avg_completion_rate"] = (metrics["avg_completion_rate"] * (metrics["total_students"] - 1) + 
                                                (completed_count / acted_count if acted_count > 0 else 0)) / metrics["total_students"]
        
        return {
            "agent_metrics": agent_metrics,
            "overall_metrics": {
                "total_agents": len(agent_metrics),
                "total_students": sum(m["total_students"] for m in agent_metrics.values()),
                "total_active_students": sum(m["active_students"] for m in agent_metrics.values()),
                "total_completed_applications": sum(m["completed_applications"] for m in agent_metrics.values()),
                "avg_engagement_rate": sum(m["avg_engagement_rate"] for m in agent_metrics.values()) / len(agent_metrics) if agent_metrics else 0,
                "avg_completion_rate": sum(m["avg_completion_rate"] for m in agent_metrics.values()) / len(agent_metrics) if agent_metrics else 0
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/reports/student-engagement")
def get_student_engagement_report(
    student_id: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: Session = Depends(get_db)
):
    """Get engagement report for a specific student."""
    try:
        # Get student
        student = db.query(StudentProfile).filter_by(student_id=student_id).first()
        if not student:
            raise HTTPException(status_code=404, detail=f"Student {student_id} not found")
        
        # Get student's actions
        actions = tracking_service.get_student_actions(student_id)
        
        # Calculate engagement metrics
        total_nudges = len(actions)
        acted_count = sum(1 for a in actions if a["action_type"] == "acted")
        completed_count = sum(1 for a in actions if a["action_completed"])
        
        # Get action timeline
        timeline = []
        for action in actions:
            timeline.append({
                "timestamp": action["action_timestamp"],
                "action_type": action["action_type"],
                "nudge_id": action["nudge_id"],
                "completed": action["action_completed"],
                "time_to_action": action["time_to_action"]
            })
        
        return {
            "student_info": {
                "student_id": student.student_id,
                "funnel_stage": student.funnel_stage,
                "application_status": student.application_status,
                "enrollment_agent_id": student.enrollment_agent_id
            },
            "engagement_metrics": {
                "total_nudges": total_nudges,
                "acted_count": acted_count,
                "completed_count": completed_count,
                "engagement_rate": acted_count / total_nudges if total_nudges > 0 else 0,
                "completion_rate": completed_count / acted_count if acted_count > 0 else 0
            },
            "action_timeline": timeline
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 