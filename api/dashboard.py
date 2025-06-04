from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Response, Depends
from sqlalchemy.orm import Session
from typing import Dict, List, Set, Optional
from datetime import datetime, timedelta
import json
import asyncio
import csv
import io
from database.session import get_db
from data.models.student_profile import StudentProfile
from data.models.engagement_history import EngagementHistory

router = APIRouter()

# Store active WebSocket connections
active_connections: Set[WebSocket] = set()

def get_db():
    db = next(get_session())
    try:
        yield db
    finally:
        db.close()

async def broadcast_dashboard_update():
    """Broadcast dashboard data to all connected clients."""
    if not active_connections:
        return
    data = await get_dashboard_data()
    for connection in active_connections:
        try:
            await connection.send_json(data)
        except WebSocketDisconnect:
            active_connections.remove(connection)

@router.websocket("/ws/dashboard")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.add(websocket)
    try:
        while True:
            # Send initial data
            data = await get_dashboard_data()
            await websocket.send_json(data)
            # Wait for 30 seconds before next update
            await asyncio.sleep(30)
    except WebSocketDisconnect:
        active_connections.remove(websocket)

@router.get("/dashboard")
async def get_dashboard_data_endpoint(db: Session = Depends(get_db)):
    """HTTP endpoint for dashboard data."""
    return await get_dashboard_data(db)

async def get_dashboard_data(db: Session):
    try:
        # Get total active students
        total_students = db.query(StudentProfile).filter(StudentProfile.last_interaction_date >= datetime.now() - timedelta(days=30)).count()
        # Get application rate
        completed_applications = db.query(StudentProfile).filter_by(application_status="completed").count()
        application_rate = f"{completed_applications * 100.0 / total_students:.1f}%" if total_students > 0 else "0.0%"
        # Get at-risk students
        at_risk_students = db.query(StudentProfile).filter(StudentProfile.dropout_risk_score >= 0.7).count()
        # Get engagement rate
        engagement_rate = f"{db.query(EngagementHistory).filter(EngagementHistory.engagement_response == 'positive').count() * 100.0 / total_students:.1f}%" if total_students > 0 else "0.0%"
        # Get funnel distribution
        funnel_distribution = [
            {"name": stage[0], "value": db.query(StudentProfile).filter_by(funnel_stage=stage[0]).count()}
            for stage in db.query(StudentProfile.funnel_stage).distinct()
        ]
        # Get engagement trends (last 30 days)
        engagement_data = db.query(EngagementHistory).filter(EngagementHistory.timestamp >= datetime.now() - timedelta(days=30)).all()
        # Process engagement trends
        engagement_trends = {}
        for row in engagement_data:
            date = row.timestamp.strftime('%Y-%m-%d')
            if date not in engagement_trends:
                engagement_trends[date] = {'date': date, 'email': 0, 'sms': 0, 'campus_visit': 0}
            engagement_trends[date][row.engagement_type] = engagement_trends[date].get(row.engagement_type, 0) + 1
        engagement_trends = list(engagement_trends.values())
        # Get application likelihood distribution
        likelihood_distribution = [
            {"likelihood": round(score[0] * 10) / 10, "count": db.query(StudentProfile).filter_by(application_likelihood_score=score[0]).count()}
            for score in db.query(StudentProfile.application_likelihood_score).distinct()
        ]
        return {
            "totalStudents": total_students,
            "applicationRate": application_rate,
            "atRiskStudents": at_risk_students,
            "engagementRate": engagement_rate,
            "funnelDistribution": funnel_distribution,
            "engagementTrends": engagement_trends,
            "likelihoodDistribution": likelihood_distribution,
            "lastUpdated": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard/export/{format}")
async def export_dashboard_data(format: str, db: Session = Depends(get_db)):
    """Export dashboard data in various formats."""
    try:
        data = await get_dashboard_data(db)
        if format == "json":
            return Response(
                content=json.dumps(data, indent=2),
                media_type="application/json",
                headers={
                    "Content-Disposition": f"attachment; filename=dashboard_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                }
            )
        elif format == "csv":
            output = io.StringIO()
            writer = csv.writer(output)
            # Write metrics
            writer.writerow(["Metric", "Value"])
            writer.writerow(["Total Students", data["totalStudents"]])
            writer.writerow(["Application Rate", data["applicationRate"]])
            writer.writerow(["At-Risk Students", data["atRiskStudents"]])
            writer.writerow(["Engagement Rate", data["engagementRate"]])
            writer.writerow([])
            # Write funnel distribution
            writer.writerow(["Funnel Stage", "Count"])
            for item in data["funnelDistribution"]:
                writer.writerow([item["name"], item["value"]])
            writer.writerow([])
            # Write engagement trends
            writer.writerow(["Date", "Email", "SMS", "Campus Visit"])
            for trend in data["engagementTrends"]:
                writer.writerow([
                    trend["date"],
                    trend["email"],
                    trend["sms"],
                    trend["campus_visit"]
                ])
            writer.writerow([])
            # Write likelihood distribution
            writer.writerow(["Likelihood", "Count"])
            for item in data["likelihoodDistribution"]:
                writer.writerow([item["likelihood"], item["count"]])
            return Response(
                content=output.getvalue(),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename=dashboard_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                }
            )
        else:
            raise HTTPException(status_code=400, detail="Unsupported export format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Dependency to get recommendation service


def get_recommendation_service():
    if recommendation_service is None:
        raise HTTPException(status_code=503, detail="Recommendation service not initialized")
    return recommendation_service


# Dependency to get likelihood service
def get_likelihood_service():
    if likelihood_service is None:
        raise HTTPException(status_code=503, detail="Likelihood service not initialized")
    return likelihood_service


# Dependency to get risk service
def get_risk_service():
    if risk_service is None:
        raise HTTPException(status_code=503, detail="Risk service not initialized")
    return risk_service


# Dependency to get dashboard service
def get_dashboard_service():
    if dashboard_service is None:
        raise HTTPException(status_code=503, detail="Dashboard service not initialized")
    return dashboard_service


# Dependency to get bulk action service
def get_bulk_action_service():
    if bulk_action_service is None:
        raise HTTPException(status_code=503, detail="Bulk action service not initialized")
    return bulk_action_service


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to the Student Engagement API"}


# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


# Student endpoints
@app.get("/api/students")
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


@app.get("/api/students/at-risk")
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


@app.get("/api/students/high-potential")
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


@app.get("/api/students/{student_id}")
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


@app.get("/api/students/{student_id}/engagements")
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
@app.get("/api/dashboard/stats")
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


@app.post("/api/recommendations", response_model=RecommendationResponse)
async def get_recommendations(
        request: RecommendationRequest,
        service: RecommendationService = Depends(get_recommendation_service)
):
    """Get personalized engagement recommendations for a student."""
    try:
        recommendations = service.get_recommendations(
            student_id=request.student_id,
            top_k=request.top_k,
            funnel_stage=request.funnel_stage,
            risk_level=request.risk_level
        )

        return RecommendationResponse(
            student_id=request.student_id,
            recommendations=recommendations
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/likelihood", response_model=LikelihoodResponse)
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


@app.post("/api/risk", response_model=RiskAssessmentResponse)
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


@app.post("/api/bulk-actions/preview", response_model=BulkActionPreviewResponse)
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


@app.post("/api/bulk-actions/apply", response_model=BulkActionApplyResponse)
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


@app.post("/api/risk-assessment")
async def assess_risk(request: RiskAssessmentRequest):
    """Assess dropout risk for a student."""
    try:
        # Get the risk assessment service
        service = get_risk_service()

        # Get the risk assessment
        risk_assessment = service.get_dropout_risk(request.student_id, request.features)

        return risk_assessment
    except Exception as e:
        logger.error(f"Error in assess_risk: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
