from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Optional, List

from database.session import get_db
from data.models import (
    EngagementHistory,
    RecommendationAction,
    RecommendationFeedbackMetrics,
    StudentProfile,
    RecommendationSettings,
    StatusChange
)
from api.services import RecommendationTrackingService

router = APIRouter(prefix="/reports", tags=["reports"])

tracking_service = RecommendationTrackingService()

@router.get("/performance")
def get_performance_report(
    start_date: datetime = None,
    end_date: datetime = None,
    db: Session = Depends(get_db)
):
    """Get performance report for agents and staff."""
    if not start_date:
        start_date = datetime.now() - timedelta(days=30)
    if not end_date:
        end_date = datetime.now()

    # Get engagement metrics
    engagement_metrics = (
        db.query(EngagementHistory)
        .filter(
            EngagementHistory.timestamp >= start_date,
            EngagementHistory.timestamp <= end_date
        )
        .all()
    )

    # Get recommendation metrics
    recommendation_metrics = (
        db.query(RecommendationAction)
        .filter(
            RecommendationAction.action_timestamp >= start_date,
            RecommendationAction.action_timestamp <= end_date
        )
        .all()
    )

    # Get student status changes
    status_changes = (
        db.query(StatusChange)
        .filter(
            StatusChange.timestamp >= start_date,
            StatusChange.timestamp <= end_date
        )
        .all()
    )

    # Calculate metrics
    total_engagements = len(engagement_metrics)
    total_recommendations = len(recommendation_metrics)
    total_status_changes = len(status_changes)

    # Calculate engagement success rate
    successful_engagements = sum(1 for e in engagement_metrics if e.engagement_response == "success")
    engagement_success_rate = (successful_engagements / total_engagements * 100) if total_engagements > 0 else 0

    # Calculate recommendation completion rate
    completed_recommendations = sum(1 for r in recommendation_metrics if r.action_completed)
    recommendation_completion_rate = (completed_recommendations / total_recommendations * 100) if total_recommendations > 0 else 0

    # Calculate funnel progression
    funnel_progression = {}
    for change in status_changes:
        if change.field == "funnel_stage":
            old_stage = change.old_value
            new_stage = change.new_value
            if old_stage not in funnel_progression:
                funnel_progression[old_stage] = {"total": 0, "progressed": 0}
            funnel_progression[old_stage]["total"] += 1
            if new_stage > old_stage:  # Assuming funnel stages are ordered
                funnel_progression[old_stage]["progressed"] += 1

    # Calculate progression rates
    progression_rates = {}
    for stage, data in funnel_progression.items():
        progression_rates[stage] = (data["progressed"] / data["total"] * 100) if data["total"] > 0 else 0

    return {
        "period": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat()
        },
        "engagement_metrics": {
            "total_engagements": total_engagements,
            "successful_engagements": successful_engagements,
            "success_rate": engagement_success_rate
        },
        "recommendation_metrics": {
            "total_recommendations": total_recommendations,
            "completed_recommendations": completed_recommendations,
            "completion_rate": recommendation_completion_rate
        },
        "funnel_metrics": {
            "total_status_changes": total_status_changes,
            "progression_rates": progression_rates
        }
    }

@router.get("/engagement")
def get_engagement_report(
    start_date: datetime = None,
    end_date: datetime = None,
    db: Session = Depends(get_db)
):
    """Get engagement report showing student interactions and responses."""
    if not start_date:
        start_date = datetime.now() - timedelta(days=30)
    if not end_date:
        end_date = datetime.now()

    # Get engagement history
    engagements = (
        db.query(EngagementHistory)
        .filter(
            EngagementHistory.timestamp >= start_date,
            EngagementHistory.timestamp <= end_date
        )
        .all()
    )

    # Get recommendation actions
    recommendations = (
        db.query(RecommendationAction)
        .filter(
            RecommendationAction.action_timestamp >= start_date,
            RecommendationAction.action_timestamp <= end_date
        )
        .all()
    )

    # Calculate engagement metrics
    engagement_types = {}
    for engagement in engagements:
        if engagement.engagement_type not in engagement_types:
            engagement_types[engagement.engagement_type] = {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "no_response": 0
            }
        
        engagement_types[engagement.engagement_type]["total"] += 1
        if engagement.engagement_response == "success":
            engagement_types[engagement.engagement_type]["successful"] += 1
        elif engagement.engagement_response == "failed":
            engagement_types[engagement.engagement_type]["failed"] += 1
        else:
            engagement_types[engagement.engagement_type]["no_response"] += 1

    # Calculate recommendation metrics
    recommendation_types = {}
    for recommendation in recommendations:
        if recommendation.action_type not in recommendation_types:
            recommendation_types[recommendation.action_type] = {
                "total": 0,
                "completed": 0,
                "dropped_off": 0
            }
        
        recommendation_types[recommendation.action_type]["total"] += 1
        if recommendation.action_completed:
            recommendation_types[recommendation.action_type]["completed"] += 1
        elif recommendation.dropoff_point:
            recommendation_types[recommendation.action_type]["dropped_off"] += 1

    return {
        "period": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat()
        },
        "engagement_types": {
            type_name: {
                "total": data["total"],
                "success_rate": (data["successful"] / data["total"] * 100) if data["total"] > 0 else 0,
                "failure_rate": (data["failed"] / data["total"] * 100) if data["total"] > 0 else 0,
                "no_response_rate": (data["no_response"] / data["total"] * 100) if data["total"] > 0 else 0
            }
            for type_name, data in engagement_types.items()
        },
        "recommendation_types": {
            type_name: {
                "total": data["total"],
                "completion_rate": (data["completed"] / data["total"] * 100) if data["total"] > 0 else 0,
                "dropoff_rate": (data["dropped_off"] / data["total"] * 100) if data["total"] > 0 else 0
            }
            for type_name, data in recommendation_types.items()
        }
    }

@router.get("/students/at-risk")
def get_at_risk_students(
    risk_threshold: float = 0.7,
    db: Session = Depends(get_db)
):
    """Get list of students at risk of dropping out."""
    # Get students with high dropout risk
    at_risk_students = (
        db.query(StudentProfile)
        .filter(StudentProfile.dropout_risk_score >= risk_threshold)
        .all()
    )

    # Get their recent engagement history
    student_engagements = {}
    for student in at_risk_students:
        recent_engagements = (
            db.query(EngagementHistory)
            .filter(
                EngagementHistory.student_id == student.student_id,
                EngagementHistory.timestamp >= datetime.now() - timedelta(days=30)
            )
            .all()
        )
        student_engagements[student.student_id] = recent_engagements

    # Get their recommendation actions
    student_recommendations = {}
    for student in at_risk_students:
        recent_recommendations = (
            db.query(RecommendationAction)
            .filter(
                RecommendationAction.student_id == student.student_id,
                RecommendationAction.action_timestamp >= datetime.now() - timedelta(days=30)
            )
            .all()
        )
        student_recommendations[student.student_id] = recent_recommendations

    return {
        "total_at_risk": len(at_risk_students),
        "students": [
            {
                "student_id": student.student_id,
                "dropout_risk_score": student.dropout_risk_score,
                "funnel_stage": student.funnel_stage,
                "last_interaction": student.last_interaction_date.isoformat() if student.last_interaction_date else None,
                "recent_engagements": len(student_engagements.get(student.student_id, [])),
                "recent_recommendations": len(student_recommendations.get(student.student_id, [])),
                "engagement_success_rate": (
                    sum(1 for e in student_engagements.get(student.student_id, []) if e.engagement_response == "success") /
                    len(student_engagements.get(student.student_id, [])) * 100
                ) if student_engagements.get(student.student_id) else 0,
                "recommendation_completion_rate": (
                    sum(1 for r in student_recommendations.get(student.student_id, []) if r.action_completed) /
                    len(student_recommendations.get(student.student_id, [])) * 100
                ) if student_recommendations.get(student.student_id) else 0
            }
            for student in at_risk_students
        ]
    }

@router.get("/staff-performance")
def get_staff_performance_report(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    staff_id: Optional[str] = Query(None),
    include_engagement: bool = Query(False, description="Include engagement metrics"),
    db: Session = Depends(get_db)
):
    """
    Get comprehensive performance report for staff/recruiters.
    Includes both basic metrics and optional engagement metrics.
    """
    try:
        query = db.query(StudentProfile)
        if staff_id:
            query = query.filter(StudentProfile.enrollment_agent_id == staff_id)
        students = query.all()
        
        # Aggregate by staff
        staff_stats = {}
        for student in students:
            agent_id = student.enrollment_agent_id or "unassigned"
            if agent_id not in staff_stats:
                staff_stats[agent_id] = {
                    "total_students": 0,
                    "active_students": 0,
                    "at_risk_students": 0,
                    "completed_applications": 0,
                    "avg_engagement_rate": 0.0,
                    "avg_completion_rate": 0.0
                }
            
            metrics = staff_stats[agent_id]
            metrics["total_students"] += 1
            
            # Active students (interacted in last 30 days)
            if student.last_interaction_date and (datetime.now() - student.last_interaction_date).days <= 30:
                metrics["active_students"] += 1
            
            # At-risk students
            if student.dropout_risk_score and student.dropout_risk_score > 0.7:
                metrics["at_risk_students"] += 1
            
            # Completed applications
            if student.application_status and student.application_status.lower() == "completed":
                metrics["completed_applications"] += 1
            
            # Engagement metrics if requested
            if include_engagement:
                actions = tracking_service.get_student_actions(student.student_id, db)
                if actions:
                    acted_count = sum(1 for a in actions if a["action_type"] == "acted")
                    completed_count = sum(1 for a in actions if a["action_completed"])
                    metrics["avg_engagement_rate"] = (metrics["avg_engagement_rate"] * (metrics["total_students"] - 1) + 
                                                   (acted_count / len(actions))) / metrics["total_students"]
                    metrics["avg_completion_rate"] = (metrics["avg_completion_rate"] * (metrics["total_students"] - 1) + 
                                                    (completed_count / acted_count if acted_count > 0 else 0)) / metrics["total_students"]
        
        # Calculate overall metrics
        overall_metrics = {
            "total_staff": len(staff_stats),
            "total_students": sum(m["total_students"] for m in staff_stats.values()),
            "total_active_students": sum(m["active_students"] for m in staff_stats.values()),
            "total_at_risk_students": sum(m["at_risk_students"] for m in staff_stats.values()),
            "total_completed_applications": sum(m["completed_applications"] for m in staff_stats.values())
        }
        
        if include_engagement:
            overall_metrics.update({
                "avg_engagement_rate": sum(m["avg_engagement_rate"] for m in staff_stats.values()) / len(staff_stats) if staff_stats else 0,
                "avg_completion_rate": sum(m["avg_completion_rate"] for m in staff_stats.values()) / len(staff_stats) if staff_stats else 0
            })
        
        return {
            "staff_metrics": staff_stats,
            "overall_metrics": overall_metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/student-engagement")
def get_student_engagement_report(
    student_id: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: Session = Depends(get_db)
):
    """
    Get comprehensive engagement report for a specific student.
    """
    try:
        student = db.query(StudentProfile).filter_by(student_id=student_id).first()
        if not student:
            raise HTTPException(status_code=404, detail=f"Student {student_id} not found")
        
        actions = tracking_service.get_student_actions(student_id, db)
        total_recommendations = len(actions)
        acted_count = sum(1 for a in actions if a["action_type"] == "acted")
        completed_count = sum(1 for a in actions if a["action_completed"])
        
        timeline = []
        for action in actions:
            timeline.append({
                "timestamp": action["action_timestamp"],
                "action_type": action["action_type"],
                "recommendation_id": action["recommendation_id"],
                "completed": action["action_completed"],
                "time_to_action": action["time_to_action"]
            })
        
        return {
            "student_info": {
                "student_id": student.student_id,
                "funnel_stage": student.funnel_stage,
                "application_status": student.application_status,
                "enrollment_agent_id": student.enrollment_agent_id,
                "risk_score": student.dropout_risk_score
            },
            "engagement_metrics": {
                "total_recommendations": total_recommendations,
                "acted_count": acted_count,
                "completed_count": completed_count,
                "engagement_rate": acted_count / total_recommendations if total_recommendations > 0 else 0,
                "completion_rate": completed_count / acted_count if acted_count > 0 else 0
            },
            "action_timeline": timeline
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/unmatched-engagements")
def unmatched_engagements(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    db: Session = Depends(get_db)
):
    """Get a list of unmatched engagements (not linked to any recommendation)."""
    query = db.query(EngagementHistory).outerjoin(
        RecommendationAction, EngagementHistory.engagement_id == RecommendationAction.recommendation_id
    ).filter(RecommendationAction.id.is_(None))
    if start_date:
        query = query.filter(EngagementHistory.timestamp >= start_date)
    if end_date:
        query = query.filter(EngagementHistory.timestamp <= end_date)
    engagements = query.all()
    return {"count": len(engagements), "engagements": [e.to_dict() for e in engagements]}

@router.get("/import-history")
def import_history(
    days: int = Query(30, description="Number of days to look back"),
    db: Session = Depends(get_db)
):
    """Get a summary of import history and status changes."""
    since = datetime.now() - timedelta(days=days)
    changes = db.query(StatusChange).filter(StatusChange.timestamp >= since).all()
    return {"import_history": [c.to_dict() for c in changes]} 