from datetime import datetime, timedelta
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from data.models.student_profile import StudentProfile
from data.models.engagement_history import EngagementHistory
from data.models.funnel_stage import FunnelStage

def calculate_application_likelihood(student: Dict, engagements: List[Dict], db: Session) -> float:
    """
    Calculate the likelihood score for a student based on their progress towards the tracking goal.
    All scoring factors are dynamically calculated from historical data.
    
    Args:
        student: Dictionary containing student information
        engagements: List of dictionaries containing engagement information
        db: Database session for fetching data
    
    Returns:
        float: Likelihood score between 0 and 1
    """
    # Get all active funnel stages
    stages = (
        db.query(FunnelStage)
        .filter(FunnelStage.is_active == True)
        .order_by(FunnelStage.stage_order)
        .all()
    )
    
    if not stages:
        return 0.0
    
    # Find tracking goal stage
    tracking_goal_stage = next(
        (stage for stage in stages if stage.is_tracking_goal),
        None
    )
    
    if not tracking_goal_stage:
        return 0.0
    
    # 1. Calculate dynamic engagement weights based on historical success
    engagement_weights = calculate_engagement_weights(db, tracking_goal_stage)
    
    # 2. Calculate engagement score
    engagement_score = calculate_engagement_score(engagements, engagement_weights)
    
    # 3. Calculate time-based score
    time_score = calculate_time_score(engagements, student['funnel_stage'], db)
    
    # 4. Calculate progress score
    progress_score = calculate_progress_score(student['funnel_stage'], stages, tracking_goal_stage)
    
    # 5. Calculate dynamic weights based on student characteristics
    weights = calculate_dynamic_weights(student, stages, db)
    
    # Calculate final score
    final_score = (
        weights['engagement'] * engagement_score +
        weights['time'] * time_score +
        weights['progress'] * progress_score
    )
    
    return max(0.0, min(1.0, final_score))

def calculate_engagement_weights(db: Session, goal_stage: FunnelStage) -> Dict[str, float]:
    """Calculate engagement weights based on historical success rates."""
    # Get all engagement types used by successful students
    successful_students = (
        db.query(StudentProfile)
        .filter(StudentProfile.funnel_stage == goal_stage.stage_name)
        .filter(StudentProfile.is_successful == True)
        .all()
    )
    
    engagement_types = (
        db.query(EngagementHistory.engagement_type)
        .filter(EngagementHistory.student_id.in_([s.student_id for s in successful_students]))
        .distinct()
        .all()
    )
    
    weights = {}
    for engagement_type in engagement_types:
        # Calculate success rate for this engagement type
        total_students = (
            db.query(StudentProfile)
            .join(EngagementHistory)
            .filter(EngagementHistory.engagement_type == engagement_type[0])
            .distinct()
            .count()
        )
        
        if total_students > 0:
            successful_students = (
                db.query(StudentProfile)
                .join(EngagementHistory)
                .filter(
                    StudentProfile.funnel_stage == goal_stage.stage_name,
                    StudentProfile.is_successful == True,
                    EngagementHistory.engagement_type == engagement_type[0]
                )
                .distinct()
                .count()
            )
            
            weights[engagement_type[0]] = successful_students / total_students
    
    # Normalize weights
    total = sum(weights.values())
    if total > 0:
        weights = {k: v/total for k, v in weights.items()}
    
    return weights

def calculate_engagement_score(engagements: List[Dict], weights: Dict[str, float]) -> float:
    """Calculate engagement score using dynamic weights."""
    if not engagements:
        return 0.0
    
    total_score = 0
    for engagement in engagements:
        engagement_type = engagement['engagement_type'].lower()
        base_score = weights.get(engagement_type, 0.1)  # Default weight for unknown types
        
        # Adjust score based on response
        if engagement.get('engagement_response'):
            response = engagement['engagement_response'].lower()
            if response in ['positive', 'interested', 'scheduled']:
                base_score *= 1.2
            elif response in ['negative', 'not_interested']:
                base_score *= 0.8
        
        total_score += base_score
    
    return total_score / len(engagements)

def calculate_time_score(engagements: List[Dict], current_stage: str, db: Session) -> float:
    """Calculate time-based score using historical stage transition data."""
    if not engagements:
        return 0.0
    
    # Get average time spent in current stage
    avg_time = (
        db.query(func.avg(EngagementHistory.timestamp - StudentProfile.created_at))
        .join(StudentProfile)
        .filter(StudentProfile.funnel_stage == current_stage)
        .scalar()
    )
    
    if not avg_time:
        return 0.5  # Default score if no historical data
    
    # Calculate time since last engagement
    last_engagement = max(engagements, key=lambda x: x['timestamp'])
    days_since_last = (datetime.now() - last_engagement['timestamp']).days
    
    # Calculate decay based on average time in stage
    avg_days = avg_time.days if avg_time else 30
    decay_rate = 1 / avg_days
    
    return max(0, 1 - (days_since_last * decay_rate))

def calculate_progress_score(current_stage: str, stages: List[FunnelStage], goal_stage: FunnelStage) -> float:
    """Calculate progress score based on position relative to goal."""
    try:
        current_stage_index = next(
            (i for i, stage in enumerate(stages) if stage.stage_name.lower() == current_stage.lower()),
            -1
        )
        goal_stage_index = stages.index(goal_stage)
        
        if current_stage_index == -1 or goal_stage_index == 0:
            return 0.0
        
        return current_stage_index / goal_stage_index
    except (ValueError, IndexError):
        return 0.0

def calculate_dynamic_weights(student: Dict, stages: List[FunnelStage], db: Session) -> Dict[str, float]:
    """Calculate dynamic weights based on student characteristics and historical data."""
    # Base weights
    weights = {
        'engagement': 0.4,
        'time': 0.3,
        'progress': 0.3
    }
    
    # Adjust weights based on time in funnel
    if 'created_at' in student:
        days_in_funnel = (datetime.now() - student['created_at']).days
        if days_in_funnel > 90:
            weights['time'] *= 1.2
            weights['engagement'] *= 0.9
    
    # Adjust weights based on stage
    current_stage_index = next(
        (i for i, stage in enumerate(stages) if stage.stage_name.lower() == student['funnel_stage'].lower()),
        -1
    )
    
    if current_stage_index > 0:
        # Later stages get more weight on progress
        progress_weight = min(0.5, 0.3 + (current_stage_index * 0.05))
        weights['progress'] = progress_weight
        weights['engagement'] = (1 - progress_weight) * 0.6
        weights['time'] = (1 - progress_weight) * 0.4
    
    return weights 