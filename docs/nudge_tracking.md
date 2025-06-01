# Nudge Tracking System

## Overview
The nudge tracking system provides a comprehensive framework for monitoring and analyzing student interactions with engagement nudges. It enables data-driven decision making by tracking how students respond to different types of nudges and using this feedback to improve future recommendations.

## Components

### 1. Database Models

#### NudgeAction
Tracks individual student interactions with nudges:
- `student_id`: Links to the student profile
- `nudge_id`: Links to the stored recommendation
- `action_type`: Type of interaction ("acted", "ignored", "untouched")
- `action_timestamp`: When the action occurred
- `time_to_action`: Time taken to respond (in seconds)
- `action_completed`: Whether the suggested action was completed
- `dropoff_point`: Where the student dropped off if not completed

#### NudgeFeedbackMetrics
Aggregates metrics for different types of nudges:
- `nudge_type`: Type of recommendation
- `total_shown`: Total number of times shown
- `acted_count`: Number of times acted upon
- `ignored_count`: Number of times ignored
- `untouched_count`: Number of times left untouched
- `avg_time_to_action`: Average time to respond
- `completion_rate`: Rate of action completion
- `dropoff_rates`: JSON tracking where students drop off

### 2. Tracking Service

The `NudgeTrackingService` provides methods for:
- Tracking nudge actions (`track_nudge_action`)
- Recording action completion (`track_completion`)
- Updating feedback metrics
- Retrieving metrics and student actions

### 3. Recommendation Integration

The recommendation system now considers feedback when generating suggestions:
- Base scores are adjusted based on historical engagement
- Completion rates influence future recommendations
- Metrics are included in recommendation responses

### 4. Reporting Endpoints

Three main reporting endpoints:
1. `/api/reports/nudge-performance`: Overall nudge effectiveness
2. `/api/reports/agent-performance`: Enrollment agent metrics
3. `/api/reports/student-engagement`: Individual student tracking

## Usage Examples

### Tracking a Nudge Action
```python
tracking_service.track_nudge_action(
    student_id="S001",
    nudge_id=123,
    action_type="acted"
)
```

### Recording Completion
```python
tracking_service.track_completion(
    student_id="S001",
    nudge_id=123,
    completed=True
)
```

### Getting Feedback Metrics
```python
metrics = tracking_service.get_feedback_metrics("application_reminder")
```

## Testing

The system includes comprehensive tests in `tests/unit/api/test_nudge_tracking.py` covering:
- Action tracking
- Completion recording
- Metric retrieval
- Student action history

## Database Migrations

The system uses Alembic for database migrations:
- Adds `enrollment_agent_id` to student profiles
- Creates `nudge_actions` table
- Creates `nudge_feedback_metrics` table 