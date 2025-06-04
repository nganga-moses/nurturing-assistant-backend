# Recommendation Tracking

This document describes the recommendation tracking system used to monitor and analyze student interactions with recommendations.

## Models

### RecommendationAction
Tracks individual student interactions with recommendations:
- `student_id`: ID of the student
- `recommendation_id`: ID of the recommendation
- `action_type`: Type of action (e.g., "viewed", "acted", "completed")
- `action_timestamp`: When the action occurred
- `time_to_action`: Time between recommendation and action
- `action_completed`: Whether the action was completed
- `dropoff_point`: Where the student dropped off (if applicable)

### RecommendationFeedbackMetrics
Tracks aggregate metrics for recommendation types:
- `recommendation_type`: Type of recommendation
- `total_shown`: Total number of times shown
- `acted_count`: Number of times acted upon
- `ignored_count`: Number of times ignored
- `untouched_count`: Number of times not interacted with
- `avg_time_to_action`: Average time to action
- `avg_time_to_completion`: Average time to completion
- `completion_rate`: Rate of completion
- `dropoff_rates`: JSON object with dropoff points and rates

## API Endpoints

### Tracking Actions
- `POST /recommendations/{recommendation_id}/track`
  - Tracks a student's action on a recommendation
  - Body: `{"student_id": "...", "action_type": "..."}`

### Tracking Completion
- `POST /recommendations/{recommendation_id}/complete`
  - Tracks whether a student completed the suggested action
  - Body: `{"student_id": "...", "completed": true/false, "dropoff_point": "..."}`

### Getting Metrics
- `GET /recommendations/feedback/metrics`
  - Gets feedback metrics for recommendation types
  - Query params: `recommendation_type` (optional)

### Getting Student Actions
- `GET /recommendations/students/{student_id}/actions`
  - Gets all actions for a specific student

## Database Schema

The recommendation tracking system uses two main tables:

1. `recommendation_actions`
   - Tracks individual student interactions
   - Foreign keys to `student_profiles` and `stored_recommendations`

2. `recommendation_feedback_metrics`
   - Stores aggregate metrics
   - Used for analyzing recommendation effectiveness

## Usage Example

```python
# Track a recommendation action
tracking_service.track_recommendation_action(
    student_id="student123",
    recommendation_id=1,
    action_type="viewed"
)

# Track completion
tracking_service.track_completion(
    student_id="student123",
    recommendation_id=1,
    completed=True
)

# Get feedback metrics
metrics = tracking_service.get_feedback_metrics()
```

## Testing

The system includes comprehensive tests in `tests/unit/api/test_recommendation_tracking.py` covering:
- Action tracking
- Completion recording
- Metric retrieval
- Student action history

## Database Migrations

The system uses Alembic for database migrations:
- Adds `enrollment_agent_id` to student profiles
- Creates `recommendation_actions` table
- Creates `recommendation_feedback_metrics` table 