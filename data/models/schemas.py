from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime


class DemographicFeatures(BaseModel):
    location: str
    age_range: str
    intended_major: str
    academic_scores: Dict[str, float]


class EngagementMetrics(BaseModel):
    """Type-specific metrics for different engagement types"""
    metrics: Dict[str, Any]


class StudentProfile(BaseModel):
    student_id: str
    demographic_features: DemographicFeatures
    application_status: str
    funnel_stage: str
    first_interaction_date: datetime
    last_interaction_date: datetime
    interaction_count: int
    application_likelihood_score: float
    dropout_risk_score: float
    last_recommended_engagement_id: Optional[str] = None
    last_recommended_engagement_date: Optional[datetime] = None

    class Config:
        schema_extra = {
            "example": {
                "student_id": "S12345",
                "demographic_features": {
                    "location": "California",
                    "age_range": "18-24",
                    "intended_major": "Computer Science",
                    "academic_scores": {"GPA": 3.8, "SAT": 1450}
                },
                "application_status": "In Progress",
                "funnel_stage": "Consideration",
                "first_interaction_date": "2025-01-15T10:30:00",
                "last_interaction_date": "2025-04-20T14:45:00",
                "interaction_count": 12,
                "application_likelihood_score": 0.75,
                "dropout_risk_score": 0.25,
                "last_recommended_engagement_id": "E789",
                "last_recommended_engagement_date": "2025-04-15T09:00:00"
            }
        }


class EngagementHistory(BaseModel):
    engagement_id: str
    student_id: str
    engagement_type: str
    engagement_content_id: str
    timestamp: datetime
    engagement_response: str
    engagement_metrics: EngagementMetrics
    funnel_stage_before: str
    funnel_stage_after: str

    class Config:
        schema_extra = {
            "example": {
                "engagement_id": "E789",
                "student_id": "S12345",
                "engagement_type": "Email",
                "engagement_content_id": "C456",
                "timestamp": "2025-04-15T09:00:00",
                "engagement_response": "opened",
                "engagement_metrics": {
                    "metrics": {
                        "open_time": "2025-04-15T09:05:23",
                        "click_through": True,
                        "time_spent": 45
                    }
                },
                "funnel_stage_before": "Interest",
                "funnel_stage_after": "Consideration"
            }
        }


class EngagementContent(BaseModel):
    content_id: str
    engagement_type: str
    content_category: str
    content_description: str
    content_features: Dict[str, Any]
    success_rate: float
    target_funnel_stage: str
    appropriate_for_risk_level: str

    class Config:
        schema_extra = {
            "example": {
                "content_id": "C456",
                "engagement_type": "Email",
                "content_category": "Program Information",
                "content_description": "Computer Science Department Overview",
                "content_features": {
                    "topics": ["curriculum", "faculty", "research"],
                    "embedding": [0.1, 0.2, 0.3, 0.4]
                },
                "success_rate": 0.68,
                "target_funnel_stage": "Interest",
                "appropriate_for_risk_level": "medium"
            }
        }


class RecommendationRequest(BaseModel):
    student_id: str
    top_k: int = 5
    funnel_stage: Optional[str] = None
    risk_level: Optional[str] = None


class RecommendationResponse(BaseModel):
    student_id: str
    recommendations: List[Dict[str, Any]]
    timestamp: datetime = Field(default_factory=datetime.now)


class LikelihoodRequest(BaseModel):
    student_id: str


class LikelihoodResponse(BaseModel):
    student_id: str
    application_likelihood: float
    timestamp: datetime = Field(default_factory=datetime.now)


class RiskAssessmentRequest(BaseModel):
    student_id: str


class RiskAssessmentResponse(BaseModel):
    student_id: str
    risk_category: str
    risk_score: float
    timestamp: datetime = Field(default_factory=datetime.now)


class AtRiskStudentsRequest(BaseModel):
    risk_threshold: float = 0.7


class AtRiskStudentsResponse(BaseModel):
    students: List[Dict[str, Any]]
    count: int
    timestamp: datetime = Field(default_factory=datetime.now)


class BulkActionRequest(BaseModel):
    action: str
    segment: str


class BulkActionPreviewResponse(BaseModel):
    students: List[Dict[str, Any]]
    count: int
    action_details: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


class BulkActionApplyResponse(BaseModel):
    success: bool
    students_affected: int
    action_details: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


class DashboardStatsResponse(BaseModel):
    total_students: int
    application_rate: float
    at_risk_count: int
    stage_distribution: Dict[str, int]
    timestamp: datetime = Field(default_factory=datetime.now)
