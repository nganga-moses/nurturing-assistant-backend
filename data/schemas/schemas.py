from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class RecommendationSettingsRequest(BaseModel):
    """Request model for updating recommendation settings."""
    mode: Optional[str] = Field(None, description="Mode of operation: 'scheduled' or 'realtime'")
    schedule_type: Optional[str] = Field(None, description="Type of schedule: 'daily' or 'weekly'")
    schedule_day: Optional[str] = Field(None, description="Day of week for weekly schedule")
    schedule_time: Optional[str] = Field(None, description="Time of day in 24-hour format (HH:MM)")
    batch_size: Optional[int] = Field(None, description="Number of recommendations to generate per batch")
    is_active: Optional[bool] = Field(None, description="Whether the schedule is active")

class RecommendationSettingsResponse(BaseModel):
    """Response model for recommendation settings."""
    id: int
    mode: str
    schedule_type: Optional[str]
    schedule_day: Optional[str]
    schedule_time: str
    batch_size: int
    last_run: Optional[datetime]
    next_run: Optional[datetime]
    is_active: bool
    created_at: datetime
    updated_at: datetime 