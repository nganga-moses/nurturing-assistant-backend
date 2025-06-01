# Data Models Directory

This directory contains data models and schemas for the application.

## Files
- `models.py`: SQLAlchemy models for database tables
- `schemas.py`: Pydantic schemas for API requests/responses

## Models
### StudentProfile
- Student demographic information
- Application status
- Funnel stage
- Risk and likelihood scores

### EngagementHistory
- Student engagement records
- Engagement types and responses
- Timestamps and metrics

### EngagementContent
- Content metadata
- Content types and formats
- Engagement rules

## Schemas
### Request Schemas
- StudentProfileCreate
- StudentProfileUpdate
- EngagementCreate
- EngagementUpdate
- ContentCreate
- ContentUpdate

### Response Schemas
- StudentProfileResponse
- EngagementResponse
- ContentResponse
- DashboardStatsResponse

## Usage
These models and schemas are used by:
1. Database operations
2. API endpoints
3. Data validation
4. Type checking 