from fastapi import APIRouter

router = APIRouter(prefix="/docs", tags=["documentation"])

@router.get("/")
def get_docs_links():
    """Get API documentation and help links."""
    return {
        "api_docs": "/api/docs",
        "redoc": "/api/redoc",
        "admin_help": "/docs/admin-help",
        "developer_help": "/docs/dev-help",
        "api_reference": "/docs/api-reference"
    }

@router.get("/admin-help")
def get_admin_help():
    """Get admin help content (markdown)."""
    return {
        "content": """
# Admin Guide

## Authentication & User Management
- **Supabase Auth**
  - Users sign up and log in using email and password (handled by Supabase)
  - Password reset and email verification are available via the API
  - Roles and permissions are managed in the backend database
- **Endpoints**
  - `/auth/signup`: Register a new user
  - `/auth/login`: Log in and receive a JWT
  - `/auth/password-reset-request`: Request a password reset email
  - `/auth/password-reset-confirm`: Confirm password reset with token
  - `/auth/verify-email-request`: Resend verification email
  - `/auth/verify-email-confirm`: Confirm email verification

## User Settings
- Configure notification preferences
- Set default views and filters
- Manage personal dashboard layout

## Data Management
- **Import Process**
  1. Prepare CSV files with required columns
  2. Use the import endpoint or admin UI
  3. Review validation results
  4. Fix any errors in the data
  5. Confirm import
- **Data Validation Rules**
  - Student IDs must be unique
  - Required fields: name, email, status
  - Date formats: YYYY-MM-DD
  - Numeric fields: no negative values

## System Configuration
- **Notification Settings**
  - Email notifications
  - In-app alerts
  - Daily/weekly reports
- **Recommendation Settings**
  - Frequency of recommendations
  - Risk thresholds
  - Engagement types to include

## Troubleshooting
- **Common Issues**
  - Import failures: Check file format and required fields
  - Missing data: Verify data source and mapping
  - Performance issues: Check database indexes
- **Error Logs**
  - View error logs at /notifications/error-logs
  - Filter by error type and date
  - Export logs for analysis
"""
    }

@router.get("/dev-help")
def get_developer_help():
    """Get developer help content (markdown)."""
    return {
        "content": """
# Developer Guide

## Authentication (Supabase)
- **Supabase Auth**
  - Handles user signup, login, password reset, and email verification
  - JWT tokens are issued by Supabase and validated in the backend
  - Use `/auth/signup`, `/auth/login`, `/auth/password-reset-request`, `/auth/password-reset-confirm`, `/auth/verify-email-request`, `/auth/verify-email-confirm`
- **Role Management**
  - User roles are stored in the backend database
  - Role-based access enforced in API endpoints

## System Architecture
- **Backend**
  - FastAPI application
  - PostgreSQL database
  - Alembic migrations
  - SQLAlchemy ORM
- **Frontend**
  - React + Vite
  - Tailwind CSS
  - Supabase Auth

## Adding New Features
- **Engagement Types**
  1. Add new type to `EngagementType` model
  2. Create migration
  3. Update admin UI
  4. Add validation rules
  5. Test with sample data
- **Integrations**
  1. Create new integration config
  2. Implement data mapping
  3. Add error handling
  4. Set up webhooks
  5. Test connection

## API Development
- **Authentication**
  - Supabase JWT tokens
  - Role-based access
  - API key management
- **Endpoints**
  - Follow RESTful conventions
  - Include input validation
  - Add proper error handling
  - Document with OpenAPI

## Database
- **Models**
  - Use SQLAlchemy models
  - Include proper indexes
  - Add foreign key constraints
  - Implement soft deletes
- **Migrations**
  - Create with Alembic
  - Test both up and down
  - Include data migrations
  - Document changes

## Testing
- **Unit Tests**
  - Test models and services
  - Mock external services
  - Cover edge cases
- **Integration Tests**
  - Test API endpoints
  - Verify database operations
  - Check authentication
"""
    }

@router.get("/api-reference")
def get_api_reference():
    """Get API reference documentation."""
    return {
        "content": """
# API Reference

## Authentication (Supabase)
- **Sign Up**: POST /auth/signup
- **Login**: POST /auth/login
- **Refresh**: POST /auth/refresh
- **Logout**: POST /auth/logout
- **Password Reset Request**: POST /auth/password-reset-request
- **Password Reset Confirm**: POST /auth/password-reset-confirm
- **Verify Email Request**: POST /auth/verify-email-request
- **Verify Email Confirm**: POST /auth/verify-email-confirm

## Users & Roles
- **List Roles**: GET /roles
- **Get Users by Role**: GET /roles/users/{role_name}
- **Assign Role**: POST /roles/assign

## Settings
- **Global Settings**: GET /settings
- **User Settings**: GET /settings/user/{user_id}
- **Update Settings**: POST /settings/user/{user_id}

## Students
- **List Students**: GET /students
- **Get Student**: GET /students/{student_id}
- **Update Student**: PUT /students/{student_id}

## Engagements
- **List Engagements**: GET /engagements
- **Create Engagement**: POST /engagements
- **Update Engagement**: PUT /engagements/{id}

## Recommendations
- **Get Recommendations**: POST /recommendations
- **Update Settings**: PUT /recommendation-settings

## Reports
- **Dashboard Stats**: GET /dashboard/stats
- **Performance Metrics**: GET /reports/performance
- **Error Logs**: GET /notifications/error-logs
"""
    } 