from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, BackgroundTasks, Response
from typing import Optional, Dict, Any, List
import os
import pandas as pd
from google.cloud import storage
from sqlalchemy.orm import Session
from database.session import get_db
from api.auth.supabase import check_admin
import logging
from datetime import datetime
import asyncio
from pydantic import BaseModel
import csv
import io

router = APIRouter(prefix="/initial-setup", tags=["initial-setup"])

logger = logging.getLogger(__name__)

class SetupMode:
    TESTING = "testing"
    PRODUCTION = "production"

class SetupStatus(BaseModel):
    status: str
    progress: float
    current_step: str
    message: str
    errors: List[str] = []
    timestamp: datetime

# Global variable to store setup status
setup_status = SetupStatus(
    status="idle",
    progress=0.0,
    current_step="",
    message="",
    timestamp=datetime.now()
)

def validate_csv_file(file_path: str, required_columns: List[str]) -> List[str]:
    """Validate a CSV file for required columns and data types."""
    errors = []
    try:
        df = pd.read_csv(file_path)
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {', '.join(missing_columns)}")
        return errors
    except Exception as e:
        return [f"Error reading CSV file: {str(e)}"]

def update_setup_status(status: str, progress: float, current_step: str, message: str, errors: List[str] = None):
    """Update the global setup status."""
    global setup_status
    setup_status = SetupStatus(
        status=status,
        progress=progress,
        current_step=current_step,
        message=message,
        errors=errors or [],
        timestamp=datetime.now()
    )

def download_from_gcs(bucket_name: str, source_blob_name: str, destination_file_name: str) -> None:
    """Download a file from Google Cloud Storage."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
    except Exception as e:
        logger.error(f"Error downloading from GCS: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to download file from GCS: {str(e)}")

def purge_database(db: Session) -> None:
    """Purge all data from the database."""
    try:
        # Delete all records from relevant tables
        db.execute("TRUNCATE TABLE student_profiles CASCADE")
        db.execute("TRUNCATE TABLE engagement_history CASCADE")
        db.execute("TRUNCATE TABLE engagement_content CASCADE")
        db.execute("TRUNCATE TABLE stored_recommendations CASCADE")
        db.execute("TRUNCATE TABLE recommendation_feedback_metrics CASCADE")
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Error purging database: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to purge database: {str(e)}")

async def run_setup(mode: str, db: Session):
    """Run the setup process asynchronously."""
    try:
        update_setup_status("running", 0.0, "Starting setup", "Initializing...")
        
        # Create temporary directory for files
        temp_dir = "temp_import"
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            if mode == SetupMode.TESTING:
                # Use local files
                students_file = "data/initial/students.csv"
                engagements_file = "data/initial/engagements.csv"
                content_file = "data/initial/content.csv"
            else:
                # Download files from GCS
                update_setup_status("running", 0.1, "Downloading files", "Downloading from GCS...")
                students_file = os.path.join(temp_dir, "students.csv")
                engagements_file = os.path.join(temp_dir, "engagements.csv")
                content_file = os.path.join(temp_dir, "content.csv")
                
                bucket_name = os.getenv("GCS_BUCKET_NAME")
                if not bucket_name:
                    raise HTTPException(status_code=500, detail="GCS_BUCKET_NAME environment variable not set")
                
                download_from_gcs(bucket_name, "initial/students.csv", students_file)
                download_from_gcs(bucket_name, "initial/engagements.csv", engagements_file)
                download_from_gcs(bucket_name, "initial/content.csv", content_file)
            
            # Validate files
            update_setup_status("running", 0.2, "Validating files", "Checking file formats...")
            errors = []
            
            # Validate students file
            student_errors = validate_csv_file(students_file, ["student_id", "demographic_features", "application_status"])
            errors.extend([f"Students file: {e}" for e in student_errors])
            
            # Validate engagements file
            engagement_errors = validate_csv_file(engagements_file, ["student_id", "engagement_type", "timestamp"])
            errors.extend([f"Engagements file: {e}" for e in engagement_errors])
            
            # Validate content file
            content_errors = validate_csv_file(content_file, ["content_id", "content_type", "metadata"])
            errors.extend([f"Content file: {e}" for e in content_errors])
            
            if errors:
                update_setup_status("failed", 0.0, "Validation failed", "File validation failed", errors)
                return
            
            # Import data and train model
            update_setup_status("running", 0.3, "Importing data", "Importing data into database...")
            from scripts.ingest_and_train import main as ingest_and_train
            ingest_and_train(
                students_csv=students_file,
                engagements_csv=engagements_file,
                content_csv=content_file
            )
            
            update_setup_status("completed", 1.0, "Setup complete", "Initial setup completed successfully")
            
        finally:
            # Clean up temporary files
            if mode == SetupMode.PRODUCTION:
                for file in [students_file, engagements_file, content_file]:
                    if os.path.exists(file):
                        os.remove(file)
                os.rmdir(temp_dir)
                
    except Exception as e:
        logger.error(f"Error during initial setup: {str(e)}")
        update_setup_status("failed", 0.0, "Setup failed", str(e), [str(e)])

@router.post("/purge")
async def purge_data(
    db: Session = Depends(get_db),
    current_user=Depends(check_admin)
):
    """Purge all data from the database."""
    try:
        update_setup_status("running", 0.0, "Purging database", "Starting database purge...")
        purge_database(db)
        update_setup_status("completed", 1.0, "Purge complete", "Database purged successfully")
        return {"status": "success", "message": "Database purged successfully"}
    except Exception as e:
        update_setup_status("failed", 0.0, "Purge failed", str(e), [str(e)])
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/setup")
async def initial_setup(
    mode: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user=Depends(check_admin)
):
    """
    Perform initial setup of the system.
    
    Args:
        mode: Either "testing" or "production"
    """
    if mode not in [SetupMode.TESTING, SetupMode.PRODUCTION]:
        raise HTTPException(status_code=400, detail="Invalid mode. Must be 'testing' or 'production'")
    
    # Start setup process in background
    background_tasks.add_task(run_setup, mode, db)
    
    return {
        "status": "started",
        "message": "Initial setup process started",
        "timestamp": datetime.now().isoformat()
    }

@router.get("/status")
async def get_setup_status():
    """Get the current status of the setup process."""
    return setup_status 

@router.get("/templates/{template_type}")
async def get_csv_template(template_type: str):
    """
    Get a sample CSV template for students, engagements, or recruiters.
    """
    output = io.StringIO()
    writer = csv.writer(output)
    
    if template_type == "students":
        writer.writerow([
            "student_id",
            "first_name",
            "last_name",
            "birthdate",
            "recruiter_id",
            "location",
            "intended_major",
            "application_status",
            "funnel_stage",
            "first_interaction_date",
            "last_interaction_date",
            "interaction_count",
            "gpa",
            "sat_score",
            "act_score"
        ])
        writer.writerow([
            "S12345",
            "Jane",
            "Doe",
            "2005-04-12",
            "R12345",
            "California",
            "Computer Science",
            "In Progress",
            "Consideration",
            "2024-01-15T10:30:00",
            "2024-04-20T14:45:00",
            "12",
            "3.8",
            "1450",
            "32"
        ])
        filename = "student_template.csv"
    elif template_type == "engagements":
        writer.writerow([
            "engagement_id",
            "student_id",
            "engagement_type",
            "engagement_content_id",
            "timestamp",
            "engagement_response",
            "open_time",
            "click_through",
            "time_spent",
            "funnel_stage_before",
            "funnel_stage_after"
        ])
        writer.writerow([
            "E789",
            "S12345",
            "Email",
            "C456",
            "2024-04-15T09:00:00",
            "opened",
            "2024-04-15T09:05:23",
            "TRUE",
            "45",
            "Interest",
            "Consideration"
        ])
        filename = "engagement_template.csv"
    elif template_type == "recruiters":
        writer.writerow([
            "first_name",
            "last_name",
            "email",
            "crm_user_id",
            "role"
        ])
        writer.writerow([
            "John",
            "Doe",
            "john.doe@university.edu",
            "R12345",
            "recruiter"
        ])
        filename = "recruiter_template.csv"
    else:
        raise HTTPException(status_code=400, detail="Invalid template type. Must be 'students', 'engagements', or 'recruiters'")
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    ) 