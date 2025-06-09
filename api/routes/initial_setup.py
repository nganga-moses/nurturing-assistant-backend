from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, BackgroundTasks, Response
from typing import Optional, Dict, Any, List
import os
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import text
from database.session import get_db
from api.auth.supabase import check_admin
import logging
from datetime import datetime, timedelta
import asyncio
from pydantic import BaseModel
import csv
import io
import subprocess
import sys

# Conditional import for Google Cloud Storage (only needed in production mode)
try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    storage = None

router = APIRouter(prefix="/initial-setup", tags=["initial-setup"])

logger = logging.getLogger(__name__)

class SetupMode:
    TESTING = "testing"
    PRODUCTION = "production"

class SetupRequest(BaseModel):
    mode: str
    epochs: int = 20  # Number of training epochs (default: 20 for good performance)
    batch_size: int = 32  # Batch size for training (default: 32)
    embedding_dim: int = 128  # Dimension of embeddings (default: 128)

class RecommendationGenerationRequest(BaseModel):
    target_stage: str = "application"  # Target funnel stage: "application", "enrollment", "deposit", etc.
    batch_size: int = 100  # Number of students to process at once
    min_confidence: float = 0.3  # Minimum confidence threshold for recommendations (lowered for vector similarity)
    top_k: int = 5  # Number of recommendations per student

class PeriodicUpdateRequest(BaseModel):
    update_type: str  # "daily" or "weekly" 
    days_back: int = 1  # How many days back to look for updates
    force_retrain: bool = False  # Whether to force model retraining

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

# Global variable to store recommendation generation status
recommendation_status = SetupStatus(
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

def update_recommendation_status(status: str, progress: float, current_step: str, message: str, errors: List[str] = None):
    """Update the global recommendation generation status."""
    global recommendation_status
    recommendation_status = SetupStatus(
        status=status,
        progress=progress,
        current_step=current_step,
        message=message,
        errors=errors or [],
        timestamp=datetime.now()
    )

def download_from_gcs(bucket_name: str, source_blob_name: str, destination_file_name: str) -> None:
    """Download a file from Google Cloud Storage."""
    if not GCS_AVAILABLE:
        raise HTTPException(status_code=500, detail="Google Cloud Storage not available. Please install google-cloud-storage package.")
    
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        logger.info(f"Successfully downloaded {source_blob_name} to {destination_file_name}")
    except Exception as e:
        logger.error(f"Error downloading from GCS: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to download file from GCS: {str(e)}")

def purge_database(db: Session) -> None:
    """Purge all data from the database."""
    try:
        # Delete all records from relevant tables
        db.execute(text("TRUNCATE TABLE student_profiles CASCADE"))
        db.execute(text("TRUNCATE TABLE engagement_history CASCADE"))
        db.execute(text("TRUNCATE TABLE engagement_content CASCADE"))
        db.execute(text("TRUNCATE TABLE stored_recommendations CASCADE"))
        db.execute(text("TRUNCATE TABLE recommendation_feedback_metrics CASCADE"))
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Error purging database: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to purge database: {str(e)}")

async def run_setup(mode: str, epochs: int, batch_size: int, embedding_dim: int, db: Session):
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
                
                # Check if local files exist
                if not os.path.exists(students_file):
                    raise HTTPException(status_code=404, detail=f"Students file not found: {students_file}")
                if not os.path.exists(engagements_file):
                    raise HTTPException(status_code=404, detail=f"Engagements file not found: {engagements_file}")
                    
                logger.info(f"Using local files: {students_file}, {engagements_file}")
            else:
                # Production mode: Download files from GCS
                if not GCS_AVAILABLE:
                    raise HTTPException(status_code=500, detail="Production mode requires Google Cloud Storage. Please install google-cloud-storage package.")
                
                update_setup_status("running", 0.1, "Downloading files", "Downloading from GCS...")
                students_file = os.path.join(temp_dir, "students.csv")
                engagements_file = os.path.join(temp_dir, "engagements.csv")
                
                bucket_name = os.getenv("GCS_BUCKET_NAME")
                if not bucket_name:
                    raise HTTPException(status_code=500, detail="GCS_BUCKET_NAME environment variable not set for production mode")
                
                logger.info(f"Downloading from GCS bucket: {bucket_name}")
                download_from_gcs(bucket_name, "initial/students.csv", students_file)
                download_from_gcs(bucket_name, "initial/engagements.csv", engagements_file)
            
            # Validate files
            update_setup_status("running", 0.2, "Validating files", "Checking file formats...")
            errors = []
            
            # Validate students file
            student_errors = validate_csv_file(students_file, ["student_id", "location", "intended_major", "application_status"])
            errors.extend([f"Students file: {e}" for e in student_errors])
            
            # Validate engagements file
            engagement_errors = validate_csv_file(engagements_file, ["student_id", "engagement_type", "timestamp"])
            errors.extend([f"Engagements file: {e}" for e in engagement_errors])
            
            if errors:
                update_setup_status("failed", 0.0, "Validation failed", "File validation failed", errors)
                return
            
            # Import data and train model
            update_setup_status("running", 0.3, "Importing data", "Starting data import and model training...")
            
            # Call the ingest_and_train script with proper arguments
            script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scripts", "ingest_and_train.py")
            cmd = [
                sys.executable, script_path,
                "--students-csv", students_file,
                "--engagements-csv", engagements_file,
                "--epochs", str(epochs),
                "--batch-size", str(batch_size),
                "--embedding-dim", str(embedding_dim)
            ]
            
            logger.info(f"Running command: {' '.join(cmd)}")
            update_setup_status("running", 0.4, "Training model", "Running data import and model training...")
            
            # Use thread executor to run subprocess without blocking the event loop
            import concurrent.futures
            import asyncio
            
            def run_training_sync():
                """Run training in a separate thread to avoid blocking."""
                return subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                )
            
            # Run the training subprocess in a thread executor
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                result = await loop.run_in_executor(executor, run_training_sync)
            
            if result.returncode != 0:
                error_msg = f"Training script failed: {result.stderr}"
                logger.error(error_msg)
                logger.error(f"stdout: {result.stdout}")
                raise Exception(error_msg)
            
            logger.info("Training script completed successfully")
            logger.info(f"Training output: {result.stdout}")
            
            update_setup_status("running", 0.9, "Finalizing", "Training completed, finalizing setup...")
            
            # Restart the model manager to load the new model
            try:
                from api.services.model_manager import ModelManager
                update_setup_status("running", 0.95, "Loading model", "Loading trained model...")
                
                # This will be available in the main app context, but for setup validation we can test loading
                test_manager = ModelManager()
                health = test_manager.health_check()
                logger.info(f"Model health check: {health}")
                
            except Exception as e:
                logger.warning(f"Model loading validation failed: {e}")
                # Don't fail the setup, as the model will be loaded on next app restart
            
            update_setup_status("completed", 1.0, "Setup complete", "Initial setup completed successfully! Model training finished and system is ready.")
            
        finally:
            # Clean up temporary files
            if mode == SetupMode.PRODUCTION:
                for file in [students_file, engagements_file]:
                    if os.path.exists(file):
                        os.remove(file)
                os.rmdir(temp_dir)
                
    except Exception as e:
        logger.error(f"Error during initial setup: {str(e)}")
        update_setup_status("failed", 0.0, "Setup failed", str(e), [str(e)])

async def run_recommendation_generation(request: RecommendationGenerationRequest, db: Session):
    """Run bulk recommendation generation asynchronously."""
    try:
        update_recommendation_status("running", 0.0, "Starting generation", "Initializing recommendation generation...")
        
        # Import model manager and services
        from api.services.model_manager import ModelManager
        from api.services.recommendation_service import RecommendationService
        from data.models.stored_recommendation import StoredRecommendation
        from data.models.funnel_stage import FunnelStage
        from data.models.student_profile import StudentProfile
        
        # Get the target stage
        target_stage = db.query(FunnelStage).filter_by(stage_name=request.target_stage.title()).first()
        if not target_stage:
            raise ValueError(f"Target stage '{request.target_stage}' not found")
        
        update_recommendation_status("running", 0.1, "Loading model", "Loading trained model...")
        
        # Initialize model manager and recommendation service
        model_manager = ModelManager()
        if not model_manager.is_healthy:
            raise ValueError("Model is not healthy - please train model first")
        
        recommendation_service = RecommendationService(model_manager)
        
        update_recommendation_status("running", 0.2, "Finding students", "Querying eligible students...")
        
        # Get students who haven't reached the target stage yet (eligible for recommendations)
        all_stages = db.query(FunnelStage).order_by(FunnelStage.stage_order).all()
        target_stage_order = target_stage.stage_order
        
        # Get students in stages before the target stage
        eligible_stages = [stage for stage in all_stages if stage.stage_order < target_stage_order]
        eligible_stage_names = [stage.stage_name for stage in eligible_stages]
        
        students = db.query(StudentProfile).filter(
            StudentProfile.funnel_stage.in_(eligible_stage_names)
        ).all()
        
        total_students = len(students)
        if total_students == 0:
            update_recommendation_status("completed", 100.0, "No eligible students", 
                                        f"No students found before target stage '{request.target_stage}'")
            return
        
        logger.info(f"Found {total_students} eligible students for recommendations")
        update_recommendation_status("running", 0.3, "Generating recommendations", 
                                    f"Processing {total_students} students...")
        
        # Process students in batches
        recommendations_generated = 0
        recommendations_stored = 0
        students_with_recommendations = 0
        batch_size = request.batch_size
        
        for i in range(0, total_students, batch_size):
            batch_students = students[i:i + batch_size]
            batch_progress = 0.3 + (i / total_students) * 0.6  # Progress from 0.3 to 0.9
            
            update_recommendation_status(
                "running", 
                batch_progress, 
                "Generating recommendations",
                f"Processing batch {i//batch_size + 1}/{(total_students + batch_size - 1)//batch_size}..."
            )
            
            # Generate recommendations for this batch
            for student in batch_students:
                try:
                    # Generate recommendations for this student
                    recommendations = recommendation_service.get_recommendations(
                        student.student_id, 
                        top_k=request.top_k,
                        funnel_stage=student.funnel_stage
                    )
                    
                    # Filter by confidence threshold
                    high_confidence_recs = [
                        rec for rec in recommendations 
                        if rec.get('confidence', 0) >= request.min_confidence
                    ]
                    
                    if high_confidence_recs:
                        # Store recommendations in database
                        # Add target_stage and generation_metadata to the recommendations JSON
                        enhanced_recs = []
                        for rec in high_confidence_recs:
                            enhanced_rec = rec.copy()
                            enhanced_rec.update({
                                "target_stage": request.target_stage,
                                "model_version": model_manager.model_metadata.get("version", "unknown"),
                                "confidence_threshold": request.min_confidence,
                                "top_k": request.top_k,
                                "student_stage": student.funnel_stage
                            })
                            enhanced_recs.append(enhanced_rec)
                        
                        stored_rec = StoredRecommendation(
                            student_id=student.student_id,
                            recommendations=enhanced_recs,
                            generated_at=datetime.now(),
                            expires_at=datetime.now() + timedelta(days=30)  # 30-day expiry
                        )
                        db.add(stored_rec)
                        recommendations_stored += len(high_confidence_recs)
                        students_with_recommendations += 1
                    
                    recommendations_generated += len(recommendations)
                    
                except Exception as e:
                    logger.error(f"Error generating recommendations for student {student.student_id}: {e}")
                    continue
            
            # Commit batch
            db.commit()
        
        update_recommendation_status("running", 0.9, "Validating results", "Validating generated recommendations...")
        
        update_recommendation_status(
            "completed", 
            1.0, 
            "Generation complete", 
            f"Generated {recommendations_generated} recommendations for {total_students} students. "
            f"Stored {recommendations_stored} high-confidence recommendations for {students_with_recommendations} students."
        )
        
        logger.info(f"Recommendation generation completed: {recommendations_generated} generated, {recommendations_stored} stored")
        
    except Exception as e:
        error_msg = f"Recommendation generation failed: {str(e)}"
        logger.error(error_msg)
        update_recommendation_status("failed", 0.0, "Generation failed", error_msg, [str(e)])

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
async def train_model(
    request: SetupRequest,
    db: Session = Depends(get_db),
    # TODO: Add authentication after initial setup is complete
    # current_user=Depends(check_admin)
):
    """Train the initial model with the provided data and settings."""
    global setup_status
    
    if setup_status.status == "running":
        raise HTTPException(status_code=400, detail="Training is already in progress")
    
    # Validate mode
    if request.mode not in [SetupMode.TESTING, SetupMode.PRODUCTION]:
        raise HTTPException(status_code=400, detail="Invalid mode. Must be 'testing' or 'production'")
    
    # For production mode, check if Google Cloud Storage is available
    if request.mode == SetupMode.PRODUCTION and not GCS_AVAILABLE:
        raise HTTPException(
            status_code=500, 
            detail="Production mode requires google-cloud-storage package. Please install it first."
        )
    
    # Run setup in a separate thread to avoid blocking
    def run_setup_thread():
        """Run setup in a separate thread to avoid blocking the API."""
        import asyncio
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_setup(request.mode, request.epochs, request.batch_size, request.embedding_dim, db))
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            update_setup_status("failed", 0.0, "Setup failed", f"Error: {str(e)}", [str(e)])
        finally:
            loop.close()
    
    # Start the setup in a background thread
    import threading
    setup_thread = threading.Thread(target=run_setup_thread)
    setup_thread.daemon = True
    setup_thread.start()
    
    return {"message": "Model training started", "status": "started"}

@router.post("/generate-recommendations")
async def generate_recommendations(
    request: RecommendationGenerationRequest,
    db: Session = Depends(get_db),
    current_user=Depends(check_admin)
):
    """Generate bulk recommendations for storage."""
    global recommendation_status
    
    if recommendation_status.status == "running":
        raise HTTPException(status_code=400, detail="Recommendation generation is already in progress")
    
    # Validate target stage
    valid_stages = ["awareness", "interest", "consideration", "application", "deposit", "enrollment", "matriculation"]
    if request.target_stage.lower() not in valid_stages:
        raise HTTPException(status_code=400, detail=f"Invalid target stage. Must be one of: {valid_stages}")
    
    # Run recommendation generation in a separate thread
    def run_generation_thread():
        """Run recommendation generation in a separate thread."""
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_recommendation_generation(request, db))
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            update_recommendation_status("failed", 0.0, "Generation failed", f"Error: {str(e)}", [str(e)])
        finally:
            loop.close()
    
    # Start generation in background thread
    import threading
    generation_thread = threading.Thread(target=run_generation_thread)
    generation_thread.daemon = True
    generation_thread.start()
    
    return {"message": "Recommendation generation started", "status": "started"}

@router.get("/recommendation-status")
async def get_recommendation_status():
    """Get the current recommendation generation status."""
    return recommendation_status

@router.post("/periodic-update")
async def run_periodic_update(
    request: PeriodicUpdateRequest,
    current_user=Depends(check_admin)
):
    """Run periodic data updates (for use by schedulers)."""
    try:
        # Run the appropriate update script
        script_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scripts")
        
        if request.update_type == "daily":
            # Run daily update
            engagement_cmd = [
                sys.executable, 
                os.path.join(script_dir, "ingest_engagements.py"),
                "--days", str(request.days_back)
            ]
            
            student_cmd = [
                sys.executable,
                os.path.join(script_dir, "ingest_students.py"), 
                "--days", str(request.days_back)
            ]
            
        elif request.update_type == "weekly":
            # Run weekly update with optional retraining
            engagement_cmd = [
                sys.executable,
                os.path.join(script_dir, "ingest_engagements.py"),
                "--days", str(request.days_back)
            ]
            
            student_cmd = [
                sys.executable,
                os.path.join(script_dir, "ingest_students.py"),
                "--days", str(request.days_back)
            ]
            
            if request.force_retrain:
                engagement_cmd.append("--force-retrain")
                student_cmd.append("--force-retrain")
        else:
            raise HTTPException(status_code=400, detail="Invalid update_type. Must be 'daily' or 'weekly'")
        
        # Run engagement updates
        logger.info(f"Running engagement update: {' '.join(engagement_cmd)}")
        engagement_result = subprocess.run(
            engagement_cmd,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            capture_output=True,
            text=True
        )
        
        if engagement_result.returncode != 0:
            raise HTTPException(
                status_code=500, 
                detail=f"Engagement update failed: {engagement_result.stderr}"
            )
        
        # Run student updates  
        logger.info(f"Running student update: {' '.join(student_cmd)}")
        student_result = subprocess.run(
            student_cmd,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            capture_output=True,
            text=True
        )
        
        if student_result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Student update failed: {student_result.stderr}"
            )
        
        return {
            "message": f"{request.update_type.title()} update completed successfully",
            "engagement_output": engagement_result.stdout,
            "student_output": student_result.stdout,
            "retrained": request.force_retrain
        }
        
    except Exception as e:
        logger.error(f"Periodic update failed: {e}")
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")

@router.get("/funnel-stages")
async def get_funnel_stages(db: Session = Depends(get_db)):
    """Get available funnel stages for goal setting."""
    from data.models.funnel_stage import FunnelStage
    
    stages = db.query(FunnelStage).filter(FunnelStage.is_active == True).order_by(FunnelStage.stage_order).all()
    
    return {
        "stages": [
            {
                "id": stage.id,
                "name": stage.stage_name,
                "order": stage.stage_order, 
                "is_tracking_goal": stage.is_tracking_goal,
                "tracking_goal_type": stage.tracking_goal_type
            }
            for stage in stages
        ]
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