from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.routes import router
from data.models import init_db
from data.sample_data import generate_sample_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Student Engagement API",
    description="API for managing student engagement and risk assessment",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database and services on startup."""
    try:
        # Initialize database
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

def setup_database():
    """Initialize the database and generate sample data if needed."""
    # Initialize database
    init_db()

    # Check if we need to generate sample data
    if os.environ.get("GENERATE_SAMPLE_DATA", "true").lower() == "true":
        # Check if database file exists and has data
        db_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'student_engagement.db')
        if not os.path.exists(db_file) or os.path.getsize(db_file) < 10000:  # If DB doesn't exist or is very small
            try:
                print("Generating sample data...")
                generate_sample_data(num_students=100, num_engagements_per_student=10, num_content_items=50)
                print("Sample data generated successfully!")
            except Exception as e:
                print(f"Error generating sample data: {e}")
                print("Continuing with existing data...")
        else:
            print("Database already exists with data. Skipping sample data generation.")
    else:
        print("Sample data generation disabled.")

if __name__ == "__main__":
    # Setup database
    setup_database()

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
