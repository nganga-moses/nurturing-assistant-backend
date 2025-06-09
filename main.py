from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.routes import api_router
from api.services.model_manager import ModelManager

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
app.include_router(api_router)

# Initialize database and model on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database and services on startup."""
    try:
        # Initialize model manager
        logger.info("Initializing model manager...")
        model_manager = ModelManager()
        app.state.model_manager = model_manager
        
        # Log model status
        health = model_manager.health_check()
        logger.info(f"Model manager initialized - Status: {health['status']}")
        
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
