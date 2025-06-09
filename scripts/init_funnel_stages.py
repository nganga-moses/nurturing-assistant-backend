import os
import sys
import logging

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_dir)

from sqlalchemy.orm import Session
from data.models.funnel_stage import FunnelStage, TrackingGoalType
from database.session import SessionLocal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_funnel_stages():
    """Initialize default funnel stages."""
    db = SessionLocal()
    try:
        # Check if we already have stages
        existing_stages = db.query(FunnelStage).count()
        if existing_stages > 0:
            logger.info("Funnel stages already exist. Skipping initialization.")
            return

        # Create default stages
        default_stages = [
            {
                "stage_name": "Awareness",
                "stage_order": 0,
                "is_tracking_goal": False,
                "tracking_goal_type": None
            },
            {
                "stage_name": "Interest",
                "stage_order": 1,
                "is_tracking_goal": False,
                "tracking_goal_type": None
            },
            {
                "stage_name": "Consideration",
                "stage_order": 2,
                "is_tracking_goal": False,
                "tracking_goal_type": None
            },
            {
                "stage_name": "Application",
                "stage_order": 3,
                "is_tracking_goal": True,
                "tracking_goal_type": TrackingGoalType.APPLICATION
            },
            {
                "stage_name": "Enrollment",
                "stage_order": 4,
                "is_tracking_goal": False,
                "tracking_goal_type": None
            }
        ]

        for stage_data in default_stages:
            stage = FunnelStage(**stage_data)
            db.add(stage)

        db.commit()
        logger.info("Successfully initialized default funnel stages")
    except Exception as e:
        db.rollback()
        logger.error(f"Error initializing funnel stages: {str(e)}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    init_funnel_stages() 