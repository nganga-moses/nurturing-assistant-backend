from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database.session import get_db

router = APIRouter(prefix="/engagement-types", tags=["engagement-types"])

@router.get("")
def list_engagement_types(db: Session = Depends(get_db)):
    """List all engagement types (stub)."""
    return {"engagement_types": []} 