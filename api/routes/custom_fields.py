from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database.session import get_db

router = APIRouter(prefix="/custom-fields", tags=["custom-fields"])

@router.get("")
def list_custom_fields(db: Session = Depends(get_db)):
    """List all custom fields (stub)."""
    return {"custom_fields": []} 