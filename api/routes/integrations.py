from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database.session import get_db

router = APIRouter(prefix="/integrations", tags=["integrations"])

@router.get("")
def list_integrations(db: Session = Depends(get_db)):
    """List all integrations (stub)."""
    return {"integrations": []} 